# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid
from wisp.accelstructs import OctreeAS
import kaolin.ops.spc as spc_ops

from wisp.ops.differential.gradients import autodiff_gradient, finitediff_gradient, tetrahedron_gradient

from ide_embedder import get_ide_embedder
from utils import l2_normalize, reflect, compute_mean_angular_error, \
    finitediff_gradient_with_grad, tetrahedron_gradient_with_grad


class MyNeuralRadianceField(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self,
                 grid: BLASGrid = None,
                 # embedder args
                 pos_embedder: str = 'none',
                 view_embedder: str = 'none',
                 pos_multires: int = 10,
                 view_multires: int = 4,
                 position_input: bool = False,
                 # decoder args
                 activation_type: str = 'relu',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 # pruning args
                 prune_density_decay: float = None,
                 prune_min_density: float = None,
                 use_blob: bool = False,
                 blob_scale: float = 5.0,
                 blob_width: float = 0.2,
                 bottleneck_dim: int = 8,
                 **kwargs,
                 ):
        super().__init__()
        self.grid = grid

        # Init Embedders
        self.position_input = position_input
        if self.position_input:
            self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires)
        else:
            self.pos_embedder, self.pos_embed_dim = None, 0
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires)
        self.bg_view_embedder, self.bg_view_embed_dim = self.init_embedder(view_embedder, 2)

        # Init Decoder
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.decoder_spatial, self.decoder_normal, self.decoder_directional, self.decoder_background = \
            self.init_decoders(activation_type, layer_type, num_layers, hidden_dim, bottleneck_dim)
        self.density_activation = torch.nn.Softplus()

        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density

        self.blob_scale = blob_scale
        self.blob_width = blob_width
        self.use_blob = use_blob

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none':
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity':
            embedder, embed_dim = torch.nn.Identity(), 0
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies)
        elif embedder_type == 'spherical':
            embedder, embed_dim = get_ide_embedder(degree=frequencies, use_kappa=False)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim, bottleneck_dim):
        """Initializes the decoder object.
        """

        decoder_spatial = BasicDecoder(input_dim=self.spatial_net_input_dim,
                                       output_dim=7,#+bottleneck_dim,
                                       activation=get_activation_class(activation_type),
                                       bias=True,
                                       layer=get_layer_class(layer_type),
                                       num_layers=num_layers,
                                       hidden_dim=hidden_dim,
                                       skip=[])

        decoder_normal = BasicDecoder(input_dim=self.spatial_net_input_dim,
                                      output_dim=3,
                                      activation=get_activation_class(activation_type),
                                      bias=True,
                                      layer=get_layer_class(layer_type),
                                      num_layers=num_layers,
                                      hidden_dim=hidden_dim,
                                      skip=[])

        decoder_directional = BasicDecoder(input_dim=self.directional_net_input_dim,
                                           output_dim=3,
                                           activation=get_activation_class(activation_type),
                                           bias=True,
                                           layer=get_layer_class(layer_type),
                                           num_layers=num_layers+1,
                                           hidden_dim=hidden_dim,
                                           skip=[])

        decoder_background = BasicDecoder(input_dim=self.background_net_input_dim,
                                          output_dim=3,
                                          activation=get_activation_class(activation_type),
                                          bias=True,
                                          layer=get_layer_class(layer_type),
                                          num_layers=1,
                                          hidden_dim=16,
                                          skip=[])

        return decoder_spatial, decoder_normal, decoder_directional, decoder_background

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            if isinstance(self.grid, HashGrid):
                density_decay = self.prune_density_decay
                min_density = self.prune_min_density

                self.grid.occupancy = self.grid.occupancy.cuda()
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points.cuda()
                res = 2.0 ** self.grid.blas_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                with torch.no_grad():
                    density = self.forward(coords=samples, ray_d=sample_views, channels="density")
                self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]

                mask = self.grid.occupancy > min_density

                _points = points[mask]

                if _points.shape[0] == 0:
                    return

                # TODO (operel): This will soon change to support other blas types
                octree = spc_ops.unbatched_points_to_octree(_points, self.grid.blas_level, sorted=True)
                self.grid.blas = OctreeAS(octree)
            else:
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid}')

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self.extra_channels = ["lambertian", "textureless", "albedo"]
        self._register_forward_function(self.features, ["density", "rgb", "normal", "normal_pred",
                                                        "predicted_normal_reg", "orientation_reg",
                                                        *self.extra_channels])
        self._register_forward_function(self.background, ["background"])

    def density_blob(self, coords, scale, width):
        d2 = (coords ** 2).sum(axis=-1, keepdim=True)
        # density = scale * torch.exp(- d2 / (width ** 2))
        d = torch.sqrt(d2)
        density = scale * (1 - d / width)
        return density

    def density(self, coords, lod_idx=None):
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(-1, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)

        # Decode high-dimensional vectors to density features.
        density_feats = self.decoder_spatial(feats)[..., 0:1]
        if self.use_blob:
            density_feats += self.density_blob(coords, self.blob_scale, self.blob_width)
        density = self.density_activation(density_feats)

        return density

    def background(self, coords, ray_d):
        embedded_view = self.bg_view_embedder(ray_d).view(-1, self.background_net_input_dim)
        color = self.decoder_background(embedded_view)
        color = torch.sigmoid(color)
        return dict(background=color)

    def features(self, coords, ray_d, lod_idx=None, channels=[],
                 ambient_ratio=0.1, shading='lambertian', light=None,
                 use_light=True, phase='coarse'):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
                - Density tensor of shape [batch, 1]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(-1, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)

        # Decode high-dimensional vectors to density features.
        spatial_feats = self.decoder_spatial(feats)

        # Density is [particles / meter], so need to be multiplied by distance
        # density ~ (batch, 1)
        density_feats = spatial_feats[..., 0:1]
        albedo_color_feats = spatial_feats[..., 1:4]
        normal_pred_feats = spatial_feats[..., 4:7]
        # normal_pred_feats = self.decoder_normal(feats)[..., 0:3]
        bottleneck = None #spatial_feats[..., 7:]

        if self.use_blob:
            density_feats += self.density_blob(coords, self.blob_scale, self.blob_width)
        density = self.density_activation(density_feats)
        albedo_color = torch.sigmoid(albedo_color_feats)
        normal_pred = l2_normalize(normal_pred_feats)
        normal_pred = torch.nan_to_num(normal_pred)

        normal = None
        predicted_normal_reg = None
        if "normal" in channels or "predicted_normal_reg" in channels:
            normal = self.normal(coords, lod_idx=lod_idx, method="finitediff with grad")
        if "predicted_normal_reg" in channels:
            predicted_normal_reg = 1 - torch.sum(normal * normal_pred, axis=-1, keepdim=True)
            # predicted_normal_reg = compute_mean_angular_error(normal, normal_pred)
        orientation_reg = None
        if "orientation_reg" in channels:
            orientation_reg = torch.sum(torch.clamp(normal_pred * ray_d, min=0) ** 2, axis=-1, keepdim=True)

        # Colors are values [0, 1] floats
        # colors ~ (batch, 3)

        albedo, lambertian, textureless = None, None, None
        if "albedo" in channels:
            shading = "albedo"
            ambient_ratio = 1.0
        elif "textureless" in channels:
            shading = "textureless"
            ambient_ratio = 0.1
        elif "lambertian" in channels:
            shading = "lambertian"
            ambient_ratio = 0.1

        if use_light:
            if light is None:
                light = 2 * torch.tensor([1., 1., 1.], device=self.device)
            if shading == 'textureless':
                albedo_color = torch.ones_like(albedo_color)
                # albedo_color = albedo_color.detach()
            # light_dir = -l2_normalize(light - coords)
            light_dir = -l2_normalize(light)
            light_diffuse = (-normal_pred * light_dir).sum(axis=-1, keepdim=True).clamp(min=0)
            colors = albedo_color * ((1 - ambient_ratio) * light_diffuse + ambient_ratio)
        else:
            ref_ray_d = reflect(ray_d, normal_pred)
            # Concatenate embedded view directions.
            if self.view_embedder is not None:
                embedded_dir = self.view_embedder(-ref_ray_d).view(-1, self.view_embed_dim)
                fdir = torch.cat([bottleneck, embedded_dir], dim=-1)
            else:
                fdir = spatial_feats
            colors = torch.sigmoid(self.decoder_directional(fdir))

        if "albedo" in channels:
            albedo = colors
        elif "textureless" in channels:
            textureless = colors
        elif "lambertian" in channels:
            lambertian = colors

        if "normal" not in channels:
            normal = None
        if "normal_pred" not in channels:
            normal_pred = None

        return dict(rgb=colors, density=density,
                    normal=normal, normal_pred=normal_pred,
                    predicted_normal_reg=predicted_normal_reg,
                    orientation_reg=orientation_reg,
                    albedo=albedo, lambertian=lambertian, textureless=textureless)

    def normal(self, coords, lod_idx=None, method="finitediff", eps=0.005):
        def f(x):
            return self.density(x, lod_idx=lod_idx)
        if method == "autograd":
            gradient = autodiff_gradient(coords, f)
        elif method == "finitediff":
            gradient = finitediff_gradient(coords, f, eps)
        elif method == "tetrahedron":
            gradient = tetrahedron_gradient(coords, f, eps)
        elif method == "finitediff with grad":
            gradient = finitediff_gradient_with_grad(coords, f, eps)
        elif method == "tetrahedron with grad":
            gradient = tetrahedron_gradient_with_grad(coords, f, eps)
        else:
            raise NotImplementedError(f"Unknown method {method}")
        normal = l2_normalize(-gradient)
        normal = torch.nan_to_num(normal)

        return normal

    @property
    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    @property
    def spatial_net_input_dim(self):
        return self.effective_feature_dim + self.pos_embed_dim

    @property
    def directional_net_input_dim(self):
        return self.bottleneck_dim + self.view_embed_dim

    @property
    def background_net_input_dim(self):
        return self.bg_view_embed_dim
