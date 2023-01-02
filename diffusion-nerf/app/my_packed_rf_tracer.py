# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.tracers import BaseTracer


class MyPackedRFTracer(BaseTracer):
    """Tracer class for sparse (packed) radiance fields.
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is differentiable, and can be employed within training loops.

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """

    def __init__(self, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white', **kwargs):
        """Set the default trace() arguments.

        Args:
            raymarch_type (str): Sample generation strategy to use for raymarch.
                'voxel' - intersects the rays with the acceleration structure cells.
                    Then among the intersected cells, each cell is sampled `num_steps` times.
                'ray' - samples `num_steps` along each ray, and then filters out samples which falls outside of occupied
                    cells of the acceleration structure.
            num_steps (int): The number of steps to use for the sampling. The meaning of this parameter changes
                depending on `raymarch_type`:
                'voxel' - each acceleration structure cell which intersects a ray is sampled `num_steps` times.
                'ray' - number of samples generated per ray, before culling away samples which don't fall
                    within occupied cells.
                The exact number of samples generated, therefore, depends on this parameter but also the occupancy
                status of the acceleration structure.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use.
        """
        super().__init__()
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.bg_color = bg_color

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Returns:
            (set): Set of channel strings.
        """
        return {"depth", "hit", "rgb", "alpha", "bg", "opacity_sum", "entropy"}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.

        Returns:
            (set): Set of channel strings.
        """
        return {"rgb", "density"}

    def trace(self, nef, channels, extra_channels, rays,
              lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white',
              stop_grad_channels=[], entropy_threshold=0.01, **kwargs):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  perform volumetric integration on those channels.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at.
            raymarch_type (str): The type of raymarching algorithm to use. Currently we support:
                                 voxel: Finds num_steps # of samples per intersected voxel
                                 ray: Finds num_steps # of samples per ray, and filters them by intersected samples
            num_steps (int): The number of steps to use for the sampling.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use. TODO(ttakikawa): Might be able to simplify / remove

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        # TODO(ttakikawa): Use a more robust method
        assert nef.grid is not None and "this tracer requires a grid"

        N = rays.origins.shape[0]

        if "depth" in channels:
            depth = torch.zeros(N, 1, device=rays.origins.device)
        else:
            depth = None

        if bg_color == 'white':
            bg = torch.ones(N, 3, device=rays.origins.device)
        elif bg_color == 'noise':
            bg = torch.random() * torch.ones(N, 3, device=rays.origins.device)
        else:
            bg = torch.zeros(N, 3, device=rays.origins.device)
        bg.requires_grad = False
        rgb = bg.clone().detach()
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)

        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1

        # By default, PackedRFTracer will attempt to use the highest level of detail for the ray sampling.
        # This however may not actually do anything; the ray sampling behaviours are often single-LOD
        # and is governed by however the underlying feature grid class uses the BLAS to implement the sampling.
        raymarch_results = nef.grid.raymarch(rays,
                                             level=nef.grid.active_lods[lod_idx],
                                             num_samples=num_steps,
                                             raymarch_type=raymarch_type)
        ridx, samples, depths, deltas, boundary = raymarch_results

        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
        # Compute the color and density for each ray and their samples
        hit_ray_d = rays.dirs.index_select(0, ridx)

        # Compute the color and density for each ray and their samples
        queried_channels = {"rgb", "density"}.union(extra_channels)
        queried_features = nef.features(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=queried_channels)
        color = queried_features["rgb"]
        density = queried_features["density"]
        density = density.reshape(-1, 1)  # Protect against squeezed return shape
        del ridx

        # Compute optical thickness
        tau = density * deltas
        del density, deltas
        ray_colors, transmittance = spc_render.exponential_integration(color, tau, boundary, exclusive=True)

        if "depth" in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(-1, 1) * transmittance, boundary)
            depth[ridx_hit, :] = ray_depth

        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[..., 0] > 0.0

        # Populate the background
        if bg_color == 'white':
            color = torch.clamp((1.0 - alpha) + ray_colors, max=1.0)
        elif bg_color == 'noise':
            color = (1.0 - alpha) * bg + alpha * ray_colors
        else:
            color = alpha * ray_colors
        rgb[ridx_hit] = color

        extra_outputs = {}
        for channel in extra_channels:
            feats = queried_features[channel]
            num_channels = feats.shape[-1]
            if channel in stop_grad_channels:
                ray_feats, transmittance = spc_render.exponential_integration(
                    feats.view(-1, num_channels), tau.detach(), boundary, exclusive=True
                )
                composited_feats = alpha.detach() * ray_feats
            else:
                ray_feats, transmittance = spc_render.exponential_integration(
                    feats.view(-1, num_channels), tau, boundary, exclusive=True
                )
                composited_feats = alpha * ray_feats
            out_feats = torch.zeros(N, num_channels, device=feats.device)
            out_feats[ridx_hit] = composited_feats
            extra_outputs[channel] = out_feats

        opacity = 1 - torch.exp(-tau)

        opacity_sum = spc_render.sum_reduce(opacity, boundary)
        out_opacity_sum = torch.zeros(N, 1, device=opacity_sum.device)
        out_opacity_sum[ridx_hit] = opacity_sum

        opacity_xlogx_sum = spc_render.sum_reduce(opacity * torch.log(opacity + 1e-10), boundary)
        entropy_ray = -opacity_xlogx_sum / (opacity_sum + 1e-10) + torch.log(opacity_sum + 1e-10)
        mask = (opacity_sum > 1e-6).detach()
        entropy_ray *= mask
        out_entropy = torch.zeros(N, 1, device=entropy_ray.device)
        out_entropy[ridx_hit] = entropy_ray

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha, bg=bg,
                            opacity_sum=out_opacity_sum, entropy=out_entropy,
                            **extra_outputs)
