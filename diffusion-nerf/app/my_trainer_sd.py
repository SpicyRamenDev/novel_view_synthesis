# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import logging as log
from tqdm import tqdm
import random
import pandas as pd
import torch
from lpips import LPIPS
from torch.utils.tensorboard import SummaryWriter
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays, RenderBuffer

import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

import time

from kaolin.render.camera import Camera
from wisp.ops.raygen.raygen import generate_centered_pixel_coords, generate_pinhole_rays

from utils import sample, spherical_to_cartesian, l2_normalize, sample_polar, sample_spherical_uniform, get_rotation_matrix
import math


class DataLoaderGenerator(object):
    def __init__(self, data_loader, num_novel_views_per_gt, num_novel_views_base):
        self.data_loader = data_loader
        self.num_novel_views_per_gt = num_novel_views_per_gt
        self.num_novel_views_base = num_novel_views_base
        if self.data_loader is None:
            self.length = self.num_novel_views_base
        else:
            self.length = len(self.data_loader) * (self.num_novel_views_per_gt + 1)

    def __iter__(self):
        if self.data_loader is None:
            return [None]*self.num_novel_views_base
        elif self.num_novel_views_per_gt == 0:
            return self.data_loader
        else:
            self.data_loader_iter = iter(self.data_loader)
            self.iteration = 1
            return self

    def __len__(self):
        return self.length

    def __next__(self):
        self.iteration += 1
        if self.iteration > self.length:
            raise StopIteration
        if self.iteration - 2 % (self.num_novel_views_per_gt + 1) == 0:
            return next(self.data_loader_iter)
        else:
            return None


from PIL import Image
import numpy as np

class MyTrainerSD(BaseTrainer):

    def __init__(self, *args,
                 diffusion=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.coarse_resolution = self.extra_args["coarse_resolution"]
        self.fine_resolution = self.extra_args["fine_resolution"]
        self.num_novel_views_per_gt = self.extra_args["num_novel_views_per_gt"]
        self.num_novel_views_base = self.extra_args["num_novel_views_base"]
        self.bg_color = self.extra_args["bg_color"] if self.extra_args["bg_color"] != 'noise' else 'black'
        self.warmup_iterations = self.extra_args["warmup_iterations"]
        self.init_lr = self.extra_args["init_lr"]
        self.end_lr = self.extra_args["end_lr"]
        self.reg_warmup_iterations = self.extra_args["reg_warmup_iterations"]
        self.reg_init_lr = self.extra_args["reg_init_lr"]
        # immutable
        self.ray_grid = dict(
            coarse=generate_centered_pixel_coords(self.coarse_resolution, self.coarse_resolution,
                                                  self.coarse_resolution, self.coarse_resolution,
                                                  device='cuda'),
            fine=generate_centered_pixel_coords(self.fine_resolution, self.fine_resolution,
                                                self.fine_resolution, self.fine_resolution,
                                                device='cuda')
        )
        if self.diffusion is not None:
            self.init_text_embeddings()

        def scheduler_function(epoch):
            iteration = epoch * self.iterations_per_epoch
            if iteration < self.warmup_iterations:
                return self.init_lr + (1 - self.init_lr) * iteration / self.warmup_iterations
            else:
                t = (iteration - self.warmup_iterations) / (self.max_iterations - self.warmup_iterations)
                return self.end_lr + 0.5 * (1 - self.end_lr) * (1 + math.cos(t * math.pi))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_function)

    def init_optimizer(self):
        """Default initialization for the optimizer.
        """

        params_dict = {name: param for name, param in self.pipeline.nef.named_parameters()}

        params = []
        decoder_params = []
        decoder_bg_params = []
        grid_params = []
        rest_params = []

        for name in params_dict:
            if name == 'decoder_background':
                decoder_bg_params.append(params_dict[name])

            elif 'decoder' in name:
                # If "decoder" is in the name, there's a good chance it is in fact a decoder,
                # so use weight_decay
                decoder_params.append(params_dict[name])

            elif 'grid' in name:
                # If "grid" is in the name, there's a good chance it is in fact a grid,
                # so use grid_lr_weight
                grid_params.append(params_dict[name])

            else:
                rest_params.append(params_dict[name])

        params.append({"params": decoder_bg_params,
                       "lr": self.lr * 0.1,
                       "weight_decay": self.weight_decay})

        params.append({"params": decoder_params,
                       "lr": self.lr,
                       "weight_decay": self.weight_decay})

        params.append({"params": grid_params,
                       "lr": self.lr * self.grid_lr_weight})

        params.append({"params": rest_params,
                       "lr": self.lr})

        self.optimizer = self.optim_cls(params, **self.optim_params)

    def init_text_embeddings(self):
        if self.dataset is not None and self.num_novel_views_per_gt == 0:
            self.text_embeddings = None
            return

        prompt, negative_prompt = self.extra_args["prompt"], self.extra_args["negative_prompt"]

        view_prompts = {
            '': ('', ''),
            'overhead': ('overhead view', ''),
            'front': ('front view', 'back view'),
            'side': ('side view', ''),
            'back': ('back view', 'front view')
        }
        self.text_embeddings = dict()
        for view, (positive, negative) in view_prompts.items():
            positive_cond = prompt if positive == '' else prompt + ', ' + positive
            negative_cond = negative_prompt  # if negative == '' else negative_prompt + ', ' + negative
            text_embedding = self.diffusion.get_text_embedding(positive_cond, negative_cond)
            self.text_embeddings[view] = text_embedding

    def get_text_embeddings(self, azimuth, polar):
        if not self.extra_args["use_view_prompt"]:
            return self.text_embeddings['']
        if polar < 30:
            view_prompt = 'overhead'
        elif 0 <= azimuth < 60:
            view_prompt = 'front'
        elif 60 <= azimuth < 180:
            view_prompt = 'side'
        elif 180 <= azimuth < 240:
            view_prompt = 'back'
        else:
            view_prompt = 'side'

        return self.text_embeddings[view_prompt]

    def init_dataloader(self):
        if self.dataset is None:
            self.iterations_per_epoch = self.extra_args["num_novel_views_base"]
            self.train_data_loader = [None] * self.iterations_per_epoch
        else:
            super().init_dataloader()
            if self.extra_args["num_novel_views_per_gt"] > 0:
                self.iterations_per_epoch *= self.extra_args["num_novel_views_per_gt"] + 1
                self.train_data_loader = DataLoaderGenerator(self.train_data_loader,
                                                             self.extra_args["num_novel_views_per_gt"],
                                                             self.extra_args["num_novel_views_base"])

    def pre_step(self):
        """Override pre_step to support pruning.
        """
        super().pre_step()

        if self.extra_args["prune_every"] > -1 and self.iteration > 0 and self.iteration % self.extra_args[
            "prune_every"] == 0:
            self.pipeline.nef.prune()

    def init_log_dict(self):
        """Custom log dict.
        """
        super().init_log_dict()
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['predicted_normal_loss'] = 0.0

    def sample_all_rays(self, camera, phase="fine"):
        return generate_pinhole_rays(camera, self.ray_grid[phase]).reshape(camera.height, camera.width, 3)

    def step_gt(self, data, phase="fine"):
        if self.total_iterations < 2000:
            lod_idx = None
        else:
            lod_idx = 8

        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        loss = 0

        with torch.cuda.amp.autocast():
            rb = self.pipeline(rays=rays,
                               lod_idx=lod_idx,
                               channels=["rgb",
                                         "predicted_normal_reg",
                                         "orientation_reg",
                                         "normal_pred"],
                               stop_grad_channels=["orientation_reg", "predicted_normal_reg"],
                               phase=phase,
                               use_light=False)

            # RGB Loss
            # rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            rgb_loss = rgb_loss.sum(-1)
            rgb_loss = rgb_loss.sum()

            loss += self.extra_args["rgb_loss"] * rgb_loss
            self.log_dict['rgb_loss'] += rgb_loss.item()

            predicted_normal_loss = rb.predicted_normal_reg.sum()
            orientation_loss = rb.orientation_reg.sum()
            loss += self.extra_args["predicted_normal_loss"] * predicted_normal_loss
            loss += self.extra_args["orientation_loss"] * orientation_loss
            opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).sum()
            loss += self.extra_args["opacity_loss"] * opacity_loss
            entropy_loss = rb.entropy.sum()
            alphas = rb.alpha.clamp(1e-5, 1 - 1e-5)
            alpha_loss = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).sum()
            loss += self.extra_args["entropy_loss"] * entropy_loss

        return loss

    def prepare_novel_view(self, phase="fine"):
        camera_dir, azimuth, polar = sample_spherical_uniform(azimuth_range=self.extra_args["azimuth_range"],
                                                              polar_range=self.extra_args["polar_range"])
        camera_offset = (2 * torch.rand(3) - 1) * self.extra_args["camera_offset"]
        camera_distance = sample(self.extra_args["camera_distance_range"])
        camera_angle = sample(self.extra_args["camera_angle_range"])
        focal_length_multiplier = sample(self.extra_args["focal_length_multiplier_range"])
        camera_up = torch.tensor([0., 1., 0.]) + torch.randn(3) * self.extra_args["camera_up_std"]
        look_at = torch.randn(3) * self.extra_args["look_at_std"]

        if phase == "fine":
            camera_distance *= 2
            focal_length_multiplier *= 1.
        if False and self.total_iterations < 1000:
            focal_length_multiplier = 1.0

        resolution = self.extra_args[f"{phase}_resolution"]
        camera_coords = camera_dir * camera_distance + camera_offset
        fov = camera_angle * torch.pi / 180
        focal_length = 0.5 * resolution * (camera_distance - 0.) * focal_length_multiplier
        # focal_length = resolution * focal_length_multiplier
        camera = Camera.from_args(
            eye=camera_coords,
            at=look_at,
            up=camera_up,
            # fov=fov,
            focal_x=focal_length,
            width=resolution, height=resolution,
            near=max(camera_distance-2, 0.0),  # 1e-2,
            far=camera_distance+2,  # 6.0,
            dtype=torch.float32,
            device='cuda'
        )

        light_dir, light_azimuth, light_polar = sample_spherical_uniform(polar_range=(30, 90))
        light_rot = get_rotation_matrix(light_azimuth, light_polar)
        light_dir = light_rot @ camera_dir
        light_distance = sample(self.extra_args["light_distance_range"])
        light = light_dir * light_distance

        text_embeddings = self.get_text_embeddings(azimuth, polar)

        rays = self.sample_all_rays(camera, phase)
        rays = rays.reshape(resolution ** 2, -1)

        return dict(rays=rays, camera=camera, light=light, text_embeddings=text_embeddings)

    def sample_rays_gaussian(self, rays, num_samples):
        idx = torch.multinomial(self.pgrid, num_samples)
        output = rays[idx].contiguous()
        return output

    def get_novel_view_render_parameters(self, phase="fine"):
        if self.total_iterations < self.extra_args["albedo_steps"]:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            rand = random.random()
            if rand < 0.75:
                if True or rand < 0.375:
                    ambient_ratio = 0.1
                    shading = 'lambertian'
                else:
                    ambient_ratio = 0.1
                    shading = 'textureless'
            else:
                shading = 'albedo'
                ambient_ratio = 1.0
        bg_color_value = torch.rand(3, device='cuda')
        return dict(shading=shading, ambient_ratio=ambient_ratio, bg_color_value=bg_color_value)

    def get_diffusion_parameters(self, phase="fine"):
        if phase == 'coarse':
            weight_type = 'constant'
            min_ratio = 0.02
            max_ratio = 0.98
        else:
            weight_type = 'quadratic'
            min_ratio = 0.02
            max_ratio = 0.60
        guidance_scale=self.extra_args["guidance_scale"]

        return dict(weight_type=weight_type, min_ratio=min_ratio, max_ratio=max_ratio, guidance_scale=guidance_scale)

    def step_novel_view(self, phase="fine"):
        if "phase" == "fine":
            lod_idx = None
        else:
            lod_idx = 8

        render_parameters = self.get_novel_view_render_parameters(phase)
        diffusion_parameters = self.get_diffusion_parameters(phase)

        scene = self.prepare_novel_view(phase=phase)
        rays = scene['rays']
        light = scene['light']
        camera = scene['camera']
        text_embeddings = scene['text_embeddings']

        nerf_parameters = dict(lod_idx=lod_idx,
                               channels=["rgb",
                                         "predicted_normal_reg",
                                         "orientation_reg"],
                               stop_grad_channels=["orientation_reg", "predicted_normal_reg"],
                               compute_entropy=True,
                               phase=phase,
                               use_light=True,
                               light=light,
                               bg_color='decoder',
                               **render_parameters)
        render_only_nerf_parameters = dict(lod_idx=lod_idx,
                                           channels=["rgb"],
                                           phase=phase,
                                           use_light=True,
                                           light=light,
                                           bg_color='decoder',
                                           **render_parameters)

        diffusion_parameters = dict(text_embeddings=text_embeddings, **diffusion_parameters)

        total_loss_value = 0

        if self.total_iterations >= self.reg_warmup_iterations:
            reg_factor = 1
        else:
            reg_factor = self.reg_init_lr + (1 - self.reg_init_lr) * self.total_iterations / self.reg_warmup_iterations

        if 0 < self.extra_args["render_batch"] < rays.shape[0]:
            image_batches = []
            with torch.no_grad():
                for ray_pack in rays.split(self.extra_args["render_batch"]):
                    rb = self.pipeline(rays=ray_pack, **render_only_nerf_parameters)
                    image_batches.append(rb.rgb[..., :3])
                    del rb
                image = torch.cat(image_batches, dim=0)
                image = image.reshape(1, camera.width, camera.height, 3)
                image = image.permute(0, 3, 1, 2).contiguous()
            with torch.cuda.amp.autocast():
                diffusion_grad = self.diffusion.step(image=image, **diffusion_parameters)
                diffusion_grad = diffusion_grad.reshape(3, -1).contiguous()

                for i, ray_pack in enumerate(rays.split(self.extra_args["render_batch"])):
                    pack_loss = 0

                    rb = self.pipeline(rays=ray_pack, **nerf_parameters)
                    pack_image = rb.rgb[..., :3]
                    pack_image = pack_image.reshape(self.extra_args["render_batch"], 3)
                    pack_image = pack_image.permute(1, 0).contiguous()
                    start, end = i * self.extra_args["render_batch"], (i + 1) * self.extra_args["render_batch"]
                    pack_diffusion_grad = diffusion_grad[..., start:end]
                    pack_diffusion_loss = (pack_diffusion_grad * pack_image).sum(1)
                    pack_diffusion_loss = pack_diffusion_loss.sum() / (64 * 64)
                    pack_loss += self.extra_args["diffusion_loss"] * pack_diffusion_loss

                    predicted_normal_loss = rb.predicted_normal_reg.mean()
                    orientation_loss = rb.orientation_reg.mean()
                    opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).mean()
                    entropy_loss = rb.entropy.mean()
                    pack_loss += reg_factor * self.extra_args["predicted_normal_loss"] * predicted_normal_loss
                    pack_loss += reg_factor * self.extra_args["orientation_loss"] * orientation_loss
                    pack_loss += self.extra_args["opacity_loss"] * opacity_loss
                    pack_loss += self.extra_args["entropy_loss"] * entropy_loss

                    self.scaler.scale(pack_loss).backward()
                    total_loss_value += pack_loss.item()

                    del rb
        else:
            loss = 0

            with torch.cuda.amp.autocast():
                rb = self.pipeline(rays=rays, **nerf_parameters)
                image = rb.rgb[..., :3]
                image = image.reshape(1, camera.width, camera.height, 3)
                image = image.permute(0, 3, 1, 2).contiguous()
                diffusion_grad = self.diffusion.step(image=image, **diffusion_parameters)
                diffusion_loss = (diffusion_grad * image).sum(1)
                diffusion_loss = diffusion_loss.sum() / (64 * 64)
                loss += self.extra_args["diffusion_loss"] * diffusion_loss

                predicted_normal_loss = rb.predicted_normal_reg.mean()
                orientation_loss = rb.orientation_reg.mean()
                opacity_loss = torch.sqrt(rb.alpha ** 2 + 0.01).mean()
                entropy_loss = rb.entropy.mean()
                loss += reg_factor * self.extra_args["predicted_normal_loss"] * predicted_normal_loss
                loss += reg_factor * self.extra_args["orientation_loss"] * orientation_loss
                loss += self.extra_args["opacity_loss"] * opacity_loss
                loss += self.extra_args["entropy_loss"] * entropy_loss

                self.scaler.scale(loss).backward()
                total_loss_value += loss.item()

        print("Iteration:", self.total_iterations,
              "Learning rate:", self.optimizer.param_groups[0]['lr'],
              "Diffusion loss: ", total_loss_value)
        # alphas = rb.alpha.clamp(1e-5, 1 - 1e-5)
        # alpha_loss = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
        #
        if self.total_iterations % 5 == 0:
            image = diffusion_grad.reshape(1, 3, camera.width, camera.height).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).contiguous()
            pil_image = Image.fromarray((image[0].detach().cpu().numpy() * 255).astype(np.uint8))
            pil_image.save("image.png")

        return total_loss_value

    def step(self, data):
        """Implement the optimization over image-space loss.
        """

        if self.total_iterations < self.max_iterations * self.extra_args["coarse_ratio"]:
            phase = "coarse"
        else:
            phase = "fine"

        self.optimizer.zero_grad()

        if data is not None:
            loss_item = self.step_gt(data, phase=phase)
        else:
            loss_item = self.step_novel_view(phase=phase)

        self.log_dict['total_loss'] += loss_item
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step(self.epoch - 1 + (self.iteration - 1) / self.iterations_per_epoch)

    def log_cli(self):
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'] / len(self.train_data_loader))

        log.info(log_text)

    def evaluate_metrics(self, rays, imgs, lod_idx, name=None):

        ray_os = list(rays.origins)
        ray_ds = list(rays.dirs)
        lpips_model = LPIPS(net='vgg').cuda()

        psnr_total = 0.0
        lpips_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for idx, (img, ray_o, ray_d) in tqdm(enumerate(zip(imgs, ray_os, ray_ds))):

                rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                rays = rays.reshape(-1, 3)
                rays = rays.to('cuda')
                rb = self.renderer.render(self.pipeline, rays, lod_idx=lod_idx)
                rb = rb.reshape(*img.shape[:2], -1)

                gts = img.cuda()
                psnr_total += psnr(rb.rgb[..., :3], gts[..., :3])
                lpips_total += lpips(rb.rgb[..., :3], gts[..., :3], lpips_model)
                ssim_total += ssim(rb.rgb[..., :3], gts[..., :3])

                out_rb = RenderBuffer(rgb=rb.rgb, depth=rb.depth, alpha=rb.alpha,
                                      gts=gts, err=(gts[..., :3] - rb.rgb[..., :3]) ** 2)
                exrdict = out_rb.reshape(*img.shape[:2], -1).cpu().exr_dict()

                out_name = f"{idx}"
                if name is not None:
                    out_name += "-" + name

                write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
                write_png(os.path.join(self.valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb.numpy())

        psnr_total /= len(imgs)
        lpips_total /= len(imgs)
        ssim_total /= len(imgs)

        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} LPIPS", lpips_total)
        log.info(log_text)

        return {"psnr": psnr_total, "lpips": lpips_total, "ssim": ssim_total}

    def render_final_view(self, num_angles, camera_distance):
        angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
        x = -camera_distance * np.sin(angles)
        y = self.extra_args["camera_origin"][1]
        z = -camera_distance * np.cos(angles)
        for d in range(self.extra_args["num_lods"]):
            out_rgb = []
            for idx in tqdm(range(num_angles + 1), desc=f"Generating 360 Degree of View for LOD {d}"):
                log_metric_to_wandb(f"LOD-{d}-360-Degree-Scene/step", idx, step=idx)
                out = self.renderer.shade_images(
                    self.pipeline,
                    f=[x[idx], y, z[idx]],
                    t=self.extra_args["camera_lookat"],
                    fov=self.extra_args["camera_fov"],
                    lod_idx=d,
                    camera_clamp=self.extra_args["camera_clamp"]
                )
                out = out.image().byte().numpy_dict()
                if out.get('rgb') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGB", out['rgb'].T, idx)
                    out_rgb.append(Image.fromarray(np.moveaxis(out['rgb'].T, 0, -1)))
                if out.get('rgba') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGBA", out['rgba'].T, idx)
                if out.get('depth') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Depth", out['depth'].T, idx)
                if out.get('normal') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Normal", out['normal'].T, idx)
                if out.get('alpha') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Alpha", out['alpha'].T, idx)
                wandb.log({})

            rgb_gif = out_rgb[0]
            gif_path = os.path.join(self.log_dir, "rgb.gif")
            rgb_gif.save(gif_path, save_all=True, append_images=out_rgb[1:], optimize=False, loop=0)
            wandb.log({f"360-Degree-Scene/RGB-Rendering/LOD-{d}": wandb.Video(gif_path)})

    def validate(self):
        return 
        self.pipeline.eval()

        # record_dict contains trainer args, but omits torch.Tensor fields which were not explicitly converted to
        # numpy or some other format. This is required as parquet doesn't support torch.Tensors
        # (and also for output size considerations)
        record_dict = {k: v for k, v in self.extra_args.items() if not isinstance(v, torch.Tensor)}
        dataset_name = os.path.splitext(os.path.basename(self.extra_args['dataset_path']))[0]
        model_fname = os.path.abspath(os.path.join(self.log_dir, f'model.pth'))
        record_dict.update({"dataset_name": dataset_name, "epoch": self.epoch,
                            "log_fname": self.log_fname, "model_fname": model_fname})
        parent_log_dir = os.path.dirname(self.log_dir)

        log.info("Beginning validation...")

        validation_split = self.extra_args.get('valid_split', 'val')
        data = self.dataset.get_images(split=validation_split, mip=self.extra_args['mip'])
        imgs = list(data["imgs"])

        img_shape = imgs[0].shape
        log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        lods = list(range(self.pipeline.nef.grid.num_lods))
        evaluation_results = self.evaluate_metrics(data["rays"], imgs, lods[-1], f"lod{lods[-1]}")
        record_dict.update(evaluation_results)
        if self.using_wandb:
            log_metric_to_wandb("Validation/psnr", evaluation_results['psnr'], self.epoch)
            log_metric_to_wandb("Validation/lpips", evaluation_results['lpips'], self.epoch)
            log_metric_to_wandb("Validation/ssim", evaluation_results['ssim'], self.epoch)

        df = pd.DataFrame.from_records([record_dict])
        df['lod'] = lods[-1]
        fname = os.path.join(parent_log_dir, f"logs.parquet")
        if os.path.exists(fname):
            df_ = pd.read_parquet(fname)
            df = pd.concat([df_, df])
        df.to_parquet(fname, index=False)

    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        # Default TensorBoard Logging
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Info', self.info)

        if self.using_wandb:
            wandb_project = self.extra_args["wandb_project"]
            wandb_run_name = self.extra_args.get("wandb_run_name")
            wandb_entity = self.extra_args.get("wandb_entity")
            wandb.init(
                project=wandb_project,
                name=self.exp_name if wandb_run_name is None else wandb_run_name,
                entity=wandb_entity,
                job_type=self.trainer_mode,
                config=self.extra_args,
                sync_tensorboard=True
            )

            for d in range(self.extra_args["num_lods"]):
                wandb.define_metric(f"LOD-{d}-360-Degree-Scene")
                wandb.define_metric(
                    f"LOD-{d}-360-Degree-Scene",
                    step_metric=f"LOD-{d}-360-Degree-Scene/step"
                )

    def post_training(self):
        """
        Override this function to change the logic which runs after the last training iteration.
        This function runs once after training ends.
        """
        self.writer.close()
        wandb_viz_nerf_angles = self.extra_args.get("wandb_viz_nerf_angles", 0)
        wandb_viz_nerf_distance = self.extra_args.get("wandb_viz_nerf_distance")
        if self.using_wandb and wandb_viz_nerf_angles != 0:
            self.render_final_view(
                num_angles=wandb_viz_nerf_angles,
                camera_distance=wandb_viz_nerf_distance
            )
            wandb.finish()
