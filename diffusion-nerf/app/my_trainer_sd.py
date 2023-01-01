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

from utils import spherical_to_cartesian, l2_normalize


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


class MyTrainerSD(BaseTrainer):

    def __init__(self, *args,
                 diffusion=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.diffusion_resolution = self.extra_args["diffusion_resolution"]
        self.camera_distance_range = self.extra_args["camera_distance_range"]
        self.light_distance_range = self.extra_args["light_distance_range"]
        self.azimuth_range = self.extra_args["azimuth_range"]
        self.elevation_range = self.extra_args["elevation_range"]
        self.camera_angle_range = self.extra_args["camera_angle_range"]
        self.focal_length_multiplier_range = self.extra_args["focal_length_multiplier_range"]
        self.camera_offset = self.extra_args["camera_offset"]
        self.camera_up_std = self.extra_args["camera_up_std"]
        self.look_at_std = self.extra_args["look_at_std"]
        self.num_novel_views_per_gt = self.extra_args["num_novel_views_per_gt"]
        self.num_novel_views_base = self.extra_args["num_novel_views_base"]
        self.bg_color = self.extra_args["bg_color"] if self.extra_args["bg_color"] != 'noise' else 'black'
        # immutable
        self.ray_grid = generate_centered_pixel_coords(self.diffusion_resolution, self.diffusion_resolution,
                                                       self.diffusion_resolution, self.diffusion_resolution, device='cuda')
        if self.diffusion is not None:
            self.init_text_embeddings()

    def init_text_embeddings(self):
        if self.dataset is not None and self.num_novel_views_per_gt == 0:
            self.text_embeddings = None
            return

        prompt, negative_prompt = self.extra_args["prompt"], self.extra_args["negative_prompt"]

        view_prompts = ['', 'overhead view', 'front view', 'side view', 'back view']
        self.text_embeddings = dict()
        for view_prompt in view_prompts:
            cond_prompt = prompt if view_prompt == '' else prompt + ', ' + view_prompt
            text_embedding = self.diffusion.get_text_embedding(cond_prompt, negative_prompt)
            self.text_embeddings[view_prompt] = text_embedding

    def get_text_embeddings(self, azimuth, elevation):
        if not self.extra_args["use_view_prompt"]:
            return self.text_embeddings['']
        if elevation > 60:
            view_prompt = 'overhead view'
        elif 45 < azimuth < 135:
            view_prompt = 'side view'
        elif azimuth < 225:
            view_prompt = 'back view'
        elif azimuth < 315:
            view_prompt = 'side view'
        else:
            view_prompt = 'front view'

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

    def sample_all_rays(self, camera):
        return generate_pinhole_rays(camera, self.ray_grid).reshape(camera.height, camera.width, 3)

    def prepare_step_gt(self, data):
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)
        mask_gts = data['masks'].to(self.device).squeeze(0)
        return rays, img_gts, mask_gts

    def prepare_step_novel(self):
        azimuth = torch.rand(1) * (self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0]
        elevation = torch.rand(1) * (self.elevation_range[1] - self.elevation_range[0]) + self.elevation_range[0]
        camera_offset = (2 * torch.rand(3) - 1) * self.camera_offset
        camera_distance = torch.rand(1) * (self.camera_distance_range[1] - self.camera_distance_range[0]) + self.camera_distance_range[0]
        camera_angle = torch.rand(1) * (self.camera_angle_range[1] - self.camera_angle_range[0]) + self.camera_angle_range[0]
        focal_length_multiplier = torch.rand(1) * (self.focal_length_multiplier_range[1] - self.focal_length_multiplier_range[0]) + self.focal_length_multiplier_range[0]
        camera_up = torch.tensor([0., 1., 0.]) + torch.randn(3) * self.camera_up_std
        look_at = torch.randn(3) * self.look_at_std

        camera_coords = spherical_to_cartesian(azimuth * torch.pi / 180, elevation * torch.pi / 180, camera_distance)
        camera_coords += camera_offset
        camera_coords = camera_coords
        fov = camera_angle * torch.pi / 180
        focal_length = self.diffusion_resolution * focal_length_multiplier
        camera = Camera.from_args(
            eye=camera_coords,
            at=look_at,
            up=camera_up,
            fov=fov,
            #focal_x=focal_length,
            width=self.diffusion_resolution, height=self.diffusion_resolution,
            near=camera_distance-1, # 1e-2,
            far=camera_distance+1, # 6.0,
            dtype=torch.float32,
            device='cuda'
        )
        rays = self.sample_all_rays(camera)
        rays = rays.reshape(self.diffusion_resolution ** 2, -1)

        light_direction = l2_normalize(l2_normalize(camera_coords) + torch.randn(3))
        light_distance = torch.rand(1) * (self.light_distance_range[1] - self.light_distance_range[0]) + self.light_distance_range[0]
        light = light_direction * light_distance

        if self.total_iterations < self.extra_args["albedo_steps"]:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            rand = random.random()
            if rand < 0.8:
                if rand < 0.4:
                    ambient_ratio = 0.1
                    shading = 'lambertian'
                else:
                    ambient_ratio = 0.05
                    shading = 'textureless'
            else:
                shading = 'albedo'
                ambient_ratio = 1.0

        text_embeddings = self.get_text_embeddings(azimuth, elevation)

        return rays, text_embeddings, shading, ambient_ratio, light

    def step(self, data):
        """Implement the optimization over image-space loss.
        """

        # Map to device
        if data is not None:
            rays, img_gts, mask_gts = self.prepare_step_gt(data)
            kwargs = dict(use_light=False)
        else:
            rays, text_embeddings, shading, ambient_ratio, light = self.prepare_step_novel()
            kwargs = dict(shading=shading, ambient_ratio=ambient_ratio, light=light, use_light=True)

        self.optimizer.zero_grad()

        loss = 0

        if self.extra_args["random_lod"]:
            # Sample from a geometric distribution
            population = [i for i in range(self.pipeline.nef.grid.num_lods)]
            weights = [2 ** i for i in range(self.pipeline.nef.grid.num_lods)]
            weights = [i / sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None

        with torch.cuda.amp.autocast():
            rb = self.pipeline(rays=rays,
                               lod_idx=lod_idx,
                               channels=["rgb",
                                         "predicted_normal_reg",
                                         "orientation_reg",
                                         "normal_pred",
                                         "entropy"],
                               stop_grad_channels=["orientation_reg"],
                               **kwargs)

            if data is not None:
                # RGB Loss
                # rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
                bg = rb.bg[..., :3]
                rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
                rgb_loss = rgb_loss.mean()

                loss += self.extra_args["rgb_loss"] * rgb_loss
                self.log_dict['rgb_loss'] += rgb_loss.item()
            else:
                image = rb.rgb[..., :3]
                image = image.reshape(1, self.diffusion_resolution, self.diffusion_resolution, 3)
                image = image.permute(0, 3, 1, 2).contiguous()
                diffusion_loss = self.diffusion.step(text_embeddings=text_embeddings,
                                                     image=image,
                                                     guidance_scale=self.extra_args["guidance_scale"])
                diffusion_loss = diffusion_loss.mean()
                loss += self.extra_args["diffusion_loss"] * diffusion_loss

            if data is not None or self.total_iterations >= self.extra_args["albedo_steps"]:
                factor = 1
            else:
                factor = 1
            predicted_normal_loss = rb.predicted_normal_reg.mean()
            orientation_loss = rb.orientation_reg.mean()
            loss += factor * self.extra_args["predicted_normal_loss"] * predicted_normal_loss
            loss += factor * self.extra_args["orientation_loss"] * orientation_loss
            opacity_loss = torch.sqrt(rb.opacity_sum ** 2 + 0.01).mean()
            loss += self.extra_args["opacity_loss"] * opacity_loss
            entropy_loss = rb.entropy.mean()
            #entropy_loss = (-rb.opacity_sum * torch.log2(rb.opacity_sum) - (1 - rb.opacity_sum) * torch.log2(1 - rb.opacity_sum)).mean()
            loss += self.extra_args["entropy_loss"] * entropy_loss

        self.log_dict['total_loss'] += loss.item()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

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
