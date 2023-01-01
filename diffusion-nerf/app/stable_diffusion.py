import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion import Diffusion


class StableDiffusion(Diffusion):
    def __init__(self, device, repo_id="stabilityai/stable-diffusion-2-1-base"):
        super().__init__(device, repo_id)

        self.vae = self.pipeline.vae
        self.vae.requires_grad_(False)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Normalize([0.5], [0.5]),
            ])

    def process_image(self, image):
        posterior = self.vae.encode(image).latent_dist
        latent = posterior.sample() * 0.18215
        return latent