import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class Diffusion(nn.Module):
    def __init__(self, device, repo_id):
        super().__init__()

        self.device = device

        self.pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_attention_slicing()
        self.pipeline.to(device)

        self.unet = self.pipeline.unet
        self.unet.requires_grad_(False)
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.transform = None

    def get_text_embedding(self, prompt, negative_prompt=""):
        with torch.no_grad():
            text_embedding = self.pipeline._encode_prompt(prompt=prompt,
                                                          device=self.device,
                                                          num_images_per_prompt=1,
                                                          do_classifier_free_guidance=True,
                                                          negative_prompt=negative_prompt)
        return text_embedding

    def process_image(self, image):
        return image

    def step(self, text_embeddings, image, guidance_scale=100,
             min_ratio=0.02, max_ratio=0.98,
             weight_type='constant'):
        detached_image = image.detach()
        detached_image.requires_grad = True
        if self.transform is not None:
            latent = self.transform(detached_image)
        latent = self.process_image(latent)

        min_step = int(self.num_train_timesteps * min_ratio)
        max_step = int(self.num_train_timesteps * max_ratio)
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        noise = torch.randn_like(latent)
        with torch.no_grad():
            noisy_latent = self.scheduler.add_noise(latent, noise, t)
            noisy_latents = torch.cat([noisy_latent] * 2)
            noise_preds = self.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_preds
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if weight_type == 'linear':
            w = (1 - self.scheduler.alphas_cumprod[t])
        elif weight_type == 'quadratic':
            w = (1 - self.scheduler.alphas_cumprod[t])**2
        else:
            w = 1
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        latent.backward(gradient=grad)

        image_grad = detached_image.grad

        return image_grad
