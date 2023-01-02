import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from diffusers import DDIMPipeline

from PIL import Image

from tqdm import tqdm
import os


def denoise_image(model, scheduler, images,
                  frac, num_steps,
                  generator=None,
                  eta=0.0,
                  use_clipped_model_output=None):
    with torch.no_grad():
        noise = torch.randn(images.shape).to(images.device)
        
        # set step values
        scheduler.timesteps = torch.linspace(frac*(scheduler.config.num_train_timesteps-1), 0, num_steps, device=images.device).round().long()
        scheduler.num_inference_steps = len(scheduler.timesteps)
        starts = scheduler.timesteps[0] * torch.ones(images.shape[0], device=images.device).long()
        
        noisy_images = scheduler.add_noise(images, noise, starts)
        
        if generator is None:
            generator = torch.Generator(images.device)
        
        for t in tqdm(scheduler.timesteps):
            model_output = model(noisy_images, t).sample
            noisy_images = scheduler.step(
                model_output, t, noisy_images, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        denoised_images = noisy_images.clamp(-1, 1)

        return denoised_images


def process_image(path, data_transforms):
    image = Image.open(path).convert('RGBA')
    image = transforms.ToTensor()(image)
    image = data_transforms(image)
    return image


def repaint(model, noise_scheduler, image, frac, num_steps):
    images = image.unsqueeze(0).to(model.device)
    r = denoise_image(model, noise_scheduler, images, frac, num_steps)
    return r[0]


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def evaluate(config, epoch, pipeline, num_inference_steps=100):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
        num_inference_steps=num_inference_steps,
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
    
def sample_images(config, model, noise_scheduler, num_inference_steps=200):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
        num_inference_steps=num_inference_steps,
    ).images

    image_grid = make_grid(images, rows=4, cols=4)
    
    return image_grid
    

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Training epoch {epoch}:'):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    push_to_hub(config, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir) 