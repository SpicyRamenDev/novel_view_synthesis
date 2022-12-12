import torch

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
        
        if generator=None:
            generator = torch.Generator(images.device)
        
        for t in tqdm(scheduler.timesteps):
            model_output = model(noisy_images, t).sample
            noisy_images = scheduler.step(
                model_output, t, noisy_images, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        denoised_images = noisy_images.clamp(-1, 1)

        return denoised_images