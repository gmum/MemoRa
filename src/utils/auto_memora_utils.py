import torch

@torch.no_grad()

def pred_cfg(pipe, x, t, guidance_scale, neg_embeds, prompt_embeds):
    e_u = pipe.unet(x, t, encoder_hidden_states=neg_embeds).sample
    e_c = pipe.unet(x, t, encoder_hidden_states=prompt_embeds).sample
    return e_u + guidance_scale * (e_c - e_u)

def sample_auto_memora_cfg(
    pipe_esd, pipe_lora,
    prompt,
    negative_prompt="",
    steps=30, height=512, width=512,
    guidance_scale=7.5,
    w=0.5,
    seed=42,
    main="lora",
):
    device = pipe_esd.device
    dtype  = next(pipe_esd.unet.parameters()).dtype

    sch = pipe_esd.scheduler
    sch.set_timesteps(steps, device=device)
    g = torch.Generator(device=device).manual_seed(seed)

    lat = torch.randn(
        (1, pipe_esd.unet.config.in_channels, height // 8, width // 8),
        device=device, dtype=dtype, generator=g
    )
    lat = lat * sch.init_noise_sigma 


    prompt_embeds, neg_embeds, *_ = pipe_esd.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if neg_embeds.dim() == 2:
        neg_embeds = neg_embeds.unsqueeze(0)

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    neg_embeds    = neg_embeds.to(device=device, dtype=dtype)

    for t in sch.timesteps:
        lat_in = sch.scale_model_input(lat, t)
        pred_esd  = pred_cfg(pipe_esd,  lat_in, t, guidance_scale, neg_embeds, prompt_embeds)
        pred_lora = pred_cfg(pipe_lora, lat_in, t, guidance_scale, neg_embeds, prompt_embeds)

        if main == "lora":
            main_pred, bad_pred = pred_lora, pred_esd
        else:
            main_pred, bad_pred = pred_esd, pred_lora

        pred_hat = bad_pred + w * (main_pred - bad_pred)
        lat = sch.step(model_output=pred_hat, timestep=t, sample=lat).prev_sample

    scale = getattr(pipe_esd.vae.config, "scaling_factor", 0.18215)
    with torch.autocast(device_type="cuda", dtype=dtype):
        img = pipe_esd.vae.decode(lat / scale).sample
    return (img.clamp(-1, 1) + 1) / 2