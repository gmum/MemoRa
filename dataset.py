import os, torch, math
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms as tfms


#settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 
original_model = "CompVis/stable-diffusion-v1-4"
esd_unet = "./ESD/diffusers-nudity-ESDu1-UNET.pt"
prompt = "a photo of the nude person"
num_inference_steps = 50
guidance_scale = 7.5

output_dir = "./dataset"
os.makedirs(output_dir, exist_ok=True)


#original SD with NSFW
pipe_original = StableDiffusionPipeline.from_pretrained(
    original_model, safety_checker=None, torch_dtype=torch_dtype
).to(device)


# ESD
vae = AutoencoderKL.from_pretrained(original_model, subfolder="vae", torch_dtype=torch_dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(original_model, subfolder="unet", torch_dtype=torch_dtype)
unet.load_state_dict(torch.load(esd_unet, map_location="cpu"))
unet = unet.to(device, dtype=torch_dtype)
tokenizer = CLIPTokenizer.from_pretrained(original_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(original_model, subfolder="text_encoder", torch_dtype=torch_dtype).to(device)
scheduler = DDIMScheduler.from_pretrained(original_model, subfolder="scheduler")

pipe_esd = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
).to(device)
pipe_esd = pipe_esd.to(dtype=torch.float16)


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images



## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)





def generate_original(pipe, prompt, seed, negative_prompt="", height=512, width=512):
    g = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height, width=width,
        generator=g,
    )
    return out.images[0]


def encode_to_latent(pipe, image: Image.Image):
    tensor_img = tfms.functional.to_tensor(image).unsqueeze(0)
    tensor_img = tensor_img.to(device, dtype=pipe.vae.dtype) * 2 - 1
    latent = pipe.vae.encode(tensor_img)
    l = 0.18215 * latent.latent_dist.sample()
    return l

def slerp(z1, z2, t):
    z1_flat = z1.reshape(1, -1)
    z2_flat = z2.reshape(1, -1)

    z1_norm = z1_flat / z1_flat.norm(dim=1, keepdim=True)
    z2_norm = z2_flat / z2_flat.norm(dim=1, keepdim=True)

    omega = torch.acos((z1_norm * z2_norm).sum(dim=1))
    so = torch.sin(omega)

    res = (torch.sin((1.0 - t) * omega) / so).unsqueeze(1) * z1_flat + (torch.sin(t * omega) / so).unsqueeze(1) * z2_flat
    return res.view_as(z1)

#first
#seed_a = 1405 #firt photo
#seed_b = 2421 #last photo

#second
seed_a = 1805 #firt photo
seed_b = 2821 #last photo

#third
#seed_a = 1705 #firt photo
#seed_b = 2721 #last photo
K = 10 #interpolation steps = number of images 

# original images
img_a = generate_original(pipe_original, prompt, seed_a)
img_b = generate_original(pipe_original, prompt, seed_b)

#inversion
z_start_a = encode_to_latent(pipe_esd, img_a)
z_start_b = encode_to_latent(pipe_esd, img_b)

inv_a = invert(
    pipe_esd,
    start_latents=z_start_a,
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    device=device
)
inv_b = invert(
    pipe_esd,
    start_latents=z_start_b,
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    device=device
)

# get the last noise
z0_a = inv_a[-1].unsqueeze(0).to(device, dtype=torch_dtype)
z0_b = inv_b[-1].unsqueeze(0).to(device, dtype=torch_dtype)

# generate photos from interpolotion
for i in range(K + 1):
    t = i / K
    z_interp = slerp(z0_a[0], z0_b[0], t)
    z_interp = z_interp.unsqueeze(0)
    z_interp = z_interp.to(device, dtype=torch_dtype)

    imgs = sample(
        pipe=pipe_esd,
        prompt=prompt,
        start_step=0, 
        start_latents=z_interp,            
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        device=device,
    )
    img = imgs[0]
    img.save(os.path.join(output_dir, f"slerp{i:02d}_t{t:.2f}_2.png"))

print("Done!")