from diffusers import StableDiffusionPipeline
import torch

base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "models/DarkGhibli_000001500.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # CPU friendly
)
pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

# Load and apply the LoRA
pipe.load_lora_weights(lora_path)
pipe.fuse_lora(lora_scale=0.8)

prompt = "Studio Ghibli Dark Fairytale, a digital painting of a black cat with glowing green eyes in a mystical forest at night"
negative_prompt = "blurry, deformed, distorted, low quality, text, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]

image.save("dark_ghibli_cat.png")
print("✅ Image saved as dark_ghibli_cat.png")
