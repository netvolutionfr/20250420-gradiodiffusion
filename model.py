from diffusers import StableDiffusionPipeline
import torch

print("CUDA dispo :", torch.cuda.is_available())
print("Nom GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU")
print("Support float16 :", torch.cuda.is_bf16_supported() or torch.cuda.is_fp16_supported())

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def generate_image(pipe, prompt):
    return pipe(prompt).images[0]
