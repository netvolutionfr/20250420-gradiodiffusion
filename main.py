from model import load_model, generate_image
import gradio as gr

# Charger le pipeline
pipe = load_model()

# Interface Gradio
demo = gr.Interface(
    fn=lambda prompt: generate_image(pipe, prompt),
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion App (Local GPU)"
)

if __name__ == "__main__":
    demo.launch(share=True)
