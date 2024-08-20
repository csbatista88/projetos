import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def transcricao(audio_path):

    # Usar o pipeline para transcrever o áudio
    result = pipe(audio_path)
    transcription = result["text"]

    return transcription

with gr.Blocks() as demo:
    gr.Markdown("## Trabalho faculdade implementando Gradio com Whisper Large v3")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Carregar Áudio", type="filepath")
        #with gr.Column():
            transcription_output = gr.Textbox(label="Transcrição")

    transcribe_button = gr.Button("Transcrever")

    transcribe_button.click(
        fn=transcricao,
        inputs=audio_input,
        outputs=transcription_output
    )
    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(label="Transcrição")
demo.launch()
