import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pyannote.audio import Pipeline
import gradio as gr
import numpy as np
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
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")


def transcricao(uploaded_audio):
    """
    Handles uploaded audio files, ensuring proper loading and transcription.

    Args:
        uploaded_audio (gradio.Upload): The uploaded audio file from the Gradio interface.

    Returns:
        list: A list containing the transcribed text.
    """

    # Validate the uploaded audio file
    if not uploaded_audio:
        return ["No audio uploaded."]

    # Handle potential file path issues
    audio_path = uploaded_audio.name  # Use the uploaded file's name

    try:
        # Load the audio from the local directory (assuming security is addressed)
        audio, sample_rate = librosa.load(audio_path, sr=None)  # Use librosa for audio loading
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error loading audio: {e}")
        return ["Error: Could not access uploaded audio."]

    # Perform pre-processing (if required) based on your model's needs

    # Transcribe the audio
    result = pipe(audio, sampling_rate=sample_rate)

    return [result["text"]]  # Return the transcribed text

demo = gr.Interface(
    title="Trabalho faculdade implementando gradio com whisperv3-large",
    fn=transcricao,
    inputs=gr.Audio(sources=["upload"]),  # Allow audio upload
    outputs="text",
)

if __name__ == "__main__":
    # Address security concerns by ensuring the temporary directory is writable
    # and consider using a dedicated directory outside the user's profile.
    # For this example, we'll assume a secure temporary directory is available.
    demo.launch()
