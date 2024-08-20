import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pyannote.audio import Pipeline
import gradio as gr
import numpy as np



device = "cuda" #if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 #if torch.cuda.is_available() else torch.float32

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
#sample = dataset[0]["audio"]

#result = pipe("teste1.wav")


#print(result["text"])

#result2 = pipe("teste1.wav", return_timestamps=True)

#print(result2["chunks"])


## bloco mono


## bloco stereo



## criando gradio

def transcricao(audio):
    result3 = pipe(audio)
    return (result3["text"])

demo = gr.Interface(
    title = 'Trabalho faculdade implementando gradio com whisperv3-large',
    fn=transcricao,
    inputs=gr.,
    outputs=["label","textbox"]
    )

demo.launch()
