import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import numpy as np

device = "cuda"  # if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16  # if torch.cuda.is_available() else torch.float32

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

def transcricao(audio):
    # O Gradio passa o áudio como um numpy array (forma de onda) e a taxa de amostragem
    waveform, sample_rate = audio

    # Certifique-se de que a taxa de amostragem seja um único valor e que a forma de onda seja do tipo float32
    if isinstance(sample_rate, (list, np.ndarray)):
        sample_rate = sample_rate[0]
    waveform = waveform.astype(np.float32)

    # Convertendo para o formato esperado
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_values)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

demo = gr.Interface(
    title='Trabalho faculdade implementando gradio com whisperv3-large',
    fn=transcricao,
    inputs="audio",
    outputs="textbox"
)

demo.launch()
