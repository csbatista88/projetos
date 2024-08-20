import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Configuração do modelo Whisper
device = "cuda"
torch_dtype = torch.float16

#model
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


def transcricao(audio_input):
    result3 = pipe(audio_input)
    transcription = result3["text"]
    return transcription

def transcricao_chunck(audio_input):
    result2 = pipe(audio_input, return_timestamps=True)
    transcription_chunck = result2["chunks"]
    return transcription_chunck


# Pipeline para diarização



def transcribe_diarization(audio_input):
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="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")
    diarization = diarization_pipeline(audio_input, num_speakers=2)
    # uma lista para armazenar os resultados
    results = []

    # Imprimindo os segmentos de fala com os respectivos identificadores de falantes
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append(f"Speaker {speaker}: {turn.start:.1f}s to {turn.end:.1f}s")
    return results

def transcricao_segmentation(audio_input):
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="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")
    diarization = diarization_pipeline(audio_input, num_speakers=2)
    audio = AudioSegment.from_wav(audio_input)

    segmentation_audio = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = audio[turn.start * 1000:turn.end * 1000]
        segment.export(f"segment_{speaker}.wav", format="wav")

        # transcrevendo segmentos
        result = pipe(f"segment_{speaker}.wav")
        segmentation_audio.append(f"Speaker {speaker}: {result['text']}")
    return segmentation_audio


def transcricao_combinada(audio_input):
    # Chamando a primeira funcao
    transcription = transcricao(audio_input)

    transcription_chunck = transcricao_chunck(audio_input)

    transcription_segmentation = transcricao_segmentation(audio_input)

    # Retornando uma tupla com os resultados
    return transcription, transcription_chunck, transcription_segmentation


with gr.Blocks() as demo:
    gr.Markdown("# projeto da faculdade. Machine learning.")
    with gr.Column():
        audio_input = gr.Audio(type="filepath", label="Upload Audio File")
        with gr.Row():
            type_audio = gr.Radio(["mono", "stereo"], label="Tipo do audio", info="Escolha antes de transcrever!")
            transcribe_button = gr.Button("transcrever")
    with gr.Row():
        transcription_output = gr.Textbox(label="Transcrição")#, lines=10)
    with gr.Row():
        transcription_result = gr.Textbox(label="Transcrição com Timestamps")#, lines=10)
    with gr.Row():
        transcription_dialization = gr.Textbox(label="Diálogo")#, lines=10)
    with gr.Row():
        download_transcription = gr.Button("Download Transcription")
        download_chunks = gr.Button("Download Chunks")

    transcribe_button.click(
        fn=transcricao_combinada,
        inputs=audio_input,
        outputs=[transcription_output, transcription_result, transcription_dialization]
    )
    transcribe_mono.click(
        fn=transcricao_segmentation,
        inputs=audio_input,
        outputs=transcription_dialization
    )

demo.launch()
