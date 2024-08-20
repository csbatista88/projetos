import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import os
import evaluate

device = "cuda"# if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16# if torch.cuda.is_available() else torch.float32

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

#modelo
asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

#accuracy

accuracy = evaluate.load("accuracy")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(transcription):
    return classifier(transcription)[0]["label"]

def transcricao(audio_path):

    # Usar o pipeline para transcrever o áudio
    result = pipe(audio_path)
    transcription = result["text"]
    return transcription

def transcricao_chunck(audio_path):
    result2 = pipe(audio_path, return_timestamps=True)
    transcription_chunck = result2["text"]
    return transcription_chunck

with gr.Blocks() as demo:
    gr.Markdown(""""
        ## Trabalho faculdade implementando Gradio com Whisper Large v3
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Carregar Áudio", type="filepath")
        #options
            radio = gr.Radio(["mono","stereo"])
            transcribe_button = gr.Button("Transcrever")
            text = gr.Textbox()
            b1 = gr.Button("Recognize Speech")
            sentiment = gr.Button("Classificando Sentimento")
            with gr.Column(visible=True):
                name_audio = gr.Textbox(label="Audio")
                label = gr.Label()
                transcription_output = gr.Textbox(label="Transcrição")

    def submit():
        return {
            transcribe_button: gr.Column(visible=True)
            }

    # transcribe_button = gr.Button("Transcrever")

    transcribe_button.click(
        fn=transcricao,
        inputs=audio_input,
        outputs=transcription_output
    )
    # text_to_sentiment
    b1.click(speech_to_text, inputs=audio_input, outputs=text)
    sentiment.click(text_to_sentiment, inputs=transcription_output, outputs=label)

    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(label="Chunk")
demo.launch()
