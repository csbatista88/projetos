import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Configuração do modelo Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

# Pipeline para diarização
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token="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")

def transcribe(audio_file, use_diarization, check_stereo):
    audio = AudioSegment.from_file(audio_file)
    result_text = ""
    chunks_text = ""
    is_stereo = False

    # Verificar se o áudio é estéreo ou mono
    if check_stereo:
        is_stereo = audio.channels > 1
        if is_stereo:
            left_channel = audio.split_to_mono()[0]
            right_channel = audio.split_to_mono()[1]

            left_channel.export("left_channel.wav", format="wav")
            right_channel.export("right_channel.wav", format="wav")

            result_left = pipe("left_channel.wav")["text"]
            result_right = pipe("right_channel.wav")["text"]

            result_text = f"Person 1 (left channel): {result_left}\nPerson 2 (right channel): {result_right}"
        else:
            result_text = "The audio is mono."

    if not is_stereo or not check_stereo:
        if use_diarization:
            diarization = diarization_pipeline(audio_file)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                chunks_text += f"Speaker {speaker}: {turn.start:.1f}s to {turn.end:.1f}s\n"
                segment = audio[turn.start * 1000:turn.end * 1000]
                segment.export(f"segment_{speaker}.wav", format="wav")
                result = pipe(f"segment_{speaker}.wav")["text"]
                result_text += f"Speaker {speaker}: {result}\n"
        else:
            result = pipe(audio_file)
            result_text = result["text"]
            chunks = result.get("chunks", [])
            for chunk in chunks:
                chunks_text += f"Chunk from {chunk['timestamp'][0]}s to {chunk['timestamp'][1]}s: {chunk['text']}\n"

    return result_text, chunks_text, is_stereo

def download_text(text):
    with open("transcription.txt", "w") as f:
        f.write(text)
    return "transcription.txt"

def download_chunks(text):
    with open("chunks.txt", "w") as f:
        f.write(text)
    return "chunks.txt"

with gr.Blocks() as demo:
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio File")
        use_diarization = gr.Checkbox(label="Use Diarization for Mono Audio", value=False)
        check_stereo = gr.Checkbox(label="Check if Audio is Stereo", value=False)
    with gr.Row():
        transcribe_button = gr.Button("Transcribe")
    with gr.Row():
        result_textbox = gr.Textbox(label="Transcription", lines=10)
        chunks_textbox = gr.Textbox(label="Chunks", lines=10)
    with gr.Row():
        download_transcription = gr.File(label="Download Transcription")
        download_chunks = gr.File(label="Download Chunks")

    transcribe_button.click(
        transcribe,
        inputs=[audio_input, use_diarization, check_stereo],
        outputs=[result_textbox, chunks_textbox, download_transcription],
    )

    result_textbox.change(
        lambda text: gr.File.update(value=download_text(text), filename="transcription.txt"),
        inputs=[result_textbox],
        outputs=download_transcription
    )

    chunks_textbox.change(
        lambda text: gr.File.update(value=download_chunks(text), filename="chunks.txt"),
        inputs=[chunks_textbox],
        outputs=download_chunks
    )

demo.launch()
