from pyannote.audio import Pipeline
import os

TOKEN = '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'
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=TOKEN)
audio_input = '/home/christian/projetos/teste1.wav'
# apply the pipeline to an audio file
diarization = pipeline(audio_input, num_speakers = 2)
# dump the diarization output to disk using RTTM format
with open("audio-test-pyannote.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
