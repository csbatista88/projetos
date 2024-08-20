import torch
from datetime import timedelta
from pyannote.audio import Pipeline
import re
from pathlib import Path
#import whisper
#from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np


from pydub import AudioSegment



from pathlib import Path
output_path = "/home/christian/projetos/" #@param {type:"string"}
output_path = str(Path(output_path))
audio_title = "Sample Order Taking" #@param {type:"string"}
#antes de tudo acrescentar .5 seconds


import locale
locale.getpreferredencoding = lambda: "UTF-8"

video_title = ""
video_id = ""



spacermilli = 0
#spacer = AudioSegment.silent(duration=spacermilli)


#audio = AudioSegment.from_wav("input.wav")

#audio = spacer.append(audio, crossfade=0)

#audio.export('input_prep.wav', format='wav')


#diarization
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))


# apply pretrained pipeline
diarization = pipeline("teste1.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...


DEMO_FILE = {'uri': 'blabla', 'audio': 'teste1.wav'}
dz = pipeline(DEMO_FILE)

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))


print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")


#transcribe

#device = "cuda"# if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16# if torch.cuda.is_available() else torch.float32



def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

#agrupando a diarização de acordo com o speaker

dzs = open('diarization.txt').read().splitlines()

groups = []
g = []
lastend = 0

for d in dzs:
  if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
    groups.append(g)
    g = []

  g.append(d)

  end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
  end = millisec(end)
  if (lastend > end):       #segment engulfed by a previous segment
    groups.append(g)
    g = []
  else:
    lastend = end
if g:
  groups.append(g)
print(*groups, sep='\n')

# salvando cada parte de augio de acordo com a diarização

audio = AudioSegment.from_wav("teste1.wav")
gidx = -1
for g in groups:
  start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
  end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
  start = millisec(start) #- spacermilli
  end = millisec(end)  #- spacermilli
  gidx += 1
  audio[start:end].export(str(gidx) + '.wav', format='wav')
  print(f"group {gidx}: {start}--{end}")



# limpando cache memory
del   DEMO_FILE, dz, pipeline # ,spacer, audio,

## trancrição usando modelo whisper-v3-large uncomment cpu if gpu cuda unavaliable
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
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

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#for i in range(len(groups)):
  #audiof = str(i) + '.wav'
 ## result = model.transcribe(audio=audiof, language='en', word_timestamps=True)#, initial_prompt=result.get('text', ""))
#  result = pipe(audiof, return_timestamps=True)
#  with open(str(i)+'.json', "w") as outfile:
#    json.dump(result, outfile, indent=4)


import json
audiof = "teste1.wav"
result = pipe(audiof, return_timestamps=True)

json_string = json.dumps(result, indent=4, ensure_ascii=False)
with open(audiof+".json", "w", encoding='utf-8') as outfile:
    outfile.write(json_string)

# html generate
speakers = {'SPEAKER_00':('Customer', '#e1ffc7', 'darkgreen'), 'SPEAKER_01':('Call Center', 'white', 'darkorange') }
def_boxclr = 'white'
def_spkrclr = 'orange'


preS = '\n\n\n\n\n\t\n\t\n\t\n\t' + \
audio_title+ \
'\n\t';
preS += '\n\t\n\t';
preS += '\n\n\n\t' + audio_title + '\n\tClick on a part of the transcription, to jump to its portion of audio, and get an anchor to it in the address\n\t\tbar\n\t\n\t\t\n\t\t\t\n\t\t\n\t\tauto-scroll: \n\t\t\t\n\t\t\n\t\n';

postS = '\t\n'

# webvtt

#import webvtt



def timeStr(t):
  return '{0:02d}:{1:02d}:{2:06.2f}'.format(round(t // 3600),
                                                round(t % 3600 // 60),
                                                t % 60)
html = list(preS)
txt = list("")
gidx = -1
for g in groups:
  shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
  shift = millisec(shift) - spacermilli #the start time in the original video
  shift = max(shift, 0)

  gidx += 1

  #captions = json.load(open(str(gidx) + '.json'))["segments"]
  with open(audiof + ".json", "r", encoding="utf-8") as json_file:
    data = json_file.read()
  captions = json.loads(data)['segments']


  if captions:
    speaker = g[0].split()[-1]
    boxclr = def_boxclr
    spkrclr = def_spkrclr
    if speaker in speakers:
      speaker, boxclr, spkrclr = speakers[speaker]

    html.append(f'\n');
    html.append('\n')
    html.append(f'{speaker}\n\t\t\t\t')

    for c in captions:
      start = shift + c['start'] * 1000.0
      start = start / 1000.0   #time resolution ot youtube is Second.
      end = (shift + c['end'] * 1000.0) / 1000.0
      txt.append(f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')

      for i, w in enumerate(c['words']):
        if w == "":
           continue
        start = (shift + w['start']*1000.0) / 1000.0
        #end = (shift + w['end']) / 1000.0   #time resolution ot youtube is Second.
        html.append(f'{w["word"]}')
    #html.append('\n')
    html.append('\n')
    html.append(f'\n')

html.append(postS)

with open(f"capspeaker.txt", "w", encoding='utf-8') as file:
  s = "".join(txt)
  file.write(s)
  print('captions saved to capspeaker.txt:')
  print(s+'\n')

with open(f"capspeaker.html", "w", encoding='utf-8') as file:    #TODO: proper html embed tag when video/audio from file
  s = "".join(html)
  file.write(s)
  print('captions saved to capspeaker.html:')
  print(s+'\n')
