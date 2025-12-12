import numpy as np
import wave
import time
import torch
from faster_whisper import WhisperModel

# generate 5s 16kHz sine wave
sr = 16000
duration = 5
t = np.linspace(0, duration, int(sr*duration), False)
wave_data = 0.5 * np.sin(2 * np.pi * 440 * t)
# convert to 16-bit PCM
pcm = (wave_data * 32767).astype('<i2')
with wave.open('bench.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm.tobytes())

print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

model = WhisperModel('small', device='cuda', compute_type='float16')
start = time.time()
segments, info = model.transcribe('bench.wav', beam_size=1)
end = time.time()
print('Transcribe time (s):', round(end - start, 3))
print('Segments sample:', segments[:1])
