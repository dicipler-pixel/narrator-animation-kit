# SCRIPT: MAKE_VOICE
# Turns a text file into narration.wav with the standing voice (Kokoro, am_michael, speed 0.92).
# Needs: pip install kokoro-onnx soundfile, plus the two model files kokoro-v1.0.onnx and voices-v1.0.bin
# (from the kokoro-onnx project's releases) placed beside this script.
# Usage: python3 make_voice.py lines.txt narration.wav     (one sentence per line)
import sys, numpy as np, soundfile as sf
from kokoro_onnx import Kokoro
k = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin'); SR = 24000
out = [np.zeros(int(0.8 * SR), np.float32)]
for line in open(sys.argv[1], encoding='utf-8'):
    if not line.strip(): continue
    y, _ = k.create(line.strip(), voice='am_michael', speed=0.92, lang='en-us')
    out += [y.astype(np.float32), np.zeros(int(0.38 * SR), np.float32)]
y = np.concatenate(out + [np.zeros(int(2 * SR), np.float32)]); sf.write(sys.argv[2], y / np.max(np.abs(y)) * 0.89, SR)
