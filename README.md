NARRATOR ANIMATION KIT v2.0 (22 October 2026) — Jeromie Beasley

New in v2 (everything from v1.0 still works the same)
  code/arm.py          the free arm is cut out of the figure and rigged as two bones:
                       upper arm about the shoulder, forearm and hand about the elbow.
                       The place the arm leaves is filled in, so the coat shows behind it.
  code/narrator_v2.py  the v1 rig plus: hand gestures toward the panel, idle arm swing,
                       beats on emphasised words, breathing and a small weight shift,
                       and a two-book panel (connection energy, its opposite, the budget).
  narrator_v2_sample.mp4  20.6 s test in the Kokoro am_michael voice:
                       hand rises at 1 s, arc at 3.3 s, mirror turn at 7.6 s,
                       cane tap at 12.75 s, hand rises again at 14.6 s while the books trade.

Make one
  1. python3 make_voice.py lines.txt narration.wav
       (kokoro-v1.0.onnx and voices-v1.0.bin beside the script, from the kokoro-onnx releases)
  2. Open code/narrator_v2.py and set the cue times in the CONFIG block:
       TAPS     = [seconds, ...]                  cane tap
       MIRRORS  = [seconds]                       mirror turn
       ARCS     = [(seconds, fx, fy), ...]        lightning to point (fx, fy) of the panel
       GESTURES = [(start, end, angle), ...]      the hand rises toward the panel and holds;
                                                  angle 40-70 degrees (70 is chest height)
       PANEL_ON = seconds                         the books appear
  3. python3 narrator_v2.py narration.wav out.mp4
  Mouth, head, glow, eyes, breathing and the beats follow the voice by themselves.

Needs: Python 3 with numpy, opencv-python, soundfile, pillow, kokoro-onnx; ffmpeg.
