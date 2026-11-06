# SCRIPT: NARRATOR_DEMO
# The narrator rig used in the Atlas Memoriae films, as one self-contained script.
# Every move is driven by the narration audio and by cues you list:
#   mouth / jaw   : follows the loudness of the voice (openv)
#   head turn/nod : slow sway, plus nods on emphasis (emph)
#   body glow     : brightens with the voice (env_slow); rare flicker
#   eyes          : glow with the voice; flare fully before an arc strike
#   cane tap      : cane lifts and strikes, a ripple runs over the floor, sparks fly  (TAPS)
#   mirror turn   : the figure turns around its vertical axis and back           (MIRRORS)
#   arc lightning : eyes shine, then an arc of energy strikes a point on the panel (ARCS)
# Usage: python3 narrator_demo.py narration.wav out.mp4 [seconds]
# Needs: numpy, opencv-python, soundfile, pillow, ffmpeg on the PATH; character.py beside this file; art/ beside code/.
import json, sys, math, subprocess, os, numpy as np, cv2, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from character import Figure, EYES
CHAR = os.path.join(HERE, '..', 'art')

# ---------------- CONFIG: change these for your film ----------------
W, H, FPS = 1920, 1080, 30
WAV = sys.argv[1]; OUTP = sys.argv[2]
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else None
TAPS = [3.0, 9.5]                 # seconds: cane tap lands 0.35 s after each time (use for proved results)
MIRRORS = [5.2]                   # seconds: start of the mirror turn (1.3 s turn, hold, 1.3 s back)
ARCS = [(12.0, 0.45, 0.40)]       # (seconds, fx, fy): arc strikes the panel at fraction (fx, fy) of PBOX
PBOX = (50, 104, 1270, 938)       # the panel box the arcs aim at (x0, y0, x1, y1)
P = 1
TOTAL = SECS if SECS else sf.info(WAV).duration
NFR = int(TOTAL * FPS)

# ---------------- audio features ----------------
y, sr = sf.read(WAV); y = y.astype(np.float32)
win = int(0.045 * sr)
openraw = np.zeros(NFR, np.float32)
for f in range(NFR):
    c = int(f / FPS * sr); seg = y[max(0, c - win // 2): c + win // 2]
    if len(seg):
        db = 20 * math.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-6)
        openraw[f] = min(1.0, max(0.0, (db + 40.0) / 22.0))
openv = np.zeros(NFR, np.float32); o = 0
for f in range(NFR):
    a = 0.65 if openraw[f] > o else 0.45
    o += a * (openraw[f] - o); openv[f] = o
def ema(x, tau):
    out = np.zeros_like(x); v = 0; k = 1 - math.exp(-1 / (FPS * tau))
    for i in range(len(x)): v += k * (x[i] - v); out[i] = v
    return out
env_slow = ema(openraw, 0.35)
emph = np.clip(ema(openraw, 0.4) - ema(openraw, 2.0), 0, None); emph = emph / max(emph.max(), 1e-6)
rng = np.random.default_rng(3 + P)
flick = np.ones(NFR, np.float32)
tt = 6.0
while tt < TOTAL:
    f0 = int(tt * FPS); flick[f0:f0 + 2] = 0.86; flick[f0 + 2:f0 + 3] = 0.95
    tt += rng.uniform(9, 17)
flick *= 1 + 0.01 * rng.standard_normal(NFR).astype(np.float32)


# ---------------- figures ----------------
src = cv2.cvtColor(cv2.imread(os.path.join(CHAR, 'figure_src.png')), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
src[int(src.shape[0] * 0.88):, int(src.shape[1] * 0.84):] = 0          # the artwork's corner mark is not part of the figure
FULL_SC = 0.88
full_img = cv2.resize(src, None, fx=FULL_SC, fy=FULL_SC, interpolation=cv2.INTER_AREA)
FH, FW = full_img.shape[:2]
FULL_POS = (1130, 24)
_l = full_img.max(axis=2)
SIL = cv2.GaussianBlur(cv2.dilate((_l > 0.06).astype(np.float32), np.ones((25, 25), np.uint8)), (0, 0), 8)[..., None]
head_img = cv2.cvtColor(cv2.imread(os.path.join(CHAR, 'head_x2.png')), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
CLOSE_POS = (1110, 0)
AX = 0.5 * (142 + 863) * FULL_SC                            # the figure's vertical axis, figure px
TIP = (235 * FULL_SC + FULL_POS[0], 1160 * FULL_SC + FULL_POS[1])   # cane tip on the floor, frame px
_XG = np.arange(FW, dtype=np.float32)[None, :].repeat(FH, 0); _YG = np.arange(FH, dtype=np.float32)[:, None].repeat(FW, 1)

def ease(u): u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)
def tap_state(t):
    """vertical offset (px), ring gain boost, spark boost for the cane tap"""
    dy = 0.0; ring = 0.0; spark = 0.0
    for ti in TAPS:
        s = t - ti
        if -0.45 < s < 0: dy -= 12 * math.sin(math.pi * (s + 0.45) / 0.45 * 0.5)
        if 0 <= s < 0.35: dy += 3 * math.exp(-s / 0.08) - 12 * max(0, 1 - s / 0.06)
        if 0 <= s < 2.0: ring += 2.6 * math.exp(-s / 0.22)
        if 0 <= s < 0.18: spark += 40
    return dy, ring, spark
def mirror_theta(t):
    for tm in MIRRORS:
        s = t - tm
        if 0 <= s < 1.3: return math.pi * ease(s / 1.3)
        if 1.3 <= s < 2.5: return math.pi
        if 2.5 <= s < 3.8: return math.pi * (1 - ease((s - 2.5) / 1.3))
    return 0.0
RX0, RX1 = int(TIP[0] - 350), min(W, int(TIP[0] + 350)); RY0, RY1 = int(TIP[1] - 90), H
def ripple(fr, t):
    act = [ti for ti in TAPS if 0 <= t - ti < 14]
    if not act: return
    lay = np.zeros((RY1 - RY0, RX1 - RX0, 3), np.float32); cx0, cy0 = TIP[0] - RX0, TIP[1] - RY0
    for ti in act:
        s = t - ti
        a = 22 + 300 * (1 - math.exp(-s / 0.55)); al = math.exp(-s / 0.8)
        if al > 0.01: cv2.ellipse(lay, (int(cx0 * 4), int(cy0 * 4)), (int(a * 4), int(a * 0.2 * 4)), 0, 0, 360, (0.45 * al, 0.8 * al, 1.0 * al), 8, cv2.LINE_AA, shift=2)
        keep = 0.22 * (1 - math.exp(-s / 0.4)) * math.exp(-s / 6.0)
        cv2.ellipse(lay, (int(cx0 * 4), int(cy0 * 4)), (int(70 * 4), int(14 * 4)), 0, 0, 360, (0.4 * keep, 0.7 * keep, 1.0 * keep), 6, cv2.LINE_AA, shift=2)
    lay = cv2.GaussianBlur(lay, (0, 0), 2.0) * 1.2 + cv2.GaussianBlur(lay, (0, 0), 9.0) * 1.1
    fr[RY0:RY1, RX0:RX1] += lay
def mirror_fig(fig, th):
    if th <= 0: return fig
    c = math.cos(th); ac = max(abs(c), 0.015)
    xs = AX + (_XG - AX) / ac
    if c < 0: xs = 2 * AX - xs
    out = cv2.remap(fig, xs, _YG, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    s = math.sin(th)
    if s > 0.02:
        sheen = np.exp(-((_XG - AX) / (12 + 300 * ac)) ** 2)[..., None] * np.array([0.25, 0.5, 0.8], np.float32) * s
        out = out * (1 + 0.5 * s) + sheen * (out.max(axis=2, keepdims=True) > 0.03)
    return out

# ---------------- the new move: the eyes shine bright, then an arc of energy strikes the panel ----------------
A_CHARGE, A_STRIKE, A_END = 0.75, 0.8, 1.55          # seconds after the cue: eyes at full shine, strike, last flicker
def arc_cue(t):
    for ta, fx, fy in ARCS:
        s_ = t - ta
        if -0.05 < s_ < 3.4: return s_, fx, fy, ta
    return None
def eye_shine(t):
    c = arc_cue(t)
    if c is None: return 0.0
    s_ = c[0]
    if s_ < A_CHARGE: return ease(s_ / A_CHARGE)
    if s_ < 1.7: return 1.0
    return 1.0 - ease((s_ - 1.7) / 0.9)
def bolt(p0, p1, rng, depth=6, rough=0.13, bow=0.12):
    p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64); v = p1 - p0; L = np.hypot(*v); nrm = np.array([-v[1], v[0]]) / max(L, 1e-9)
    if nrm[1] > 0: nrm = -nrm                           # bow upward, an arc
    pts = [p0, p1]; r = rough
    for _ in range(depth):
        out = [pts[0]]
        for a_, b_ in zip(pts, pts[1:]):
            w_ = b_ - a_; l_ = np.hypot(*w_); n_ = np.array([-w_[1], w_[0]]) / max(l_, 1e-9)
            out += [(a_ + b_) / 2 + n_ * rng.uniform(-1, 1) * r * l_, b_]
        pts = out; r *= 0.62
    u = np.linspace(0, 1, len(pts))[:, None]
    return np.array(pts) + nrm[None, :] * (bow * L * np.sin(np.pi * u))
def draw_arc(fr, t, eyes):
    c = arc_cue(t)
    if c is None or eyes is None: return
    s_, fx, fy, ta = c
    E = eye_shine(t); ex = 0.5 * (eyes[0][0] + eyes[1][0]); ey = 0.5 * (eyes[0][1] + eyes[1][1])
    # eye flare: a bloom and a thin horizontal streak on each eye
    if E > 0.01:
        X0, Y0, X1, Y1 = int(ex - 260), int(ey - 90), int(ex + 260), int(ey + 90)
        lay = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32)
        for (x, y) in eyes:
            cx_, cy_ = int((x - X0) * 4), int((y - Y0) * 4)
            cv2.circle(lay, (cx_, cy_), int(4 * (3 + 5 * E)), (0.9 * E, 0.97 * E, 1.0 * E), -1, cv2.LINE_AA, shift=2)
            cv2.ellipse(lay, (cx_, cy_), (int(4 * (24 + 105 * E)), 4 * 2), 0, 0, 360, (0.35 * E, 0.75 * E, 1.0 * E), -1, cv2.LINE_AA, shift=2)
        lay = cv2.GaussianBlur(lay, (0, 0), 2.0) * 1.1 + cv2.GaussianBlur(lay, (0, 0), 10.0) * 1.6 + cv2.GaussianBlur(lay, (0, 0), 26.0) * 1.2
        fr[max(Y0, 0):Y1, X0:X1] += lay[max(0, -Y0):, :]
    if s_ < A_STRIKE - 0.04: return
    tx_, ty_ = PBOX[0] + fx * (PBOX[2] - PBOX[0]), PBOX[1] + fy * (PBOX[3] - PBOX[1])
    k = s_ - A_STRIKE; seed = int(ta * 1000) % 100000
    # the strike itself, redrawn every two frames, with two re-strikes
    if k < A_END - A_STRIKE:
        fi = int(k * 15); rng = np.random.default_rng(seed * 50 + fi)
        restrike = any(0 <= k - r_ < 0.07 for r_ in (0.0, 0.28, 0.52))
        I = 1.0 if restrike else (0.45 + 0.4 * rng.random()) * (1 - 0.6 * k / (A_END - A_STRIKE))
        p0 = (ex - 16, ey + 6); main = bolt(p0, (tx_, ty_), rng)
        paths = [(main, 1.0)]
        for _ in range(3):
            j = int(rng.uniform(0.25, 0.8) * (len(main) - 1)); a0 = main[j]; d_ = main[min(j + 6, len(main) - 1)] - a0
            ang = math.atan2(d_[1], d_[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 0.8); Lb = rng.uniform(0.1, 0.22) * np.hypot(tx_ - p0[0], ty_ - p0[1])
            paths.append((bolt(a0, a0 + Lb * np.array([math.cos(ang), math.sin(ang)]), rng, depth=4, rough=0.25, bow=0.0), 0.55))
        allp = np.vstack([q for q, _ in paths]); X0, Y0 = int(max(0, allp[:, 0].min() - 60)), int(max(0, allp[:, 1].min() - 60))
        X1, Y1 = int(min(W, allp[:, 0].max() + 60)), int(min(H, allp[:, 1].max() + 60))
        core = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32); gl = np.zeros_like(core)
        for q, wgt in paths:
            pts = ((q - [X0, Y0]) * 4).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(core, [pts], False, (0.85 * wgt, 0.95 * wgt, 1.0 * wgt), max(1, int(2 * wgt)), cv2.LINE_AA, shift=2)
            cv2.polylines(gl, [pts], False, (0.25 * wgt, 0.62 * wgt, 1.0 * wgt), max(2, int(7 * wgt)), cv2.LINE_AA, shift=2)
        lay = core * 1.3 + cv2.GaussianBlur(gl, (0, 0), 3.5) * 1.4 + cv2.GaussianBlur(gl, (0, 0), 16.0) * 1.5
        fr[Y0:Y1, X0:X1] += lay * I
    # the impact: a flash, a ring running out over the panel, and sparks thrown off
    fl = math.exp(-k / 0.35); al = math.exp(-k / 0.6)
    if al > 0.02:
        R0 = 360; X0, Y0, X1, Y1 = int(max(0, tx_ - R0)), int(max(0, ty_ - R0)), int(min(W, tx_ + R0)), int(min(H, ty_ + R0))
        lay = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32); cx_, cy_ = int((tx_ - X0) * 4), int((ty_ - Y0) * 4)
        cv2.circle(lay, (cx_, cy_), int(4 * (10 + 18 * fl)), (0.9 * fl, 0.97 * fl, 1.0 * fl), -1, cv2.LINE_AA, shift=2)
        rr_ = 24 + 210 * (1 - math.exp(-k / 0.45))
        cv2.circle(lay, (cx_, cy_), int(4 * rr_), (0.3 * al, 0.7 * al, 1.0 * al), 5, cv2.LINE_AA, shift=2)
        rng = np.random.default_rng(seed)
        for _ in range(14):
            ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(120, 340); d0 = 20 + sp * k; d1 = d0 + 16 + 30 * fl
            a2 = al * rng.uniform(0.5, 1.0)
            cv2.line(lay, (int(cx_ + 4 * d0 * math.cos(ang)), int(cy_ + 4 * d0 * math.sin(ang))), (int(cx_ + 4 * d1 * math.cos(ang)), int(cy_ + 4 * d1 * math.sin(ang))),
                     (0.6 * a2, 0.9 * a2, 1.0 * a2), 2, cv2.LINE_AA, shift=2)
        lay = lay + cv2.GaussianBlur(lay, (0, 0), 6.0) * 1.4 + cv2.GaussianBlur(lay, (0, 0), 30.0) * 1.3 * fl
        fr[Y0:Y1, X0:X1] += lay


def blend(dst, layer, alpha=1.0, x=0, y=0):
    rgb, a = layer; h, w = a.shape[:2]
    y1, x1 = min(H, y + h), min(W, x + w)
    if y1 <= y or x1 <= x: return
    r = dst[y:y1, x:x1]
    r *= (1 - a[:y1 - y, :x1 - x] * alpha); r += rgb[:y1 - y, :x1 - x] * alpha
def screen(dst, fig, x, y):
    h, w = fig.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(W, x + w), min(H, y + h)
    f = np.clip(fig[y0 - y:y1 - y, x0 - x:x1 - x], 0, 1)
    r = dst[y0:y1, x0:x1]; dst[y0:y1, x0:x1] = 1 - (1 - r) * (1 - f)


# ---------------- background ----------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
g = np.exp(-(((xx - 1500) / 700) ** 2 + ((yy - 520) / 620) ** 2))
BG = np.stack([0.012 + 0.03 * g, 0.022 + 0.07 * g, 0.038 + 0.12 * g], -1).astype(np.float32)


class Renderer:
    def __init__(self, seed=0):
        self.Ffull = Figure(full_img, (0, 0), FULL_SC)
        self.Fclose = Figure(head_img, (280, 40), 2.0, feather=(60, 260))
        self.Ffull.rng = np.random.default_rng(seed + 11); self.Fclose.rng = np.random.default_rng(seed + 12)
    def figs(self, f, t, need):
        dy, ringb, sparkb = tap_state(t)
        p = dict(openv=float(openv[f]),
                 angle_deg=0.75 * math.sin(2 * math.pi * 0.09 * t) + 0.45 * math.sin(2 * math.pi * 0.21 * t + 1.1)
                           + 0.35 * math.sin(2 * math.pi * 0.47 * t + 2.0) * float(env_slow[f]),
                 nod_src=1.3 * float(emph[f]) + 0.35 * math.sin(2 * math.pi * 0.23 * t),
                 glow_gain=0.12 + 0.22 * float(env_slow[f]), eye_gain=0.25 + 0.55 * float(openv[f]),
                 flicker=float(flick[f]), dt=1 / FPS)
        E = eye_shine(t); p['eye_gain'] += 9.0 * E; p['glow_gain'] += 0.3 * E
        out = {'dy': int(round(dy))}
        th_ = math.radians(p['angle_deg']); nod_ = p['nod_src'] * FULL_SC
        out['eyes'] = [(FULL_POS[0] + self.Ffull.fwd(ex_, ey_, th_, nod_)[0], FULL_POS[1] + dy + self.Ffull.fwd(ex_, ey_, th_, nod_)[1]) for ex_, ey_ in EYES]
        if 'full' in need:
            fig = self.Ffull.render(t, spark_rate=0.25 + 1.4 * float(emph[f]) + sparkb, ring_gain=0.75 + 0.6 * float(env_slow[f]) + ringb, **p)
            if t < 1.6:
                hh = fig.shape[0]; edge = hh * (1 - t / 1.3)
                rows = np.arange(hh, dtype=np.float32)[:, None, None]
                m = np.clip((rows - edge) / 30, 0, 1)
                line = np.exp(-((rows - edge) / 5) ** 2) * 0.8 * (t < 1.3)
                fig = fig * m + line * SIL * np.array([0.5, 0.8, 1.0], np.float32)
            out['full'] = mirror_fig(fig, mirror_theta(t))
        if 'close' in need:
            out['close'] = self.Fclose.render(t, spark_rate=0, ring_gain=0.75 + 0.6 * float(env_slow[f]), **p)
        return out

def frame(R, f):
    t = f / FPS; figs = R.figs(f, t, {'full'})
    fr = BG.copy()
    c = np.array([0.15, 0.23, 0.34], np.float32)       # the panel outline, so you can see where arcs land
    fr[PBOX[1], PBOX[0]:PBOX[2]] = c; fr[PBOX[3], PBOX[0]:PBOX[2]] = c; fr[PBOX[1]:PBOX[3], PBOX[0]] = c; fr[PBOX[1]:PBOX[3], PBOX[2]] = c
    ripple(fr, t)
    screen(fr, figs['full'], FULL_POS[0], FULL_POS[1] + figs['dy'])
    draw_arc(fr, t, figs.get('eyes'))
    return fr

if __name__ == '__main__':
    R = Renderer()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-i', WAV, '-t', f'{TOTAL:.3f}', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', OUTP]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(NFR):
        p.stdin.write((np.clip(frame(R, f), 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait(); print('wrote', OUTP, NFR, 'frames')
