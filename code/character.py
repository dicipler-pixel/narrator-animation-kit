# SCRIPT: CHARACTER_ANIMATION
# Animates the supplied hologram figure: head turn and nod about the neck,
# jaw drop with a dark mouth opening, voice-driven glow, eye glow,
# a turning light on the cane ring, and occasional sparks.
import numpy as np, cv2

# Landmarks in SOURCE pixels (figure_src.png, 896 x 1200)
MOUTH = (445.0, 217.0); MOUTH_HW = 15.0; LENS_R = 13.5; JAW_MAX = 5.0
PIVOT = (455.0, 262.0)
EYES = [(444.0, 168.0), (472.0, 167.0)]
RING = (235.0, 1111.0); RING_R = (44.0, 12.0)
HEAD_Y0, HEAD_Y1 = 238.0, 266.0       # head weight 1 above Y0, 0 below Y1
JAW_X = (412.0, 426.0, 466.0, 480.0)  # ramp in, plateau, ramp out
JAW_Y = (217.0, 244.0, 262.0)         # hard top at mouth line, plateau, fade

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

class Figure:
    def __init__(self, img_rgb01, origin, sc, feather=None):
        """img_rgb01: float32 HxWx3 in [0,1]; origin: source coords of pixel (0,0); sc: px per source px."""
        self.base = img_rgb01.astype(np.float32)
        H, W = self.base.shape[:2]; self.H, self.W = H, W
        self.ox, self.oy = origin; self.sc = sc
        if feather is not None:
            fx, fb = feather  # side feather px, bottom feather px
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            m = smooth(0, fx, xx) * smooth(W - 1, W - 1 - fx, xx) * smooth(H - 1, H - 1 - fb, yy) * smooth(0, 8, yy)
            self.base *= m[..., None]
        b = self.base
        lum = b.max(axis=2)
        hi = np.clip((lum - 0.42) / 0.58, 0, 1)[..., None] * np.array([0.55, 0.8, 1.0], np.float32)
        s = max(sc, 0.3)
        self.glow = cv2.GaussianBlur(hi, (0, 0), 5 * s) * 0.9 + cv2.GaussianBlur(hi, (0, 0), 16 * s) * 0.6
        # head/jaw region box in image px
        self.hb = self._box(290, 70, 640, 300)
        x0, y0, x1, y1 = self.hb
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        self.gx, self.gy = xx, yy
        sy = yy / sc + self.oy; sx = xx / sc + self.ox
        self.wh = 1.0 - smooth(HEAD_Y0, HEAD_Y1, sy)
        dxs = sx - MOUTH[0]
        lens = np.clip(1 - (dxs / LENS_R) ** 2, 0, 1)
        chin = np.clip(1 - (dxs / 24.0) ** 2, 0, 1) ** 1.5
        a = 1 - smooth(JAW_Y[0] + 4.0, JAW_Y[0] + 15.0, sy)
        wx = a * lens + (1 - a) * chin
        wy = smooth(JAW_Y[0], JAW_Y[0] + 1.2, sy) * (1 - smooth(JAW_Y[1], JAW_Y[2], sy))
        self.wj = wx * wy
        self.P = self.p(*PIVOT)
        # scanlines
        self.scan = (0.5 + 0.5 * np.cos(np.arange(H, dtype=np.float32) * 2 * np.pi / 4.0))
        self.rng = np.random.default_rng(7)
        self.sparks = []

    def p(self, sx, sy):
        return ((sx - self.ox) * self.sc, (sy - self.oy) * self.sc)

    def _box(self, sx0, sy0, sx1, sy1):
        x0, y0 = self.p(sx0, sy0); x1, y1 = self.p(sx1, sy1)
        x0 = int(max(0, np.floor(x0))); y0 = int(max(0, np.floor(y0)))
        x1 = int(min(self.W, np.ceil(x1))); y1 = int(min(self.H, np.ceil(y1)))
        return x0, y0, x1, y1

    def fwd(self, sx, sy, th, nod):
        """forward head transform of a source point (image px out)."""
        x, y = self.p(sx, sy); cx, cy = self.P
        c, s = np.cos(th), np.sin(th)
        dx, dy = x - cx, y - cy
        return cx + c * dx - s * dy, cy + s * dx + c * dy + nod

    def render(self, t, openv, angle_deg, nod_src, glow_gain, eye_gain, ring_gain, spark_rate, flicker, dt):
        out = self.base.copy()
        sc = self.sc
        # ---- head turn + nod + jaw (backward map) ----
        x0, y0, x1, y1 = self.hb
        th = np.deg2rad(angle_deg); cx, cy = self.P
        c, s = np.cos(-th), np.sin(-th)
        dx, dy = self.gx - cx, self.gy - (cy + nod_src * sc)
        rx = cx + c * dx - s * dy; ry = cy + s * dx + c * dy
        jaw = openv * JAW_MAX * sc
        mapx = self.gx + self.wh * (rx - self.gx)
        mapy = self.gy + self.wh * (ry - self.gy) - self.wj * jaw
        out[y0:y1, x0:x1] = cv2.remap(self.base, mapx.astype(np.float32), mapy.astype(np.float32),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # ---- mouth opening: the gap the jaw warp opened below the upper lip ----
        if openv > 0.04:
            mx, my = self.fwd(MOUTH[0], MOUTH[1], th, nod_src * sc)
            R = LENS_R * sc
            X0, X1 = max(int(mx - R - 2), 0), int(mx + R + 3)
            Y0, Y1 = max(int(my - 3), 0), int(my + jaw + 4)
            yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
            prof = np.clip(1 - ((xx - mx) / R) ** 2, 0, 1)
            top = my - 0.2 * sc + 0.9 * sc * (1 - prof)          # upper lip line, corners slightly lower
            bot = my + jaw * prof                                 # dropped lower lip
            soft = 0.35 + 0.35 * sc
            m = np.clip((yy - top) / soft, 0, 1) * np.clip((bot - yy) / soft, 0, 1)
            depth = np.clip((yy - top) / np.maximum(bot - top, 0.5), 0, 1)[..., None]
            col = np.array([0.01, 0.02, 0.04], np.float32) * (1 - depth) + np.array([0.07, 0.11, 0.17], np.float32) * depth
            if openv > 0.4:
                tb = np.clip(1 - (yy - top) / np.maximum(0.28 * (bot - top), 0.6), 0, 1) * np.clip(prof * 1.6 - 0.4, 0, 1)
                col = col + tb[..., None] * np.array([0.40, 0.48, 0.56], np.float32) * min(1.0, (openv - 0.4) * 1.6)
            m = m[..., None] * 0.94
            out[Y0:Y1, X0:X1] = out[Y0:Y1, X0:X1] * (1 - m) + col * m
        # ---- glow ----
        out += self.glow * glow_gain
        # ---- eyes ----
        for ex, ey in EYES:
            px, py = self.fwd(ex, ey, th, nod_src * sc)
            self._blob(out, px, py, 2.4 * sc, np.array([0.6, 0.8, 1.0]) * eye_gain * 0.6)
            self._blob(out, px, py, 7.0 * sc, np.array([0.3, 0.55, 1.0]) * eye_gain * 0.18)
        # ---- cane ring ----
        rc = self.p(*RING)
        if 0 <= rc[1] < self.H:
            self._ring(out, t, rc, ring_gain)
            self._sparks(out, rc, spark_rate, dt)
        # ---- hologram scanlines + flicker ----
        out *= (1.0 - 0.07 * self.scan[:, None, None]) * flicker
        return out

    def _blob(self, img, x, y, sig, col):
        r = int(3 * sig) + 2
        X0, Y0, X1, Y1 = int(x - r), int(y - r), int(x + r), int(y + r)
        if X1 <= 0 or Y1 <= 0 or X0 >= self.W or Y0 >= self.H: return
        X0c, Y0c = max(X0, 0), max(Y0, 0); X1c, Y1c = min(X1, self.W), min(Y1, self.H)
        yy, xx = np.mgrid[Y0c:Y1c, X0c:X1c].astype(np.float32)
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig * sig))
        img[Y0c:Y1c, X0c:X1c] += g[..., None] * col.astype(np.float32)

    def _ring(self, img, t, rc, gain):
        sc = self.sc; rx, ry = RING_R[0] * sc, RING_R[1] * sc
        pad = int(24 * sc) + 6
        X0, Y0 = int(rc[0] - rx - pad), int(rc[1] - ry - pad)
        X1, Y1 = int(rc[0] + rx + pad), int(rc[1] + ry + pad)
        X0c, Y0c, X1c, Y1c = max(X0, 0), max(Y0, 0), min(X1, self.W), min(Y1, self.H)
        if X1c <= X0c or Y1c <= Y0c: return
        layer = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32)
        n = 90
        for k in range(n):
            ph = 2 * np.pi * k / n
            a = (ph - 1.9 * t) % (2 * np.pi); b = (ph + 1.2 * t + 2.0) % (2 * np.pi)
            w = np.exp(-min(a, 2 * np.pi - a) ** 2 / 0.35) + 0.6 * np.exp(-min(b, 2 * np.pi - b) ** 2 / 0.25)
            if w < 0.03: continue
            px = rc[0] + rx * np.cos(ph) - X0; py = rc[1] + ry * np.sin(ph) - Y0
            cv2.circle(layer, (int(px * 4), int(py * 4)), int(max(1, 1.6 * sc) * 4),
                       (0.7 * w, 0.9 * w, 1.0 * w), -1, lineType=cv2.LINE_AA, shift=2)
        lay = cv2.GaussianBlur(layer, (0, 0), 1.2 * max(sc, 0.5)) * 1.3 + cv2.GaussianBlur(layer, (0, 0), 5 * max(sc, 0.5)) * 0.9
        img[Y0c:Y1c, X0c:X1c] += lay[Y0c - Y0:Y1c - Y0, X0c - X0:X1c - X0] * gain

    def _sparks(self, img, rc, rate, dt):
        sc = self.sc
        if self.rng.random() < rate * dt:
            ph = self.rng.uniform(0, 2 * np.pi)
            start = (rc[0] + RING_R[0] * sc * np.cos(ph), rc[1] + RING_R[1] * sc * np.sin(ph))
            L = self.rng.uniform(18, 46) * sc; ang = ph + self.rng.uniform(-0.5, 0.5)
            pts = [start]; nseg = 6
            for i in range(1, nseg + 1):
                f = i / nseg
                px = start[0] + np.cos(ang) * L * f + self.rng.normal(0, 3 * sc)
                py = start[1] + np.sin(ang) * L * f * 0.45 + self.rng.normal(0, 2 * sc) - 4 * sc * f
                pts.append((px, py))
            self.sparks.append([np.array(pts, np.float32), int(self.rng.integers(3, 6))])
        keep = []
        for pts, life in self.sparks:
            if life <= 0: continue
            x0, y0 = pts.min(0) - 12 * sc; x1, y1 = pts.max(0) + 12 * sc
            X0, Y0, X1, Y1 = int(max(x0, 0)), int(max(y0, 0)), int(min(x1, self.W)), int(min(y1, self.H))
            if X1 > X0 and Y1 > Y0:
                lay = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32)
                q = ((pts - [X0, Y0]) * 4).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(lay, [q], False, (0.8, 0.93, 1.0), max(1, int(1.2 * sc)), lineType=cv2.LINE_AA, shift=2)
                g = lay * 1.2 + cv2.GaussianBlur(lay, (0, 0), 3 * max(sc, 0.5)) * 1.4
                img[Y0:Y1, X0:X1] += g * (0.5 + 0.5 * life / 5)
            keep.append([pts, life - 1])
        self.sparks = keep
