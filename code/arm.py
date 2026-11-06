# SCRIPT: NARRATOR_ARM_RIG
# Cuts the narrator's free arm (the hand that is not on the cane) out of the figure,
# fills the place it leaves with an inpainted background, and poses it as two rigid bones:
# upper arm about the shoulder, forearm and hand about the elbow. Angles in degrees;
# a positive forearm angle swings the hand up and across the body, toward the panel.
import numpy as np, cv2

# landmarks in SOURCE pixels of figure_src.png (896 x 1200)
SHOULDER = (570.0, 315.0)
ELBOW = (577.0, 470.0)
ARM_POLY = [(545, 285), (600, 290), (612, 320), (618, 400), (622, 480), (620, 560), (612, 620), (602, 645),
            (592, 665), (590, 700), (580, 727), (540, 727), (524, 700), (529, 660), (539, 640), (537, 600),
            (534, 540), (529, 470), (525, 400), (524, 330), (530, 300)]
JOINT_HALF = 14.0          # half-width of the soft elbow blend, source px


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class Arm:
    def __init__(self, img, sc):
        """img: the figure at scale sc (float32 HxWx3 in [0,1])."""
        self.sc = sc; H, W = img.shape[:2]; self.H, self.W = H, W
        poly = (np.array(ARM_POLY, np.float32) * sc).astype(np.int32)
        hard = np.zeros((H, W), np.uint8); cv2.fillPoly(hard, [poly], 255)
        soft = cv2.GaussianBlur(hard.astype(np.float32) / 255, (0, 0), 1.6 * sc)
        yy = np.arange(H, dtype=np.float32)[:, None] / sc
        wf = smooth(ELBOW[1] - JOINT_HALF, ELBOW[1] + JOINT_HALF, yy) * np.ones((1, W), np.float32)
        self.up = img * (soft * (1 - wf))[..., None]
        self.fo = img * (soft * wf)[..., None]
        # the place the arm leaves: inpaint the figure where the (dilated) arm was
        hole = cv2.dilate(hard, np.ones((13, 13), np.uint8))
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        filled = cv2.inpaint(u8, hole, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255
        # the inpainted fill is dimmed so it reads as the dark side of the coat, not a copy of the arm
        hm = cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 3)[..., None]
        self.rest = img * (1 - hm) + filled * hm * 0.3
        self.S = (SHOULDER[0] * sc, SHOULDER[1] * sc); self.E = (ELBOW[0] * sc, ELBOW[1] * sc)

    @staticmethod
    def _rot(c, deg):
        return cv2.getRotationMatrix2D(c, -deg, 1.0)   # cv2 is counter-clockwise for +angle; ours is clockwise on screen

    @staticmethod
    def _compose(A, B):
        A3 = np.vstack([A, [0, 0, 1]]); B3 = np.vstack([B, [0, 0, 1]]); return (A3 @ B3)[:2]

    def hand_point(self, a_up, a_fo, src=(560.0, 690.0)):
        """where a source point on the hand lands, image px"""
        M = self._compose(self._rot(self.S, a_up), self._rot(self.E, a_fo))
        x, y = src[0] * self.sc, src[1] * self.sc
        return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]

    def pose(self, a_up, a_fo):
        """the figure with the arm posed; screen-blended so the overlap at the elbow does not show a seam"""
        Mu = self._rot(self.S, a_up)
        Mf = self._compose(Mu, self._rot(self.E, a_fo))
        up = cv2.warpAffine(self.up, Mu, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=0)
        fo = cv2.warpAffine(self.fo, Mf, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=0)
        arm = 1 - (1 - np.clip(up, 0, 1)) * (1 - np.clip(fo, 0, 1))
        # the arm sits in front of the coat: it hides a little of what is behind it, then adds its light
        cover = np.clip(arm.max(axis=2, keepdims=True) * 1.6, 0, 0.6)
        return self.rest * (1 - cover) + arm
