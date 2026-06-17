import cv2
import numpy as np
from skimage import transform as trans
import string
import time
from typing import List,Tuple,Union, Optional, Dict, Any
np.set_printoptions(suppress=True)

class FaceAligner:
    def __init__(self):
        self.arcface_dst: np.ndarray = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32,
        )
        self.dst: np.ndarray = self.arcface_dst.copy()  # Initialize dst here
        self._adjust_dst()

    def _adjust_dst(self) -> None:
        self.dst = self.arcface_dst * 1.0
        self.dst[:, 0] += 0

    def estimate_norm(self, lmk: np.ndarray) -> np.ndarray:
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, self.dst)
        return tform.params[0:2, :]

    def norm_crop(self, img: np.ndarray, landmark: np.ndarray, reco_input: int) -> np.ndarray:
        start = time.time()
        M: np.ndarray = self.estimate_norm(landmark)
        end = time.time()
        print(f"result Align and cropp Face: time: {end - start}")
        return cv2.warpAffine(img, M, (reco_input, reco_input)), M
