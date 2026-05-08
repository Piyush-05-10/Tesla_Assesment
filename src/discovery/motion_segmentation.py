import cv2
import numpy as np


class MotionSegmenter:

    def __init__(self, flow_threshold=25.0, morph_kernel_size=5, use_flow=False):
        self.flow_threshold = flow_threshold
        self.morph_kernel_size = morph_kernel_size
        self.use_flow = use_flow

    def compute_mask(self, frame_a, frame_b):
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

        if self.use_flow:
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        else:
            mag = np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32))

        _, mask = cv2.threshold(mag, self.flow_threshold, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask
