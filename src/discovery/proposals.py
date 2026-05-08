import cv2
import numpy as np


class ProposalGenerator:

    def __init__(self, min_area_ratio=0.001, max_area_ratio=0.50, expand_pixels=15, crop_size=(224, 224)):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.expand_pixels = expand_pixels
        self.crop_size = crop_size

    def extract_boxes(self, mask, frame_shape):
        h, w = frame_shape[:2]
        frame_area = h * w

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            if area < self.min_area_ratio * frame_area:
                continue
            if area > self.max_area_ratio * frame_area:
                continue

            x1 = max(0, bx - self.expand_pixels)
            y1 = max(0, by - self.expand_pixels)
            x2 = min(w, bx + bw + self.expand_pixels)
            y2 = min(h, by + bh + self.expand_pixels)
            boxes.append((x1, y1, x2, y2))

        return boxes

    def extract_crops(self, frame, boxes):
        crops = []
        for x1, y1, x2, y2 in boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, self.crop_size)
            crops.append(crop)
        return crops
