import numpy as np
from typing import List,Tuple,Union, Optional, Dict, Any

def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    num_kps = distance.shape[1] // 2    
    px = points[:, 0].reshape(-1, 1) + distance[:, 0::2]
    py = points[:, 1].reshape(-1, 1) + distance[:, 1::2]
    return np.stack([px, py], axis=-1).reshape(points.shape[0], num_kps * 2)
        

def generate_anchor_centers_batch(heights: np.ndarray, widths: np.ndarray, strides: np.ndarray, num_anchors: int, center_cache: dict) -> List[np.ndarray]:
    all_anchor_centers = []
    for height, width, stride in zip(heights, widths, strides):
        key = (height, width, stride, num_anchors)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers
        all_anchor_centers.append(anchor_centers)
    return all_anchor_centers
