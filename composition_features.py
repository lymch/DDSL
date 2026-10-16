"""Product-specific component features used by the composition branch."""
import os

import cv2
import numpy as np
import torch
from PIL import Image

PRODUCTS = {
    "breakfast_box": (7, 6, False),
    "juice_bottle": (9, 12, True),
    "pushpins": (16, 15, False),
    "screw_bag": (7, 6, False),
    "splicing_connectors": (10, 11, True),
    "pcba1": (9, 24, True),
    "pcba2": (12, 33, True),
}

def get_logic_hist_features(mask, num_classes=5):
    """Return foreground class areas divided by H*W, shape (B, K-1, 1)."""
    batch_size, h, w = mask.shape
    all_geo_feats = []
    for k in range(1, num_classes):
        comp_mask = (mask == k).float()
        area = comp_mask.sum(dim=(1, 2))
        norm_area = area / (h * w)
        comp_geo = torch.stack([norm_area], dim=1)
        all_geo_feats.append(comp_geo)
    geo_features = torch.stack(all_geo_feats, dim=1)
    return geo_features

def get_logic_pos_features(masks, dataset_type="splicing_connectors"):
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy().astype(np.int32)
        else:
            masks = masks.astype(np.int32)

        batch_size = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        batch_results = []

        for b in range(batch_size):
            single_mask = masks[b]
            if dataset_type == "splicing_connectors":
                logic_config = {
                    3: {"name": "yellow_wire", "triplet": (1, 3, 2)},
                    6: {"name": "blue_wire", "triplet": (4, 6, 5)},
                    9: {"name": "orange_wire", "triplet": (7, 9, 8)}
                }
                focus_wire_id = _get_wire_main_id(single_mask, logic_config.keys())
                if focus_wire_id is None:
                    batch_results.append([0, 0])
                    continue
                id_L, id_W, id_R = logic_config[focus_wire_id]["triplet"]
                mask_L = (single_mask == id_L).astype(np.uint8) * 255
                mask_W = (single_mask == id_W).astype(np.uint8) * 255
                mask_R = (single_mask == id_R).astype(np.uint8) * 255
                ratio_L = _calculate_joint_ratio(mask_W, mask_L, 'left')
                ratio_R = _calculate_joint_ratio(mask_W, mask_R, 'right')
                batch_results.append([ratio_L, ratio_R])
            if dataset_type == "juice_bottle":
                logic_config = {
                    2: {"name": "yellow_liquid", "triplet": (5, 2, 8)},
                    3: {"name": "white_liquid", "triplet": (6, 3, 8)},
                    4: {"name": "brown_liquid", "triplet": (7, 4, 8)}
                }
                focus_liq_id = _get_liq_main_id(single_mask, logic_config.keys())
                if focus_liq_id is None:
                    batch_results.append([0.0, 0.0, 0.0, 0.0])
                    continue
                id_top, id_liq, id_bot = logic_config[focus_liq_id]["triplet"]

                mask_top = (single_mask == id_top).astype(np.uint8) * 255
                mask_bottom = (single_mask == id_bot).astype(np.uint8) * 255

                x_top, y_top  = _get_mask_center_xy(mask_top, label_type = 'top')
                x_bot, y_bot = _get_mask_center_xy(mask_bottom, label_type = 'bottom')

                rel_top_x = x_top / W
                rel_top_y = y_top/ H
                rel_bot_x = x_bot / W
                rel_bot_y = y_bot / H
                batch_results.append([float(rel_top_x), float(rel_top_y), float(rel_bot_x), float(rel_bot_y)])
            if dataset_type == "pcba1" or dataset_type == "pcba2":
                if dataset_type == "pcba1":
                    target_labels = [1, 2, 3, 4, 5, 6, 7, 8]
                elif dataset_type == "pcba2":
                    target_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                current_image_results = []
                for label_id in target_labels:
                    if label_id not in single_mask:
                        current_image_results.extend([0.0, 0.0])
                        continue
                    mask_component = (single_mask == label_id).astype(np.uint8) * 255
                    x_comp, y_comp = _get_mask_center_xy(mask_component, label_type='component')
                    rel_x = x_comp / W
                    rel_y = y_comp / H
                    current_image_results.extend([float(rel_x), float(rel_y)])
                batch_results.append(current_image_results)
        return batch_results

def _get_mask_center_xy(mask, label_type):
    m = cv2.moments(mask)
    if m["m00"] == 0:
        return 0.0, 0.0
    count = np.count_nonzero(mask)
    if label_type == 'top':
        count_ths = 1000
    elif label_type == 'bottom':
        count_ths = 500
    else:
        count_ths = 0
    if count <= count_ths:
        return 0.0, 0.0
    x_center = m["m10"] / m["m00"]
    y_center = m["m01"] / m["m00"]
    return x_center, y_center

def _get_wire_main_id(mask, ids):
    max_area = 0
    focus_id = None
    for idx in ids:
        area = np.sum(mask == idx)
        if area > max_area:
            max_area = area
            focus_id = idx
    return focus_id

def _get_liq_main_id(mask, ids):
        focus_id = None
        max_area = 1500
        for idx in ids:
            area = np.sum(mask == idx)
            if area > max_area:
                max_area = area
                focus_id = idx
        return focus_id

def _calculate_joint_ratio(mask_w, mask_t, joint_pos='left', dilation_size=10):
        if not np.any(mask_t) or not np.any(mask_w):
            return 0
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated_w = cv2.dilate(mask_w, kernel, iterations=1)
        intersection = cv2.bitwise_and(dilated_w, mask_t)
        m = cv2.moments(intersection)
        if m["m00"] == 0:
            return 0
        x_joint = int(m["m10"] / m["m00"])
        y_joint = m["m01"] / m["m00"]
        if joint_pos == 'left':
            column = x_joint - 5
        else:
            column = x_joint + 5
        terminal_column = (mask_t[:, column] if 0 <= column < mask_t.shape[1]
                           else np.zeros(mask_t.shape[0], dtype=mask_t.dtype))
        y_indices = np.where(terminal_column > 0)[0]
        if len(y_indices) == 0:
            search_range = 5
            for offset in range(1, search_range):
                for side in [-1, 1]:
                    check_x = x_joint + offset * side
                    if 0 <= check_x < mask_t.shape[1]:
                        y_indices = np.where(mask_t[:, check_x] > 0)[0]
                        if len(y_indices) > 0: break
                if len(y_indices) > 0: break
        if len(y_indices) == 0:
            return 0
        y_min_local = y_indices.min()
        y_max_local = y_indices.max()
        local_height = y_max_local - y_min_local
        if local_height <= 0:
            return 0
        ratio = (y_joint - y_min_local) / local_height
        return ratio
def read_seg_maps(seg_map_path, image_names):
    maps = []
    for name in image_names:
        path = os.path.join(seg_map_path, *str(name).replace("\\", "/").split("/"))
        with Image.open(path) as image:
            mask = np.asarray(image).copy()
        if mask.ndim != 2:
            raise ValueError("Expected an indexed, single-channel component map: " + path)
        maps.append(mask)
    return np.stack(maps, axis=0)
