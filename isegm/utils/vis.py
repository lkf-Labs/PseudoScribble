from functools import lru_cache

import cv2
import numpy as np
import random
from pathlib import Path

def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)

# plot the user click points on the image
def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        #  determine whether the click is a pseudo-click, with a radius of 1 for pseudo-sribbles

        if pradius != 1:
            image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)
        else:
            image[int(p[0]), int(p[1])] = color

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
# overlay the semi-transparent gland binary image and clicks on the image
def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(0, 0, 255), radius=5):
    result = img.copy()
    # color===(B,G,R)
    if mask is not None and len(clicks_list) !=0 and clicks_list[0].is_pseudo == False:
        palette = get_palette(np.max(mask) + 1)

        palette[1][0] = 255
        palette[1][1] = 180
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result

# overlay the false positive and false positive regions on the original image
def draw_img_with_fp_fn(img, fn_map, fp_map, output, sample_id):
    fp_map = np.logical_and(fp_map == 255, output == 255).astype(np.uint8) * 255
    fn_map = np.logical_and(fn_map == 255, output == 0).astype(np.uint8) * 255
    fn_areas_num, fn_areas = cv2.connectedComponents(fn_map, connectivity=8)
    fp_areas_num, fp_areas = cv2.connectedComponents(fp_map, connectivity=8)
    #count the number of pixels for each label
    fn_areas_counts = np.bincount(fn_areas.flatten())
    fp_areas_counts = np.bincount(fp_areas.flatten())
    max_fn_label = np.argmax(fn_areas_counts[1:]) + 1
    max_fn_area = fn_areas_counts[max_fn_label]
    max_fp_label = np.argmax(fp_areas_counts[1:]) + 1
    max_fp_area = fp_areas_counts[max_fp_label]
    img_fp_fn = img.copy()
    img_max = img.copy()
    img_blue = img.copy()
    blue = (160,100,0)

    for index in range(1,fn_areas_num):
        indices = np.argwhere(fn_areas == index)
        color = tuple(random.randint(0, 255) for _ in range(3))
        if index == max_fn_label: max_fn_color = color
        img_fp_fn[indices[:, 0], indices[:, 1], :] = color
        img_blue[indices[:, 0], indices[:, 1], :] = blue

    for index in range(1,fp_areas_num):
        indices = np.argwhere(fp_areas == index)
        color = tuple(random.randint(0, 255) for _ in range(3))
        if index == max_fp_label: max_fp_color = color
        img_fp_fn[indices[:, 0], indices[:, 1], :] = color
        img_blue[indices[:, 0], indices[:, 1], :] = blue

    path = Path('./experiments')
    save_path = path/'scribble_generation'
    save_path.mkdir(parents=True, exist_ok=True)
    sample_path = save_path / f'{sample_id}.jpg'
    cv2.imwrite(str(sample_path), img_fp_fn)

    if max_fn_area > max_fp_area:
        indices = np.argwhere(fn_areas == max_fn_label)
        img_max[indices[:, 0], indices[:, 1], :] = max_fn_color
    else:
        indices = np.argwhere(fp_areas == max_fp_label)
        img_max[indices[:, 0], indices[:, 1], :] = max_fp_color

    sample_path = save_path / f'max_{sample_id}.jpg'
    cv2.imwrite(str(sample_path), img_max)

    sample_path = save_path / f'blue_{sample_id}.jpg'
    cv2.imwrite(str(sample_path), img_blue)

    return ;



def draw_with_fpmask(img, mask=None, alpha=0.4):
    result = img.copy()
    # color===(B,G,R)
    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        palette[1] = [255, 0, 255]
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)



    return result