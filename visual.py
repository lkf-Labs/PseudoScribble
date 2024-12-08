import cv2
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from isegm.utils.vis import draw_with_blend_and_clicks, draw_with_fpmask


def get_prediction_vis_callback( prob_thresh):
    path = Path('./experiments')
    save_path = path/'unet_logs'
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_mask, image_name, click_indx, clicks_list, pseudo_clicks_list):
        sample_path = save_path / f'{image_name}_{click_indx}.jpg'
        image_with_mask = draw_with_blend_and_clicks(image, pred_mask, clicks_list=clicks_list)
        image_with_mask = draw_with_blend_and_clicks(image_with_mask, pred_mask, clicks_list=pseudo_clicks_list, pos_color=(255, 150, 220), neg_color=(0,255,255),radius=1)
        cv2.imwrite(str(sample_path), image_with_mask)

    return callback


def get_prediction_vis(sample_id, click_indx, image, output, clicks_list, pseudo_clicks_list):
    path = Path('./experiments')
    save_path = path/'output'
    save_path.mkdir(parents=True, exist_ok=True)
    sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
    image_with_mask = draw_with_blend_and_clicks(image, output > 0.49, clicks_list=clicks_list)
    image_with_mask = draw_with_blend_and_clicks(image_with_mask, output > 0.49,
                                                 clicks_list=pseudo_clicks_list, pos_color=(255, 255, 0),
                                                 neg_color=(0, 255, 255), radius=1)

    cv2.imwrite(str(sample_path), image_with_mask)



def get_prediction_vis_with_fpmask(image, gt_mask, pred_mask, image_name, click_indx, clicks_list, fp_mask):
    path = Path('./experiments')
    save_path = path / 'unet_logs'
    save_path.mkdir(parents=True, exist_ok=True)
    sample_path = save_path / f'{image_name}_{click_indx}.jpg'
    image_with_mask = draw_with_blend_and_clicks(image, pred_mask, clicks_list=clicks_list)
    image_with_mask = draw_with_fpmask(image_with_mask, fp_mask )

    cv2.imwrite(str(sample_path), image_with_mask)


