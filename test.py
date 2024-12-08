import os
import random
import sys
import pickle
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm

from isegm.data.datasets.glans import MeibomianGlandsDataset
from isegm.data.datasets.mgd1k import MGD1KDataset
from isegm.inference.clicker import Clicker
from time import time
from visual import get_prediction_vis_callback, get_prediction_vis, get_prediction_vis_with_fpmask
from isegm.utils.vis import draw_img_with_fp_fn

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset, Progressive_Merge


def main():
    # ----------------------------------test stage parameter configuration area-----------------------------------------
    cuda_id = 0
    dataset_name = 'MGD-1K'  # tested datset name
    model_name = 'mgd1k_pse' # tested model name
    save_hist_pkl = False # choose whether to save the data for plotting the histogram
    save_miou_pkl = False # choose whether to save the data for plotting the line chart
    progressive_mode = False # choose whether to use progressive merging
    max_clicks = 2  # limit the number of clicks
    # ------------------------------------------------------------------------------------------------------------------

    model = utils.load_is_model(f'weights/{model_name}.pth', f'cuda:{cuda_id}')
    if dataset_name == 'MGD-1K':
        dataset = MGD1KDataset(f"./datasets/MGD-1K", split='test')
    else:
        dataset = MeibomianGlandsDataset("./datasets/MeibomianGlands", split='test')
    zoom_in_params = None
    predictor_params = {}
    predictor_params['net_clicks_limit'] = 740*350
    predictor = get_predictor(model, 'NoBRS', f'cuda:{cuda_id}',
                              prob_thresh=0.49,
                              predictor_params=predictor_params,
                              zoom_in_params=zoom_in_params)
    callback = get_prediction_vis_callback(0.49)
    # record the current GPU memory allocation before testing
    torch.cuda.reset_max_memory_allocated(device=cuda_id)
    torch.cuda.empty_cache()
    iou_list = []
    dice_list = []
    Iou_list = []
    time_list = []
    biou_list = []
    assd_list = []
    hd_list = []
    # record the data for later plotting of line charts and histograms
    mIoU_list = []
    my_dict = {i: [] for i in range(len(dataset))}
    memory_allocated_list = []
    # iterate through each item in the test set
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        image = sample.image
        gt_mask = sample.gt_mask
        clicker = Clicker(gt_mask=gt_mask)
        pseudo_clicker = Clicker(gt_mask=gt_mask)
        pred_mask = fp_map = fn_map = np.zeros_like(gt_mask)
        with torch.no_grad():
            predictor.set_input_image(image)
            for click_indx in range(max_clicks):
                # ===========================****user interaction phase: the user performs a click***===================
                clicker.make_next_click(pred_mask)
                pred_probs, fp_map_probs, fn_map_probs,  execution_time = predictor.get_prediction(clicker, pseudo_clicker=pseudo_clicker, generate_pseu=True) # the generate_pseu parameter is used to control whether the data should be fed into the false positive and false negative decoders
                time_list.append(execution_time)
                pred_mask = pred_probs > 0.49
                fn_map = (fn_map_probs > 0.49 ).astype(np.uint8) * 255
                fp_map = (fp_map_probs > 0.49 ).astype(np.uint8) * 255
                output = (pred_probs > 0.49).astype(np.uint8) * 255
                #  *** visualization and performance metrics before scribbling ***
                # get_prediction_vis(sample.image_name, click_indx, image, pred_probs, clicker.clicks_list, pseudo_clicker.clicks_list)
                # with open(r'experiments/metrics/unet_metrics.txt', 'a') as f:
                #     iou = utils.get_iou(gt_mask, pred_mask)
                #     f.write('id:{}  clicks:{} before scribble-----  '.format(sample.image_name, click_indx + 1) + str(iou) + '\n')
                # overlay the false positive and false positive regions on the original image
                # draw_img_with_fp_fn(image, fn_map, fp_map, output, index)
                prev_mask = pred_mask
                # ==========================****model simulation phase: the model performs a pseudo scribble***=========
                pseudo_clicker.make_next_pseudo_scribble(fn_map, fp_map, output)
                # pseudo_clicker.make_next_pseudo_click(fn_map, fp_map, output)
                pred_probs, fp_map_probs, fn_map_probs, execution_time = predictor.get_prediction(clicker, pseudo_clicker=pseudo_clicker, generate_pseu = False)
                pred_mask = pred_probs > 0.49
                if save_miou_pkl:
                    my_dict[index].append(utils.get_iou(gt_mask, pred_mask))
                # ====================== progressive merging ================================================
                # if progressive_mode:
                #     pseudo_clicks = pseudo_clicker.get_clicks()
                #     if len(pseudo_clicks) >= 0:
                #         last_click = pseudo_clicks[-1]
                #         last_y, last_x = last_click.coords[0], last_click.coords[1]
                #         pred_mask = Progressive_Merge(pred_mask, prev_mask, last_y, last_x)
                #         predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)
                # ==========================***visualize the test results***========================================
                # if click_indx == 0 or click_indx == 1 or  click_indx ==3 or click_indx ==4 or click_indx == 6 or click_indx ==9 or click_indx == 14:
                #     callback(image, gt_mask, pred_mask, sample.image_name, click_indx, clicker.clicks_list, pseudo_clicks_list=pseudo_clicker.clicks_list)
                #     with open(r'experiments/metrics/unet_metrics.txt', 'a') as f:
                #         iou = utils.get_iou(gt_mask, pred_mask)
                #         f.write('id:{}  clicks:{} after scribble -----  '.format(sample.image_name, click_indx+1)+str(iou)+'\n')

        # get the current GPU memory allocation during testing
        memory_allocated = torch.cuda.max_memory_allocated(device=cuda_id)
        memory_allocated_list.append(memory_allocated)
        # clear the cache to avoid memory leaks
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device=cuda_id)
        iou = utils.get_iou(gt_mask, pred_mask)
        iou_list.append(iou)
        dice = utils.get_dice(gt_mask, pred_mask)
        dice_list.append(dice)
        biou = utils.get_biou(gt_mask, pred_mask)
        biou_list.append(biou)
        assd = utils.get_assd(gt_mask, pred_mask)
        assd_list.append(assd)
        hd = utils.get_hd(gt_mask, pred_mask)
        hd_list.append(hd)
    with open(rf'datasets/{dataset_name}/metrics/pse_scribble.txt', 'a') as f:
        f.write('clicks = '+str(max_clicks)+'  model: '+model_name+'.pth'+'\n')
        f.write('IoU = '+str(round(sum(iou_list)/len(iou_list),4))+'\n')
        f.write('dice = '+str(round(sum(dice_list) / len(dice_list), 4)) + '\n')
        f.write('biou = '+str(round(sum(biou_list)/len(biou_list),4))+'\n')
        f.write('assd = '+str(round(sum(assd_list)/len(assd_list),4))+'\n')
        f.write('hd = ' + str(round(sum(hd_list) / len(hd_list), 4)) + '\n')
    # calculate the average GPU memory consumption
    avg_memory_allocated = sum(memory_allocated_list) / len(memory_allocated_list)
    print(memory_allocated_list)
    print(len(memory_allocated_list))
    max_memory_allocated_gb = avg_memory_allocated / (1024 ** 3)
    print(f"Maximum GPU memory allocated during testing: {max_memory_allocated_gb:.2f} GB")

    if save_hist_pkl:
        folder_path = 'visual_analysis'
        file_name = f'{dataset_name}_model_pseudoscribble_clicks_{max_clicks}.pkl'
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, 'wb') as file:
            pickle.dump(iou_list, file)

        print(f"List has been saved to {file_path}")


    if save_miou_pkl:
        for i in range(max_clicks):
            miou = 0
            for key, value in my_dict.items():
                miou += value[i]
            miou = miou / len(my_dict.keys())
            mIoU_list.append(miou)

        folder_path = 'visual_analysis'
        file_name = f'{dataset_name}_model_pseudoscribble_clicks_{max_clicks}_plot.pkl'
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, 'wb') as file:
            pickle.dump(mIoU_list, file)

        print(f"List has been saved to {file_path}")



if __name__ == '__main__':
    main()




