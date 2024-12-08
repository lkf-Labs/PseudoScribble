import random
import sys
import torch
import numpy as np
from tqdm import tqdm

from isegm.data.datasets.mgd1k import MGD1KDataset

sys.path.insert(0, '..')
from isegm.data.datasets.glans import MeibomianGlandsDataset
from isegm.inference.clicker import Clicker
from visual import get_prediction_vis_callback, get_prediction_vis
from isegm.inference import utils
from isegm.inference.predictors import get_predictor


def main():
    cuda_id = 0
    model_name = "mgd1k_pse"
    dataset_name = 'MGD-1K'
    model = utils.load_is_model(f'../weights/{model_name}.pth', f'cuda:{cuda_id}')
    if dataset_name == 'MGD-1K':
        dataset = MGD1KDataset(f"../datasets/MGD-1K", split='test')
    else:
        dataset = MeibomianGlandsDataset("../datasets/MeibomianGlands", split='test')
    zoomin_params= None
    # sum = 0x
    predictor_params = {}
    predictor_params['net_clicks_limit'] = 740*350
    predictor = get_predictor(model, 'NoBRS', f'cuda:{cuda_id}',
                              prob_thresh=0.49,
                              predictor_params=predictor_params,
                              zoom_in_params=zoomin_params)
    callback = get_prediction_vis_callback(0.49)
    # set the noc hyperparameter
    max_clicks = 5
    target_iou = 0.6500
    # max_clicks = 10
    # target_iou = 0.7000
    # max_clicks = 20
    # target_iou = 0.7500

    iou_list = []
    dice_list = []
    acc_list = []
    fnr_list = []
    fpr_list = []
    Iou_list = []
    time_list = []
    clicks_list = []
    total_clicks = 0
    NoF_num = 0
    max_iou_up = 0
    # iterate through each item in the test set
    random_num = random.randint(0, 39)
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        image = sample.image
        gt_mask = sample.gt_mask
        min_clicks = 15
        clicker = Clicker(gt_mask=gt_mask)
        pseudo_clicker = Clicker(gt_mask=gt_mask)
        pred_mask = fp_map = fn_map = np.zeros_like(gt_mask)
        with torch.no_grad():
            predictor.set_input_image(image)
            for click_indx in range(max_clicks):
                total_clicks += 1
                clicker.make_next_click(pred_mask)
                pred_probs, fp_map_probs, fn_map_probs,  execution_time = predictor.get_prediction(clicker, pseudo_clicker=pseudo_clicker, generate_pseu=True)
                time_list.append(execution_time)
                pred_mask = pred_probs > 0.49
                fn_map = (fn_map_probs > 0.49 ).astype(np.uint8) * 255
                fp_map = (fp_map_probs > 0.49 ).astype(np.uint8) * 255
                output = (pred_probs > 0.49).astype(np.uint8) * 255
                pseudo_clicker.make_next_pseudo_scribble(fn_map, fp_map, output)
                pred_probs, fp_map_probs, fn_map_probs, execution_time = predictor.get_prediction(clicker, pseudo_clicker=pseudo_clicker, generate_pseu=False)

                pred_mask = pred_probs > 0.49
                iou = utils.get_iou(gt_mask, pred_mask)
                if iou >= target_iou:
                    clicks_list.append(click_indx+1)
                    break
        if click_indx == max_clicks-1:
            clicks_list.append(max_clicks)

    with open(rf'../datasets/{dataset_name}/metrics/pse_scribble.txt', 'a') as f:
        f.write('clicks = '+str(max_clicks)+'  model'+str(model_name)+'.pth'+'\n')
        f.write(f'Noc@{target_iou * 100} = '+str(round(sum(clicks_list)/len(clicks_list),4))+'\n')

if __name__ == '__main__':
    main()




