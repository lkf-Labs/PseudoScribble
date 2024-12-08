import sys
import torch
import numpy as np
from tqdm import tqdm

from isegm.data.datasets.glans import MeibomianGlandsDataset
from isegm.inference.clicker import Clicker
from time import time


sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.predictors import get_predictor

model = utils.load_is_model('../weights/test.pth', 'cuda:0')
dataset = MeibomianGlandsDataset("../datasets/MGD-1K",split='test')
zoomin_params= None
predictor_params = {}
predictor_params['net_clicks_limit'] = 740*350
predictor = get_predictor(model, 'NoBRS', 'cuda:0',
                              prob_thresh=0.49,
                              predictor_params=predictor_params,
                              zoom_in_params=zoomin_params)
# limit the number of clicks
max_clicks = 3
total_clicks = 0
start_time = time()
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
            total_clicks += 1
            clicker.make_next_click(pred_mask)
            pred_probs, fp_map_probs, fn_map_probs, execution_time = predictor.get_prediction(clicker,pseudo_clicker=pseudo_clicker, generate_pseu=True)
            fn_map = (fn_map_probs > 0.49).astype(np.uint8) * 255
            fp_map = (fp_map_probs > 0.49).astype(np.uint8) * 255
            output = (pred_probs > 0.49).astype(np.uint8) * 255
            # pseudo scribbles
            pseudo_clicker.make_next_pseudo_scribble(fn_map, fp_map, output)
            # pseudo_clicker.make_next_pseudo_click(fn_map, fp_map, output)
            pred_probs, fp_map_probs, fn_map_probs, execution_time = predictor.get_prediction(clicker, pseudo_clicker=pseudo_clicker, generate_pseu=False)
            pred_mask = pred_probs > 0.49
            iou = utils.get_iou(gt_mask, pred_mask)
            if iou >= 0.7000:  break;

end_time = time()
elapsed_time = end_time - start_time
mean_spc = elapsed_time / total_clicks
print(total_clicks)
print(mean_spc * 1000)

