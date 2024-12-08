import random

import numpy as np
from copy import deepcopy
import cv2


class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    # refer to previous work, the robot user simulates real user-generated clicks during the testing stage
    def _get_next_click(self, pred_mask, padding=True):
        # obtain false positives and false positive regions
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
        # calculate the Euclidean distance between the error regions and the background
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool_)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)





    # pseudo scribble generation strategy: let the model simulate real user generate scribble annotations based on the predicted error regions
    def make_next_pseudo_scribble(self, fn_map, fp_map, output):
        assert  fn_map is not None
        assert  fp_map is not None
        fp_map = np.logical_and(fp_map==255, output==255).astype(np.uint8)*255
        fn_map = np.logical_and(fn_map==255, output==0).astype(np.uint8)*255

        fn_areas_num, fn_areas = cv2.connectedComponents(fn_map, connectivity=8)
        fp_areas_num, fp_areas = cv2.connectedComponents(fp_map, connectivity=8)

        # count the number of pixels for each label
        fn_areas_counts = np.bincount(fn_areas.flatten())
        fp_areas_counts = np.bincount(fp_areas.flatten())
        # find the label with the most pixels (excluding the background label)
        if len(fn_areas_counts) > 1:
            # output the number of pixels of the largest area
            max_fn_label = np.argmax(fn_areas_counts[1:]) + 1
            max_fn_area = fn_areas_counts[max_fn_label]
        else:
            max_fn_area = 0
        if len(fp_areas_counts) > 1:
            max_fp_label = np.argmax(fp_areas_counts[1:]) + 1
            max_fp_area = fp_areas_counts[max_fp_label]
        else:
            max_fp_area = 0
        if max_fn_area == 0 and max_fp_area == 0:
            return ;
        # generate one pseudo scribble within the largest error region
        is_positive = max_fn_area > max_fp_area

        if is_positive:
            # create a mask image with the same shape as the largest connected component
            fn_max_area = np.uint8(fn_areas == max_fn_label)
            indices = np.where(fn_max_area)
            coords = list(zip(indices[1], indices[0]))
            for coord in coords:
                coord_x, coord_y = coord
                self.add_click(Click(is_positive=is_positive, coords=(coord_y, coord_x), is_pseudo=True)) # scribbles are represented by a series of clicks
        else:
            # create a mask image with the same shape as the largest connected component
            fp_max_area = np.uint8(fp_areas == max_fp_label)
            indices = np.where(fp_max_area)
            coords = list(zip(indices[1], indices[0]))
            for coord in coords:
                coord_x, coord_y = coord
                self.add_click(Click(is_positive=is_positive, coords=(coord_y, coord_x), is_pseudo=True)) # scribbles are represented by a series of clicks




    # Design the pseudo-click generation strategy based on the paper: PseudoClick: Interactive Image Segmentation with Click Imitation
    def make_next_pseudo_click(self, fn_map, fp_map, output, padding=True):
        assert  fn_map is not None
        assert  fp_map is not None
        fp_mask = np.logical_and(fp_map==255, output==255).astype(np.uint8)*255
        fn_mask = np.logical_and(fn_map==255, output==0).astype(np.uint8)*255
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        self.add_click(Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]), is_pseudo=True))



class Click:
    def __init__(self, is_positive, coords, indx=None, is_pseudo=False):
        self.is_positive = is_positive # determine whether it is generated by the model virtually
        self.coords = coords
        self.indx = indx
        self.is_pseudo = is_pseudo

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
