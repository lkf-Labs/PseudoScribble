import torch
import torch.nn as nn
import numpy as np
import cv2
import os


from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize, CurvesMap
from isegm.model.modifiers import LRMult


class ISModel(nn.Module):
    def __init__(self, use_rgb_conv=True, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=False, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d, is_unet=False,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.pseudo_dist_map = DistMaps(norm_radius=3, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)
        self.curves_map = CurvesMap()
        # additional input channels
        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1
        # other input options
        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=64,
                                            kernel_size=3, stride=2, padding=1)
            self.maps_transform.apply(LRMult(0.1))
        else:
            # conv1s
            self.rgb_conv = None
            # determine whether the model is a U-Net architecture
            if is_unet:
                mt_layers =[
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                # U2NET does not use downsampling in the first step
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            else:
                mt_layers = [
                    nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                    nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                    ScaleLayer(init_value=0.05, lr_mult=1)
                ]
            self.maps_transform = nn.Sequential(*mt_layers)
            self.fn_fp_maps_transform = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                #  U2NET does not use downsampling in the first step
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            )
        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0,
                                               cpu_mode=cpu_dist_maps, use_disks=use_disks))
        else:
            self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)
    # points = batch×2n（positive、negative）×3(x y index）
    def forward(self, image, points, pseudo_points=None):
        # Separate the previous mask from the image
        image, prev_mask, prev_fp_mask, prev_fn_mask = self.prepare_input(image)
        # obtain the feature map of the additional input, including the concatenation of the mask and click map
        coord_features, fp_coord_features, fn_coord_features = self.get_coord_features(image, prev_mask, points, pseudo_points, prev_fp_mask, prev_fn_mask)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            outputs = self.backbone_forward(x)
        else:
            # conv1s input scheme
            coord_features = self.maps_transform(coord_features)
            # if fp_coord_features is not None and fn_coord_features is not None:
            #     # fp_coord_features = self.maps_transform(fp_coord_features)
            #     # coord_features = torch.cat((fp_coord_features, coord_features), dim=1)
            #     fp_fn_coord_features = self.fn_fp_maps_transform(fp_coord_features)
            #     coord_features = torch.cat((fp_fn_coord_features, coord_features), dim=1)
            if fn_coord_features is not None:
                fn_coord_features = self.fn_fp_maps_transform(fn_coord_features)
                coord_features = torch.cat((fn_coord_features, coord_features), dim=1)
            # if fp_coord_features is not None:
            #     fp_coord_features = self.maps_transform(fp_coord_features)
            #     coord_features = torch.cat((fp_coord_features, coord_features), dim=1)
            # fed into the backbone network
            outputs = self.backbone_forward(image, coord_features)

        # upsample to restore to the original resolution
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        return outputs

    def prepare_input(self, image):
        prev_mask = None
        prev_fp_mask = prev_fn_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:4, :, :]
            if image.size()[1] > 4:
                prev_fp_mask = image[:, 4:5, :, :]
                prev_fn_mask = image[:, 5:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()
                if prev_fp_mask is not None and prev_fn_mask is not None:
                    prev_fp_mask = (prev_fp_mask > 0.5).float()
                    prev_fn_mask = (prev_fn_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask, prev_fp_mask, prev_fn_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError


    # encode interactive information
    def get_coord_features(self, image, prev_mask, points, pseudo_points, prev_fp_mask, prev_fn_mask):
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            # disk transform for user clicks
            coord_features = self.dist_maps(image, points)
            if pseudo_points != None:
                # curve transform for model sribbles
                pseudo_coord_features = self.curves_map(image, pseudo_points)
                coord_features = torch.logical_or(coord_features, pseudo_coord_features).to(torch.int)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        fp_coord_features = fn_coord_features = None
        if prev_fp_mask is not None and prev_fn_mask is not None:
            fp_coord_features = coord_features[:, 1:, :, :]
            fn_coord_features = coord_features[:, 1:, :, :]
        return coord_features, fp_coord_features, fn_coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
