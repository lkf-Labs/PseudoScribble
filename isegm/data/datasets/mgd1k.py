import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class MGD1KDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val','test'}

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "img"
        self._mask_path = self.dataset_path / "label"
        self.dataset_split = split
        with open(self.dataset_path / f'{split}/{split}.txt', 'r') as f:
            self.dataset_samples = [name.strip() for name in f.readlines()]

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]
        image_path = str(self._images_path / f'{sample_id}.png')
        mask_path = str(self._mask_path / f'{sample_id}.png')
        # read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (740, 350), interpolation=cv2.INTER_LINEAR)
        # read the mask and convert it to a grayscale image
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY)
        # uniformly resize to 740×350
        instances_mask = cv2.resize(instances_mask, (740, 350), interpolation=cv2.INTER_LINEAR)
        instances_mask[instances_mask > 127] = 255
        instances_mask[instances_mask <= 127] = 0
        instances_mask = instances_mask.astype(np.int32)
        objects_ids = np.unique(instances_mask)
        # gland id = 255
        objects_ids = [x for x in objects_ids if x != 0]

        return DSample(sample_id, image, instances_mask, objects_ids=objects_ids, sample_id=index)
