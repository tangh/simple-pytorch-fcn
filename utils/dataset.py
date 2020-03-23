import os
import collections

import numpy as np
from PIL import Image
import scipy
import torch
from torch.utils import data


class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        '__background__',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = os.path.join(self.root, 'VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = os.path.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split
            )
            for line in open(imgsets_file):
                line = line.strip()
                img_file = os.path.join(
                    dataset_dir, 'JPEGImages/%s.jpg' % line
                )
                lbl_file = os.path.join(
                    dataset_dir, 'SegmentationClass/%s.png' % line)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]

        img_file = data_file['img']
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)

        lbl_file = data_file['lbl']
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1

        return self.to_tensor(img, lbl)

    def to_tensor(self, img, lbl):
        """
        to unnormalized torch tensor
        img: float, mean substracted
        target: long
        """
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def to_numpy(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()

        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):
    """
    VOC 2011 segval intersects with SBD train.
    Care must be taken for proper evaluation
    by excluding images from the train or val splits.
    This class use non-intersecting val set defined in the seg11valid.txt
    """

    def __init__(self, root, split='seg11valid', transform=False):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform
        )
        imgsets_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "seg11valid.txt"
        )
        dataset_dir = os.path.join(self.root, 'VOCdevkit/VOC2012')
        for line in open(imgsets_file):
            line = line.strip()
            img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % line)
            lbl_file = os.path.join(
                dataset_dir, 'SegmentationClass/%s.png' % line
            )
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class SBDClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = os.path.join(self.root, 'benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = os.path.join(dataset_dir, '%s.txt' % split)
            for line in open(imgsets_file):
                line = line.strip()
                img_file = os.path.join(dataset_dir, 'img/%s.jpg' % line)
                lbl_file = os.path.join(dataset_dir, 'cls/%s.mat' % line)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]

        img_file = data_file['img']
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)

        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1

        return self.to_tensor(img, lbl)


if __name__ == "__main__":
    # sbd_dataset = SBDClassSeg(root="../datasets")
    voc_dataset = VOC2011ClassSeg(root=os.path.abspath("./datasets"))
    loader = iter(voc_dataset)
    img, target = voc_dataset.to_numpy(*next(loader))
    import cv2
    cv2.imshow("image", img[:, :, ::-1])
    cv2.imshow("target", target.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
