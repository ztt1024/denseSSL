import random
import math
import numpy as np

from torchvision.datasets import ImageFolder
import torch

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            elif self.pred_shape == 'rcc':
                mask = np.ones((H, W), dtype=bool)
                bboxs = torch.tensor(get_bboxs_in_grid())
                cut_ratio = self.get_pred_ratio()
                mask = RCC(mask, bboxs, (W, H), cut_ratio)
                mask = ~mask
            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)

def get_bboxs_in_grid(scale=(0.12, 0.3), ratio=(0.5, 2), grid_num=3):
    '''
    This function is used to generate bboxs in each grid
    scale: scale of bbox area
    ratio: aspect ratio of bbox
    grid_num: number of grid in each row and column
    '''
    bboxs_ls = np.zeros([grid_num**2, 4])
    iw, ih = 1, 1
    N_grid = grid_num ** 2
    for gid in range(N_grid):
        gw , gh = iw / grid_num, ih / grid_num
        gx, gy = gid % grid_num, gid // grid_num
        gx1, gx2 = gx * iw / grid_num + gw/4, (gx + 1) * iw / grid_num - gw/4
        gy1, gy2 = gy * ih / grid_num + gh/4, (gy + 1) * ih / grid_num - gh/4

        scale, ratio = scale, ratio

        target_area = random.uniform(*scale)
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = math.sqrt(target_area * aspect_ratio)
        h = math.sqrt(target_area / aspect_ratio)

        xc = random.uniform(gx1, gx2)
        yc = random.uniform(gy1, gy2)

        x1, y1 = max(0, xc - w / 2), max(0, yc - h / 2)
        x2, y2 = min(1, xc + w / 2), min(1, yc + h / 2)
        bboxs_ls[gid] = np.array([x1,y1,x2,y2])
    return bboxs_ls

def RCC(msk, resized_bboxs, view_size, cutout_ratio=0.3):
    '''
    This function is used to generate RCC mask
    msk: mask to be processed
    resized_bboxs: bboxs in the image
    view_size: size of the image
    cutout_ratio: ratio of cutout
    '''
    b_size = (resized_bboxs[:, 2] - resized_bboxs[:, 0]) * (resized_bboxs[:, 3] - resized_bboxs[:, 1])
    _, b_ls = b_size.sort(descending=True)
    for bid in b_ls:
        bbox = resized_bboxs[bid]

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = round(float(x1) * view_size[0])
        x2 = min(round(float(x2) * view_size[1]), view_size[0]-1)
        y1 = round(float(y1) * view_size[0])
        y2 = min(round(float(y2) * view_size[1]), view_size[0]-1)

        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        cur_size = msk[y1:y2 + 1, x1:x2 + 1].sum()
        cur_ratio = (1 - cur_size / area)

        if cur_ratio > cutout_ratio:
            msk[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 1
            cur_ratio = 0
        cur_rcc_ratio = cutout_ratio - cur_ratio

        target_area = cur_rcc_ratio * area
        ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
        w = math.sqrt(target_area * ratio)
        h = math.sqrt(target_area / ratio)

        if w == 0 or h == 0:
            continue

        cut_x1 = random.random()*(x2-x1-w) + x1
        cut_y1 = random.random()*(y2-y1-h) + y1
        cut_x2 = cut_x1 + w - 1
        cut_y2 = cut_y1 + h - 1

        cut_x1 = round(cut_x1)
        cut_y1 = round(cut_y1)
        cut_x2 = min(round(cut_x2), x2)
        cut_y2 = min(round(cut_y2), y2)

        msk[cut_y1:cut_y2+1, cut_x1:cut_x2+1] = 0
    return msk