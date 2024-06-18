import torch
from kornia.augmentation import Normalize, ColorJitter, RandomGrayscale, RandomSolarize

class aug1(torch.nn.Module):
    def __init__(self):
        super(aug1, self).__init__()
        self.nor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.colorjitter = ColorJitter(0.4, 0.4, 0.2, 0.1, p=.8)
        self.gray = RandomGrayscale(p=.2)

    def forward(self, img):
        img = self.colorjitter(img)
        img = self.gray(img)
        img = self.nor(img)
        return img


class aug2(torch.nn.Module):
    def __init__(self):
        super(aug2, self).__init__()
        self.nor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.colorjitter = ColorJitter(0.4, 0.4, 0.2, 0.1, p=.8)
        self.gray = RandomGrayscale(p=.2)
        self.solarize = RandomSolarize(0,0,p=.2)

    def forward(self, img):
        img = self.colorjitter(img)
        img = self.gray(img)
        img = self.solarize(img)
        img = self.nor(img)
        return img