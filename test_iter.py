# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import LoadDatasetFromFolder, DA_DatasetFromFolder, calMetric_iou
import numpy as np
import random
from utils.cdan import ConditionalDomainAdversarialLoss
from utils.metric import cal_average
from utils.set_seed import set_seed
from utils.domain_discriminator import DomainDiscriminator
from utils.teacher import EMATeacher
from utils.sam import SAM
from utils.masking import Masking
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from return_models import return_models
import itertools
from loss.losses import cross_entropy
from argparse import ArgumentParser
from models.unet.unet_model import UNet
from utils.data import ForeverDataIterator

url = ''
url2 = ''
url3 = ''
url4 = ''


# train_set = DA_DatasetFromFolder(url, url2, crop=False)
# target_set = DA_DatasetFromFolder(url3, url4, crop=False)
# train_loader = DataLoader(dataset=train_set, num_workers=3, batch_size=11, shuffle=True)
# target_loader = DataLoader(dataset=target_set, num_workers=3, batch_size=4, shuffle=True)
    
    
# train_source_iter = ForeverDataIterator(train_loader)
# train_target_iter = ForeverDataIterator(target_loader)
# a = next(train_source_iter)
# b = next(train_target_iter)
    
# for i in range(100):
#     a = next(train_source_iter)
#     print(a[0].shape)
#     b = next(train_target_iter)
#     print(a[0].shape)
device = torch.device('cuda:0')
a = torch.cuda.device_count()

print(a)