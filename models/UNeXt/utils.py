import argparse
import torch.nn as nn
import torch
import math


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
#获得，反转的target的值，也就是把0变成云，把1变成背景的mask
# 创建掩码
def rever_label(label_target):
    zero_mask = (label_target == 0).to(label_target.device)
    one_mask = (label_target == 1).to(label_target.device)
    # 替换零值和一值
    label_target_change = torch.where(one_mask, torch.tensor(0), torch.where(zero_mask, torch.tensor(1), label_target)).to(label_target.device)
    #0 是 云  1 是背景 return的
    return label_target_change.to(label_target.device)

def return_0_cloud_embedding(label_target_change,embedding):
    label_target_change = label_target_change # 0是云，1是背景
    embedding = label_target_change*embedding
    return embedding.to(embedding.device)

def return_0_cloud_embedding_mean_std(embedding):
    # 获取非零值的掩码
    nonzero_mask = (embedding != 0).to(embedding.device)

    # 获取非零值
    nonzero_values = embedding[nonzero_mask].to(embedding.device)

    #print(nonzero_values,"-=-=-=-")
    # 计算均值和标准差
    mean = torch.mean(nonzero_values).to(embedding.device)
    std = torch.std(nonzero_values).to(embedding.device)
    if torch.isnan(std).any() or ((std < 0).any()) or torch.isnan(mean).any():
        mean = torch.tensor(0.0).to(embedding.device)
        std = torch.tensor(0.0).to(embedding.device)
    return mean.to(embedding.device),std.to(embedding.device)
    #print("Mean:", mean,)
    #print("Standard Deviation:", std)
    
#输入，source的embedding，背景变成0，直接用原来的label乘，就行。
def change_embedding_background(embedding,label,mean,std,weight_source_embedding):
    # print("-=-=-=-=-=-") .to(embedding.device)
    # print(type(mean),type(std))
    # print(mean.shape,std.shape)
    # print((mean),(std))
    embedding_background_0 = embedding * label #背景变成0
    background_mask = (embedding_background_0 == 0).to(embedding.device)
    new_values = torch.normal(mean.item(), std.item(), size=embedding.shape).to(embedding.device)
    #print(weight_source_embedding)
    new_values = new_values*torch.Tensor([weight_source_embedding]).to(embedding.device) + embedding
    
    # 根据B的背景区域，填充新的张量值到A中
    new_embedding = torch.where(background_mask, new_values, embedding).to(embedding.device)
    return new_embedding.to(embedding.device)

# #
#         if temp.shape[0] == 20:
#             label_source    , label_target     = label.chunk(2, dim=0)
#             embedding_source, embedding_target = temp_out5.chunk(2, dim=0)
    
#             mean,std = return_0_cloud_embedding_mean_std(return_0_cloud_embedding(rever_label(label_target),embedding_target))

#             embedding_source = change_embedding_background(embedding_source,label_source,mean,std,weight_source_embedding)

#             out_5 = torch.cat((embedding_source, embedding_target), dim=0)
#             out_5 = self.classifer_5(out_5)
#torch.cat((source, target), dim=0)
#predictions_2.chunk(2, dim=0)