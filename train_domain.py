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
#mutual = Mutual_info_reg(input_channels=64, channels=128, size=128,device=device, latent_size=2).to(device)
import argparse
from utils.data import ForeverDataIterator
from utils.masking import flip_values#(matrix, image,flip_prob):
import torchvision.transforms as transforms




#training options
parser = argparse.ArgumentParser(description='Training Change Detection Network')
##color_aug brightness contrast saturation hue
parser.add_argument('--color_aug', default='False', type=str, help='train epoch number')
parser.add_argument('--brightness', default=0.5, type=float, help='train epoch number')
parser.add_argument('--contrast', default=0.5, type=float, help='train epoch number')
parser.add_argument('--saturation', default=0.5, type=float, help='train epoch number')
parser.add_argument('--hue', default=0.5, type=float, help='train epoch number')
parser.add_argument('--mix_result', default='True', type=str, help='train epoch number')
#color_aug_dual
parser.add_argument('--deep_sup', default='True', type=str, help='train epoch number')

parser.add_argument('--color_aug_dual', default='False', type=str, help='train epoch number')

#source_using_mask  source_mask_patch source_mask_ratio
parser.add_argument('--source_using_mask', default='False', type=str, help='train epoch number')
parser.add_argument('--source_mask_patch', default=1, type=int, help='train epoch number')
parser.add_argument('--source_mask_ratio', default=0.2, type=float, help='train epoch number')


parser.add_argument('--weight_source_embedding', default=0.00001, type=float, help='train epoch number')
#--weight_source_embedding  0.00001


# training parameters   
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=10, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=5, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=6, type=int, help='num of workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0,1", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
#/CLCD_Google
# path for loading data from folder/home/wei/Dataset/BCDD//home/liwei/gutai2/home/wei/SSDnew/Dataset/
#parser.add_argument('--hr1_train', default='/root/Dataset/BCDD/train/A', type=str, help='image at t1 in training set')
#parser.add_argument('--hr2_train', default='/root/Dataset/BCDD/train/B', type=str, help='image at t2 in training set')
#parser.add_argument('--lab_train', default='/root/Dataset/BCDD/train/label', type=str, help='label image in training set')

#parser.add_argument('--hr1_val', default='/root/Dataset/BCDD/val/A', type=str, help='image at t1 in validation set')
#parser.add_argument('--hr2_val', default='/root/Dataset/BCDD/val/B', type=str, help='image at t2 in validation set')
#parser.add_argument('--lab_val', default='/root/Dataset/BCDD/val/label', type=str, help='label image in validation set')

# network saving 
#parser.add_argument('--model_dir', default='./epochs/BCDD/', type=str, help='model save path')
#HRC_WHU   #L8SPARCS    #CloudS26
parser.add_argument('--dataset_name_source', required=False,default='CloudS26', type=str, help='model save path')
parser.add_argument('--dataset_name_target', required=False,default='L8SPARCS', type=str, help='model save path')

parser.add_argument('--val_data', required=False,default='HRC_WHU', type=str, help='model save path')

parser.add_argument('--model_name', required=False,default='UNext_S', type=str, help='model save path')
#python train.py --dataset_name 'L8SPARCS' --gpu_id '1' --batchsize 8 --model_name 'Unet';
parser.add_argument('--using_mask', default='True', type=str, help='label image in validation set')
parser.add_argument('--mask_patch', default=1, type=int, help='label image in validation set')
parser.add_argument('--mask_ratio', default=0.2, type=float, help='label image in validation set')
#weight_source_loss
parser.add_argument('--weight_source_loss', default=0.99, type=float, help='should be small than 1')

args = parser.parse_args()
# all_input   input_900
#L8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCSL8SPARCS
parser.add_argument('--hr1_train', default='/root/cloud_mask/'+args.dataset_name_source+'/all_input', type=str, help='image at t1 in training set')
parser.add_argument('--lab_train', default='/root/cloud_mask/'+args.dataset_name_source+'/all_output', type=str, help='label image in training set')

parser.add_argument('--hr2_train', default='/root/cloud_mask/'+args.dataset_name_target+'/input_900', type=str, help='image at t1 in training set')
parser.add_argument('--lab2_train', default='/root/cloud_mask/'+args.dataset_name_target+'/output_900', type=str, help='label image in training set')



parser.add_argument('--hr1_val', default='/root/cloud_mask/'+args.val_data+'/all_input', type=str, help='image at t1 in validation set')
parser.add_argument('--lab_val', default='/root/cloud_mask/'+args.val_data+'/all_output', type=str, help='label image in validation set')




parser.add_argument('--hr2_val', default='/root/cloud_mask/'+args.val_data+'/all_input', type=str, help='image at t1 in validation set')
parser.add_argument('--lab2_val', default='/root/cloud_mask/'+args.val_data+'/all_output', type=str, help='label image in validation set')
args = parser.parse_args()
parser.add_argument('--model_dir', default='./epochs/'+args.dataset_name_source+'_'+args.dataset_name_target+'/'+args.model_name+'/'+str(args.weight_source_embedding)+'-'+args.source_using_mask+'-'+str(args.source_mask_patch)+'-'+ str(args.source_mask_ratio)+'-'+args.using_mask+'-'+str(args.mask_patch)+'-'+str(args.mask_ratio)+str(args.weight_source_loss)+'/', type=str, help='model save path')



args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# mse = torch.nn.MSELoss(reduction='mean')

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

# set seeds
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)
#
base_optimizer = torch.optim.Adam
if args.using_mask == 'True':
    masking  = Masking(block_size=args.mask_patch, ratio=args.mask_ratio,color_jitter_s=0.2,color_jitter_p=0.2, blur=True,mean=torch.Tensor([0,0,0]),std=torch.Tensor([1,1,1]))
#source_using_mask  source_mask_patch source_mask_ratio
if args.source_using_mask == 'True': 
    masking_source  = Masking(block_size=args.source_mask_patch, ratio=args.source_mask_ratio,color_jitter_s=0.2,color_jitter_p=0.2, blur=True,mean=torch.Tensor([0,0,0]),std=torch.Tensor([1,1,1]))

if args.color_aug == 'True':
    color_aug = transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue)
#color_aug brightness contrast saturation hue
weight_source_embedding = args.weight_source_embedding



if __name__ == '__main__':
    mloss = 0

    #
    # load data
    train_set = DA_DatasetFromFolder(args.hr1_train, args.lab_train, crop=False)
    target_set = DA_DatasetFromFolder(args.hr2_train, args.lab2_train, crop=False)
    #target_set = ConcatDataset([target_set, target_set,target_set])
    
    val_set = LoadDatasetFromFolder(args, args.hr2_train, args.lab2_train)
    
    print("train_set:len,target_set:len, val_set:len",len(train_set),len(target_set),len(val_set))
    
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    target_loader = DataLoader(dataset=target_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True,drop_last=True)
    val_loader = DataLoader(dataset=target_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)
    
    train_source_iter = ForeverDataIterator(train_loader,device=device)
    train_target_iter = ForeverDataIterator(target_loader,device=device)

    
    print(args)
    print("train_loader:len,target_loader:len, val_loader:len",len(train_loader),len(target_loader),len(val_loader))
    #val_loader = train_loader

    # define model
    #CDNet = UNet(3,2)
    CDNet = return_models(args.model_name)
    CDNet = CDNet.to(device, dtype=torch.float)
    
    CDNet.load_state_dict(torch.load("/root/mxx_code/cloud_mask_code/epochs/HRC_WHU_CloudS26/UNext_S/1e-14-False-1-0.4-True-1-0.251.0/netCD_epoch_best_0.532469778670386.pth"))
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))
        
    teacher = EMATeacher(CDNet, alpha=0.1, pseudo_label_weight=None)
    
    if torch.cuda.device_count() > 2:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # set optimization
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr= args.lr, betas=(0.9, 0.999))
    #optimizer = torch.nn.DataParallel(optimizer, device_ids=range(torch.cuda.device_count()))
    
    CDcriterionCD = cross_entropy().to(device, dtype=torch.float)
    loss_list_all = []

    loss_list_all_1 = []
    loss_list_all_2 = []
    loss_list_all_3 = []
    # training
    
    
    for epoch in range(0, args.num_epochs + 1):
        
        #train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'edge_loss':0, 'CD_loss':0, 'loss': 0 }

        CDNet.train()
        
        #for i,((source, source_label), (target, target_label)) in enumerate(zip(train_loader,target_loader)):
            #
        train_bar = tqdm(train_loader)
        for i, _  in enumerate(train_bar):
        #for i  in range(2):
            running_results['batch_sizes'] += args.batchsize
            #
            source, source_label = next(train_source_iter)
            target, target_label = next(train_target_iter)
            
            #(1) 获得 dataloader 的数据
            source       =  source.to(device, dtype=torch.float)
            source_label =  source_label.to(device, dtype=torch.float)
            target       =  target.to(device, dtype=torch.float)
            #print(source_label)
            #print(source_label.shape)
            target_label =  target_label.to(device, dtype=torch.float)
            #rand_num = random.random()
            if args.source_using_mask == 'True':
                source = masking_source(source)
            else:
                source = source
    # 判断随机数是否小于设定的概率
            # if rand_num < 0.2:
            #     source =  masking(source)
            #(2) masking target data input
            
            
            if args.using_mask == 'True':
                target_masked=  masking(target)
            else:
                target_masked = target
                
            # if args.color_aug_dual == 'True':
            #     #if epoch <= 30:
            #         target =  color_aug(target)
            #         #source =  color_aug(source)
                
            
            if args.color_aug == 'True':
                target_masked = color_aug(target_masked)
            # (3) using teacher model to generate pseudo-label
            if epoch == 0:
                teacher.update_weights(CDNet, 0)
            #teacher.update_weights(CDNet, epoch * len(train_loader) + i)
            #print(i,"-=-=-=-=")
            teacher.update_weights(CDNet, i+epoch)
            if args.deep_sup == 'True':
                pseudo_label_target = teacher(target)[0]
            else:
                pseudo_label_target = teacher(target)
            
            
            
            # (4) 获得一个正经的source的有监督的loss # 计算源数据的MSE损失
            combined_data = torch.cat((source, target), dim=0) #64,5,16,193
            combined_label = torch.cat((source_label, pseudo_label_target), dim=0)
            
            
            if args.deep_sup == 'True':
                predictions,predictions_2,predictions_3 = CDNet(combined_data,combined_label,weight_source_embedding)#64,3
            else:    
                predictions = CDNet(combined_data)
            if args.deep_sup == 'True':
                pred_source, pred_target     = predictions.chunk(2, dim=0)
                pred_source_2, pred_target_2 = predictions_2.chunk(2, dim=0)
                pred_source_3, pred_target_3 = predictions_3.chunk(2, dim=0)
            else:         
                pred_source, pred_target = predictions.chunk(2, dim=0)  #分割成两个块
            
            
            
            #print(source_label.shape,"before")
            source_label = torch.argmax(source_label, 1).unsqueeze(1).float()
            #
            pseudo_label_target = torch.argmax(pseudo_label_target, 1).unsqueeze(1).float()
            #print(source_label.shape,"after")
            #print(source_label)
            if args.deep_sup == 'True':
                loss_source  = CDcriterionCD(pred_source, source_label)
                loss_source += CDcriterionCD(pred_source_2, source_label)
                loss_source += CDcriterionCD(pred_source_3, source_label)
                
                loss_source += CDcriterionCD(pred_target, pseudo_label_target)
                loss_source += CDcriterionCD(pred_target_2, pseudo_label_target)
                loss_source += CDcriterionCD(pred_target_3, pseudo_label_target)
            else:
                loss_source = CDcriterionCD(pred_source, source_label) 
            
            
            
            
            
            # (5) 使用mask掉的输入，来预测mask的输出，不需要hidden feature
            
            
            
            #pseudo_label_target = torch.argmax(pseudo_label_target, 1).unsqueeze(1).float()
            #print(pseudo_label_target.shape) # 1 512 512； 0是无云 1 是有云
            #让1  有百分之10的概率变成 0  原本是0的全变成1  
            target_masked_masked = flip_values(pseudo_label_target, target_masked,args.mask_ratio)
            if args.deep_sup == 'True':
                pred_target_masked,pred_target_masked_2,pred_target_masked_3 = CDNet(target_masked_masked)
            else:
                pred_target_masked = CDNet(target_masked_masked)
            #二次 mask
            loss_masking  = CDcriterionCD(pred_target_masked, pseudo_label_target)
            loss_masking += CDcriterionCD(pred_target_masked_2, pseudo_label_target)
            loss_masking += CDcriterionCD(pred_target_masked_3, pseudo_label_target)
            
            # loss_masking 是 伪label的预测 和 mask后的 预测的label
            loss = loss_masking + args.weight_source_loss * loss_source
            # loss = loss_mse_source
            # print(loss)
            # base_optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            optimizer.step()
            #
            #CDNet.zero_grad()
            
            
            
            #第二个大步骤
            if args.deep_sup == 'True':
                # The code snippet is assigning the output of the `CDNet` function called with the
                # `combined_data` argument to three variables `y`, `y_2`, and `y_3`.
                y,y_2,y_3 = CDNet(combined_data,combined_label,weight_source_embedding) 
            else:
                y= CDNet(combined_data) #  x 包含，source 和 target
            if args.deep_sup == 'True':
                y_s, y_t = y.chunk(2, dim=0)
                y_s_2, y_t_2 = y_2.chunk(2, dim=0)
                y_s_3, y_t_3 = y_3.chunk(2, dim=0)
        
       
            else:
                y_s, y_t = y.chunk(2, dim=0)
       
            if args.deep_sup == 'True':
                loss_source  = CDcriterionCD(y_s, source_label)
                loss_source += CDcriterionCD(y_s_2, source_label)
                loss_source += CDcriterionCD(y_s_3, source_label)
            else:
                loss_source = CDcriterionCD(y_s, source_label)
             
            # print(loss_source.shape,"loss_source")
            # print(loss_source,"loss_source")
            
            if args.deep_sup == 'True':
                y_t_masked,y_t_masked_2,y_t_masked_3 = CDNet(target_masked,weight_source_embedding)
                
            else:
                y_t_masked = CDNet(target_masked)
          
            #transfer_loss = domain_adv_loss(y_s, f_s, y_t, f_t) +   
            #print(y_t.shape,"y_t") #torch.Size([10, 2, 512, 512]) torch.Size([10, 1, 512, 512])
            #mcc 这个mcc，好像做定位回归 用不到
            #y_t = torch.argmax(y_t, 1).unsqueeze(1).float()
            #loss2 = CDcriterionCD(y_t_masked, y_t) 
            if args.deep_sup == 'True':
                loss2 = F.mse_loss(y_t_masked, y_t)
                loss2 += F.mse_loss(y_t_masked_2, y_t_2)
                loss2 += F.mse_loss(y_t_masked_3, y_t_3)
            else:
                loss2 = F.mse_loss(y_t_masked, y_t)
            
            #domain_acc = domain_adv_loss.domain_discriminator_accuracy # 这个是区分二元分类的准确度，也就是分： 1.source 2.target，做展示的，并不是真正的loss
            loss  = args.weight_source_loss * loss_source + loss2
            
            loss.backward()
            
            # base_optimizer.step()
            # optimizer.step()
            optimizer.step()
        
            running_results['CD_loss'] += loss.item() * args.batchsize

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, args.num_epochs,
                    running_results['CD_loss'] / running_results['batch_sizes'],
                    ))

        # eval
        CDNet.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            inter_1, unin_1 = 0,0
            inter_2, unin_2 = 0,0
            inter_3, unin_3 = 0,0
            valing_results = {'batch_sizes': 0, 'IoU': 0,'IoU_1':0,'IoU_2':0,'IoU_3':0}

            for hr_img1, label in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                hr_img1 = hr_img1.to(device, dtype=torch.float)
               
                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()
                if args.mix_result == 'True':
                    cd_map_1,cd_map_2,cd_map_3 = CDNet(hr_img1)
                    cd_map = cd_map_1 + cd_map_2 + cd_map_3
                else:
                    cd_map = CDNet(hr_img1)
                #print(cd_map.shape)
                cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()
                cd_map_1 = torch.argmax(cd_map_1, 1).unsqueeze(1).float()
                cd_map_2 = torch.argmax(cd_map_2, 1).unsqueeze(1).float()
                cd_map_3 = torch.argmax(cd_map_3, 1).unsqueeze(1).float()
                
                gt_value = (label > 0).float()
                prob = (cd_map > 0).float()
                prob = prob.cpu().detach().numpy()

                prob_1 = (cd_map_1 > 0).float()
                prob_1 = prob_1.cpu().detach().numpy()
                
                prob_2 = (cd_map_2 > 0).float()
                prob_2 = prob_2.cpu().detach().numpy()
                
                prob_3 = (cd_map_3 > 0).float()
                prob_3 = prob_3.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                
                
                result = np.squeeze(prob)
                
                result_1 = np.squeeze(prob_1)
                result_2 = np.squeeze(prob_2)
                result_3 = np.squeeze(prob_3)
                
                intr, unn = calMetric_iou(gt_value, result)
                inter = inter + intr
                unin  = unin + unn
                
                intr_1, unn_1 = calMetric_iou(gt_value, result_1)
                inter_1 = inter_1 + intr_1
                unin_1  = unin_1 + unn_1
                
                intr_2, unn_2 = calMetric_iou(gt_value, result_2)
                inter_2 = inter_2 + intr_2
                unin_2  = unin_2 + unn_2
                
                intr_3, unn_3 = calMetric_iou(gt_value, result_3)
                inter_3 = inter_3 + intr_3
                unin_3  = unin_3 + unn_3
                
                valing_results['IoU'] = (inter * 1.0 / unin)
                valing_results['IoU_1'] = (inter_1 * 1.0 / unin_1)
                valing_results['IoU_2'] = (inter_2 * 1.0 / unin_2)
                valing_results['IoU_3'] = (inter_3 * 1.0 / unin_3)
                
                val_bar.set_description(
                    desc='IoU: %.4f' % (  valing_results['IoU'],))
                # val_bar.set_description(
                #     desc='IoU_1: %.4f' % (  valing_results['IoU_1'],))
                # val_bar.set_description(
                #     desc='IoU_2: %.4f' % (  valing_results['IoU_2'],))
                # val_bar.set_description(
                #     desc='IoU_3: %.4f' % (  valing_results['IoU_3'],))
#valing_results['IoU_1'],valing_results['IoU_2'],valing_results['IoU_3']



        # save model parameters
        val_loss = valing_results['IoU']
        loss_list_all.append(val_loss)
        loss_list_all_1.append(valing_results['IoU_1'])
        loss_list_all_2.append(valing_results['IoU_2'])
        loss_list_all_3.append(valing_results['IoU_3'])
        #print("loss_list_all",loss_list_all)
        print("IoU Change",loss_list_all)
        print("IoU_1 Change",loss_list_all_1)
        print("IoU_2 Change",loss_list_all_2)
        print("IoU_3 Change",loss_list_all_3)
        if val_loss > mloss or epoch==1:
            mloss = val_loss
            torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_best.pth')
            if val_loss >= 0.40:
                torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_best_{}.pth'.format(str(val_loss)))
        #

        #
        aug_par = args.color_aug+'-'+str(args.brightness)+'-'+str(args.contrast)+'-'+str(args.saturation)+'-'+str(args.hue)
        with open("/root/cloud_mask_code/output/{}.txt".format(args.model_name), "a") as file:
            file.write("Setting is : " + str(args) + "\n")
            file.write("Epoch is : " + str(epoch) + "\n")
            file.write("IoU is : " + str(loss_list_all) + "\n")
        if epoch >= 110:  
            with open("/root/cloud_mask_code/output/{}_last_10_epoch.txt".format(args.model_name), "a") as file:
                file.write("Setting is : " + str(args) + "\n")
                #file.write("Epoch is : " + str(epoch) + "\n")
                file.write("IoU is : " + str(loss_list_all) + "\n")
