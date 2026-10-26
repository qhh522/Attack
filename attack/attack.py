import os
import time
import torch
import mmcv
import cv2
import torch.nn.functional as F
import random
from mmengine.registry import init_default_scope

import logging
from PIL import Image

from utils.utils import *
from omegaconf import OmegaConf

logging.basicConfig(filename='w_try0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sibling_attack(im_np, pert, bboxes_tgt, labels_tgt,det_model, det_model2, config):

    epochs = config.attack['outer_loops']
    alpha = config.attack['alpha']
    eps = config.attack['eps']
    INNER_MAX_EPOCH = config.attack['inner_loops']
    magic = config.attack['gamma']

    
    
    for i in range(1, epochs+1):
        pert_init = pert.detach().clone()
        pre = time.time()
        if i % 2 != 0:
            INNER_LR = magic
            pert_list = []
            for j in range(INNER_MAX_EPOCH):
                loss = det_model.loss(im_np, pert, bboxes_tgt, labels_tgt)
                det_model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    pert = pert - 1.5 * INNER_LR *torch.sign(pert.grad)
                    pert = pert.clamp(min=-10, max=10)
                    pert.requires_grad = True
                pert_list.append(pert)

            SEG_grad_temp_list = []
            #seg_model=model_train_seg()
            for pert in pert_list:
                loss = det_model2.loss(im_np, pert, bboxes_tgt, labels_tgt)
                det_model2.zero_grad()
                loss.backward()
                grad = pert.grad
                SEG_grad_temp_list.append(grad.clone())

            aggr_grad_pic = torch.zeros_like(pert)
            for SEG_grad_temp in SEG_grad_temp_list:
                aggr_grad_pic += SEG_grad_temp

            seg_pert = pert_list[-1]
            loss = det_model2.loss(im_np, seg_pert, bboxes_tgt, labels_tgt)
            det_model2.zero_grad()
            loss.backward()
            seg_grad=seg_pert.grad

            w = 0.00005
            sign_grad = torch.sign(seg_grad + w * aggr_grad_pic)
            pert = pert_init - magic * alpha * sign_grad
            pert = pert.clamp(min=-10, max=10)
            pert.requires_grad = True

            print("[Epoch-%d](DET-branch) SEG loss: %f, time cost: %fs" % (i, loss.item(), time.time() - pre))
            logging.info("[Epoch-%d](DET-branch) SEG loss: %f, time cost: %fs" % (i, loss.item(), time.time() - pre))
        else:
            INNER_LR =  1 - magic
            pert_list = []

            for j in range(INNER_MAX_EPOCH):
                loss = det_model2.loss(im_np, pert, bboxes_tgt, labels_tgt)
                det_model2.zero_grad()
                loss.backward()
                with torch.no_grad():
                    pert = pert - 1.5 * INNER_LR *torch.sign(pert.grad)
                    pert = pert.clamp(min=-10, max=10)
                    pert.requires_grad = True
                pert_list.append(pert)

            DET_grad_temp_list = []
            for pert in pert_list:
                loss = det_model.loss(im_np, pert, bboxes_tgt, labels_tgt)
                det_model.zero_grad()
                loss.backward()
                grad = pert.grad
                DET_grad_temp_list.append(grad.clone())

            aggr_grad_pic = torch.zeros_like(pert)
            for DET_grad_temp in DET_grad_temp_list:
                aggr_grad_pic += DET_grad_temp

            det_pert = pert_list[-1]
            loss = det_model.loss(im_np, det_pert, bboxes_tgt, labels_tgt)
            det_model.zero_grad()
            loss.backward()
            det_grad=det_pert.grad

            w = 0.00005
            sign_grad = torch.sign(det_grad + w * aggr_grad_pic)

            pert = pert_init - (1- magic) * alpha * sign_grad
            pert = pert.clamp(min=-10, max=10)
            pert.requires_grad = True     

            print("[Epoch-%d](SEG-branch) DET loss: %f, time cost: %fs" % (i, loss.item(), time.time() - pre))
            logging.info("[Epoch-%d](SEG-branch) DET loss: %f, time cost: %fs" % (i, loss.item(), time.time() - pre))

    return pert


if __name__ == '__main__':
    config = OmegaConf.load('attack/config.yaml')
    gpu = config.attack['gpu0']
    dataset_name = config.dataset['dataset_name']
    device = torch.device('cuda:' + str(gpu))
    n_labels=80
    success_num=0
    black_success_num=0
    total=0
    attack_img_paths = obtain_attacker_victim(config)

    epochs = config.attack['outer_loops']
    alpha = config.attack['alpha']
    eps = config.attack['eps']
    INNER_MAX_EPOCH = config.attack['inner_loops']
    magic = config.attack['gamma']
    w = 0.00005
    logging.info(f'w: {w}')
    logging.info(f'epochs: {epochs}')
    logging.info(f'alpha: {alpha}')
    logging.info(f'eps: {eps}')
    logging.info(f'INNER_MAX_EPOCH: {INNER_MAX_EPOCH}')
    logging.info(f'magic: {magic}')

    det_model=model_train(config_file='mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',checkpoint_file='work_dirs_coco/faster-rcnn_r50_fpn_1x_coco/epoch_12.pth',device=device)
    det_model2=model_train(config_file='mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py',checkpoint_file='work_dirs_co/retinanet_r50_fpn_1x_coco/epoch_12.pth',device=device)
    black_model=model_train(config_file='mmdetection/configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py',checkpoint_file='mmdetection/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth',device=device)
    

    for attack_img_path in attack_img_paths:
        print("========",attack_img_path, "========")
        logging.info("======== %s ========" % attack_img_path)
        
        im_np_init = np.array(Image.open(attack_img_path).convert('RGB'))
        det = black_model.det(im_np_init)
        print("det:",det)
        logging.info("det: %s" % det)
        det_det=det_model.det(im_np_init)
        print("det_det:",det_det)
        logging.info("det_det: %s" % det_det)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]

        if len(det)==0:
            continue
        all_categories = set(labels.astype(int))
        victim_idx = random.randint(0,len(det)-1)
        victim_class = int(det[victim_idx,4])
        print("victim_class:",victim_class)
        logging.info("victim_class: %d" % victim_class)

        select_n = 1
        target_pool = list(set(range(n_labels)) - all_categories)
        target_pool = np.random.permutation(target_pool)[:select_n]
        target_class = int(target_pool[0])
        print("target_class:",target_class)
        logging.info("target_class: %d" % target_class)

        tar_gt_det = det.copy()
        tar_gt_det[victim_idx, 4] = target_class
        target_clean = tar_gt_det[victim_idx,:][None]
        bboxes_tgt = tar_gt_det[:,:4].astype(np.float32)
        labels_tgt = tar_gt_det[:,4].astype(int).copy()
        print("tar_gt_det:",tar_gt_det)
        logging.info("tar_gt_det: %s" % tar_gt_det)


        im = torch.from_numpy(im_np_init).permute(2,0,1).unsqueeze(0).float().to(device)
        im_np = im.squeeze().cpu().numpy().transpose(1, 2, 0)
        adv = torch.clone(im)
        pert = adv - im
        pert.requires_grad = True

        pert = sibling_attack(im_np, pert, bboxes_tgt, labels_tgt, det_model, det_model2, config)
        adv = (im + pert).clip(0, 255)
        adv_np = adv.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
        det_adv=det_model.det(adv_np)
        
        black_det_adv=black_model.det(adv_np)
        print("black_det_adv:",black_det_adv)
        logging.info("black_det_adv: %s" % black_det_adv)
        success_num=success_num+is_success(det_adv, target_clean)
        black_success_num=black_success_num+is_success(black_det_adv, target_clean)
        total=total+1
        print("total:",total)
        print("success_num:",success_num)
        print("black_success_num:",black_success_num)
        print("success_rate:",success_num/total)
        print("black_success_rate:",black_success_num/total)
        logging.info("total: %d" % total)
        logging.info("success_num: %d" % success_num)
        logging.info("black_success_num: %d" % black_success_num)
        logging.info("success_rate: %f" % (success_num / total))
        logging.info("black_success_rate: %f" % (black_success_num / total))

        if is_success(det_adv, target_clean) == 1:
            with open("successful_attacks_sib.txt", "a") as file:
                file.write(attack_img_path + "\n")
        
        if is_success(black_det_adv, target_clean) == 1:
            with open("black_successful_attacks_sib.txt", "a") as file:
                file.write(attack_img_path + "\n")

        save_dir = './' + dataset_name +  '_results_adv_images/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = save_dir + attack_img_path.split('/')[-1].split('.')[0] + '.png'
        adv_np = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, adv_np)
        print("Save adversarial image to - ", save_path)
        logging.info("Save adversarial image to - %s" % save_path)

