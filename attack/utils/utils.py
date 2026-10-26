import numpy as np
import torch
from torch.nn import functional as F
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmcv.transforms import Compose
from mmdet.models import DetDataPreprocessor

from mmseg.apis import init_model
from mmseg.apis import inference_model
from mmengine.dataset import Compose as Compose_seg
from mmseg.models import SegDataPreProcessor

CITYSCAPES_BBOX_LABEL_NAMES = (
    'person', 
    'rider', 
    'car', 
    'truck',
    'bus', 
    'train',
    'motorcycle', 
    'bicycle'
    )

def output2det(outputs, im, conf_thres=0.5, dataset='voc'):

    labels = outputs.pred_instances.labels.cpu().numpy()
    scores = outputs.pred_instances.scores.cpu().numpy()
    bboxes = outputs.pred_instances.bboxes.cpu().numpy()

    det = np.hstack((bboxes[:, :2], bboxes[:, 2:], labels.reshape(-1, 1), scores.reshape(-1, 1)))


    # if det is empty
    if len(det) == 0: 
        return np.zeros([0,6])

    det = det[det[:,-1] >= conf_thres]
    
    if dataset == 'voc':
        # map the labels from coco to voc
        voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]
        for idx, item in enumerate(det):
            if item[4] not in voc2coco:
                item[4] = -1
            else:
                det[idx,4] = voc2coco.index(item[4])
        det = det[det[:,4] != -1]


    # make the value in range
    m, n, _ = im.shape
    for item in det:
        item[0] = min(max(item[0],0),n)
        item[2] = min(max(item[2],0),n)
        item[1] = min(max(item[1],0),m)
        item[3] = min(max(item[3],0),m)
    return det

def get_conf_thres(model):
    """assign a different confidence threshold for every model
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
    Returns:
        conf_thres (~ float): the confidence threshold
    """
    model_type = model.cfg.model.type
    if model_type in ['FasterRCNN']: # frcnn, foveabox, Libra R-CNN, GN+WS 
        conf_thres = 0.7
    elif model_type in ['RetinaNet', 'SingleStageDetector']: # retina, ssd
        conf_thres = 0.5
    elif model_type in ['YOLOV3']: # yolov3        
        conf_thres = 0.5
    elif model_type in ['FCOS']:       
        conf_thres = 0.2
    elif model_type in ['FOVEA']:
        conf_thres = 0.3
    else:
        conf_thres = 0.1
    return conf_thres

def get_det(model, im):
    """input an image to a model and get the detection
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
        im (~ numpy.ndarray): input image (in rgb format)
        dataset (str): if use 'voc', only the labels within the voc dataset will be returned
    Returns:
        det (~ numpy.ndarray): nx6 array
    """
    result = inference_detector(model, im)
    #conf_thres = 0.3
    conf_thres = get_conf_thres(model)
    det = output2det(result, im, conf_thres, 'coco')
    return det

def get_test_data(model, im):
    cfg = model.cfg
    cfg = cfg.copy()
    cfg.test_pipeline[0].type = 'LoadImageFromNDArray'
    #cfg.test_pipeline[1].scale = (1344, 800)
    test_pipeline = Compose(cfg.test_pipeline)
    data = dict(img=im)
    data = test_pipeline(data)
    return data


def get_train_data(im, pert, data, bboxes, labels,device):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format) / with grad
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """
    device = device

    # BELOW IS TRAIN
    data_train = data.copy()

    '''
    data_transform: test_pipeline: the first step: ATTENTION!!!!!!!!!!!!!!!

    data_test=data.copy()
    data_test['inputs']=F.interpolate(data_test['inputs'].unsqueeze(0), size=(1024, 2048), mode='bilinear')
    data_test['inputs'] = data_test['inputs'].squeeze(0).transpose(0, 1).transpose(1, 2)
    print("data_test$$$$$$$$$$$$$$$$$$$$$$",data_test['inputs'])
    
    '''

    '''change init img to adv img'''

    img = torch.from_numpy(im.copy().transpose((2, 0, 1)))[None].float().to(device).contiguous()
    img = (img + pert).clamp(0,255)
    
    '''data_train['inputs']=img'''

    image_sizes=data_train['data_samples'].img_shape
    ori_sizes = data_train['data_samples'].ori_shape
    img=F.interpolate(img,size=image_sizes, mode='bilinear').squeeze(0)
    w_scale = image_sizes[1] / ori_sizes[1]
    h_scale = image_sizes[0] / ori_sizes[0]
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    gt_bboxes = bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, image_sizes[1])
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, image_sizes[0])

    data_train['data_samples'].gt_instances.bboxes = torch.from_numpy(gt_bboxes).to(device)
    data_train['data_samples'].gt_instances.labels = torch.from_numpy(labels).to(device)

    preprocessor = DetDataPreprocessor(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    data_train['inputs']=img
    data_train['inputs'] = [data_train['inputs']]
    data_train['data_samples'] = [data_train['data_samples']]
    data_train=preprocessor(data_train)
    data_train['inputs']=data_train['inputs'].to(device)
    data_train['data_samples'][0].gt_instances.bboxes=data_train['data_samples'][0].gt_instances.bboxes.to(device)
    data_train['data_samples'][0].gt_instances.labels=data_train['data_samples'][0].gt_instances.labels.to(device)
    data_train['data_samples'][0].ignored_instances.bboxes=data_train['data_samples'][0].ignored_instances.bboxes.to(device)
    data_train['data_samples'][0].ignored_instances.labels=data_train['data_samples'][0].ignored_instances.labels.to(device)
    #print("data_train:",data_train)
    return data_train



def get_loss_from_dict(loss_dict):
    
    if 'loss_rpn_cls' in loss_dict:
        losses = loss_dict['loss_cls'] + loss_dict['loss_bbox'] + sum(loss_dict['loss_rpn_cls']) + sum(loss_dict['loss_rpn_bbox'])
    else:
        losses = sum(loss_dict['loss_cls']) + sum(loss_dict['loss_bbox'])
        #losses = loss_dict['positive_bag_loss']+loss_dict['negative_bag_loss']
    return losses


class model_train(torch.nn.Module):

    def __init__(self,config_file, checkpoint_file, device) -> None:
        super().__init__()
        self.config_file=config_file
        #'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        self.checkpoint_file=checkpoint_file
        #'mmdetection/epoch_12.pth'
        self.device = device
        self.conf_thres = 0.5

        model_train=init_detector(config_file, checkpoint_file, device)
        #model_train.cfg = mmcv.Config.fromfile(config_file)  # save the config in the model for convenience
        model_train.to(device)
        model_train.eval()
        self.model = model_train
        

    def forward(self, x):
        """inference model using image x
        Args:
            x (numpy.ndarray): input image
            result (list): a list of output from mmdet model
        """
        result = inference_detector(self.model, x)
        return result

    def loss(self, x, pert, bboxes_tgt, labels_tgt):
        """get the loss

        args:
            x (numpy.ndarray):
            pert (tensor):             

        """
        data = get_test_data(self.model, x)
        data_train = get_train_data(x, pert.to(self.device), data, bboxes_tgt, labels_tgt,self.device)

        loss_dict = self.model(**data_train,mode='loss')
        loss = get_loss_from_dict(loss_dict)
        return loss


    def det(self, x):
        """inference model using image x, get the processed output as detection
        
        args:
            x (numpy.ndarray): input image
        """
        result = inference_detector(self.model, x)
        det = output2det(result, x, self.conf_thres ,'coco')
        return det


def get_test_data_seg(model, im):

    cfg = model.cfg
    cfg = cfg.copy()
    cfg.test_pipeline[0].type = 'LoadImageFromNDArray'
    new_step = {'type': 'LoadAnnotations'}
    insert_index = 2
    cfg.test_pipeline.insert(insert_index, new_step)
    test_pipeline = Compose_seg(cfg.test_pipeline)
    data = dict(img=im)
    data = test_pipeline(data)
    return data

    
def get_train_data_seg(im, pert, data, tgt_seg,device):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format) / with grad
        pert (torch.tensor): perturbation
        data (test_data): data fromat forged by test_pipeline
        tgt_seg (np.ndarray): desired semantic segmentation labels
    Returns:
        data_train (): train data format
    """

    # get model device
    device = device

    # BELOW IS TRAIN
    data_train = data.copy()

    img = torch.from_numpy(im.copy().transpose((2, 0, 1)))[None].float().to(device).contiguous()
    img = (img + pert).clamp(0,255)
    image_sizes = data_train['data_samples'].img_shape
    img = F.interpolate(img, size=image_sizes, mode='bilinear', align_corners=True).squeeze(0)

    #data_train['data_samples'].gt_sem_seg.data = torch.tensor(tgt_seg).to(device)
   
    preprocessor = SegDataPreProcessor(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    data_train['inputs']=img
    data_train['inputs'] = [data_train['inputs']]
    data_train['data_samples'] = [data_train['data_samples']]
    data_train=preprocessor(data_train)
    data_train['inputs']=data_train['inputs'].to(device)
    data_train['data_samples'][0].gt_sem_seg.data=torch.from_numpy(tgt_seg).to(device)
    
    return data_train

def get_loss_from_dict_seg(loss_dict):
    """Return the correct loss based on the model type
    Args:
        model_name (~ str): the mmdet model name, eg: 'Faster R-CNN', 'YOLOv3', 'RetinaNet', 'FreeAnchor' ...
        loss_dict (~ dict): the loss of the model, stored in a dictionary
    Returns:
        losses (~ torch.Tensor): the summation of the loss
    """

    if 'aux.loss_ce' in loss_dict.keys():
        losses = loss_dict['decode.loss_ce'] + loss_dict['aux.loss_ce']
    else:
        losses = loss_dict['decode.loss_ce']
    return losses

class model_train_seg(torch.nn.Module):
    """return a model in train mode, such that we can get the loss
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """

    def __init__(self, device) -> None:
        super().__init__()
        config_file='mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb4-80k_coco-stuff164k-512x512.py'
        checkpoint_file='mmsegmentation/iter_80000.pth'
        model_train=init_model(config_file, checkpoint_file, device)

        model_train.to(device)
        model_train.eval()
        self.model = model_train
        self.device = device

    def forward(self, x):
        """inference model using image x
        Args:
            x (numpy.ndarray): input image
            result (list): a list of output from mmdet model
        """
        result = inference_model(self.model, x)
        return result

    def loss(self, x, pert, target):
        """get the loss

        args:
            x (numpy.ndarray):
            pert (tensor):             

        """
        data = get_test_data_seg(self.model, x)
        data_train = get_train_data_seg( x, pert.to(self.device), data, target,self.device)
        loss_dict = self.model( **data_train,mode='loss')
        # print(f"loss_dict: {loss_dict}")
        loss = get_loss_from_dict_seg(loss_dict)
        return loss

    def seg(self, x):
        """inference model using image x, get the processed output as detection
        
        args:
            x (numpy.ndarray): input image
        """
        seg = inference_model(self.model, x)

        return seg

def get_target_seg(tar_gt_seg, gt_seg,victim_class,target_class):
    class_labels = np.unique(gt_seg)
    for label in class_labels:
        label_map = (gt_seg==label)
        label_map_mask = np.zeros_like(gt_seg).astype(np.uint8)
        label_map_mask[label_map] = 1

        if label == victim_class:
            tar_gt_seg[label_map] = target_class
        else:
            tar_gt_seg[label_map] = gt_seg[label_map]
    return tar_gt_seg


'''def obtain_attacker_victim(config):

    dataset_dir = config.dataset['dataset_dir']
    dataset_txt = config.dataset['dataset_txt']

    attack_img_paths = []

    with open(dataset_txt) as fin:
        img_names = fin.readlines()

        for img_index,img_name in enumerate(img_names):
            img_path = dataset_dir + '/' + img_name.strip()
            attack_img_paths.append(img_path) 

    return attack_img_paths'''

def get_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1 (numpy.ndarray): x1,y1,x2,y2
        bbox2 (numpy.ndarray): x1,y1,x2,y2
    Returns:
        iou (float): iou in [0, 1]
    """
    w1,h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2,h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    try:
        assert all([w1,h1,w2,h2])
    except:
        return 0

    # determine the coordinates of the intersection rectangle
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = w1*h1
    area2 = w2*h2
    iou = area_inter / float(area1 + area2 - area_inter)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def is_success(detections, target_clean, iou_threshhold=0.3):
    """ see if the detection has target label at the corresponding location with IOU > 0.3
    Args:
        detections (np.ndarray): a list of detected objects. Shape (n,6)
        target_clean (np.ndarray): a single object, our desired output. Shape (1,6) - [xyxy,cls,score]
    Returens:
        (bool): whether the detection is a success or not
    """
    if len(detections) == 0:  # Check if detections is empty
        return True


    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > iou_threshhold and items[4] == target_clean[0][4]:
            return True
    return False