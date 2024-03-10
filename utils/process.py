import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('metrics-saliency-maps')

from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from .img_process import colourise_heatmaps, super_imposed_image
from .fmgcam import FMGCAM

class Process:
    def __init__(self, model, model_with_softmax, device, preprocess, last_conv_layer, image_width=224, image_height=224):
        self.model = model
        self.model_with_softmax = model_with_softmax
        self.device = device
        self.preprocess = preprocess
        self.last_conv_layer = last_conv_layer
        self.image_width = image_width
        self.image_height = image_height
        self.fmgcam = FMGCAM(model, last_conv_layer, device)
        
    # def make_prediction(self, img, img_tensor, class_count, cam_type, class_rank_index=None,
    #                     enhance=True, alpha=0.8, act_mode="relu"):
        
    #     preds, sorted_pred_indices, gradients, activations = self.fmgcam.get_model_pred_with_grads(
    #         img_tensor, class_count=class_count, class_rank_index=class_rank_index, cam_type=cam_type
    #     )
    #     heatmaps = self.fmgcam.generate_grad_FMCAM(gradients, activations, enhance=enhance, act_mode=act_mode, class_rank_index=class_rank_index)

    #     heatmaps = colourise_heatmaps(heatmaps)
            
    #     super_imp_img = super_imposed_image(heatmaps, img, alpha=alpha, image_width=self.image_width, image_height=self.image_height)
        
    #     return preds, sorted_pred_indices, super_imp_img

    # def make_prediction_iterative(self, img, img_tensor, class_count=1, enhance_0=True, enhance=False, 
    #                               image_width=224, image_height=224, alpha_0=0.8, alpha=0.7, act_mode="relu"):
        
    #     super_imp_images = [img.resize((image_width, image_height)).convert("RGB")]
        
    #     preds, sorted_pred_indices, super_imp_image_0 = self.make_prediction(
    #         self.model, img, img_tensor, self.last_conv_layer, 
    #         class_count=class_count, class_rank_index=None, enhance=enhance_0, act_mode=act_mode,
    #         image_width=image_width, image_height=image_width, alpha=alpha_0, cam_type="fmgcam"
    #     )
        
    #     super_imp_images.append(super_imp_image_0)
        
    #     for i in range(0, class_count):
    #         _, _, super_imp_image = self.make_prediction(
    #             self.model, img, img_tensor, self.last_conv_layer, 
    #             class_count=class_count, class_rank_index=i, enhance=enhance, act_mode=act_mode,
    #             image_width=image_width, image_height=image_width, alpha=alpha, cam_type="gcam"
    #         )
    #         super_imp_images.append(super_imp_image)
            
    #     return preds, sorted_pred_indices, super_imp_images


    def get_cam_score_batch_mstep(self, img_path_batch, class_count, img_pp_size,
                                  class_rank_index, enhance, cam_type, act_mode="relu"):
        '''
        Get a batch of cams
        '''
        iauc_mean_list = []
        ic_mean_list = []
        dauc_mean_list = []
        dc_mean_list = []
        
        
        for i in tqdm(range(len(img_path_batch))):
            img = Image.open(img_path_batch[i])
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(img_pp_size, resample=Image.BICUBIC)
            img_tensor = self.preprocess(img).to(self.device)
        
            _, _, gradients, activations = self.fmgcam.get_model_pred_with_grads(
                img_tensor, cam_type=cam_type, class_count=class_count, 
                class_rank_index=class_rank_index
            )
            
            heatmaps = self.fmgcam.generate_grad_FMCAM(
                gradients, activations, class_rank_index=class_rank_index, cam_type=cam_type,
                enhance=enhance, act_mode=act_mode, filter_max_activation=False
            )
            
            heatmaps = np.array(heatmaps)
                    
            
            insertion = Insertion(
                data_replace_method="blur",bound_max_step=True,batch_size=1
            )

            result_dic = insertion(
                self.model_with_softmax,
                img_tensor.unsqueeze(dim=0),
                torch.from_numpy(heatmaps).unsqueeze(dim=0),
                [class_rank_index]
            )
            
            if not np.isnan(result_dic["iauc"]):
                iauc_mean_list.append(result_dic["iauc"])

            if not np.isnan(result_dic["ic"]):
                ic_mean_list.append(result_dic["ic"])

            deletion = Deletion(
                data_replace_method="blur",bound_max_step=True,batch_size=1
            )

            result_dic = deletion(
                self.model_with_softmax,
                img_tensor.unsqueeze(dim=0),
                torch.from_numpy(heatmaps).unsqueeze(dim=0),
                [class_rank_index]
            )
            
            if not np.isnan(result_dic["dauc"]):
                dauc_mean_list.append(result_dic["dauc"])

            if not np.isnan(result_dic["dc"]):
                dc_mean_list.append(result_dic["dc"])
            
        return iauc_mean_list, ic_mean_list, dauc_mean_list, dc_mean_list

    def get_scores_mstep(self, valid_img_paths, class_count, img_pp_size, target_classes, cam_type, enhance, act_mode="relu"):
        
        print("CAM Type: ", cam_type)
        iauc_means = []
        ic_means = []
        dauc_means = []
        dc_means = []
        
        for c_idx in target_classes:
            print("Target Class: ", c_idx)
            iauc_mean_list, ic_mean_list, dauc_mean_list, dc_mean_list = self.get_cam_score_batch_mstep(
                valid_img_paths, class_count, img_pp_size, class_rank_index=c_idx, enhance=enhance, 
                act_mode=act_mode, cam_type=cam_type
            )
            
            iauc_means.append(np.mean(iauc_mean_list))
            ic_means.append(np.mean(ic_mean_list))
            dauc_means.append(np.mean(dauc_mean_list))
            dc_means.append(np.mean(dc_mean_list))
            
        
        return iauc_means, ic_means, dauc_means, dc_means

