import torch
from torch.nn import functional as F

class FMGCAM():
    def __init__(self, model, last_conv_layer, device):
        self.model = model
        self.last_conv_layer = last_conv_layer
        self.device = device
    
    def get_model_pred_with_grads(self, img_tensor, class_count=3, class_rank_index=None, cam_type="fmgcam"):
        grad_list = []
        act_list = []
        
        for train_param in self.model.parameters():
            train_param.requires_grad = True
            
        gradients = None
        activations = None

        def hook_backward(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output

        def hook_forward(module, args, output):
            nonlocal activations
            activations = output
            
            
        hook_backward = self.last_conv_layer.register_full_backward_hook(hook_backward, prepend=False)
        hook_forward = self.last_conv_layer.register_forward_hook(hook_forward, prepend=False)
        
        self.model.eval()
        
        preds =  self.model(img_tensor.to(self.device).unsqueeze(0))
        
        # Sort prediction indices
        sorted_pred_indices = torch.argsort(preds, dim=1, descending=True).squeeze(0)
        
        if cam_type == "fmgcam":
            # Iterate through the top prediction indices
            for rank in range(class_count):
                preds[:, sorted_pred_indices[rank]].backward(retain_graph=True)
                grad_list.append(gradients)
                act_list.append(activations)
            
        elif cam_type == "gcam":
            
            if class_rank_index is not None:
                preds[:, sorted_pred_indices[class_rank_index]].backward(retain_graph=True)
                grad_list.append(gradients)
                act_list.append(activations)
            else:    
                # Iterate through the top prediction indices
                for rank in range(class_count):
                    preds[:, sorted_pred_indices[rank]].backward(retain_graph=True)
                    grad_list.append(gradients)
                    act_list.append(activations)
        
        hook_backward.remove()
        hook_forward.remove()
        
        for train_param in self.model.parameters():
            train_param.requires_grad = False
        
        return F.softmax(preds, dim=1).squeeze().detach().cpu().numpy(), sorted_pred_indices, grad_list, act_list
    
    
    def generate_grad_FMCAM(self, gradients_list, activations_list, enhance, class_rank_index=None, act_mode="relu", cam_type="fmgcam", filter_max_activation=True):
        heatmaps = []        
        
        for index, activations in enumerate(activations_list):
            gradients = gradients_list[index]

            avg_pooled_gradients = torch.mean(
                gradients[0], # Size [1, 1024, 7, 7]
                dim=[0, 2, 3]
            )

            # Weighting acitvation features (channels) using its related calculated Gradient
            for i in range(activations.size()[1]):
                activations[:, i, :, :] *= avg_pooled_gradients[i]

            # average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze()

            heatmaps.append(heatmap.unsqueeze(0).detach().cpu())

        if cam_type == "fmgcam":

            # Concatenation of activation maps based on top n classes
            heatmaps = torch.cat(heatmaps)
            
            if filter_max_activation:
                # Filter the heatmap based on the maximum weighted activation along the channel axis
                hm_mask_indices = heatmaps.argmax(dim=0).unsqueeze(0)

                hm_3d_mask = torch.cat([hm_mask_indices for _ in range(heatmaps.size()[0])])

                hm_3d_mask = torch.cat(
                    [(hm_3d_mask[index] == (torch.ones_like(hm_3d_mask[index])*index)).unsqueeze(0) for index in range(heatmaps.size()[0])]
                ).long()

                heatmaps *= hm_3d_mask
        
            # L2 normalization of the heatmap
            if enhance:
                heatmaps = F.normalize(heatmaps, p=2, dim=1)
            
            if class_rank_index is not None:
                heatmaps = heatmaps[class_rank_index].unsqueeze(0)

        elif cam_type == "gcam":
            heatmaps = heatmaps[0]

        # Activation on top of the heatmap
        if act_mode == "relu":
            heatmaps = F.relu(heatmaps)
        elif act_mode == "gelu":
            heatmaps = F.gelu(heatmaps)
        elif act_mode == "elu":
            heatmaps = F.elu(heatmaps)
    
        # Min-max normalization of the heatmap
        heatmaps = (heatmaps - torch.min(heatmaps))/(torch.max(heatmaps) - torch.min(heatmaps))
        
        return heatmaps.detach().cpu().numpy()
