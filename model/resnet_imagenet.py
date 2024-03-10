import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50V2ImageNet(nn.Module):
    def __init__(self, activation=None, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        self.model = resnet50(weights=weights)
        self.weights = weights
        if activation is not None:
            self.model.fc = nn.Sequential(self.model.fc, activation) # Expected activation is nn.Softmax(-1)
        
    def forward(self, x):
        x = self.model(x)
        return x

# # Model
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
# weights = ResNet50_Weights.DEFAULT
# preprocess = weights.transforms()

# # Model with softmax
# model_with_softmax = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
# weights = ResNet50_Weights.DEFAULT
# preprocess = weights.transforms()
# # Modification of the model to add a softmax layer at the bottom
# model_with_softmax.fc = nn.Sequential(model_with_softmax.fc, nn.Softmax(1))