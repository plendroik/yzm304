import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

# MODEL 1: Base Custom CNN (LeNet-5 Inspired)
class Model1_LeNet5(nn.Module):
    def __init__(self):
        super(Model1_LeNet5, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 6, kernel_size=5)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(6, 16, kernel_size=5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        # CIFAR-10 images are 32x32, after 2 pools they are 5x5 if kernel 5x5 used correctly
        # We use Adaptive Pool to ensure 5x5 regardless of input size (within reason)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5, 120)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(84, 10))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# MODEL 2: Regularized Custom CNN (Same dimensions as Model 1)
class Model2_EnhancedLeNet5(nn.Module):
    def __init__(self):
        super(Model2_EnhancedLeNet5, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 6, kernel_size=5)),
            ('bn1', nn.BatchNorm2d(6)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(6, 16, kernel_size=5)),
            ('bn2', nn.BatchNorm2d(16)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5, 120)),
            ('relu3', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),
            ('fc3', nn.Linear(84, 10))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# MODEL 3: Standard Literature CNN (AlexNet)
def get_model3_alexnet(pretrained=False):
    """
    Literature Model: AlexNet
    CIFAR-10 için sıfırdan eğitilebilir veya pretrained kullanılabilir.
    """
    model = models.alexnet(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 10)
    return model

# MODEL 4: Hybrid Model Logic
# Not: Model 4 bir PyTorch sınıfı değil, VGG16 (Feature Extractor) + SVM kombinasyonudur.
# Bu modelin mantığı src/hybrid.py içerisindedir.

# MODEL 5: Full Literature CNN (VGG16) - Model 4 ile kıyaslanacak
def get_model5_vgg16(pretrained=True):
    """
    Model 4 (Hibrit) ile kıyaslanacak tam CNN mimarisi.
    """
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 10)
    return model
