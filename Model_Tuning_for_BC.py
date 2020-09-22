import torchvision.models as models

# Alexnet
model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
                      nn.Linear(9216, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, 2),                   
                      nn.LogSoftmax(dim=1))
model.classifier

# Resnet18
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512,100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100,2)),
    ('output', nn.LogSoftmax(dim=1))
]))
#
model.fc = fc

# Resnet50
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048,256)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(256,2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.fc = fc

# DenseNet161
model = models.densenet161(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
                      nn.Linear(2208, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, 2),                   
                      nn.LogSoftmax(dim=1))
model.classifier

#VGG19
model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, 2),                   
                      nn.LogSoftmax(dim=1))
model.classifier

#VGG16
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, 2),                   
                      nn.LogSoftmax(dim=1))
model.classifier
