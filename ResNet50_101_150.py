from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel, inchannel, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inchannel, inchannel, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.conv3 = nn.Conv2d(inchannel, outchannel, 1, 1, bias=False)
        
        self.shortcut = shortcut
        

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
    
        if self.shortcut is None:
            residual = x
        else:
            residual = self.shortcut(x)
        
        out += residual
        out = F.relu(out)
        
        return out
    
    
class ResNet50(nn.Module):
    
    def __init__(self):
        super(ResNet50, self).__init__()
        
        self.pre = nn.Sequential(
                nn.Conv2d(1, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        
        self.pool = nn.MaxPool2d(3, stride=2)
        
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        
        self.avg = nn.AvgPool2d(3, stride=2)
        
        self.fc = nn.Linear(100352, 1000)
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride):
        
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.pre(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x