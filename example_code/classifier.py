import torch
from torch import nn

class SimpleConvNN(nn.Module):
    def __init__(self):
        super(SimpleConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,
                               bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.logstic = nn.Linear(128, 40)
        self.output_layer = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.mean(x.view(x.size(0),x.size(1),-1),dim=2)
        out = self.logstic(x)
        # out = self.output_layer(out)
        return out


class Neuralnetwork(nn.Module):
    """ 
    Pytorch中神经网络模块化接口
    """
    def __init__(self, in_dim, n_class):
        """ 
        dim is the 
        """
        super(Neuralnetwork, self).__init__()
        # 卷积层池化层
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            # nn.BatchNorm2d(16,affine=True),
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2))
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(10304, 1024),
            nn.Dropout(0.5) ,
            nn.Linear(1024, 512), 
            nn.Dropout(0.5),
            nn.Linear(512, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# torch.save(model.state_dict(), './logstic.pth')
