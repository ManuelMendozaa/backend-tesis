import torch
from torch.nn import Module, Sequential                                          # Base model
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Linear     # Arquitecture blocks
from torch.nn import BatchNorm2d, Dropout                                        # Regularization
from torch.nn import functional as F                                             # Activations

class BasicConv2d(Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out, inplace=True)

# --------------------------------  Inception  ---------------------------------
class Inception(Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # Branches
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        outputs = [out1, out2, out3, out4]
        return torch.cat(outputs, 1)

class Output(Module):
    def __init__(self, in_channels, num_classes):
        super(Output, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = Linear(2048, 1024)
        self.fc2 = Linear(1024, num_classes)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (4, 4))

        out = self.conv(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, 0.7, training=self.training)
        out = self.fc2(out)

        return out

# --------------------------------  GoogleNet  ---------------------------------
class GoogleNet(Module):
    def __init__(self, num_classes=7, kernel_init=None):
        super(GoogleNet, self).__init__()
        self.kernel_init = kernel_init
        blocks = [ BasicConv2d, Inception, Output ]

        conv_block = blocks[0]
        inception_block = blocks[1]
        output_block = blocks[2]

        # First layers
        self.conv1 = conv_block(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception part
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.aux1 = output_block(512, num_classes)
        self.aux2 = output_block(528, num_classes)
        
        # Output
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.2)
        self.fc = Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        props = x

        x = self.dropout(x)
        x = self.fc(x)

        return x, props

class ContinuousModule(Module):
    def __init__(self, kernel_init=None):
        super(ContinuousModule, self).__init__()
        # Arquitecture
        self.fc1 = Linear(1024, 2)
        self.dropout = Dropout(p=0.5)

        self.kernel_init = kernel_init
        self.init_weights()

    def init_weights(self):
        if self.kernel_init is not None:
            for m in self.modules():
                if isinstance(m, Conv2d) or isinstance(m, Linear):                  
                    if self.kernel_init == "xavier_normal":
                        init.xavier_normal_(m.weight)
                    elif self.kernel_init == "kaiming_normal":
                        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))

        return x

def googlenet():
    model = GoogleNet(num_classes=7)

    import os
    path = "C:/Users/manue/Documents/Tesis/Code/api/backend/app/models/googlenet_v3_iter9482.pth"
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cont_model = ContinuousModule()
    path2 = "C:/Users/manue/Documents/Tesis/Code/api/backend/app/models/arousal_valence_v2_iter700.pth"
    checkpoint2 = torch.load(path2, map_location=torch.device('cpu'))
    cont_model.load_state_dict(checkpoint2["model_state_dict"])
    cont_model.eval()

    return model, cont_model
