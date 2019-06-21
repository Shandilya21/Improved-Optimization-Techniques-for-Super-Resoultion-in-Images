import torch
import torch.nn as nn
import math
import numpy as np 

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        #self.features = nn.Sequential(
        
        # input is (3) x 96 x 96
        #Block_1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_1 = nn.LeakyReLU(0.2, inplace=True)

     	# state size. (64) x 96 x 96
        #Block_2
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)          
        self.bn_1 = nn.BatchNorm2d(64)
        self.activation_2 = nn.LeakyReLU(0.2, inplace=True)

       	# state size. (64) x 96 x 96
        #Block_3
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)            
        self.bn_2 = nn.BatchNorm2d(128)
        self.activation_3 = nn.LeakyReLU(0.2, inplace=True)
            
        # state size. (64) x 48 x 48
        #Block_4
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(128)
        self.activation_4 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (128) x 48 x 48
        #Block_5
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_4 = nn.BatchNorm2d(256)
        self.activation_5 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (256) x 24 x 24
        #Block_6
        self.conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_5 = nn.BatchNorm2d(256)
        self.activation_6 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (256) x 12 x 12
        #Block_7
        self.conv_7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_6 = nn.BatchNorm2d(512)
        self.activation_7 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (512) x 12 x 12
        #Block_8
        self.conv_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)           
        self.bn_7 = nn.BatchNorm2d(512)
        self.activation_8 = nn.LeakyReLU(0.2, inplace=True)
            

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

    	# Define all the layers along with parameters accordingly, refer initilization function

        out_0 = self.conv_1(input)
        out_0 = self.activation_1(out_0)
        out_1 = self.conv_2(out_0)
        out_1 = self.bn_1(out_1)
        out_1 = self.activation_2(out_1)
        out_2 = self.conv_3(out_1)
        out_2 = self.bn_2(out_2)
        out_2 = self.activation_3(out_2)
        out_3 = self.conv_4(out_2)
        out_3 = self.bn_3(out_3)
        out_3 = self.activation_4(out_3)
        out_4 = self.conv_5(out_3)
        out_4 = self.bn_4(out_4)
        out_4 = self.activation_5(out_4)
        out_5 = self.conv_6(out_4)
        out_5 = self.bn_5(out_5)
        out_5 = self.activation_6(out_5)
        out_6 = self.conv_7(out_5)
        out_6 = self.bn_6(out_6)
        out_6 = self.activation_7(out_6)
        out_7 = self.conv_8(out_6)
        out_7 = self.bn_7(out_7)
        out_7 = self.activation_8(out_7)


        # state size. (512) x 6 x 6
        out_0 = out_0.view(out_0.size(0), -1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_3 = out_3.view(out_3.size(0), -1)
        out_4 = out_4.view(out_4.size(0), -1)
        out_5 = out_5.view(out_5.size(0), -1)
        out_6 = out_6.view(out_6.size(0), -1)
        out_7 = out_7.view(out_7.size(0), -1)


        # state size. (512 x 6 x 6)

        out_8 = self.fc1(out_7)
        # out_layer = self.activation_8(out_layer)


        # state size. (1024)
        out_8 = self.LeakyReLU(out_8)

        out_9 = self.fc2(out_8)

        out_9 = self.sigmoid(out_9)

        return out_0, out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8,out_9.view(-1, 1).squeeze(1)




# model = _NetG()
# input = np.zeros((32,3,))

# criterion = nn.MSELoss(size_average=False)
# model_1 = _NetD()
# input = np.ones((32,3,96,96))
# target = np.zeros((32,3,48,48))
# input_layer = torch.tensor(input).float()
# outmost = model_1.forward(input_layer)
# print(outmost)

# loss = criterion(input, target)
# print(loss)




