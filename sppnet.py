import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def spatial_pyramid_pooling(prev_conv, num_sample, prev_conv_size, out_pool_size):
    """
    prev_conv: 이전 conv layer의 output tensor
    num_sample: 이미지의 batch 수 => N
    prev_conv_size: 이전 conv layer의 output tensor의 width와 height이다.
    out_pool_size: a int vector of expected output size of max pooling layer, [1,2,4] 라는 배열을 넣는다
    :return: a tensor vector (1xn) is the concentration of multi-level pooling
    """
    #
    # detail : 하나의 이미지를 4x4개, 2x2개, 1x1개의 영역으로 쪼개는데 이러한 것들을 할 때 pooling을 한다.
    # pooling이 되면 4x4의 한조각이 1픽셀, 2x2의 한 조각이 한 픽셀, 1x1 한조각이 한 픽셀씩 된다.
    for i in range(len(out_pool_size)): #루프 3번을 돈다.
        h, w = prev_conv_size
        h_window = math.ceil(h / out_pool_size[i])
        w_window = math.ceil(w / out_pool_size[i])
        h_stride = math.floor(h / out_pool_size[i])
        w_stride = math.floor(w / out_pool_size[i])
        #print("h_w, w_w, h_s, w_s", h_window,
        #                            w_window,
        #                            h_stride,
        #                            w_stride)
        max_pool = nn.MaxPool2d(kernel_size=(h_window, w_window), stride=(h_stride, w_stride)) # ceiling_window 사이즈 만큼 pooling 그리고 floor_stride 사이즈만큼 stride
        x = max_pool(prev_conv)           # 윗줄 모델 -> 바로 적용
        if i == 0:                        # 간단한 예외처리
            spp = x.view(num_sample, -1)  # 맨 처음 i=0
        else :
            spp = torch.cat((spp, x.view(num_sample, -1)), 1) # pooling 거친 것
    # print("print>>",spp.shape) => 1x5376
    return spp


class SPPNet(nn.Module):

    def __init__(self, n_classes=102, init_weights=True):
        super(SPPNet, self).__init__()
        """
        c1 : [3, 96 ,11 ,11 ]
        c2 : [96, 256 ,5 ,5 ]
        c3 : [256, 384 ,3 ,3 ]
        c4 : [384, 384 ,3 ,3 ]
        c5 : [384, 256 ,3 ,3 ]
        fc6:[spatial_pool_dim*256,4096]
        fc7 : [4096, 4096]
        out : [4096, n_classes]
        """
        self.output_num = [4, 2, 1]

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3)

        # fc1에 넣는 것은 spatial_pyramid_pooling 함수의 output이다.
        self.fc1 = nn.Linear(sum([i*i for i in self.output_num]) * 256, 4096) # 1 + 4 + 16의 region이 각각 나오고 256은 feature의 수이다. -> (5376,4096)

        # >>  16x256 + 4x156 + 1x256 다 더한 것
        self.fc2 = nn.Linear(4096, 4096)

        self.out = nn.Linear(4096, n_classes)

        if init_weights: # weight initialize
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # torch.Size([N, C, H, W])
        #print(x.size())

        x = F.relu(self.conv1(x))
        x = F.local_response_norm(x, size=4)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.local_response_norm(x, size=4)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        spp = spatial_pyramid_pooling(x, # 이전 레이어의 outut
                                      x.size(0), # N
                                      [int(x.size(2)), int(x.size(3))], # H, W
                                      self.output_num) #[4, 2, 1]

        fc1 = F.relu(self.fc1(spp)) #1x5376인 상태이다.
        fc2 = F.relu(self.fc2(fc1))

        output = self.out(fc2)
        return output
