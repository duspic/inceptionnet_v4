from torch import nn, cat
import torch.nn.functional as F


class InceptionNet(nn.Module):

    def __init__(self):

        super(InceptionNet, self).__init__()
        
        self.stem = self.Stem()

        self.inception_A = self.InceptionA()
        self.inception_B = self.InceptionB()
        self.inception_C = self.InceptionC()

        self.reduction_A = self.ReductionA()
        self.reduction_B = self.ReductionB()

        self.avg_pool = nn.AdaptiveAvgPool1d(1536)
        self.dropout = nn.Dropout(p=0.2)

        return

    def forward(self,x):


        x = self.stem.fwd(x)

        x = self.inception_A.fwd(x)
        x = self.inception_A.fwd(x)
        x = self.inception_A.fwd(x)
        x = self.inception_A.fwd(x)

        x = self.reduction_A.fwd(x)

        x = self.inception_B.fwd(x)
        x = self.inception_B.fwd(x)
        x = self.inception_B.fwd(x)
        x = self.inception_B.fwd(x)
        x = self.inception_B.fwd(x)
        x = self.inception_B.fwd(x)
        x = self.inception_B.fwd(x)

        x = self.reduction_B.fwd(x)

        x = self.inception_C.fwd(x)
        x = self.inception_C.fwd(x)
        x = self.inception_C.fwd(x)
        
        x = self.avg_pool(x)

        x = self.dropout(x)

        x = F.softmax(x, dim = 1)

        return x

    class Stem():
        def __init__(self):

            self.conv_a1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = (3,3), stride=2)
            self.conv_a2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size = (3,3), stride=1)
            self.conv_a3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = (3,3), stride=1)
            self.conv_a4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3), stride=2)

            self.conv_b1 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,1), stride=1)
            self.conv_b2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3), stride=1)

            self.conv_c1 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,1), stride=1)
            self.conv_c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,1), stride=1)
            self.conv_c3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,7), stride=1)
            self.conv_c4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3), stride=1)

            self.conv_d1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), stride=1)

            self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
            self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2)

            return
        
        def fwd(self,x):

            # input dim     (299 x 299 x 3)
            x = self.conv_a1(x)
            # after conv1   (149 x 149 x 32)
            x = self.conv_a2(x)
            # after conv2   (147 x 147 x 32)
            x = self.conv_a3(x)
            # after conv3   (147 x 147 x 64)
            # splits left and right (x1,x2)

            x1 = self.max_pool1(x)
            x2 = self.conv_a4(x)
            x = cat((x1,x2),3)
            # after concatenation (73 x 73 x 160)
            # splits left and right (x1,x2)

            x1 = self.conv_b1(x)
            x1 = self.conv_b2(x1)
            x2 = self.conv_c1(x)
            x2 = self.conv_c2(x2)
            x2 = self.conv_c3(x2)
            x2 = self.conv_c4(x2)
            x = cat((x1,x2),3)
            # after concatenation (71 x 71 x 192)

            x1 = self.conv_d1(x)
            x2 = self.max_pool2(x)
            x = cat((x1,x2),3)
            # after concatenation (35 x 35 x 384)
        
            return x

    class InceptionA():

        def __init__(self):

            self.conv_a1 = nn.Conv2d(in_channels=384, out_channels=96, kernel_size = (1,1), stride=1)

            self.conv_b1 = nn.Conv2d(in_channels=384, out_channels=96, kernel_size = (1,1), stride=1)

            self.conv_c1 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size = (1,1), stride=1)
            self.conv_c2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size = (3,3), stride=1)

            self.conv_d1 = nn.Conv2d(in_channels=384, out_channels=96, kernel_size = (1,1), stride=1)
            self.conv_d2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size = (3,3), stride=1)
            self.conv_d3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size = (3,3), stride=1)

            self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)
        
        def fwd(self,x):

            x1 = self.avg_pool(x)
            x1 = self.conv_a1(x1)

            x2 = self.conv_b1(x)

            x3 = self.conv_c1(x)
            x3 = self.conv_c2(x3)

            x4 = self.conv_d1(x)
            x4 = self.conv_d2(x4)
            x4 = self.conv_d3(x4)

            x = cat((x1,x2,x3,x4),3)

            return x

    class InceptionB():

        def __init__(self):
            self.conv_a1 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size = (1,1), stride=1)

            self.conv_b1 = nn.Conv2d(in_channels=1024, out_channels=384, kernel_size = (1,1), stride=1)

            self.conv_c1 = nn.Conv2d(in_channels=1024, out_channels=192, kernel_size = (1,1), stride=1)
            self.conv_c2 = nn.Conv2d(in_channels=192, out_channels=224, kernel_size = (7,1), stride=1)
            self.conv_c3 = nn.Conv2d(in_channels=224, out_channels=256, kernel_size = (1,7), stride=1)

            self.conv_d1 = nn.Conv2d(in_channels=1024, out_channels=192, kernel_size = (1,1), stride=1)
            self.conv_d2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size = (1,7), stride=1)
            self.conv_d3 = nn.Conv2d(in_channels=192, out_channels=224, kernel_size = (7,1), stride=1)
            self.conv_d4 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size = (1,7), stride=1)
            self.conv_d5 = nn.Conv2d(in_channels=224, out_channels=256, kernel_size = (7,1), stride=1)

            self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)
        
        def fwd(self,x):

            x1 = self.avg_pool(x)
            x1 = self.conv_a1(x1)

            x2 = self.conv_b1(x)

            x3 = self.conv_c1(x)
            x3 = self.conv_c2(x3)
            x3 = self.conv_c3(x3)

            x4 = self.conv_d1(x)
            x4 = self.conv_d2(x4)
            x4 = self.conv_d3(x4)
            x4 = self.conv_d4(x4)
            x4 = self.conv_d5(x4)

            x = cat((x1,x2,x3,x4),3) 

            return x

    class InceptionC():
         
        def __init__(self):

            self.conv_a1 = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size = (1,1), stride=1)

            self.conv_b1 = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size = (1,1), stride=1)

            self.conv_c1 = nn.Conv2d(in_channels=1536, out_channels=384, kernel_size = (1,1), stride=1)
            self.conv_c_left = nn.Conv2d(in_channels=384, out_channels=256, kernel_size = (1,3), stride=1)
            self.conv_c_right = nn.Conv2d(in_channels=384, out_channels=256, kernel_size = (3,1), stride=1)

            self.conv_d1 = nn.Conv2d(in_channels=1536, out_channels=384, kernel_size = (1,1), stride=1)
            self.conv_d2 = nn.Conv2d(in_channels=384, out_channels=448, kernel_size = (1,3), stride=1)
            self.conv_d3 = nn.Conv2d(in_channels=448, out_channels=512, kernel_size = (3,1), stride=1)
            self.conv_d_left = nn.Conv2d(in_channels=512, out_channels=256, kernel_size = (3,1), stride=1)
            self.conv_d_right = nn.Conv2d(in_channels=512, out_channels=256, kernel_size = (1,3), stride=1)

            self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)
        
        def fwd(self,x):

            x1 = self.avg_pool(x)
            x1 = self.conv_a1(x1)

            x2 = self.conv_b1(x)

            x3 = self.conv_c1(x)
            x3_1 = self.conv_c_left(x3)
            x3_2 = self.conv_c_right(x3)

            x4 = self.conv_d1(x)
            x4 = self.conv_d2(x4)
            x4 = self.conv_d3(x4)
            x4_1 = self.conv_d_left(x4)
            x4_2 = self.conv_d_right(x4)

            x = cat((x1,x2,x3_1,x3_2,x4_1,x4_2),3)

            return x
    
    class ReductionA():
        
        def __init__(self):

            self.conv_a1 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size = (3,3), stride=2)

            self.conv_b1 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size = (1,1), stride=1)
            self.conv_b2 = nn.Conv2d(in_channels=192, out_channels=224, kernel_size = (3,3), stride=1)
            self.conv_b3 = nn.Conv2d(in_channels=224, out_channels=256, kernel_size = (3,3), stride=2)

            self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)

        def fwd(self,x):

            x1 = self.max_pool(x)

            x2 = self.conv_a1(x)

            x3 = self.conv_b1(x)
            x3 = self.conv_b2(x3)
            x3 = self.conv_b3(x3)
            
            x = cat((x1,x2,x3),3)

            return x

    class ReductionB():

        def __init__(self):

            self.conv_a1 = nn.Conv2d(in_channels=1024, out_channels=192, kernel_size = (3,3), stride=1)
            self.conv_a2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size = (3,3), stride=2)

            self.conv_b1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size = (1,1), stride=1)
            self.conv_b2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size = (1,7), stride=1)
            self.conv_b3 = nn.Conv2d(in_channels=256, out_channels=320, kernel_size = (7,1), stride=1)
            self.conv_b4 = nn.Conv2d(in_channels=320, out_channels=320, kernel_size = (3,3), stride=2, padding=0)

            self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)

        def fwd(self,x):

            x1 = self.max_pool(x)

            x2 = self.conv_a1(x)
            x2 = self.conv_a2(x2)

            x3 = self.conv_b1(x)
            x3 = self.conv_b2(x3)
            x3 = self.conv_b3(x3)
            x3 = self.conv_b4(x3)
            
            x = cat((x1,x2,x3),3)

            return x