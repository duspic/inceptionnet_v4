from torch import nn, cat
import torch.nn.functional as F

# objects of class ConvBlock are initialized instead of normal conv2d layers, as all of them share the same batch normalization layer and activation function
class ConvBlock():

    # dict contains all the information usually provided to Conv2d layers
    def __init__(self,dict):

        self.conv = nn.Conv2d(

            in_channels=dict['in_channels'], 
            out_channels=dict['out_channels'], 
            kernel_size=dict['kernel_size'], 
            stride=dict['stride']

                  )
        
        self.bnorm = nn.BatchNorm2d(num_features=dict['out_channels'], eps=0.001)
    
    # fwd function is called in InceptionNet.foward() to tidy up the look and combat code repetition
    def fwd(self,x):
        
        x = self.conv(x)
        x = self.bnorm(x)
        x = F.relu(x)




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

            self.conv_a1 = ConvBlock({'in_channels':3, 'out_channels':32, 'kernel_size':(3,3), 'stride':2})
            self.conv_a2 = ConvBlock({'in_channels':32, 'out_channels':32, 'kernel_size':(3,3), 'stride':1})
            self.conv_a3 = ConvBlock({'in_channels':32, 'out_channels':64, 'kernel_size':(3,3), 'stride':1})
            self.conv_a4 = ConvBlock({'in_channels':64, 'out_channels':96, 'kernel_size':(3,3), 'stride':2})

            self.conv_b1 = ConvBlock({'in_channels':160, 'out_channels':64, 'kernel_size':(1,1), 'stride':1})
            self.conv_b2 = ConvBlock({'in_channels':64, 'out_channels':96, 'kernel_size':(3,3), 'stride':1})

            self.conv_c1 = ConvBlock({'in_channels':160, 'out_channels':64, 'kernel_size':(1,1), 'stride':1})
            self.conv_c2 = ConvBlock({'in_channels':64, 'out_channels':64, 'kernel_size':(7,1), 'stride':1})
            self.conv_c3 = ConvBlock({'in_channels':64, 'out_channels':64, 'kernel_size':(1,7), 'stride':1})
            self.conv_c4 = ConvBlock({'in_channels':64, 'out_channels':96, 'kernel_size':(3,3), 'stride':1})

            self.conv_d1 = ConvBlock({'in_channels':192, 'out_channels':192, 'kernel_size':(3,3), 'stride':1})

            self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
            self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2)

            return
        
        def fwd(self,x):

            # input dim     (299 x 299 x 3)
            x = self.conv_a1.fwd(x)
            # after conv1   (149 x 149 x 32)
            x = self.conv_a2.fwd(x)
            # after conv2   (147 x 147 x 32)
            x = self.conv_a3.fwd(x)
            # after conv3   (147 x 147 x 64)
            # splits left and right (x1,x2)

            x1 = self.max_pool1(x)
            x2 = self.conv_a4.fwd(x)
            x = cat((x1,x2),3)
            # after concatenation (73 x 73 x 160)
            # splits left and right (x1,x2)

            x1 = self.conv_b1.fwd(x)
            x1 = self.conv_b2.fwd(x1)
            x2 = self.conv_c1.fwd(x)
            x2 = self.conv_c2.fwd(x2)
            x2 = self.conv_c3.fwd(x2)
            x2 = self.conv_c4.fwd(x2)
            x = cat((x1,x2),3)
            # after concatenation (71 x 71 x 192)

            x1 = self.conv_d1.fwd(x)
            x2 = self.max_pool2(x)
            x = cat((x1,x2),3)
            # after concatenation (35 x 35 x 384)
        
            return x

    class InceptionA():

        def __init__(self):

            self.conv_a1 = ConvBlock({'in_channels':384, 'out_channels':96, 'kernel_size':(1,1), 'stride':1}) 

            self.conv_b1 = ConvBlock({'in_channels':384, 'out_channels':96, 'kernel_size':(1,1), 'stride':1}) 

            self.conv_c1 = ConvBlock({'in_channels':384, 'out_channels':96, 'kernel_size':(1,1), 'stride':1}) 
            self.conv_c2 = ConvBlock({'in_channels':64, 'out_channels':96, 'kernel_size':(3,3), 'stride':1})

            self.conv_d1 = ConvBlock({'in_channels':384, 'out_channels':96, 'kernel_size':(1,1), 'stride':1}) 
            self.conv_d2 = ConvBlock({'in_channels':64, 'out_channels':96, 'kernel_size':(3,3), 'stride':1})
            self.conv_d3 = ConvBlock({'in_channels':96, 'out_channels':96, 'kernel_size':(3,3), 'stride':1})

            self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)
        
        def fwd(self,x):

            x1 = self.avg_pool(x)
            x1 = self.conv_a1.fwd(x1)

            x2 = self.conv_b1.fwd(x2)

            x3 = self.conv_c1.fwd(x)
            x3 = self.conv_c2.fwd(x3)

            x4 = self.conv_d1.fwd(x)
            x4 = self.conv_d2.fwd(x4)
            x4 = self.conv_d3.fwd(x4)

            x = cat((x1,x2,x3,x4),3)

            return x

    class InceptionB():

        def __init__(self):
            self.conv_a1 = ConvBlock({'in_channels':1024, 'out_channels':128, 'kernel_size':(1,1), 'stride':1})            

            self.conv_b1 = ConvBlock({'in_channels':1024, 'out_channels':384, 'kernel_size':(1,1), 'stride':1})

            self.conv_c1 = ConvBlock({'in_channels':1024, 'out_channels':192, 'kernel_size':(1,1), 'stride':1})
            self.conv_c2 = ConvBlock({'in_channels':192, 'out_channels':224, 'kernel_size':(7,1), 'stride':1})
            self.conv_c3 = ConvBlock({'in_channels':224, 'out_channels':256, 'kernel_size':(1,7), 'stride':1})

            self.conv_d1 = ConvBlock({'in_channels':1024, 'out_channels':192, 'kernel_size':(1,1), 'stride':1}) 
            self.conv_d2 = ConvBlock({'in_channels':192, 'out_channels':192, 'kernel_size':(1,7), 'stride':1})
            self.conv_d3 = ConvBlock({'in_channels':192, 'out_channels':224, 'kernel_size':(7,1), 'stride':1})
            self.conv_d4 = ConvBlock({'in_channels':224, 'out_channels':224, 'kernel_size':(1,7), 'stride':1})
            self.conv_d5 = ConvBlock({'in_channels':224, 'out_channels':256, 'kernel_size':(7,1), 'stride':1})

            self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)
        
        def fwd(self,x):

            x1 = self.avg_pool(x)
            x1 = self.conv_a1.fwd(x1)

            x2 = self.conv_b1.fwd(x)

            x3 = self.conv_c1.fwd(x)
            x3 = self.conv_c2.fwd(x3)
            x3 = self.conv_c3.fwd(x3)

            x4 = self.conv_d1.fwd(x)
            x4 = self.conv_d2.fwd(x4)
            x4 = self.conv_d3.fwd(x4)
            x4 = self.conv_d4.fwd(x4)
            x4 = self.conv_d5.fwd(x4)

            x = cat((x1,x2,x3,x4),3) 

            return x

    class InceptionC():
         
        def __init__(self):

            self.conv_a1 = ConvBlock({'in_channels':1536, 'out_channels':256, 'kernel_size':(1,1), 'stride':1})

            self.conv_b1 = ConvBlock({'in_channels':1536, 'out_channels':256, 'kernel_size':(1,1), 'stride':1})

            self.conv_c1 = ConvBlock({'in_channels':1536, 'out_channels':384, 'kernel_size':(1,1), 'stride':1})
            self.conv_c_left = ConvBlock({'in_channels':384, 'out_channels':256, 'kernel_size':(1,3), 'stride':1})
            self.conv_c_right = ConvBlock({'in_channels':384, 'out_channels':256, 'kernel_size':(3,1), 'stride':1})

            self.conv_d1 = ConvBlock({'in_channels':1536, 'out_channels':384, 'kernel_size':(1,1), 'stride':1})
            self.conv_d2 = ConvBlock({'in_channels':384, 'out_channels':448, 'kernel_size':(1,3), 'stride':1})
            self.conv_d3 = ConvBlock({'in_channels':448, 'out_channels':512, 'kernel_size':(3,1), 'stride':1})
            self.conv_d_left = ConvBlock({'in_channels':512, 'out_channels':256, 'kernel_size':(3,1), 'stride':1})
            self.conv_d_right = ConvBlock({'in_channels':512, 'out_channels':256, 'kernel_size':(1,3), 'stride':1})

            self.avg_pool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)
        
        def fwd(self,x):

            x1 = self.avg_pool(x)
            x1 = self.conv_a1.fwd(x1)

            x2 = self.conv_b1.fwd(x)

            x3 = self.conv_c1.fwd(x)
            x3_1 = self.conv_c_left.fwd(x3)
            x3_2 = self.conv_c_right.fwd(x3)

            x4 = self.conv_d1.fwd(x)
            x4 = self.conv_d2.fwd(x4)
            x4 = self.conv_d3.fwd(x4)
            x4_1 = self.conv_d_left.fwd(x4)
            x4_2 = self.conv_d_right.fwd(x4)

            x = cat((x1,x2,x3_1,x3_2,x4_1,x4_2),3)

            return x
    
    class ReductionA():
        
        def __init__(self):

            self.conv_a1 = ConvBlock({'in_channels':384, 'out_channels':384, 'kernel_size':(3,3), 'stride':2})

            self.conv_b1 = ConvBlock({'in_channels':384, 'out_channels':192, 'kernel_size':(1,1), 'stride':1})
            self.conv_b2 = ConvBlock({'in_channels':192, 'out_channels':224, 'kernel_size':(3,3), 'stride':1})
            self.conv_b3 = ConvBlock({'in_channels':224, 'out_channels':256, 'kernel_size':(3,3), 'stride':2})

            self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)

        def fwd(self,x):

            x1 = self.max_pool(x)

            x2 =self.conv_a1.fwd(x)

            x3 = self.conv_b1.fwd(x)
            x3 = self.conv_b2.fwd(x3)
            x3 = self.conv_b3.fwd(x3)
            
            x = cat((x1,x2,x3),3)

            return x

    class ReductionB():

        def __init__(self):

            self.conv_a1 = ConvBlock({'in_channels':1024, 'out_channels':192, 'kernel_size':(3,3), 'stride':1})
            self.conv_a2 = ConvBlock({'in_channels':192, 'out_channels':192, 'kernel_size':(3,3), 'stride':2})

            self.conv_b1 = ConvBlock({'in_channels':1024, 'out_channels':256, 'kernel_size':(1,1), 'stride':1})
            self.conv_b2 = ConvBlock({'in_channels':256, 'out_channels':256, 'kernel_size':(1,7), 'stride':1})
            self.conv_b3 = ConvBlock({'in_channels':256, 'out_channels':320, 'kernel_size':(7,1), 'stride':1})
            self.conv_b4 = ConvBlock({'in_channels':320, 'out_channels':320, 'kernel_size':(3,3), 'stride':2})

            self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)

        def fwd(self,x):

            x1 = self.max_pool(x)

            x2 = self.conv_a1.fwd(x)
            x2 = self.conv_a2.fwd(x2)

            x3 = self.conv_b1.fwd(x)
            x3 = self.conv_b2.fwd(x3)
            x3 = self.conv_b3.fwd(x3)
            x3 = self.conv_b4.fwd(x3)
            
            x = cat((x1,x2,x3),3)

            return x