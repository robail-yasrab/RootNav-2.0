
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(feat_in)
        self.conv1 = nn.Conv2d(feat_in, feat_out // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feat_out // 2)
        self.conv2 = nn.Conv2d(feat_out // 2, feat_out // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(feat_out // 2)
        self.conv3 = nn.Conv2d(feat_out // 2, feat_out, kernel_size=1, bias=False)
        
        self.downsample = None
        if feat_in != feat_out:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(feat_in),
                nn.ReLU(True),
                nn.Conv2d(feat_in, feat_out, kernel_size=1, stride=1, bias=False),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out, True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, True)
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.relu(out, True)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, num_blocks, num_feats, depth):
        super(Hourglass, self).__init__()
        self.num_blocks = num_blocks
        self.num_feats = num_feats
        self.depth = depth

        for d in range(depth):
            self.add_module("upper_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
            self.add_module("lower_in_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))

            # Additional blocks at bottom of hourglass
            if d == 0:
                self.add_module("lower_plus_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
            
            self.add_module("lower_out_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
        
    def _forward(self, depth, inp):
         # Upper branch
        up1 = inp
        up1 = self._modules['upper_branch_' + str(depth)](up1)

        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['lower_in_branch_' + str(depth)](low1)

        if depth > 0:
            low2 = self._forward(depth - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['lower_plus_branch_' + str(depth)](low2)

        low3 = low2
        low3 = self._modules['lower_out_branch_' + str(depth)](low3)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')

        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth - 1, x)

class MultiHourglass(nn.Module):
    def __init__(self, num_feats, num_blocks, input_channels, output_channels):
        super(MultiHourglass, self).__init__()

        self.init_conv1 =  nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.init_bn1 = nn.BatchNorm2d(32)


        self.init_res1 = ResBlock(32,64)
        self.init_res2 = ResBlock(64,128)
        self.init_res3 = ResBlock(128,256)

        self.core_hourglass1 = Hourglass(num_blocks, 256, 4)

        self.init_res4 = ResBlock(256, 128) 

        #self.init_conv_union =  nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        #self.init_bn_union = nn.BatchNorm2d(256)

        #self.tail_deconv1 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        #self.tail_bn1 = nn.BatchNorm2d(256)
        #self.init_res7 = ResBlock(256, 128) 

        self.tail_deconv2 = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False)
        self.tail_bn2 = nn.BatchNorm2d(128)

        self.init_res8 = ResBlock(128, 128)

        self.init_conv4 =  nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x = self.init_conv1(x)      
        x = self.init_bn1(x)
        x = F.relu(x, True)   

        x = self.init_res1(x)
        conv_512 = x

        #x = F.max_pool2d(x, 2)

        x = self.init_res2(x)
        x = self.init_res3(x)          
        #conv_256 = x
        x = self.core_hourglass1(x)
        x = self.init_res4(x) 
        ###################
        #x = torch.cat((x,conv_256),1)
        #x= self.init_conv_union(x)
        #x= self.init_bn_union(x) 
        #x = F.relu(x, True)
        ###################
        #x = self.tail_deconv1(x)
        #x = self.tail_bn1(x)
        #x = F.relu(x, True)
        
        #x = self.init_res7(x) 


        x = self.tail_deconv2(x)
        x = self.tail_bn2(x)
        x = F.relu(x, True)

        x = self.init_res8(x)
 

        out2 = self.init_conv4(x) 


        return out2

def hg1(**kwargs):
    model = MultiHourglass(num_feats=256, num_blocks=1, input_channels=3, output_channels=4)
    return model