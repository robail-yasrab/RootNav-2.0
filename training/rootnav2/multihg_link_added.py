
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

        self.init_conv1 =  nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.init_bn1 = nn.BatchNorm2d(64)
        
        self.init_conv2 =  nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.init_bn2 = nn.BatchNorm2d(128)
        self.init_res2 = ResBlock(128,128)
        self.init_res3 = ResBlock(128,num_feats)

        self.core_hourglass = Hourglass(num_blocks, num_feats, 4)
        
        self.tail_deconv1 = nn.ConvTranspose2d(num_feats, 64, 4, 2, 1, bias=False)
        self.tail_bn1 = nn.BatchNorm2d(64)
        

        #self.tail_res1 = ResBlock(128, 64)
        
        self.tail_deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.tail_bn2 = nn.BatchNorm2d(64)
        
        self.tail_res2 = ResBlock(64, 32)


        #self.rtail_deconv1 = nn.ConvTranspose2d(num_feats, 128, 4, 2, 1, bias=False)
        #self.rtail_bn1 = nn.BatchNorm2d(128)
        

        #self.rtail_res1 = ResBlock(128, 64)
        
        self.rtail_deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.rtail_bn2 = nn.BatchNorm2d(64)
        
        self.rtail_res2 = ResBlock(64, 32)
        self.init_conv3 =  nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)
        self.init_conv4 =  nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)



    def forward(self, x):

        #print (x.size())
        x = self.init_conv1(x)
        conv_x = x
        x = self.init_bn1(x)
        x = F.relu(x, True)
        #x = self.init_res1(x)
        x = self.init_conv2(x)
        #print ("Skip", x.size())
        x = self.init_bn2(x)
        x = F.max_pool2d(x, 2)
        x = self.init_res2(x)
        x = self.init_res3(x)
        
       
        x = self.core_hourglass(x)
        


        x = self.tail_deconv1(x)
        x = self.tail_bn1(x)
        x = F.relu(x, True)
        x = torch.cat((x,conv_x),1)


        split = x
        #print x.shape 

        x = self.tail_deconv2(x)
        x = self.tail_bn2(x)
        x = F.relu(x, True)

        x = self.tail_res2(x)

        out1 = x

        # Right tail
        x = self.rtail_deconv2(split)
        x = self.rtail_bn2(x)
        x = F.relu(x, True)
        
        x = self.rtail_res2(x)
        
        #out2 = x
        #out = torch.cat((out1,out2),1)
        out1 = self.init_conv3(out1)        
        out2 = self.init_conv4(x)        

        return out1, out2

def hg1(**kwargs):
    model = MultiHourglass(num_feats=256, num_blocks=1, input_channels=3, output_channels=4)
    return model