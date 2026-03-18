import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .pvtv2 import pvt_v2_b2
from torch.nn.init import trunc_normal_, constant_


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()
        
        self.gate_1 = nn.Conv2d(in_planes*3, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes*3, 1, kernel_size=1, bias=True)
        self.gate_3 = nn.Conv2d(in_planes*3, 1, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):

        cat_fea = torch.cat([x1,x2,x3], dim=1)
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)
        att_vec_3  = self.gate_3(cat_fea)
        att_vec_cat  = torch.cat([att_vec_1, att_vec_2, att_vec_3], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        att_soft_1, att_soft_2,att_soft_3 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :],att_vec_soft[:, 2:3, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2 + x3 * att_soft_3
        
        return x_fusion
    
class DilationGate(nn.Module):
    def __init__(self, in_planes,out_planes):
        super(DilationGate, self).__init__()

        self.dcConv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1), nn.BatchNorm2d(out_planes),
            nn.Conv2d(out_planes, out_planes, 3, padding=1,dilation=1), nn.BatchNorm2d(out_planes), nn.ReLU(True))
        self.dcConv2 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1), nn.BatchNorm2d(out_planes),
            nn.Conv2d(out_planes, out_planes, 3, padding=3,dilation=3), nn.BatchNorm2d(out_planes), nn.ReLU(True))
        self.dcConv3 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1), nn.BatchNorm2d(out_planes),
            nn.Conv2d(out_planes, out_planes, 3, padding=5,dilation=5), nn.BatchNorm2d(out_planes), nn.ReLU(True))
        
        self.gate = GateFusion(out_planes*2)

        self.catconv = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes, 3,stride=1,padding=1), nn.BatchNorm2d(out_planes),
            nn.Conv2d(out_planes, out_planes//2, 3,stride=1,padding=1), nn.BatchNorm2d(out_planes//2), nn.ReLU(True))

    def forward(self, x, glocal_f):
        _,_,h,w = x.shape
        x1 = self.dcConv1(x)
        x2 = self.dcConv2(x)
        x3 = self.dcConv3(x)
        glocal_f = F.interpolate(glocal_f,size=(h,w),mode="bilinear", align_corners=False)
        x1_ = torch.cat((x1,glocal_f),dim=1)
        x2_ = torch.cat((x2,glocal_f),dim=1)
        x3_ = torch.cat((x3,glocal_f),dim=1)
        x_fusion = self.gate(x1_,x2_,x3_)
        x_fusion = self.catconv(x_fusion)
        return x_fusion

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, _ = x1.size()
        _, seq_len2, _ = x2.size()

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / (self.k_dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        
        return output 

class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, 3, padding=1, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            ) 
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        out = out + x        
        return out   
class ConvolutionalAttention(nn.Module): 
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_(self.kv, std=0.001)
        trunc_normal_(self.kv3, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                constant_(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_(m.weight, val=1.)
            constant_(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                constant_(m.bias, val=0.)


    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)   
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)  
        x = x.reshape([x_shape[0], self.inter_channels, h, w]) 
        return x

    def forward(self, x):
        x = self.norm(x)
        x1 = F.conv2d(x,self.kv,bias=None,stride=1,padding=(3,0))  
        x1 = self._act_dn(x1)  
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), bias=None, stride=1,padding=(3,0))  
        x3 = F.conv2d(x,self.kv3,bias=None,stride=1,padding=(0,3)) 
        x3 = self._act_dn(x3)
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3)) 
        x= x1+x3
        return x  
class GlobalDeepModule(nn.Module):
    def __init__(self, channels):
        super(GlobalDeepModule, self).__init__()
        self.cconv = BasicConv2d(64,64,1)
        self.catConv2 = nn.Sequential(nn.Conv2d(channels[2], 64, 3,stride=1, padding=1),nn.BatchNorm2d(64), nn.ReLU(True))
        self.catConv3 = nn.Sequential(nn.Conv2d(channels[1], 64, 3,stride=1, padding=1),nn.BatchNorm2d(64), nn.ReLU(True))
        self.catConv4 = nn.Sequential(nn.Conv2d(channels[0], 64, 3,stride=1, padding=1),nn.BatchNorm2d(64), nn.ReLU(True))

        self.ca2 = ConvolutionalAttention(64,64)
        self.ca3 = ConvolutionalAttention(64,64)
        self.ca4 = ConvolutionalAttention(64,64)
        self.gate = GateFusion(64)
        self.rca = RCA(64)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dc_cat_conv = nn.Sequential(nn.Conv2d(64,32,1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 64, kernel_size=1),nn.BatchNorm2d(64))
        self.outconv = nn.Sequential(nn.Conv2d(64, 64, 3,stride=1, padding=1),nn.BatchNorm2d(64), nn.ReLU(True))

    def forward(self, x2,x3,x4):
        x2_ = self.catConv2(x2)
        x3 = F.interpolate(x3,scale_factor=2,mode='bilinear')
        x3_ = self.catConv3(x3)
        x4 = F.interpolate(x4,scale_factor=4,mode='bilinear')
        x4_ = self.catConv4(x4)
        x22 = self.ca2(x2_)
        x33 = self.ca3(x3_)
        x44 = self.ca4(x4_)
        f234 = self.gate(x22,x33,x44)
        f234_ = self.cconv(f234)

        
        b,c,h,w = f234_.shape
        g_x = self.avg(f234_)
        g_x_ = self.sigmoid(self.dc_cat_conv(g_x))

        x = f234_ * g_x_ + f234_
        x = self.outconv(x)
        return x
    


def dct_2d(x):
    """
    2D Discrete Cosine Transform (DCT-II).
    :param x: Input tensor of shape [H, W].
    :return: DCT coefficients of the same shape as input.
    """
    return torch.fft.fft2(x, norm='ortho').real


def idct_2d(x):
    """
    Inverse 2D Discrete Cosine Transform (IDCT).
    :param x: Input tensor of shape [H, W].
    :return: Reconstructed tensor of the same shape as input.
    """
    return torch.fft.ifft2(x, norm='ortho').real


class DCTChannelBlock2D(nn.Module):
    def __init__(self, channels, height, width):
        super(DCTChannelBlock2D, self).__init__()
        # Frequency-based attention mechanism
        self.fc = nn.Sequential(
            nn.Linear(channels, channels * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels, bias=False),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm([height, width], eps=1e-6)

    def forward(self, x):
        """
        Forward pass.
        :param x: Input tensor of shape [B, C, H, W].
        :return: Output tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.size()
        freq_list = []
        for i in range(C):
            # Apply DCT to each channel's spatial feature map
            freq = dct_2d(x[:, i, :, :])
            freq_list.append(freq)
        
        # Stack all frequency maps back
        freq_stack = torch.stack(freq_list, dim=1)  # Shape: [B, C, H, W]
        
        # Normalize frequency maps
        freq_stack = self.layer_norm(freq_stack)
        
        # Learn attention weights for channels
        channel_weights = freq_stack.mean(dim=(2, 3))  # Global pooling to [B, C]
        channel_weights = self.fc(channel_weights)    # Shape: [B, C]
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Apply attention to input
        return x * channel_weights

    
class SCINet(nn.Module):
    def __init__(self,num_classes=8):
        super(SCINet, self).__init__()
        self.backbone = pvt_v2_b2()
        path = '/pretrain/pvt_v2_b2.pth'
        channels=[512, 320, 128, 64]
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        print('Model pvtv2 created, param count: %d'+' backbone: ', sum([m.numel() for m in self.backbone.parameters()]))

        self.globalmodule = GlobalDeepModule(channels)


        self.rca1 = DCTChannelBlock2D(channels[-1],128,128)
        self.rca2 = DCTChannelBlock2D(channels[-2],64,64)
        self.rca3 = DCTChannelBlock2D(channels[-3],32,32)
        self.rca4 = DCTChannelBlock2D(channels[-4],16,16)
        
        self.f12Conv = DilationGate(channels[-1],channels[-1])
        self.f23Conv = DilationGate(channels[-2],channels[-2])
        self.f34Conv = DilationGate(channels[-3],256)

        self.f4Conv = DilationGate(channels[-4],256)
        self.glocalConv = nn.Sequential(
            nn.Conv2d(64, 128, 3,stride=1, padding=1), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3,stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.glocalConv1 = nn.Sequential(
            nn.Conv2d(128, 256, 3,stride=1, padding=1), nn.BatchNorm2d(256),nn.ReLU(True))
        self.out_head_glocal = nn.Conv2d(64, num_classes, 1)
        self.out_head1 = nn.Conv2d(32, num_classes, 1)
        self.out_head2 = nn.Conv2d(64, num_classes, 1)
        self.out_head3 = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):      
        # encoder
        x1, x2, x3, x4 = self.backbone(x)

        global_f = self.globalmodule(x2,x3,x4) 
                
        f12 = self.rca1(x1)
        f23 = self.rca2(x2)
        f34 = self.rca3(x3)
        f4 = self.rca4(x4)

        global_f_ = self.glocalConv(global_f)
        global_f_ = self.f4Conv(f4,global_f_)
        global_f_ = self.glocalConv1(global_f_)
        g34 = self.f34Conv(f34,global_f_)
        g34_ = F.interpolate(g34,scale_factor=2, mode='bilinear')
        g23 = self.f23Conv(f23,g34_)
        g23_ = F.interpolate(g23,scale_factor=2, mode='bilinear')
        g12 = self.f12Conv(f12,g23_)

        output_g = self.out_head_glocal(global_f)

        p1 = self.out_head1(g12)
        p2 = self.out_head2(g23)
        p3 = self.out_head3(g34)

        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')
        output_g = F.interpolate(output_g, scale_factor=8, mode='bilinear')

        
        return [p1,p2,p3,output_g]