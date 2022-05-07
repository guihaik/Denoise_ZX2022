import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
from einops import rearrange
from torch.nn.parameter import Parameter
import numpy as np
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # print(q.shape)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # print(q.shape)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x






##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings


class CE(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10, shape=64, p_len=64, in_channels=16
                 , inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False, num_edge=50):
        super(CE, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE
        self.num_edge = num_edge
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=ksize ** 2 * inter_channels, out_features=(ksize ** 2 * inter_channels) // 4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=ksize ** 2 * inter_channels, out_features=(ksize ** 2 * inter_channels) // 4),
            nn.ReLU()
        )
        self.thr_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=ksize, stride=stride_1,
                                  padding=0)
        self.bias_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=ksize, stride=stride_1,
                                   padding=0)

    def G_src2dis(self, tpk):
        l, k = tpk.shape
        dis = []
        src = list((tpk.view(-1)).storage())
        for i in range(l):
            dis += [i] * k
        return src, dis

    def forward(self, b):
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = b1

        raw_int_bs = list(b1.size())  # b*c*h*w
        b4, _ = same_padding(b, [self.ksize, self.ksize], [self.stride_1, self.stride_1], [1, 1])
        soft_thr = self.thr_conv(b4).view(raw_int_bs[0], -1)
        soft_bias = self.bias_conv(b4).view(raw_int_bs[0], -1)

        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                            strides=[self.stride_2, self.stride_2],
                                                            rates=[1, 1],
                                                            padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        y = []
        w, h = raw_int_bs[2], raw_int_bs[3]
        _, paddings = same_padding(b3[0, 0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize],
                                   [self.stride_2, self.stride_2], [1, 1])
        for xi, wi, pi, thr, bias in zip(patch_112_group_2, patch_28_group, patch_112_group, soft_thr, soft_bias):
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            wi = self.fc1(wi.view(wi.shape[1], -1))
            xi = self.fc2(xi.view(xi.shape[1], -1)).permute(1, 0)
            score_map = torch.matmul(wi, xi)
            score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
                                       math.ceil(h / self.stride_2))
            b_s, l_s, h_s, w_s = score_map.shape
            yi = score_map.view(l_s, -1)
            mask = F.relu(yi - yi.mean(dim=1, keepdim=True) * thr.unsqueeze(1) + bias.unsqueeze(1))
            mask_b = (mask != 0.).float()
            yi = yi * mask
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mask_b
            pi = pi.view(h_s * w_s, -1)
            yi = torch.mm(yi, pi)
            yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize),
                                          padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (self.ksize, self.ksize), padding=paddings[0],
                                                 stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize),
                                                padding=paddings[0], stride=self.stride_1)
            zi = zi / out_mask
            y.append(zi)
        # print("sssssssssssssssssss")

        y = torch.cat(y, dim=0)
        return y
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除�?个人觉得出来的图更好�?
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置�?
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # 输入图的通道，这里我设置�?
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    return edge_detect



class BasicgrapBlock(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[8, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[4, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(BasicgrapBlock, self).__init__()


        self.down = DownSample(dim, dim)
        self.CE1 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.CE2 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.CE3 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.CE4 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.conv = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)

        self.up = UpSample(dim, dim)
        self.conv1x1 = nn.Conv2d(dim , dim, kernel_size=1, bias=bias)





    def forward(self, input):

       res = self.down(input)
       # print(res.shape)
       res1 = self.CE1(res)
       res2 = self.CE2(res)
       res3 = self.CE3(res)
       res4 = self.CE4(res)
       # print(res4.shape)
       ce_branch = torch.cat([res1, res2, res3, res4], dim=1)
       ce_branch = self.conv(ce_branch)
       ce_branch = self.up(ce_branch)

       res = self.conv1x1(ce_branch)
       res += input
       return res


class BasicBlock(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[8, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[4, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(BasicBlock, self).__init__()
        self.trans = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type)

        self.down = DownSample(dim, dim)
        self.CE1 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.CE2 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.CE3 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.CE4 = CE(in_channels=dim * 2, inter_channels=dim // 2)
        self.conv = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)

        self.up = UpSample(dim, dim)
        self.conv1x1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)





    def forward(self, input):
       # print("sssssssssssssss")
       # print(input)
       t = self.trans(input)

       res = self.down(input)
       # print(res.shape)
       res1 = self.CE1(res)
       res2 = self.CE2(res)
       res3 = self.CE3(res)
       res4 = self.CE4(res)
       # print(res4.shape)
       ce_branch = torch.cat([res1, res2, res3, res4], dim=1)
       ce_branch = self.conv(ce_branch)
       ce_branch = self.up(ce_branch)
       res = torch.cat([t, ce_branch], dim=1)
       res = self.conv1x1(res)
       res += input
       return res



       res = self.conv(res)



       return res + input




class Gformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[4, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Gformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.sobel = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
        # 将sobel算子转换为适配卷积操作的卷积核
        self.sobel_kernel = self.sobel_kernel.reshape((1, 1, 3, 3))
        # 卷积输出通道，这里我设置�?
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=1)
        # 输入图的通道，这里我设置�?
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=0)

        self.sobel.weight.data = torch.from_numpy(self.sobel_kernel)

        # self.sobel_kernel = torch.from_numpy(self.sobel_kernel)

        self.Gbasic = nn.Sequential(*[
            BasicgrapBlock(dim=dim) for i in range(num_blocks[0])])

        self.Fbasic = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.conv_edge = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        #### For Dual-Pixel Defocus Deblurring Task ####
        # self.dual_pixel_task = dual_pixel_task
        # if self.dual_pixel_task:
        #     self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        g = self.sobel(inp_img)
        # g = F.conv2d(inp_img,self.sobel_kernel,stride=1,padding=1)
        # print(g)
        g = self.conv_edge(g)
        g = self.Gbasic(g)

        f = self.patch_embed(inp_img)
        f = self.Fbasic(f)
        res = torch.cat([f, g], dim=1)
        out = self.output(res)
        return out + inp_img



class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_feature
        self.out_features = out_feature
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        print(self.weight.shape)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BigKernelRestormerBlock(TransformerBlock):
    def __init__(self, dim=32, num_heads=4, ffn_expansion_factor=1.2, norm_layer='WithBias', bias=False,
                 dilated_rate=4, kernel_size=5):
        super().__init__(dim, num_heads, ffn_expansion_factor, norm_layer, bias)
        padding = (kernel_size * dilated_rate - dilated_rate) // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding, dilated_rate, dim)

    def forward(self, x):
        return self.conv(super().forward(x))


class PDConvFuse(nn.Module):
    def __init__(self,input_feature=256,feature_num=2,bias=True):
        super(PDConvFuse,self).__init__()
        self.pwconv = nn.Conv2d(feature_num * input_feature, input_feature, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(input_feature, input_feature, 3, 1, 1, bias=bias, groups=input_feature)
        self.act_layer = nn.GELU()
    def forward(self,*x):
        return self.dwconv(self.act_layer(self.pwconv(torch.cat(x, 1))))

class Denoiser(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=4,
                 dim=48,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 f_number = 32,
                 num_layers = 32
                 ):
        self.num_layers = num_layers
        super(Denoiser, self).__init__()

        self.pre_conv = nn.Sequential(
                                        nn.Conv2d(inp_channels,f_number,5,padding=2),
                                        nn.Conv2d(f_number,f_number,5,padding=2,groups=f_number),
                                        nn.Conv2d(f_number,f_number,1,padding=0)
        )
        self.trans_blocks = nn.Sequential(*[
            BigKernelRestormerBlock() for i in range(self.num_layers)])

        self.skip_connect_blocks = [
            PDConvFuse(f_number, feature_num=2)
            for _ in range(math.ceil(self.num_layers / 2) - 1)
        ]
        self.skip_connect_blocks = nn.Sequential(*self.skip_connect_blocks)

        self.post_conv = nn.Sequential(
            nn.Conv2d(f_number, f_number, 5, padding=2, groups=f_number),
            nn.Conv2d(f_number, f_number, 1, padding=0),
            nn.Conv2d(f_number, out_channels, 5, padding=2),
        )

    def forward(self, x):

        shortcut = x
        x = self.pre_conv(x)

        skip_features = []
        idx_skip = 1
        for idx, b in enumerate(self.trans_blocks):
            if idx > math.floor(self.num_layers / 2):
                x = self.skip_connect_blocks[idx_skip - 1](x, skip_features[-idx_skip])
                idx_skip += 1

            x = b(x)

            if idx < (math.ceil(self.num_layers / 2) - 1):
                skip_features.append(x)

        x = self.post_conv(x)

        return x + shortcut



if __name__ == '__main__':
        pic = torch.rand(1, 4, 512, 512).cuda()
        model = Denoiser().cuda()
        res = model(pic)
        print(res.shape)
        #from ptflops import get_model_complexity_info

        #acs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True, print_per_layer_stat=True,
        #                                         verbose=True)

        #print(macs)
        #print(params)
        # Trans =  TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2.66, bias=False,
        #                  LayerNorm_type='WithBias')
        # CE1 = CE(in_channels=64, inter_channels=64)
        #
        # res = Trans(pic)
        # print(res.shape)
        #
        # res1 = CE1(pic)
        # model = GraphConvolution(3, 64)
        # res = model(pic,pic)
        # print(res.shape)


