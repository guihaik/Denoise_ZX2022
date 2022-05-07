import torch
import torch.nn as nn
import numbers
import torch.nn.functional as F
from einops import rearrange

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

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

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.down(x)
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        x = self.up(x)
        return x



class CTDBlock(nn.Module):
    def __init__(self,in_channels,output_channels,head,ffn_expansion_factor,bias,LayerNorm_type,num_blocks):
        super(CTDBlock, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_channels, output_channels)
        self.encoder_level = nn.Sequential(*[
            TransformerBlock(dim=output_channels, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.down = nn.MaxPool2d(kernel_size=2)  ## From Level 1 to Level 2
    def forward(self,x,down_):
        x = self.patch_embed(x)
        x = self.encoder_level(x)
        if down_:
            x = self.down(x)
        return x

class UTBlock(nn.Module):
    def __init__(self,in_channels,output_channels,head,ffn_expansion_factor,bias,LayerNorm_type,num_blocks):
        super(CTDBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, output_channels, 2, stride=2)  ## From Level 1 to Level 2
        self.encoder_level = nn.Sequential(*[
            TransformerBlock(dim=output_channels, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

    def forward(self,x,up_):
        if up_:
            x = self.up(x)
        x = self.encoder_level(x)
        return x

class SSA(nn.Module):
    def __init__(self, in_channel, strides=1):
        super(SSA, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 8, kernel_size=3, stride=strides, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=strides, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv11 = nn.Conv2d(in_channel, 8, kernel_size=1, stride=strides, padding=0)

    def forward(self, input1, input2):
        input1 = input1.permute(0, 2, 3, 1)
        input2 = input2.permute(0, 2, 3, 1)
        cat = torch.cat([input1, input2], 3)
        cat = cat.permute(0, 3, 1, 2)
        out1 = self.relu1(self.conv1(cat))
        out1 = self.relu2(self.conv2(out1))
        out2 = self.conv11(cat)
        conv = (out1 + out2).permute(0, 2, 3, 1)
        H, W, K, batch_size = conv.shape[1], conv.shape[2], conv.shape[3],conv.shape[0]
        # print(conv.shape)
        V = conv.reshape(batch_size,H * W, K)
        # print("V  : ",V.shape)
        Vtrans = torch.transpose(V, 2, 1)
        # Vtrans = V.transpose(2, 1)
        # print("Vtrans  : ",Vtrans.shape)
        Vinverse = torch.inverse(torch.bmm(Vtrans, V))
        Projection = torch.bmm(torch.bmm(V, Vinverse), Vtrans)
        # print("Projection  : ",Projection.shape)
        H1, W1, C1,batch_size = input1.shape[1], input1.shape[2], input1.shape[3], input1.shape[0]
        X1 = input1.reshape(batch_size, H1 * W1, C1)
        # print("X1  : ",X1.shape)
        Yproj = torch.bmm(Projection, X1)
        Y = Yproj.reshape(batch_size, H1, W1, C1)
        Y = Y.permute(0, 3, 1, 2)
        return Y

class Uformer(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=4,
                 dim=32,
                 num_blocks=[1,2, 4, 2, 1],
                 num_refinement_blocks=4,
                 heads=[2, 4, 8, 4, 2],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Uformer, self).__init__()

        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1 = Downsample()  ## From Level 1 to Level 2

        self.patch_embed2 = OverlapPatchEmbed(dim, 2*dim)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2 = Downsample()  ## From Level 2 to Level 3

        self.patch_embed3 = OverlapPatchEmbed(2*dim, 2*2 * dim)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2*2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3 = Downsample()

        self.patch_embed4 = OverlapPatchEmbed(2*2 * dim, 2* 2 * 2 * dim)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 * 2*2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])



        self.up3 = nn.ConvTranspose2d(dim*2*2*2, dim*2*2, 2, stride=2)  ## From Level 4 to Level 3
        self.patch_decode3 = OverlapPatchEmbed(2*2 * 2 * dim, 2*2 * dim)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=2*2 * dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])



        self.up2 = nn.ConvTranspose2d(dim*2*2, dim*2, 2, stride=2)  ## From Level 4 to Level 3
        self.patch_decode2 = OverlapPatchEmbed(2*2 * dim, 2*dim)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=2*dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up1 = nn.ConvTranspose2d(dim * 2 , dim , 2, stride=2)  ## From Level 4 to Level 3
        self.patch_decode1 = OverlapPatchEmbed( 2 * dim,  dim)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim= dim, num_heads=heads[4], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[4])])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.relu1(self.patch_embed1(inp_img))
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_down1 = self.down1(out_enc_level1)

        inp_enc_level2 = self.relu2(self.patch_embed2(out_down1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_down2 = self.down2(out_enc_level2)

        inp_enc_level3 = self.relu3(self.patch_embed3(out_down2))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_down3 = self.down3(out_enc_level3)

        inp_enc_level4 = self.relu4(self.patch_embed4(out_down3))
        out_enc_level4 = self.encoder_level4(inp_enc_level4)

        inp_dec_level3 = self.up3(out_enc_level4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.relu4(self.patch_decode3(inp_dec_level3))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)


        inp_dec_level2 = self.up2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.relu4(self.patch_decode2(inp_dec_level2))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.relu5(self.patch_decode1(inp_dec_level1))
        inp_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(inp_dec_level1)
        return out_dec_level1+inp_img
if __name__ == '__main__':
        pic = torch.rand(1, 4, 512, 512).cuda()
        model = Uformer().cuda()
        res = model(pic)
        print(res.shape)