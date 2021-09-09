import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
from . import spectral_norm as SN


####################
# Degradation Simulator
####################


class DegNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, n_deg_lr = 6, n_deg_hr = 10, n_rec = 10, upscale=4, is_train = True,
                 mode = 'CNA', act_type = 'leakyrelu', output='flr'):
        super(DegNet, self).__init__()
        self.output = output
        rb_blocks_lr = [B.ResNetBlock(nf, nf, nf, norm_type=None, act_type=act_type, \
                                      mode=mode, res_scale=1) for _ in range(n_deg_lr)]
        n_upscale = int(math.log(upscale, 2))
        # upsample_block = B.upconv_blcok
        upsample_block = B.pixelshuffle_block
        upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        rb_blocks_hr = [B.ResNetBlock(nf, nf, nf, norm_type=None, act_type=act_type, \
                                      mode=mode, res_scale=1) for _ in range(n_deg_hr)]

        self.deg_net = B.sequential(
            B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None),
            B.ShortcutBlock(
                B.sequential(*rb_blocks_lr, B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None,
                                                         mode=mode))),
            *upsampler,
            B.ShortcutBlock(
                B.sequential(*rb_blocks_hr, B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None,
                                                         mode=mode))),
            B.conv_block(nf, in_nc, kernel_size=3, norm_type=None, act_type=None)
        )

        res_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=None, act_type=act_type, \
                                    mode=mode, res_scale=1) for _ in range(n_rec)]
        self.rec_net0 = B.sequential(
            B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None),
            B.ShortcutBlock(B.sequential(*res_blocks, B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None,
                                                                   mode=mode)))
        )
        self.rec_net1 = B.sequential(
            B.conv_block(nf, nf, kernel_size=4, stride=4, norm_type=None, act_type=act_type),
            B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        )

        if not is_train: # No need to BP to variable
            for k, v in self.deg_net.named_parameters():
                v.requires_grad = False
            for k, v in self.rec_net0.named_parameters():
                v.requires_grad = False
            for k, v in self.rec_net1.named_parameters():
                v.requires_grad = False

    def forward(self, x, mode):
        # x[0]: lr; x[1]: hr; output: flr | deg
        if mode == "flr": # fake LR
            deg = self.deg_net(x[0])
            out = x[1] * deg
            out = self.rec_net0(out)
            out = self.rec_net1(out)
        elif mode == "deg": # degradation map
            out = self.deg_net(x[0])

        return out


####################
# Generator
####################


class SRResNetLH(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb_lr = 6, nb_hr = 10, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNetLH, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks_lr = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb_lr)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        resnet_blocks_hr = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type, \
                                       mode=mode, res_scale=res_scale) for _ in range(nb_hr)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks_lr, LR_conv)),\
            *upsampler, B.ShortcutBlock(B.sequential(*resnet_blocks_hr, HR_conv0)), HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class EDSR(nn.Module):
    def __init__(self, in_nc, out_nc, nf, n_resblocks =32, upscale=4, res_scale=0.1):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        url_name = 'r{}f{}x{}'.format(n_resblocks, nf, upscale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        # define head module
        m_head = [nn.Conv2d(in_nc, nf, kernel_size, padding=(kernel_size // 2), bias=True)]
        # define body module
        m_body = [
            B.ResBlock(
                nn.Conv2d, nf, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(nf, nf, kernel_size, padding=(kernel_size // 2), bias=True))
        # define tail module
        m_tail = [
            B.Upsampler(nn.Conv2d, upscale, nf, act=False),
            nn.Conv2d(nf, out_nc, kernel_size, padding=(kernel_size // 2), bias=True)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x


class RCAN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, n_resgroups = 10, n_resblocks =20, upscale=4, res_scale=1, reduction =16):
        super(RCAN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)

        modules_head = [nn.Conv2d(in_nc, nf, kernel_size, padding=(kernel_size // 2), bias=True)]
        modules_body = [B.ResidualGroup(nn.Conv2d, nf, kernel_size, reduction, act=act, res_scale=res_scale,
                                        n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(nf, nf, kernel_size, padding=(kernel_size // 2), bias=True))
        modules_tail = [B.Upsampler(nn.Conv2d, upscale, nf, act=False),
            nn.Conv2d(nf, out_nc, kernel_size, padding=(kernel_size // 2), bias=True)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


####################
# Discriminator
####################


# VGG style Discriminator
class Discriminator_VGG(nn.Module):
    def __init__(self, in_size, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG, self).__init__()
        # features
        # hxw, c
        # in_size, feature64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # in_size/2, 32
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # in_size/4, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # in_size/8, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # in_size/16, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # in_size/32, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        out_size = int(in_size/32)
        self.classifier = nn.Sequential(
            nn.Linear(512 * out_size * out_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

