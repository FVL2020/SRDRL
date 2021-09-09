import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch

logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1.0):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1.0, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################

# Degradation Simulator for degradation reconstruction loss
def define_R(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_R']
    which_model = opt_net['which_model_R']

    if which_model == 'deg_net':  # Degradation Simulator
        netR = arch.DegNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], n_deg_lr=opt_net['n_deg_lr'],
                           n_deg_hr=opt_net['n_deg_hr'], n_rec=opt_net['n_rec'], upscale=opt_net['scale'],
                           is_train=opt_net['is_train'], output = opt_net['output'])
    else:
        raise NotImplementedError('Degradation Simulator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        init_weights(netR, init_type='kaiming', scale=0.1)
    else:
        netR.eval()  # No need to train
    if gpu_ids:
        assert torch.cuda.is_available()
        netR = nn.DataParallel(netR)
    return netR


# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'deg_net':  # Degradation Simulator
        netG = arch.DegNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], n_deg_lr=opt_net['n_deg_lr'],
                           n_deg_hr=opt_net['n_deg_hr'], n_rec=opt_net['n_rec'], upscale=opt_net['scale'],
                           is_train=opt_net['is_train'], output = opt_net['output'])
    elif which_model == 'sr_resnet':  # SRResNet
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
                             nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
                             act_type=opt_net['act_type'], mode=opt_net['mode'], upsample_mode=opt_net['upsample_mode'])
    elif which_model == 'edsr':  # EDSR
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], res_scale=opt_net['res_scale'], \
                             nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
                             act_type=opt_net['act_type'], mode=opt_net['mode'], upsample_mode=opt_net['upsample_mode'])
    elif which_model == 'rcan':  # RCAN
        netG = arch.RCAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                         res_scale=opt_net['res_scale'], upscale=opt_net['scale'], n_resgroups=opt_net['n_resgroups'],
                         n_resblocks=opt_net['n_resblocks'])
    elif which_model == 'sr_resnet_lh':  # SRResNet_L6H10
        netG = arch.SRResNetLH(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
                             nb_lr=opt_net['nb_lr'], nb_hr=opt_net['nb_hr'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
                             act_type=opt_net['act_type'], mode=opt_net['mode'], upsample_mode=opt_net['upsample_mode'])
    elif which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'],
                            norm_type=opt_net['norm_type'],
                            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG


# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    if which_model == 'discriminator_vgg':
        netD = arch.Discriminator_VGG(in_size=opt_net['input_size'], in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                                          norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                          act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD

def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        assert torch.cuda.is_available()
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF
