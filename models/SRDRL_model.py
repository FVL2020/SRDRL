import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss

logger = logging.getLogger('base')


class SRDRLModel(BaseModel):
    def __init__(self, opt):
        super(SRDRLModel, self).__init__(opt)
        train_opt = opt['train']

        self.netG = networks.define_G(opt).to(self.device)  # Generator

        if self.is_train:
            self.print_freq = opt['logger']['print_freq']
            self.netG.train()

            self.l_gan_w = train_opt['gan_weight']  # gan loss weight
            if self.l_gan_w: # use gan loss
                self.netD = networks.define_D(opt).to(self.device)
                self.netD.train()

            self.l_deg_w = train_opt['degradation_weight']  # degradation reconstruction loss weight
            if self.l_deg_w: # use degradation reconstruction loss
                self.netR = networks.define_R(opt).to(self.device)

            self.l_fea_w = train_opt['feature_weight']  # perceptual loss weight
            if self.l_fea_w: # use VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

        self.load()  # load G, D and R if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # pixel loss for G
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logging.info('Remove pixel loss.')
                self.cri_pix = None

            # feature loss for G
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
            else:
                logging.info('Remove feature loss.')
                self.cri_fea = None

            # gan loss for G,D
            if train_opt['gan_weight'] > 0:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            else:
                logging.info('Remove gan loss.')

            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # D
            if self.l_gan_w:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                                                    weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device) # real LR

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device) # real HR

    def optimize_parameters(self, step):
        # G
        if self.l_gan_w:
            for p in self.netD.parameters():
                p.requires_grad = False
        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L) # feed forward

        # Generator total loss
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix

            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            if self.l_gan_w:  # gan loss for G
                pred_g_fake = self.netD(self.fake_H)
                pred_d_real = self.netD(self.var_H).detach()
                l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                          self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                l_g_total += l_g_gan

            if self.l_deg_w: # degradation reconstruction loss
                real_L = self.netR((self.var_L, self.var_H)).detach()
                fake_L = self.netR((self.var_L, self.fake_H))
                l_deg_total = 0
                if self.cri_pix:  # pixel part
                    l_deg_pix = self.l_deg_w * self.l_pix_w * self.cri_pix(fake_L, real_L)
                    l_deg_total += l_deg_pix
                if self.cri_fea:  # feature part
                    real_fea_L = self.netF(real_L).detach()
                    fake_fea_L = self.netF(fake_L)
                    l_deg_fea = self.l_deg_w * self.l_fea_w * self.cri_fea(fake_fea_L, real_fea_L)
                    l_deg_total += l_deg_fea
                l_g_total += l_deg_total

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        if self.l_gan_w:
            for p in self.netD.parameters():
                p.requires_grad = True
            self.optimizer_D.zero_grad()
            pred_d_real = self.netD(self.var_H)
            pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2 # Discriminator total loss

            l_d_total.backward()
            self.optimizer_D.step()

        # set log
        if step % self.print_freq == 0:
            # G
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:
                    self.log_dict['l_g_pix'] = l_g_pix.item()
                    if self.l_deg_w:
                        self.log_dict['l_r_pix'] = l_deg_pix.item()
                if self.cri_fea:
                    self.log_dict['l_g_fea'] = l_g_fea.item()
                    if self.l_deg_w:
                        self.log_dict['l_r_fea'] = l_deg_fea.item()
                if self.l_gan_w:
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                self.log_dict['l_g_total'] = l_g_total.item()
            # D
            if self.l_gan_w:
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
                self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # G, Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.is_train:
            if self.l_gan_w:  # D, Discriminator
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                     self.netD.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD.__class__.__name__)

                logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            if self.l_deg_w:  # R, Degradation Simulator
                s, n = self.get_network_description(self.netR)
                if isinstance(self.netR, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netR.__class__.__name__,
                                                     self.netR.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netR.__class__.__name__)

                logger.info('Network R structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            if self.l_fea_w:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        # R
        load_path_R = self.opt['path']['pretrain_model_R']
        if load_path_R is not None:
            logger.info('Loading pretrained model for R [{:s}] ...'.format(load_path_R))
            self.load_network(load_path_R, self.netR)
        # G
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        # D
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.l_gan_w:
            self.save_network(self.netD, 'D', iter_step)
