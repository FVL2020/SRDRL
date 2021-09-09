import logging
logger = logging.getLogger('base')

def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'degnet': # degradation simulator
        from .DEGNET_model import DEGNETModel as M
    elif model == 'srdrl': # SR network
        from .SRDRL_model import SRDRLModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
