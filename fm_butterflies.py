import argparse
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import torch
from utils.data import ButterfliesDataset, CustomTransform
from models.UNet import UNet
from models.DiT import DiT
from utils.plot import plot_butterflies_over_time
from utils.myparser import getYamlConfig
from torch.utils.data import DataLoader
from models.flow_matching import FM_model
from models.diffusion import DDPM_model

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/gen_model_bflies.log"),
                        logging.StreamHandler(sys.stdout)]
                    )
def get_backbone_cfg(cfg, gen_model_key, backbone_key):
    """
    Navigate the nested config to the right backbone node.
    e.g. cfg.GEN_MODEL.DDPM.UNET  or  cfg.GEN_MODEL.FM.DIT
    """
    gen_cfg = getattr(cfg.GEN_MODEL, gen_model_key)   # FM | DDPM node
    return getattr(gen_cfg, backbone_key)              # UNET | DIT node

def build_unet(backbone_cfg):
    return UNet(
        input_channels          = backbone_cfg.INPUT_CHANNELS,
        output_channels         = backbone_cfg.OUTPUT_CHANNELS,
        base_channels           = backbone_cfg.BASE_CH,
        base_channels_multiples = backbone_cfg.BASE_CH_MULT,
        apply_attention         = backbone_cfg.APPLY_ATTENTION,
        dropout_rate            = backbone_cfg.DROPOUT_RATE,
        time_multiple           = backbone_cfg.TIME_EMB_MULT,
        total_time_steps        = backbone_cfg.TOTAL_TIME_STEPS,
        time_emb_max_frec       = backbone_cfg.TIME_EMB_MAX_FREC,
    )

def build_dit(backbone_cfg, img_size):
    return DiT(
        input_channels    = backbone_cfg.INPUT_CHANNELS,
        output_channels   = backbone_cfg.OUTPUT_CHANNELS,
        img_size          = img_size,
        patch_size        = backbone_cfg.PATCH_SIZE,
        hidden_size       = backbone_cfg.HIDDEN_SIZE,
        depth             = backbone_cfg.DEPTH,
        num_heads         = backbone_cfg.NUM_HEADS,
        mlp_ratio         = backbone_cfg.MLP_RATIO,
        dropout_rate      = backbone_cfg.DROPOUT_RATE,
        time_multiple     = backbone_cfg.TIME_EMB_MULT,
        time_emb_max_frec = backbone_cfg.TIME_EMB_MAX_FREC,
        total_time_steps  = backbone_cfg.TOTAL_TIME_STEPS,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train and sample a|from flow matching model for butterflies data.")
    parser.add_argument('--config_yml_file', type=str, default='config/Butterflies.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--task', type=str, default='TRAIN', help='Task to perform: TRAIN | SAMPLING')
    parser.add_argument('--backbone', type=str, default='UNet', help='Backbone to use: UNet | DiT')
    parser.add_argument('--generative_model', type=str, default='FM', help='Model to use: FM | DDPM')

    args = parser.parse_args()
    cfg = getYamlConfig(args.config_yml_file)

    gen_model_key   = args.generative_model.upper()   # FM | DDPM
    backbone_key    = args.backbone.upper()            # UNET | DIT
    backbone_cfg = get_backbone_cfg(cfg, gen_model_key, backbone_key)

    # Build backbone
    if backbone_key == 'UNET':
        backbone = build_unet(backbone_cfg)
        logging.info("Using UNet backbone.")
    elif backbone_key == 'DIT':
        backbone = build_dit(backbone_cfg, img_size=cfg.DATASET.IMAGE_SIZE)
        logging.info("Using DiT backbone.")
    else:
        raise ValueError(f"Unknown backbone '{args.backbone}'. Choose UNET or DiT.")

    # Build generative model
    fm_model   = FM_model(cfg, backbone, backbone_key)
    ddpm_model = DDPM_model(cfg, backbone, backbone_key)

    img_size = cfg.DATASET.IMAGE_SIZE
    n        = cfg.SAMPLING.NUM_SAMPLES
    in_ch    = backbone.input_channels

    # Task dispatch
    if args.task.upper() == "TRAIN":
        bflies_data = ButterfliesDataset(transform=CustomTransform(img_size))
        batched_bflies_data = DataLoader(bflies_data, batch_size=cfg.DATASET.BATCH_SIZE, **cfg.DATASET.params)
        if gen_model_key == "FM":
            fm_model.train(batched_bflies_data)
        elif gen_model_key == "DDPM":
            ddpm_model.train(batched_bflies_data)
        else:
            raise ValueError(f"Unknown generative model '{args.generative_model}'. Choose FM or DDPM.")
    elif args.task.upper() == "SAMPLING":
        if gen_model_key == "FM":
            xt = torch.randn((n, in_ch, img_size, img_size))
            fm_model.sampling(xt, plot_butterflies_over_time)
        elif gen_model_key == "DDPM":
            xt = torch.randn((n, in_ch, img_size, img_size)).clamp(-1, 1)
            ddpm_model.sampling(xt, plot_butterflies_over_time)
        else:
            raise ValueError(f"Unknown generative model '{args.generative_model}'. Choose FM or DDPM.")
    else:
        raise ValueError(f"Unknown task '{args.task}'. Choose TRAIN or SAMPLING.")