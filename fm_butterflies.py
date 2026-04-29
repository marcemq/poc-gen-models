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

def build_unet(cfg):
    return UNet(
        input_channels          = cfg.UNET.INPUT_CHANNELS,
        output_channels         = cfg.UNET.OUTPUT_CHANNELS,
        base_channels           = cfg.UNET.BASE_CH,
        base_channels_multiples = cfg.UNET.BASE_CH_MULT,
        apply_attention         = cfg.UNET.APPLY_ATTENTION,
        dropout_rate            = cfg.UNET.DROPOUT_RATE,
        time_multiple           = cfg.UNET.TIME_EMB_MULT,
        total_time_steps        = cfg.UNET.TOTAL_TIME_STEPS,
        time_emb_max_frec       = cfg.UNET.TIME_EMB_MAX_FREC,
    )

def build_dit(cfg):
    return DiT(
        input_channels    = cfg.DIT.INPUT_CHANNELS,
        output_channels   = cfg.DIT.OUTPUT_CHANNELS,
        img_size          = cfg.DATASET.IMAGE_SIZE,
        patch_size        = cfg.DIT.PATCH_SIZE,
        hidden_size       = cfg.DIT.HIDDEN_SIZE,
        depth             = cfg.DIT.DEPTH,
        num_heads         = cfg.DIT.NUM_HEADS,
        mlp_ratio         = cfg.DIT.MLP_RATIO,
        dropout_rate      = cfg.DIT.DROPOUT_RATE,
        time_multiple      = cfg.DIT.TIME_EMB_MULT,
        time_emb_max_frec = cfg.DIT.TIME_EMB_MAX_FREC,
        total_time_steps  = cfg.DIT.TOTAL_TIME_STEPS,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train and sample a|from flow matching model for butterflies data.")
    parser.add_argument('--config_yml_file', type=str, default='config/Butterflies.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--task', type=str, default='TRAIN', help='Task to perform: TRAIN | SAMPLING')
    parser.add_argument('--backbone', type=str, default='UNet', help='Backbone to use: UNet | DiT')
    parser.add_argument('--generative_model', type=str, default='FM', help='Model to use: FM | DDPM')

    args = parser.parse_args()
    cfg = getYamlConfig(args.config_yml_file)
    # Build backbone
    backbone_key = args.backbone.upper()
    if backbone_key == 'UNET':
        backbone = build_unet(cfg)
        logging.info("Using UNet backbone.")
    elif backbone_key == 'DIT':
        backbone = build_dit(cfg)
        logging.info("Using DiT backbone.")
    else:
        raise ValueError(f"Unknown backbone '{args.backbone}'. Choose UNET or DiT.")

    fm_model = FM_model(cfg, backbone, backbone_key)
    ddpm_model = DDPM_model(cfg, backbone, backbone_key)
    gen_model_key = args.generative_model.upper()

    if args.task.upper() == "TRAIN":
        bflies_data = ButterfliesDataset(transform=CustomTransform(cfg.DATASET.IMAGE_SIZE))
        batched_bflies_data = DataLoader(bflies_data, batch_size=cfg.DATASET.BATCH_SIZE, **cfg.DATASET.params)
        if gen_model_key == "FM":
            fm_model.train(batched_bflies_data)
        elif gen_model_key == "DDPM":
            ddpm_model.train(batched_bflies_data)
        else:
            raise ValueError(f"Unknown generative model '{args.generative_model}'. Choose FM or DDPM.")
    elif args.task.upper() == "SAMPLING":
        img_size = cfg.DATASET.IMAGE_SIZE
        n        = cfg.SAMPLING.NUM_SAMPLES
        in_ch    = backbone.input_channels

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