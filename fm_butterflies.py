import argparse
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import torch
from utils.data import ButterfliesDataset, CustomTransform
from models.UNet import UNet
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train and sample a|from flow matching model for butterflies data.")
    parser.add_argument('--config_yml_file', type=str, default='config/Butterflies.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--task', type=str, default='TRAIN', help='Task to perform: TRAIN | SAMPLING')
    parser.add_argument('--model', type=str, default='FM', help='Model to use: FM | DDPM')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file)
    model_unet = UNet(input_channels = cfg.UNET.INPUT_CHANNELS,
                      output_channels= cfg.UNET.OUTPUT_CHANNELS,
                      base_channels           = cfg.UNET.BASE_CH,
                      base_channels_multiples = cfg.UNET.BASE_CH_MULT,
                      apply_attention         = cfg.UNET.APPLY_ATTENTION,
                      dropout_rate            = cfg.UNET.DROPOUT_RATE,
                      time_multiple           = cfg.UNET.TIME_EMB_MULT,
                      total_time_steps        = cfg.UNET.TOTAL_TIME_STEPS,
                      time_emb_max_frec       = cfg.UNET.TIME_EMB_MAX_FREC,
                      )

    fm_model = FM_model(cfg, model_unet)
    ddpm_model = DDPM_model(cfg, model_unet)

    if args.task == "TRAIN":
        bflies_data = ButterfliesDataset(transform=CustomTransform(cfg.DATASET.IMAGE_SIZE))
        batched_bflies_data = DataLoader(bflies_data, batch_size=cfg.DATASET.BATCH_SIZE, **cfg.DATASET.params)
        if args.model == "FM":
            fm_model.train(batched_bflies_data)
        else:
            ddpm_model.train(batched_bflies_data)
    elif args.task == "SAMPLING":
        if args.model == "FM":
            xt = torch.randn((cfg.SAMPLING.NUM_SAMPLES,cfg.UNET.INPUT_CHANNELS,cfg.DATASET.IMAGE_SIZE,cfg.DATASET.IMAGE_SIZE))
            fm_model.sampling(xt, plot_butterflies_over_time)
        else:
            xt = torch.randn((cfg.SAMPLING.NUM_SAMPLES,cfg.UNET.INPUT_CHANNELS,cfg.DATASET.IMAGE_SIZE,cfg.DATASET.IMAGE_SIZE)).clamp(-1, 1)
            ddpm_model.sampling(xt, plot_butterflies_over_time)