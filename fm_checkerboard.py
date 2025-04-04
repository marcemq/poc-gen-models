import argparse
import logging
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import torch
from utils.data import CheckerboardDataset
from models.MLP import MLP
from utils.plot import plot_checkerboard_over_time
from utils.myparser import getYamlConfig
from torch.utils.data import DataLoader
from models.flow_matching_model import FM_model

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/checkerboard.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train and sample a|from flow matching model for checkerboard data.")
    parser.add_argument('--config_yml_file', type=str, default='config/Checkerboard.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--task', type=str, default='TRAIN', help='Task to perform: TRAIN | SAMPLING')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file)
    model_mlp = MLP(channels_data=cfg.MODEL.INPUT_CHANNELS, layers=cfg.MODEL.LAYERS, channels=cfg.MODEL.CHANNELS, channels_t=cfg.MODEL.CHANNELS_T)
    cboard_data = CheckerboardDataset(cfg_ds=cfg.DATASET)
    fm_model = FM_model(cfg, model_mlp)

    if args.task == "TRAIN":
        batched_cboard_data = DataLoader(cboard_data, batch_size=cfg.DATASET.BATCH_SIZE, **cfg.DATASET.params)
        fm_model.train(batched_cboard_data)
    elif args.task == "SAMPLING":
        xt = torch.randn((cfg.SAMPLING.NUM_SAMPLES, cfg.MODEL.INPUT_CHANNELS))
        fm_model.sampling(xt, plot_checkerboard_over_time, cboard_data)