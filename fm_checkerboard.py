import argparse
import logging
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.data import CheckerboardDataset
from models.MLP import MLP
from utils.plot import plot_checkerboard_over_time
from utils.myparser import getYamlConfig
from torch.utils.data import DataLoader
from models.flow_matching_model import train, sampling

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/checkerboard.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a flow matching model for checkerboard data.")
    parser.add_argument('--config_yml_file', type=str, default='config/Checkerboard.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--task', type=str, default='TRAIN', help='Task to perform: TRAIN | SAMPLING')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file)
    model = MLP(channels_data=cfg.MODEL.CHANNELS_DATA, layers=cfg.MODEL.LAYERS, channels=cfg.MODEL.CHANNELS, channels_t=cfg.MODEL.CHANNELS_T)
    cboard_data = CheckerboardDataset(cfg_ds=cfg.DATASET)

    if args.task == "TRAIN":
        batched_cboard_data = DataLoader(cboard_data, batch_size=cfg.DATASET.BATCH_SIZE, **cfg.DATASET.params)
        train(cfg, model, batched_cboard_data)
    elif args.task == "SAMPLING":
        sampling(cfg, model, plot_checkerboard_over_time, cboard_data)