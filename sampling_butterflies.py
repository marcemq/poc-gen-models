import argparse
import logging
import sys
import torch
import tqdm
from models.unet import UNet
from torch.utils.data import DataLoader
from utils.utils import create_directory
from utils.plot import plot_butterflies_over_time
from utils.data import ButterfliesDataset, CustomTransform


logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/sampling.log"),
                        logging.StreamHandler(sys.stdout)]
                    )
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8) # 32, 16, 8, 4
    APPLY_ATTENTION = (False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128

def sampling(plot_steps, num_samples=16):
    create_directory("images")
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fm_unet_model = UNet(
    input_channels          = 3,
    output_channels         = 3,
    base_channels           = ModelConfig.BASE_CH,
    base_channels_multiples = ModelConfig.BASE_CH_MULT,
    apply_attention         = ModelConfig.APPLY_ATTENTION,
    dropout_rate            = ModelConfig.DROPOUT_RATE,
    time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    checkpoint = torch.load('saved_models/fm_unet_model.pth', map_location=torch.device('cpu'))
    fm_unet_model.load_state_dict(checkpoint['model_state_dict'])
    fm_unet_model.to(device)
    fm_unet_model.eval().requires_grad_(False)

    xt = torch.randn((num_samples,3,64,64), device=device)
    steps = 1000
    xt_over_time = []
    xt_over_time.append((0, xt))
    pbar = tqdm.tqdm(range(1, steps + 1), desc="Sampling")
    for i, t in enumerate(torch.linspace(0, 1, steps, device=device), start=1):
        pred = fm_unet_model(xt, (t*1000).expand(xt.size(0)).long())
        xt = xt + (1 / steps) * pred
        if i % plot_steps == 0:
            xt_over_time.append((t, xt))
        pbar.update(1)

    pbar.close()
    logging.info('Done sampling')

    plot_butterflies_over_time(xt_over_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a flow matching model for checkerboard data.")
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate, default value 16')
    parser.add_argument('--plot_steps', type=int, default=20, help='Var to tell plot sampling every certain steps')
    args = parser.parse_args()

    sampling(args.plot_steps, args.num_samples)