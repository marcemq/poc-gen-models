import argparse
import logging
import sys
import torch
import tqdm
from utils.data import CheckerboardDataset
from models.MLP import MLP
from utils.utils import create_directory
from utils.plot import plot_checkerboard_over_time

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/sampling.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

def sampling(cboard, plot_steps):
    create_directory("images")
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_matching_model = MLP(layers=5, channels=512)
    checkpoint = torch.load('saved_models/flow_matching_mlp_model.pth', map_location=torch.device('cpu'))
    flow_matching_model.load_state_dict(checkpoint['model_state_dict'])
    flow_matching_model.to(device)
    flow_matching_model.eval().requires_grad_(False)

    xt = torch.randn(1000, 2, device=device)
    steps = 1000
    xt_over_time = []
    xt_over_time.append((0, xt))
    pbar = tqdm.tqdm(range(1, steps + 1), desc="Sampling")
    for i, t in enumerate(torch.linspace(0, 1, steps, device=device), start=1):
        pred = flow_matching_model(xt, t.expand(xt.size(0)))
        xt = xt + (1 / steps) * pred
        if i % plot_steps == 0:
            xt_over_time.append((t, xt))
        pbar.update(1)

    pbar.close()
    logging.info('Done sampling')

    plot_checkerboard_over_time(xt_over_time, cboard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a flow matching model for checkerboard data.")
    parser.add_argument('--N', type=int, default=1000, help='Number of points to sample for dataset.')
    parser.add_argument('--x_min', type=int, default=-4 ,help='min value over x axis')
    parser.add_argument('--x_max', type=int, default=4 ,help='max value over x axis')
    parser.add_argument('--y_min', type=int, default=-4 ,help='min value over y axis')
    parser.add_argument('--y_max', type=int, default=4 ,help='max value over y axis')
    parser.add_argument('--length', type=int, default=4, help='Length of checkboard pattern')
    parser.add_argument('--plot_steps', type=int, default=20, help='Var to tell plot sampling every certain steps')
    args = parser.parse_args()

    cboard = CheckerboardDataset(N=args.N, x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max, length=args.length)
    sampling(cboard, args.plot_steps)