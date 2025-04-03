import argparse
import logging
import sys
import os
import torch
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.data import CheckerboardDataset
from models.MLP import MLP
from utils.utils import create_directory
from utils.plot import plot_checkerboard

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/training.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

def train(cboard):
    create_directory("saved_models")
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data generation and plot
    plot_checkerboard(cboard)
    # model definition
    model = MLP(layers=5, channels=512).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data = torch.Tensor(cboard.sampled_points).to(device)
    training_steps = 100_000
    batch_size = 64
    pbar = tqdm.tqdm(range(training_steps), desc="Training")
    losses = []
    for i in pbar:
        x1 = data[torch.randint(data.size(0), (batch_size,))]
        x0 = torch.randn_like(x1, device=device)
        target = x1 - x0
        t = torch.rand(x1.size(0), device=device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = model(xt, t)
        loss = ((target - pred)**2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict()
    }, 'saved_models/flow_matching_mlp_model.pth')

    logging.info("Done training! \n Trained moded saved at saved_models dir")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a flow matching model for checkerboard data.")
    parser.add_argument('--N', type=int, default=1000, help='Number of points to sample for dataset.')
    parser.add_argument('--x_min', type=int, default=-4 ,help='min value over x axis')
    parser.add_argument('--x_max', type=int, default=4 ,help='max value over x axis')
    parser.add_argument('--y_min', type=int, default=-4 ,help='min value over y axis')
    parser.add_argument('--y_max', type=int, default=4 ,help='max value over y axis')
    parser.add_argument('--length', type=int, default=4, help='Length of checkboard pattern')
    args = parser.parse_args()

    cboard = CheckerboardDataset(N=args.N, x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max, length=args.length)
    train(cboard)