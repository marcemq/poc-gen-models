import argparse
import logging
import sys
import os
import torch
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from utils.data import ButterfliesDataset, CustomTransform
from models.unet import UNet
from utils.utils import create_directory


logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/training.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8) # 32, 16, 8, 4
    APPLY_ATTENTION = (False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128

def train(batched_train_data, epochs):
    create_directory("saved_models")
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model definition
    model_unet = UNet(
    input_channels          = 3,
    output_channels         = 3,
    base_channels           = ModelConfig.BASE_CH,
    base_channels_multiples = ModelConfig.BASE_CH_MULT,
    apply_attention         = ModelConfig.APPLY_ATTENTION,
    dropout_rate            = ModelConfig.DROPOUT_RATE,
    time_multiple           = ModelConfig.TIME_EMB_MULT,
    ).to(device)
    optim = torch.optim.AdamW(model_unet.parameters(), lr=1e-4)

    training_steps = 100_000
    pbar = tqdm.tqdm(range(training_steps), desc="Training")
    losses = []
    for epoch in range(1, epochs + 1):
        for batch in batched_train_data:
            x1 = batch.to(device)
            x0 = torch.randn_like(x1, device=device)
            target = x1 - x0
            t = torch.rand(x1.size(0), device=device).view(-1, 1, 1, 1)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            pred = model_unet(xt, (t*1000).long().view(-1,1))
            loss = ((target - pred)**2).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.set_postfix(loss=loss.item())
            losses.append(loss.item())

    torch.save({
    'model_state_dict': model_unet.state_dict(),
    'optimizer_state_dict': optim.state_dict()
    }, 'saved_models/fm_unet_model.pth')

    logging.info("Done training! \n Trained moded saved at saved_models dir")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a flow matching model for Butterflies dataset.")
    parser.add_argument('--epochs', type=int, default=10000, help='Epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size used for training')
    args = parser.parse_args()

    train_data=ButterfliesDataset(transform=CustomTransform())
    batched_train_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=6)
    train(batched_train_data, args.epochs)