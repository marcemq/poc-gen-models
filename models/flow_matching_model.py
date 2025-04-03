import logging
import torch
import tqdm
from utils.utils import create_directory

def train(cfg, model, batched_train_data):
    logging.info("Init training ...")
    create_directory(cfg.MODEL.MODEL_SAVE_DIR)
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_batches = len(batched_train_data)
    total_steps = cfg.TRAIN.EPOCHS * num_batches  # Total training steps
    pbar = tqdm.tqdm(total=total_steps, desc="Training")
    losses = []
    for epoch in range(cfg.TRAIN.EPOCHS):
        for batch in batched_train_data:
            x1 = batch.to(device).float()
            x0 = torch.randn_like(x1, device=device)
            target = x1 - x0
            t = torch.rand(x1.size(0), device=device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1

            pred = model(xt, (t*cfg.MODEL.TIME_EMB_MAX_POS).long().view(-1))
            loss = ((target - pred)**2).mean()

            loss.backward()
            optim.step()
            optim.zero_grad()

            pbar.set_postfix(loss=loss.item())
            losses.append(loss.item())

    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict()
    }, f'{cfg.MODEL.SAVE_DIR}/{cfg.MODEL.MODEL_NAME}')

    logging.info(f"Done training! \n Trained moded saved at {cfg.MODEL.MODEL_SAVE_DIR} dir")

def sampling(cfg, trained_fm_model, plot_func):
    logging.info("Init Sampling ...")
    create_directory(cfg.MODEL.OUTPUT_DIR)
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(f'{cfg.MODEL.SAVE_DIR}/{cfg.MODEL.MODEL_NAME}', map_location=torch.device('cpu'))
    trained_fm_model.load_state_dict(checkpoint['model_state_dict'])
    trained_fm_model.to(device)
    trained_fm_model.eval().requires_grad_(False)

    xt = torch.randn(cfg.SAMPLING.NUM_SAMPLES, 2, device=device)
    steps = cfg.SAMPLING.STEPS
    xt_over_time = []
    xt_over_time.append((0, xt))
    pbar = tqdm.tqdm(range(1, steps + 1), desc="Sampling")
    for i, t in enumerate(torch.linspace(0, 1, steps, device=device), start=1):
        pred = trained_fm_model(xt, t.expand(xt.size(0)))
        xt = xt + (1 / steps) * pred
        if i % cfg.PLOT.PLOT_STEPS == 0:
            xt_over_time.append((t, xt))
        pbar.update(1)

    pbar.close()
    logging.info('Done sampling')

    plot_func(xt_over_time, cboard)