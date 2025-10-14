import logging, os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda import amp
from torchmetrics import MeanMetric

from utils.utils import create_directory
from utils.data import inverse_transform
from models.ddpm.ddpm import DDPM

class DDPM_model:
    def __init__(self, cfg, denoiser):
        self.cfg = cfg
        self.denoiser = denoiser
        self.ddpm_sampler = DDPM(timesteps = self.cfg.GEN_MODEL.DDPM.TIMESTEPS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.cfg.GEN_MODEL.DDPM.TRAIN.LR)
        self.denoiser.to(self.device)
        self.ddpm_sampler.to(self.device)
        torch.manual_seed(42)

    def train_step(self, batch:torch.Tensor, model:nn.Module, forwardsampler:DDPM):
        # Sample a timestep uniformly
        t = torch.randint(low=0, high=forwardsampler.timesteps, size=(batch.shape[0],), device=batch.device)
        # Apply forward noising process on original images, up to step t (sample from q(x_t|x_0))
        x_noisy, eps_true = forwardsampler(batch, t)
        with amp.autocast():
            # Our prediction for the denoised image
            eps_predicted = model(x_noisy, t)
            # Deduce the loss
            loss          = F.mse_loss(eps_predicted, eps_true)
        return loss

    def train_one_epoch(self, sampler, batched_train_dataloader, scaler, epoch):
        loss_record = MeanMetric()
        # Set in training mode
        self.denoiser.train()

        with tqdm(total=len(batched_train_dataloader), dynamic_ncols=True) as tq:
            tq.set_description(f"Train :: Epoch: {epoch}/{self.cfg.GEN_MODEL.DDPM.TRAIN.EPOCHS}")
            # Scan the batches
            for batch in batched_train_dataloader:
                tq.update(1)
                x = batch.to(self.device).float()
                # Evaluate loss
                loss = self.train_step(x, self.denoiser, sampler)

                # Backpropagation and update
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_value = loss.detach().item()
                loss_record.update(loss_value)

                tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

            mean_loss = loss_record.compute().item()

            tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

        return mean_loss

    def train(self, batched_train_dataloader):
        # Loss
        loss_fn = nn.MSELoss()
        scaler = amp.GradScaler()

        logging.info("Init training ...")
        create_directory(self.cfg.DATA_FS.SAVE_DIR)

        # Training loop
        for epoch in range(1, self.cfg.GEN_MODEL.DDPM.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            gc.collect()

            # Training step
            self.train_one_epoch(self.ddpm_sampler, batched_train_dataloader, scaler, epoch=epoch)

        # Save checkpoint of best model
        checkpoint_dict = {
            "opt": self.optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model": self.denoiser.state_dict()
        }
        model_path = f'{self.cfg.DATA_FS.SAVE_DIR}/{self.cfg.GEN_MODEL.DDPM.NAME}'
        torch.save(checkpoint_dict, model_path)
        logging.info(f"DDPM : Done training! \n Trained model saved at {self.cfg.DATA_FS.SAVE_DIR}")
        del checkpoint_dict

    def _load_trained_model(self):
        model_path = f'{self.cfg.DATA_FS.SAVE_DIR}/{self.cfg.GEN_MODEL.DDPM.NAME}'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.denoiser.load_state_dict(checkpoint['model'])
        self.denoiser.to(self.device)
        self.denoiser.eval().requires_grad_(False)

    @torch.inference_mode()
    def sampling(self, x_T, plot_func, *args):
        model_prefix = self.cfg.GEN_MODEL.DDPM.NAME.split("_")[0]
        logging.info("Init Sampling ...")
        create_directory(self.cfg.DATA_FS.OUTPUT_DIR)

        self._load_trained_model()

        self.denoiser.eval()
        xt_over_time = [(0, x_T)]
        pbar = tqdm.tqdm(range(1, self.ddpm_sampler.timesteps + 1), desc="Sampling")
        # Denoising steps
        for t in reversed(range(self.ddpm_sampler.timesteps)):
            t_tensor = torch.as_tensor(t, dtype=torch.long, device=self.device).reshape(-1).expand(x_T.shape[0])
            eps_pred = self.denoiser(x_T, t_tensor)
            x_T = self.ddpm_sampler.timesteps.step_backward(eps_pred, x_T, t)
            x_T = inverse_transform(x_T).type(torch.uint8)
            if t % self.cfg.PLOT.PLOT_STEPS == 0:
                xt_over_time.append((t, x_T))
            pbar.update(1)

        pbar.close()
        logging.info('Done sampling.')
        plot_func(xt_over_time, model_prefix, self.cfg.DATA_FS.OUTPUT_DIR, self.cfg.PLOT.NAME, self.cfg.PLOT.FPS, *args)