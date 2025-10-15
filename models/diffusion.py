import logging, os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda import amp
from torchmetrics import MeanMetric

from utils.utils import create_directory
from utils.plot import plot_epoch_loss
from utils.data import inverse_transform
from models.ddpm.ddpm import DDPM

class DDPM_model:
    def __init__(self, cfg, denoiser):
        self.cfg = cfg
        self.denoiser = denoiser
        self.ddpm_sampler = DDPM(timesteps = self.cfg.GEN_MODEL.DDPM.TIMESTEPS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.cfg.GEN_MODEL.DDPM.TRAIN.LR)
        self.scaler = amp.GradScaler()
        self.denoiser.to(self.device)
        self.ddpm_sampler.to(self.device)
        torch.manual_seed(42)

    def train_step(self, batch:torch.Tensor, denoiser_model:nn.Module, forwardsampler:DDPM):
        # Sample a timestep uniformly
        t = torch.randint(low=0, high=forwardsampler.timesteps, size=(batch.shape[0],), device=batch.device)
        # Apply forward noising process on original images, up to step t (sample from q(x_t|x_0))
        x_noisy, eps_true = forwardsampler(batch, t)
        with amp.autocast():
            # Our prediction for the denoised image
            eps_predicted = denoiser_model(x_noisy, t)
            # Deduce the loss
            loss          = F.mse_loss(eps_predicted, eps_true)
        return loss

    def train_one_epoch(self, sampler, batched_train_dataloader, epoch):
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
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.detach().item()
                loss_record.update(loss_value)

                tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

            mean_loss = loss_record.compute().item()

            tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

        return mean_loss

    def train(self, batched_train_dataloader):
        logging.info("Init training ...")
        create_directory(self.cfg.DATA_FS.SAVE_DIR)

        epoch_loss = []
        best_loss  = 1e6

        # Training loop
        for epoch in range(1, self.cfg.GEN_MODEL.DDPM.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            gc.collect()

            # Training step
            epoch_mean_loss = self.train_one_epoch(self.ddpm_sampler, batched_train_dataloader, epoch=epoch)
            epoch_loss.append(epoch_mean_loss)

            # Save checkpoint DDPM model
            if epoch_mean_loss < best_loss:
                best_loss = epoch_mean_loss
                checkpoint_dict = {
                    "opt": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "model": self.denoiser.state_dict()
                }
                model_path = f'{self.cfg.DATA_FS.SAVE_DIR}/{self.cfg.GEN_MODEL.DDPM.NAME}'
                torch.save(checkpoint_dict, model_path)
                logging.info(f"DDPM training: checkpoint saved at epoch:{epoch}")
                del checkpoint_dict

        logging.info(f"DDPM: Done training! \n Trained model saved at {self.cfg.DATA_FS.SAVE_DIR}")
        model_prefix = self.cfg.GEN_MODEL.DDPM.NAME.split("_")[0]
        plot_epoch_loss(epoch_loss, self.cfg.DATA_FS.OUTPUT_DIR, model_prefix)

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
        x_T = x_T.to(self.device).float()
        xt_over_time = [(self.ddpm_sampler.timesteps, x_T)]
        # Denoising steps
        for t in tqdm(iterable=reversed(range(0, self.ddpm_sampler.timesteps)),
                          dynamic_ncols=False,total=self.ddpm_sampler.timesteps,
                          desc="Sampling :: ", position=0):
            t_tensor = torch.as_tensor(t, dtype=torch.long, device=self.device).reshape(-1).expand(x_T.shape[0])
            eps_pred = self.denoiser(x_T, t_tensor)
            x_T = self.ddpm_sampler.step_backward(eps_pred, x_T, t)
            if t % self.cfg.PLOT.PLOT_STEPS == 0:
                xt_over_time.append((t, x_T))

        logging.info('Done sampling.')
        plot_func(xt_over_time, model_prefix, self.cfg.DATA_FS.OUTPUT_DIR, self.cfg.PLOT.NAME, self.cfg.PLOT.FPS, *args)