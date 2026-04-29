import logging, os
import torch
import tqdm
from utils.utils import create_directory
from utils.plot import plot_epoch_loss

class FM_model:
    def __init__(self, cfg, backbone, backbone_name):
        self.cfg = cfg
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim = torch.optim.AdamW(self.backbone.parameters(), lr=self.cfg.GEN_MODEL.FM.TRAIN.LR)
        self.backbone.to(self.device)

    def train(self, batched_train_data):
        logging.info("Init training ...")
        create_directory(self.cfg.DATA_FS.SAVE_DIR)
        torch.manual_seed(42)

        num_batches = len(batched_train_data)
        total_steps = self.cfg.GEN_MODEL.FM.TRAIN.EPOCHS * num_batches
        pbar = tqdm.tqdm(total=total_steps, desc="Training")
        losses = []

        for epoch in range(self.cfg.GEN_MODEL.FM.TRAIN.EPOCHS):
            for batch in batched_train_data:
                x1 = batch.to(self.device).float()
                x0 = torch.randn_like(x1, device=self.device)
                target = x1 - x0

                t = torch.rand(x1.size(0), device=self.device)
                t = t.view(-1, 1, 1, 1) if x1.dim() > 2 else t.view(-1, 1)

                xt = (1 - t) * x0 + t * x1
                pred = self.backbone(xt, (t * self.cfg.GEN_MODEL.FM.TIME_MAX_POS).long().view(-1))
                loss = ((target - pred) ** 2).mean()

                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                losses.append(loss.item())

        model_path = f'{self.cfg.DATA_FS.SAVE_DIR}/{self.cfg.GEN_MODEL.FM.NAME.format(self.backbone_name)}'
        torch.save({'model_state_dict': self.backbone.state_dict(), 'optimizer_state_dict': self.optim.state_dict()}, model_path)
        logging.info(f"FM : Done training! \n Trained model saved at {self.cfg.DATA_FS.SAVE_DIR}")

        model_prefix = self.cfg.GEN_MODEL.DDPM.NAME.split("_")[0] +"_"+ self.backbone_name
        plot_epoch_loss(losses, self.cfg.DATA_FS.OUTPUT_DIR, model_prefix)

    def _load_trained_model(self):
        model_path = f'{self.cfg.DATA_FS.SAVE_DIR}/{self.cfg.GEN_MODEL.FM.NAME}'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(checkpoint['model_state_dict'])
        self.backbone.to(self.device)
        self.backbone.eval().requires_grad_(False)

    def sampling(self, xt, plot_func, *args):
        model_prefix = self.cfg.GEN_MODEL.FM.NAME.split("_")[0]
        logging.info("Init Sampling ...")
        output_dir = f"self.cfg.DATA_FS.OUTPUT_DIR/FM_{self.backbone_name}"
        create_directory(output_dir)

        self._load_trained_model()
        xt = xt.to(self.device).float()

        steps = self.cfg.SAMPLING.STEPS
        xt_over_time = [(0, xt)]
        pbar = tqdm.tqdm(range(1, steps + 1), desc="Sampling")

        for i, t in enumerate(torch.linspace(0, 1, steps, device=self.device), start=1):
            time_indices = (t * self.cfg.GEN_MODEL.FM.TIME_MAX_POS).clamp(0, self.cfg.GEN_MODEL.FM.TIME_MAX_POS-1).long()
            time_indices = time_indices.to(self.device).expand(xt.size(0))
            pred = self.backbone(xt, time_indices)
            xt = xt + (1 / steps) * pred
            if i % self.cfg.PLOT.PLOT_STEPS == 0:
                xt_over_time.append((t, xt))
            pbar.update(1)

        pbar.close()
        logging.info('Done sampling.')
        plot_func(xt_over_time, model_prefix, output_dir, self.cfg.PLOT.NAME, self.cfg.PLOT.FPS, *args)