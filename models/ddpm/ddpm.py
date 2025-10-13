import torch
from models.ddpm.forward import ForwardSampler
# This is a class that derives from ForwardSampler
# and that will add the posibility to do one step backwards
class DDPM(ForwardSampler):
    # This will implement one step back in the reverse process
    def step_backward(self, predicted_noise:torch.Tensor, xnoise:torch.Tensor, timestep: int):
        # Noise from normal distribution
        z  = torch.randn_like(xnoise) if timestep > 0 else torch.zeros_like(xnoise)
        beta_t                     = self.beta[timestep].reshape(-1, 1, 1, 1)
        one_by_sqrt_alpha_t        = self.one_by_sqrt_alpha[timestep].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[timestep].reshape(-1, 1, 1, 1)
        # Use the formula above to sample a denoised version from the noisy one
        xdenoised = (
            one_by_sqrt_alpha_t
            * (xnoise - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
        return xdenoised