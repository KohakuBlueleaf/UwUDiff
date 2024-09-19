import os
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class LogAdditionalLosses(Callback):

    def __init__(self, loss_name_mapping: dict[str, str], ema_decay: float = 0.99):
        super().__init__()
        self.ema_decay = ema_decay
        self.loss_name_mapping = loss_name_mapping
        self.state = {}
        for name, logged_name in self.loss_name_mapping.items():
            self.state[f"ema_{logged_name}"] = 0

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        aux_output = outputs["aux_output"]

        for name, logged_name in self.loss_name_mapping.items():
            loss = getattr(aux_output, name).detach().mean()
            ema = self.state[f"ema_{logged_name}"]
            ema = ema * self.ema_decay + (1 - self.ema_decay) * loss.item()
            self.state[f"ema_{logged_name}"] = ema

            pl_module.log(
                f"train/{logged_name}", loss.item(), on_step=True, logger=True
            )
            pl_module.log(
                f"train/ema_{logged_name}",
                ema,
                on_step=True,
                logger=True,
                prog_bar=True,
            )


class PlotValLossPerTimestep(Callback):
    # Note this is also called after sanity check at the beginning of training

    def __init__(
        self, n_diffusion_time_steps: int | None = None, loss_key: str = "losses"
    ):
        super().__init__()
        self.n_diffusion_time_steps = n_diffusion_time_steps
        self.loss_key = loss_key

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        n_diffusion_time_steps = (
            self.n_diffusion_time_steps or pl_module.n_diffusion_time_steps
        )
        self.validation_timestep_counts = torch.zeros(
            n_diffusion_time_steps, device=pl_module.device
        )
        self.validation_timestep_losses = torch.zeros_like(
            self.validation_timestep_counts
        )
        self.validation_timestep_squared_losses = torch.zeros_like(
            self.validation_timestep_counts
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, idx):

        n_diffusion_time_steps = (
            self.n_diffusion_time_steps or pl_module.n_diffusion_time_steps
        )
        _, aux_output = outputs
        losses = getattr(aux_output, self.loss_key)
        # Timesteps could be of type float
        timesteps = aux_output.timesteps.long()
        for timestep in range(n_diffusion_time_steps):
            relevant_indices = timesteps == timestep
            self.validation_timestep_counts[timestep] += relevant_indices.sum()
            self.validation_timestep_losses[timestep] += losses[relevant_indices].sum()
            self.validation_timestep_squared_losses[timestep] += (
                losses[relevant_indices] ** 2
            ).sum()

    def on_validation_epoch_end(self, trainer, pl_module):

        if dist.is_initialized():
            validation_timestep_counts = torch.sum(
                pl_module.all_gather(self.validation_timestep_counts), dim=0
            )
            validation_timestep_losses = torch.sum(
                pl_module.all_gather(self.validation_timestep_losses), dim=0
            )
            validation_timestep_squared_losses = torch.sum(
                pl_module.all_gather(self.validation_timestep_squared_losses), dim=0
            )
        else:
            validation_timestep_counts = self.validation_timestep_counts
            validation_timestep_losses = self.validation_timestep_losses
            validation_timestep_squared_losses = self.validation_timestep_squared_losses

        if pl_module.global_rank == 0:
            valid_timesteps = validation_timestep_counts > 0
            timestep_counts = validation_timestep_counts[valid_timesteps]

            # Compute mean loss per timestep
            avg_loss_per_timestep = (
                validation_timestep_losses[valid_timesteps] / timestep_counts
            )

            # Compute standard deviation using accumulated squared losses
            mean_square_loss_per_timestep = (
                validation_timestep_squared_losses[valid_timesteps] / timestep_counts
            )
            std_loss_per_timestep = torch.sqrt(
                mean_square_loss_per_timestep - avg_loss_per_timestep**2
            )

            fig = plt.figure(figsize=(12, 8))
            timesteps = torch.nonzero(valid_timesteps).flatten().cpu()

            # Plot average loss per timestep
            plt.plot(timesteps, avg_loss_per_timestep.cpu())

            # Plot standard deviation as shaded region
            plt.fill_between(
                timesteps,
                (avg_loss_per_timestep - std_loss_per_timestep).cpu(),
                (avg_loss_per_timestep + std_loss_per_timestep).cpu(),
                alpha=0.2,
            )
            plt.xlabel("Timestep")
            plt.ylabel("Loss")

            image_path = os.path.join(
                # trainer.log_dir,  # this default to .
                trainer.logger.experiment.dir,
                "plots",
                f"{self.loss_key}_per_timestep-training_step-{trainer.global_step}.png",
            )
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            # print(image_path)

            plt.savefig(image_path)
            plt.close(fig)
            trainer.logger.log_image(
                f"val/{self.loss_key}_per_timestep",
                [image_path],
                step=trainer.global_step,
            )
