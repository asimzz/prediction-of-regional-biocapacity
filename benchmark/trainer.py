import pickle
import torch
import torch.nn as nn
from enum import Enum


class ModelTrainer:
    def __init__(self, model, model_name,train_loader, val_loader):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        outputs = [self.model.validation_step(batch) for batch in self.val_loader]
        return self.model.validation_epoch_end(outputs)

    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def train(
        self, epochs, save_path="", max_lr=0.01, weight_decay=1e-4, grad_clip=0.1
    ):
        history = []
        train_loader = self.train_loader
        val_loader = self.val_loader

        optimizer = torch.optim.Adam(
            self.model.parameters(), max_lr, weight_decay=weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = self.model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(self._get_lr(optimizer))
                sched.step()

            # Validation phase
            result = self.evaluate(val_loader)
            result["train_loss"] = torch.stack(train_losses).mean().item()
            result["lrs"] = lrs
            self.model.epoch_end(epoch, result)
            history.append(result)

        # save the model parameters
        torch.save(
            {
                "epochs": epochs,
                f"{self.model_name}": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "weight_decay": weight_decay,
                "scheduler": sched,
                "learning_rates": lrs,
                "grad_clip": grad_clip,
                "train_losses": train_losses,
            },
            save_path,
        )
        with open(save_path, "wb") as handle:
            pickle.dump(history, handle)
        return history
