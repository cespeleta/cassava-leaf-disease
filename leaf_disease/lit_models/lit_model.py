from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy


class LitModel(LightningModule):
    def __init__(
        self,
        pytorch_model: nn.Module,
        num_classes: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.pytorch_model = pytorch_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["pytorch_model"])

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def compute_loss(self, logits, true_labels) -> torch.Tensor:
        return F.cross_entropy(logits, true_labels)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        features, true_labels = batch
        logits = self(features)

        loss = self.compute_loss(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def forward(self, x) -> Any:
        return self.pytorch_model(x)

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.valid_acc(predicted_labels, true_labels)
        self.log("val_acc", self.valid_acc, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self) -> Any:
        # https://www.kaggle.com/code/vishnus/cassava-pytorch-lightning-starter-notebook-0-895/notebook
        optimizer = torch.optim.AdamW(
            self.pytorch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, eta_min=0
        )
        return [optimizer], [scheduler]

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.trainer.max_epochs,
        #     T_mult=1,
        #     eta_min=5e-6,
        #     last_epoch=-1,
        # )

        # return [optimizer], [scheduler]
