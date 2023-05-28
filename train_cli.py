from pathlib import Path

from lightning.pytorch.callbacks import (
    # EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from leaf_disease.lit_models.lit_model import LitModel


def cli_main():
    monitor, mode = "val_acc", "max"
    callbacks = [
        # EarlyStopping(monitor=monitor, patience=1, verbose=True, mode=mode),
        ModelCheckpoint(
            filename="best", monitor=monitor, mode=mode, verbose=True, save_last=True
        ),
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
    ]

    cli = LightningCLI(
        model_class=LitModel,
        # datamodule_class=LeafImageDataModule,
        trainer_defaults={
            "callbacks": callbacks,
            "max_epochs": 3,
            "accelerator": "mps",
            "devices": 1,
            "logger": TensorBoardLogger(
                save_dir="logs", name="default", sub_dir="tf_logs"
            ),
            "deterministic": True,
            "gradient_clip_val": 0.1,  # from Kaggle Notebook
            "num_sanity_val_steps": 1,
        },
        run=False,
    )
    # If True, we plot the computation graph in tensorboard
    cli.trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    cli.trainer.logger._default_hp_metric = None

    print(f"{cli.trainer.logger.name=}")
    print(f"{cli.datamodule.transforms.image_size=}")
    print(f"{cli.datamodule.batch_size=}")
    print(f"{cli.model.pytorch_model.model=}")
    print(f"{cli.model.pytorch_model.weights=}")

    # Start training the model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Evaluate datasets with best model and save metrics a file
    dataloaders = [
        cli.datamodule.train_dataloader(),
        cli.datamodule.val_dataloader(),
        # cli.datamodule.test_dataloader(),
    ]
    best = cli.trainer.validate(cli.model, dataloaders=dataloaders, ckpt_path="best")

    # Save best model metrics
    checkpoints_path = Path(cli.trainer.checkpoint_callback.best_model_path).parent
    print(f"best_model_path: {checkpoints_path}")

    with open(checkpoints_path / "training_metrics.csv", "w") as f:
        f.write(f"{best}")


if __name__ == "__main__":
    cli_main()

    # python src/main_cli.py --config configs/config_mf.yaml
