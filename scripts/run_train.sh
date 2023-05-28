: '
Experiments with Resnet models.
'

EXPERIMENT_NAME="resnet18"

echo "Running experiment: $EXPERIMENT_NAME"


poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model resnet18 \
    --model.pytorch_model.weights IMAGENET1K_V1 \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.max_epochs 10

poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model resnet18 \
    --model.pytorch_model.weights IMAGENET1K_V1 \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.max_epochs 10 \
    --data leaf_disease.datasets.dataset.LeafImageDataModule \
    --data.image_size 512


EXPERIMENT_NAME="resnext50_32x4d"
echo ======================================
echo "Running experiment: $EXPERIMENT_NAME"
echo ======================================

poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model resnext50_32x4d \
    --model.pytorch_model.weights IMAGENET1K_V1 \
    --data.batch_size 32 \
    --trainer.logger.name ${EXPERIMENT_NAME}

poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model resnext50_32x4d \
    --model.pytorch_model.weights IMAGENET1K_V2 \
    --data.batch_size 32 \
    --trainer.logger.name ${EXPERIMENT_NAME}

echo "Running experiment with image size 512"
EXPERIMENT_NAME="resnext50_32x4d"
poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model resnext50_32x4d \
    --model.pytorch_model.weights IMAGENET1K_V2 \
    --data.batch_size 32 \
    --data.image_size 512 \
    --trainer.logger.name ${EXPERIMENT_NAME}
    



# Depsues con distinta loss o optimizers