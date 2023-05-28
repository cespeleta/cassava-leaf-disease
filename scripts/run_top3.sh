: '
Evaluate different network arquitechtures models:

After running the experiments the best models are:
- efficientnet_b1
- efficientnet_b4
- resnext50_32x4d
'

EXPERIMENT_NAME="run-top3-512"
IMG_SIZE=512
echo "Running experiment: $EXPERIMENT_NAME"

declare -a models=("resnext50_32x4d")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model $model_name \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name \
    --data.batch_size 32 \
    --data.transforms.image_size $IMG_SIZE
done

declare -a models=("efficientnet_b1")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.efficientnet.Efficientnet \
    --model.pytorch_model.model $model_name \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name \
    --data.batch_size 8 \
    --data.transforms.image_size $IMG_SIZE
done

declare -a models=("efficientnet_b4")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.efficientnet.Efficientnet \
    --model.pytorch_model.model $model_name \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name \
    --data.batch_size 8 \
    --data.transforms.image_size $IMG_SIZE
done
