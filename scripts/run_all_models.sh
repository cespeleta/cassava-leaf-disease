: '
Evaluate different network arquitechtures models:

After running the experiments the best models are:
- efficientnet_b1
- efficientnet_b4
- resnext50_32x4d
'

EXPERIMENT_NAME="run-all-models"
echo "Running experiment: $EXPERIMENT_NAME"


echo "========================"
echo "Running ResNet models"
echo "========================"

declare -a models=("resnet18" "resnet34" "resnet101" "resnext50_32x4d")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.Resnet \
    --model.pytorch_model.model $model_name \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name
done

echo "=================================="
echo "Running Resnet SSL and SWSL models"
echo "=================================="

declare -a models=("resnet18_ssl" "resnet50_ssl" "resnet18_swsl" "resnet18_swsl")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.resnet.ResnetSSL \
    --model.pytorch_model.model $model_name \
    --model.pytorch_model.weights null \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name
done


echo "==========================="
echo "Running Efficientnet models"
echo "==========================="

declare -a models=("efficientnet_b0" "efficientnet_b1")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.efficientnet.Efficientnet \
    --model.pytorch_model.model $model_name \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name
done


# Reduce batch size in this models
declare -a models=("efficientnet_b4" "efficientnet_b5")
for model_name in "${models[@]}"
do
   echo "Running model $model_name"
   poetry run python train_cli.py \
    --config configs/config.yaml \
    --model.pytorch_model leaf_disease.models.efficientnet.Efficientnet \
    --model.pytorch_model.model $model_name \
    --trainer.logger.name ${EXPERIMENT_NAME} \
    --trainer.logger.version $model_name \
    --data.batch_size 32
done



# poetry run python train_cli.py \
#     --config configs/cfg_eff.yaml \
#     --model.pytorch_model.model efficientnet_b0 \
#     --trainer.logger.name ${EXPERIMENT_NAME} \
#     --trainer.logger.version efficientnet_b0

# poetry run python train_cli.py \
#     --config configs/cfg_eff.yaml \
#     --model.pytorch_model.model efficientnet_b1 \
#     --trainer.logger.name ${EXPERIMENT_NAME} \
#     --trainer.logger.version efficientnet_b1

# poetry run python train_cli.py \
#     --config configs/cfg_eff.yaml \
#     --model.pytorch_model.model efficientnet_b4 \
#     --trainer.logger.name ${EXPERIMENT_NAME} \
#     --trainer.logger.version efficientnet_b4

# poetry run python train_cli.py \
#     --config configs/cfg_eff.yaml \
#     --model.pytorch_model.model efficientnet_b5 \
#     --trainer.logger.name ${EXPERIMENT_NAME} \
#     --trainer.logger.version efficientnet_b5