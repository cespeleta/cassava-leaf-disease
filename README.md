# cassava-leaf-disease

Kaggle competition to distinguish between several diseases that cause material harm to the food supply of many African countries

To run different set of models to evaluate which arquitechtures work better than the others:

```bash
./scripts/run_all_models.sh
```

Best model arquitechrures are `resnext50_32x4d`, `efficientnet_b1` and `efficientnet_b4`. To run this arquitechtures for 10 epochs and evalute them further:

```bash
./scripts/run_top3.sh
```

Experiment results

|                 | CV     |
| --------------- | ------ |
| resnext50_32x4d | 0.9097 |
| efficientnet_b1 | 0.9044 |
| efficientnet_b4 | 0.9052 |
