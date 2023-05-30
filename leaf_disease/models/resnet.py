import torch
import torch.nn as nn


class Resnet(nn.Module):
    repo_or_dir: str = "pytorch/vision"

    def __init__(
        self,
        num_classes: int = 5,
        model: str = "resnet18",
        weights: str = "DEFAULT",  # "ResNet18_Weights.IMAGENET1K_V1",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = model
        self.weights = weights

        backbone = self._get_backbone()

        # Replace fc layer with correct number of outputs
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify leaf iamges (5 image classes)
        self.classifier = nn.Linear(num_filters, self.num_classes)

    def _get_backbone(self):
        return torch.hub.load(self.repo_or_dir, model=self.model, weights=self.weights)

    def forward(self, image):
        """Train all network with input images."""
        representations = self.feature_extractor(image).flatten(1)
        return self.classifier(representations)


class ResnetSSL(Resnet):
    """Semi-Supervised and Semi-Weakly Supervised ImageNet Models.

    This project includes the semi-supervised and semi-weakly supervised ImageNet models
    introduced in "Billion-scale Semi-Supervised Learning for Image Classification"
    https://arxiv.org/abs/1905.00546.

    "Semi-supervised" (SSL) ImageNet models are pre-trained on a subset of
    unlabeled YFCC100M public image dataset and fine-tuned with the ImageNet1K
    training dataset, as described by the semi-supervised training framework in the
    paper mentioned above. In this case, the high capacity teacher model was trained
    only with labeled examples.

    https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    """

    repo_or_dir: str = "facebookresearch/semi-supervised-ImageNet1K-models"

    def _get_backbone(self):
        return torch.hub.load(self.repo_or_dir, model=self.model)
