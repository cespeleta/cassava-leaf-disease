import torch
import torch.nn as nn


class Efficientnet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        model: str = "efficientnet_b0",
        weights: str = "IMAGENET1K_V1",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = model
        self.weights = weights

        backbone = torch.hub.load(
            "pytorch/vision", model=self.model, weights=self.weights
        )
        # Update last layer of the model
        # classifier = (0) Dropout + (1) Linear
        in_features = backbone.classifier[-1].in_features

        self.backbone = backbone
        self.backbone.classifier[-1] = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

    def forward(self, image):
        """Train all network with input images."""
        return self.backbone(image)
