import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aggregators import BaseAggregator

class CLAMMIL(BaseAggregator):
    def __init__(self, num_classes, input_dim=1024, dropout=0.,  **kwargs):
        super(BaseAggregator, self).__init__()
        fc = [nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, h, coords=None):
        h = self.fc(h)
        h = h.mean(dim=1)
        logits = self.classifier(h)  # K x 2
        return logits