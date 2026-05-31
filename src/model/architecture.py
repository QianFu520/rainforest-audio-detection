"""TinyCNN architecture for binary audio detection.

The model takes a single-channel mel spectrogram and outputs one
probability (meaningful vs. not meaningful).

Expected input shape: (batch, 1, 64, 128)
    - 1 channel (mono mel spectrogram)
    - 64 mel bins
    - 128 time frames

The classifier's first linear layer (64 * 8 * 16) assumes this input
size: three 2x2 max-pools reduce 64x128 down to 8x16. If the input
spectrogram shape changes, this linear layer must be updated.
"""

import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x