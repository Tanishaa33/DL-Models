
import torch
import torch.nn as nn


# 1. CNN Model Architecture

class PlantDiseaseCNN(nn.Module):

    def __init__(self, num_classes=16):

        super(PlantDiseaseCNN, self).__init__()

        # 2. Convolutional Feature Extraction

        self.features = nn.Sequential(

            # First Convolution Block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth Convolution Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            # Number of channels increases: 3 → 32 → 64 → 128 → 256.
            # Image dimensions decrease: 224 → 112 → 56 → 28 → 14.


        # 3. Adaptive Average Pooling

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Fully Connected Classification

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    # 5. Forward Propagation

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x


# 6. Model Testing

if __name__ == "__main__":

    model = PlantDiseaseCNN(num_classes=16)

    dummy_input = torch.randn(32, 3, 224, 224)

    output = model(dummy_input)

    print("Model Architecture:")
    print(model)

    print("\nInput Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)
