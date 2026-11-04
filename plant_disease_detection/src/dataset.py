
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 1. Dataset Configuration

DATASET_PATH = r"D:\deep_learning\DL-Models\plant_disease_detection\data\PlantVillage"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# 2. Check Dataset Path

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset folder not found!")

print("Dataset path found successfully!")

# 3. Image Transformations

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Load Dataset

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

# 5. Get Class Names

class_names = dataset.classes
num_classes = len(class_names)

print("\nTotal Images:", len(dataset))
print("Total Classes:", num_classes)

print("\nClass Names:")

for i, name in enumerate(class_names):
    print(i, ":", name)

# 6. Split Dataset

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


# 7. Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 8. Dataset Information

print("\nDataset Loaded Successfully!")

print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

print("Training Batches:", len(train_loader))
print("Validation Batches:", len(val_loader))
