
import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from model import PlantDiseaseCNN


# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. PATH CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "PlantVillage"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model.pth"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# 3. CHECK PATHS
# ==========================================

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


# ==========================================
# 4. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 5. LOAD SINGLE DATASET
# ==========================================

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

if len(dataset) == 0:
    raise ValueError("No images found in dataset!")

class_names = dataset.classes

num_classes = len(class_names)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("\nDataset loaded successfully!")

print("Dataset path:", DATASET_PATH)

print("Total images:", len(dataset))

print("Number of classes:", num_classes)

print("\nClasses:")

for index, name in enumerate(class_names):
    print(index, name)


# ==========================================
# 6. LOAD TRAINED MODEL
# ==========================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model_classes = state_dict[
    "classifier.4.weight"
].shape[0]

if model_classes != num_classes:
    raise ValueError(
        f"Model output classes: {model_classes}, "
        f"Dataset classes: {num_classes}"
    )

model = PlantDiseaseCNN(
    num_classes=num_classes
)

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("\nModel loaded successfully!")


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

total_loss = 0.0

print("\nEvaluating model...")


with torch.inference_mode():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================
# 8. CALCULATE METRICS
# ==========================================

evaluation_loss = total_loss / len(dataset)

accuracy = accuracy_score(
    all_labels,
    all_predictions
) * 100

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
) * 100

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
) * 100

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
) * 100


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("       MODEL EVALUATION RESULTS")
print("======================================")

print("Evaluation: Full dataset")

print("Total images:", len(dataset))

print(f"Loss:      {evaluation_loss:.4f}")

print(f"Accuracy:  {accuracy:.2f}%")

print(f"Precision: {precision:.2f}%")

print(f"Recall:    {recall:.2f}%")

print(f"F1 Score:  {f1:.2f}%")

print("======================================")


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

print("\nCLASSIFICATION REPORT\n")

print(report)

with open(
    os.path.join(RESULT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Plant Disease Detection - Confusion Matrix")

plt.xlabel("Predicted Disease")

plt.ylabel("Actual Disease")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. SAVE METRICS
# ==========================================

with open(
    os.path.join(RESULT_DIR, "evaluation_metrics.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write("PLANT DISEASE MODEL EVALUATION\n")

    file.write("==============================\n")

    file.write("Evaluation dataset: Full dataset\n")

    file.write(f"Total images: {len(dataset)}\n")

    file.write(f"Loss: {evaluation_loss:.4f}\n")

    file.write(f"Accuracy: {accuracy:.2f}%\n")

    file.write(f"Precision: {precision:.2f}%\n")

    file.write(f"Recall: {recall:.2f}%\n")

    file.write(f"F1 Score: {f1:.2f}%\n")


# ==========================================
# 13. COMPLETION
# ==========================================

print("\nEvaluation completed successfully!")

print("Results saved at:", RESULT_DIR)

print("\nGenerated files:")

print("1. confusion_matrix.png")

print("2. classification_report.txt")

print("3. evaluation_metrics.txt")

print(
    "\nWARNING: These are full-dataset results, "
    "not independent test results."
)
