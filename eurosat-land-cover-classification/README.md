
#  EuroSAT Land Use and Land Cover Classification Using CNN

##  Project Overview

This project focuses on Land Use and Land Cover (LULC) Classification using Deep Learning and Convolutional Neural Networks (CNN).

A custom CNN architecture is implemented using PyTorch to classify Sentinel-2 satellite images from the EuroSAT dataset into 10 different land-cover categories.

##  Dataset

The project uses the EuroSAT dataset, which contains satellite images collected from the Sentinel-2 satellite.

### Land Cover Classes

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

Dataset: https://github.com/phelber/EuroSAT

##  Model Architecture

A custom Convolutional Neural Network (CNN) is developed using PyTorch.

The model learns spatial features from satellite images and performs multiclass classification across 10 land-cover categories.

##  Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- kaggle Notebook

## 📈 Model Performance

| Evaluation Metric | Accuracy |
|---|---:|
| Training Accuracy | 93.42% |
| Validation Accuracy | 91.22% |
| Test Accuracy | 92.19% |

##  Project Structure

```text
eurosat-land-cover-classification/
│
├── EuroSAT_CNN_Land_Cover_Classification.ipynb
├── EuroSAT_CNN_Checkpoint.pth
├── README.md
├── requirements.txt
│
└── results/
    ├── confusion_matrix.png
    ├── accuracy_plot.png
    └── loss_plot.png
```


##  Results and Evaluation

The model achieved 92.19% test accuracy on the EuroSAT dataset.

The confusion matrix is used to analyze correct and incorrect predictions across all 10 land-cover classes.

##  Future Improvements

- Improve classification accuracy through hyperparameter tuning.
- Experiment with ResNet50, VGG16 and MobileNetV2.
- Apply data augmentation techniques.
- Implement transfer learning.
- Perform detailed error analysis.
- Develop a web-based satellite image classification application.


##  Conclusion

This project implements a PyTorch-based CNN model for classifying 10 land-cover categories using the EuroSAT satellite image dataset. The model achieved **92.19% test accuracy**, demonstrating the effectiveness of deep learning for satellite image classification.
