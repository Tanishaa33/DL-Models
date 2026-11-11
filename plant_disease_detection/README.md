
# 🌿 Plant Disease Detection Using CNN

An AI-powered Plant Disease Detection System built using **PyTorch, Convolutional Neural Networks (CNN), and Streamlit**.

This project identifies plant leaf diseases from uploaded images using a trained deep learning model. Users can upload an image or provide an image URL to predict whether a leaf is healthy or affected by a disease.

---

## 🚀 Live Demo

Try the application here:

### 🌱 [Plant Disease Detection - Live Demo](YOUR_STREAMLIT_APP_URL)

**Live Application:** Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit application URL.

### Application Preview

![Plant Disease Detection UI](assets/plant_disease_ui.png)

---

## 📌 Project Overview

Plant diseases can significantly affect agricultural productivity and crop quality.

This project uses a Convolutional Neural Network (CNN) to automatically classify plant leaf images into healthy or diseased categories.

The trained model analyzes the uploaded image and predicts the corresponding disease class along with its prediction confidence.

### Key Objectives

- Detect plant leaf diseases automatically.
- Classify healthy and diseased leaves.
- Provide prediction confidence scores.
- Create an interactive web application using Streamlit.
- Make deep learning-based plant disease detection accessible through a simple user interface.

---

## ✨ Features

- **Image Upload:** Upload plant leaf images in JPG, JPEG, and PNG formats.
- **Image URL Prediction:** Provide a publicly accessible image URL.
- **Disease Classification:** Identify the predicted plant disease.
- **Healthy Leaf Detection:** Identify healthy plant leaves.
- **Confidence Score:** Display the model's prediction confidence.
- **Top 3 Predictions:** Display the three most likely disease classes.
- **Interactive UI:** Simple and user-friendly Streamlit interface.
- **CPU/GPU Support:** Automatically detects available CUDA GPU or CPU.

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| CNN | Image classification model |
| Torchvision | Image preprocessing |
| Scikit-learn | Model evaluation metrics |
| NumPy | Numerical operations |
| Pillow (PIL) | Image processing |
| Streamlit | Web application interface |
| Requests | Image URL downloading |

---

## 📂 Project Structure

```text
plant_disease_detection/
│
├── data/
│   └── PlantVillage/
│
├── models/
│   └── model.pth
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
│
├── results/
│   └── evaluation_results.txt
│
├── assets/
│   └── plant_disease_ui.png
│
├── requirements.txt
│
└── README.md
```

---

## 🌱 Dataset Information

The project uses the PlantVillage dataset for training and evaluating the CNN model.

**Dataset Statistics:**

- Total Images: 41,276
- Number of Classes: 15
- Image Size: 224 × 224 pixels
- Dataset Split: 80% Training and 20% Validation

### Supported Plant Disease Classes

| No. | Disease Class |
|---|---|
| 1 | Pepper Bell Bacterial Spot |
| 2 | Pepper Bell Healthy |
| 3 | Potato Early Blight |
| 4 | Potato Late Blight |
| 5 | Potato Healthy |
| 6 | Tomato Bacterial Spot |
| 7 | Tomato Early Blight |
| 8 | Tomato Late Blight |
| 9 | Tomato Leaf Mold |
| 10 | Tomato Septoria Leaf Spot |
| 11 | Tomato Spider Mites |
| 12 | Tomato Target Spot |
| 13 | Tomato Yellow Leaf Curl Virus |
| 14 | Tomato Mosaic Virus |
| 15 | Tomato Healthy |

---

## 📊 Model Evaluation Results

The trained CNN model achieved the following evaluation results:

| Evaluation Metric | Result |
|---|---:|
| Loss | 0.0208 |
| Accuracy | 99.32% |
| Precision | 99.32% |
| Recall | 99.32% |
| F1 Score | 99.31% |

**Evaluation Dataset:** 20,638 images.

**Note:** These are the previously obtained evaluation results. The evaluation split and image independence should be verified before treating them as final validation or test performance.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd plant_disease_detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Trained Model

Place your trained model checkpoint inside:

```text
models/model.pth
```

The model checkpoint is required to run predictions.

### 5. Run the Streamlit Application

Navigate to the source directory:

```bash
cd src
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 🖥️ How to Use the Application

1. Open the Plant Disease Detection web application.
2. Upload a plant leaf image or provide an image URL.
3. Click the **Predict Disease** button.
4. Wait for the CNN model to analyze the image.
5. View the predicted disease or healthy status.
6. Check the prediction confidence percentage.
7. View the top three predicted classes.

---

## 🔍 Model Architecture

The project uses a custom Convolutional Neural Network (CNN) implemented using PyTorch.

The model learns visual features from plant leaf images and uses them to classify the input into one of the 15 supported classes.

Image preprocessing includes:

- Resizing images to 224 × 224 pixels.
- Converting images into PyTorch tensors.
- Normalizing images using ImageNet mean and standard deviation values.

---

## 📈 Future Improvements

- Add Grad-CAM visualization to highlight disease-affected regions.
- Support additional plant species and disease classes.
- Improve generalization using independently collected field images.
- Add disease information and recommended management practices.
- Create prediction history and downloadable reports.
- Improve the user interface with interactive dashboards.

---

## ⚠️ Limitations

- The model supports only the 15 classes used during training.
- Predictions may be unreliable for unrelated images or unsupported plant species.
- High confidence does not guarantee correct classification.
- Real-world field performance requires testing on independently collected images.
- Predictions should not replace expert agricultural diagnosis.

---

## 👨‍💻 Author

**Your Name**

GitHub: [Your GitHub Profile](YOUR_GITHUB_PROFILE_URL)

LinkedIn: [Your LinkedIn Profile](YOUR_LINKEDIN_PROFILE_URL)

---

## ⭐ Acknowledgements

- PlantVillage Dataset
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn

---

If you find this project useful, consider giving the repository a ⭐ on GitHub!
