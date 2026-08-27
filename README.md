# 🐶 VGG16 Doggy Door Classifier

An AI-powered "smart doggy door" built with **PyTorch** and a **pre-trained VGG16** (ImageNet) model. The project starts with general animal classification and progresses to a **personalized doggy door** using transfer learning — trained to recognize a specific dog (Bo) and keep every other animal (including other dogs) out.

---

## Project Overview

This project is split into two stages:

### Stage 1 — General Animal Classifier (`05_jupyterlabb.ipynb`)
Uses an out-of-the-box, ImageNet pre-trained VGG16 model to:
- Classify any input image and return the top-3 predicted ImageNet classes.
- Implement doggy-door access logic based on ImageNet class index ranges:
  - **Dogs** (indices `151–268`) → *"Doggy come on in!"*
  - **Cats** (indices `281–285`) → *"Kitty stay inside!"*
  - **Anything else** → *"You're not a dog! Stay outside!"*

### Stage 2 — Personalized Doggy Door (`05b_presidential_doggy_door.ipynb`)
Fine-tunes VGG16 via **transfer learning** to solve a binary classification problem: *is this Bo, or not?*
- VGG16 convolutional base is **frozen** and used as a fixed feature extractor.
- A new linear classification head (`nn.Linear(1000, 1)`) is added on top.
- Trained with `BCEWithLogitsLoss` + `Adam` optimizer on a small custom image dataset.
- Data augmentation (random rotation, crop, flip, color jitter) applied to improve generalization on a small dataset.

---

## Repository Structure

```
vgg16-doggy-door-classifier/
├── 05_jupyterlabb.ipynb              # Stage 1: General VGG16-based classifier
├── 05b_presidential_doggy_door.ipynb # Stage 2: Transfer learning for personalized detection
├── utils.py                          # Helper utilities
├── data/
│   ├── doggy_door_images/            # Sample images (dog, cat, bear) for Stage 1 testing
│   │   ├── happy_dog.jpg
│   │   ├── brown_bear.jpg
│   │   └── sleepy_cat.jpg
│   ├── imagenet_class_index.json     # ImageNet class index → label mapping
│   └── presidential_doggy_door/      # Custom dataset for Stage 2
│       ├── train/
│       │   ├── bo/
│       │   └── not_bo/
│       └── valid/
│           ├── bo/
│           └── not_bo/
└── .gitignore
```

---

## Tech Stack

- **Python 3**
- **PyTorch** & **Torchvision** (VGG16, pre-trained ImageNet weights)
- **Matplotlib / PIL** for image handling and visualization
- **Jupyter Notebook**

---

## How It Works

1. **Preprocessing** — Images are loaded, resized, and normalized using the exact transforms VGG16 was trained with (`VGG16_Weights.DEFAULT.transforms()`).
2. **Inference (Stage 1)** — The frozen pre-trained model outputs ImageNet class probabilities; the top prediction's index determines door behavior.
3. **Transfer Learning (Stage 2)** — VGG16's convolutional layers are frozen, and only a new linear layer is trained on the small custom "Bo vs. not Bo" dataset, making the model recognize one specific dog.

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision matplotlib pillow numpy
```

### Run
1. Clone the repo:
   ```bash
   git clone https://github.com/Snipy19/vgg16-doggy-door-classifier.git
   cd vgg16-doggy-door-classifier
   ```
2. Open `05_jupyterlabb.ipynb` in Jupyter/VS Code to run the general classifier.
3. Open `05b_presidential_doggy_door.ipynb` to train and test the personalized doggy door model.

> A CUDA-enabled GPU is recommended for training Stage 2, but the notebooks will fall back to CPU automatically if unavailable.

---

## Results

- **Stage 1**: Correctly distinguishes dogs, cats, and other animals using raw ImageNet predictions — no training required.
- **Stage 2**: After fine-tuning on a small labeled dataset (10 images per class), the model learns to specifically identify Bo with high validation accuracy across training epochs.

---

## 📄 License

This project is for educational purposes as part of an NVIDIA Deep Learning coursework module.
