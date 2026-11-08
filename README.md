# Car Brand Classifier (MobileNetV2)

Image classifier for 10 car brands, built with transfer learning on **MobileNetV2**.
Developed for the *Problem Solving with Deep Learning* course of the MBA in BI & Data Science at Ibmec.

## Overview

The notebook `img_classifier.ipynb` covers the whole pipeline:

1. **Data collection** – downloads up to 1,000 images per brand with `icrawler` (Google Images) and splits them into `train` / `val` / `test` (70% / 15% / 15%).
2. **Model** – MobileNetV2 pre-trained on ImageNet (frozen backbone) + `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout(0.5)` → `Dense(10, softmax)`.
3. **Training** – 10 epochs with the backbone frozen, using data augmentation (rotation, shifts, shear, zoom, horizontal flip).
4. **Fine-tuning** – the backbone is unfrozen (first 100 layers kept frozen) and trained for 5 more epochs with `Adam(1e-5)`.
5. **Inference** – `predict_image(path)` prints the predicted brand and its probability for a single image.

**Brands:** Audi, BMW, Ferrari, Ford, Lamborghini, Mercedes, Porsche, Tesla, Toyota, Volkswagen.

## Tech stack

- Python 3, TensorFlow / Keras
- icrawler (image download)
- NumPy, Matplotlib

## Project structure

```
.
├── img_classifier.ipynb   # Download, training, fine-tuning and inference
├── requirements.txt
├── LICENSE
└── README.md
```

The `dataset/` folder is created by the notebook and is not versioned:

```
dataset/
├── train/<brand>/
├── val/<brand>/
└── test/<brand>/
```

## Getting started

```bash
git clone https://github.com/ggcds/car-brand-classifier.git
cd car-brand-classifier

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook img_classifier.ipynb
```

Run the cells in order. The first cell downloads and organizes the images (this can take a while and depends on Google Images availability).

Example of single-image inference (last cell):

```python
predict_image("dataset/test/toyota/000020.png")
```

## Notes

- Images are scraped from the web, so the dataset is noisy (wrong brands, interiors, logos). This limits the achievable accuracy.
- The dataset is not redistributed in this repository.

## Possible improvements

- More and cleaner images per class
- Advanced augmentation (MixUp, CutMix, Albumentations)
- Other backbones (EfficientNet, ResNet)
- Learning-rate scheduling (`ReduceLROnPlateau`, cosine decay) and callbacks (`EarlyStopping`, `ModelCheckpoint`)
- Confusion matrix and per-class metrics on the test set
- Grad-CAM to inspect what the model focuses on

## Author

**Guilherme Barros Caldas** – MBA in BI & Data Science, Ibmec

## License

[MIT](LICENSE)
