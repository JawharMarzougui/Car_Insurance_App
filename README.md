# Car Insurance Project

## Overview

The Car Insurance Project provides features for car damage assessment and insurance prediction. It includes:

1. **Fake Image Detection**
    - Uses a VGG16-based model to distinguish AI-generated and real damaged car images.

2. **Damage Severity Classification**
    - Employs a fine-tuned MobileNetV2 model to classify damage severity as minor, moderate, or severe.

3. **Damaged Parts Detection**
    - Utilizes YOLOv8 for detecting damaged parts in car images with bounding boxes.

4. **Repair Cost Prediction**
    - Implements linear regression for predicting repair costs.

## Dataset

The project uses AI-generated images, Car_DD dataset for real damaged cars, Kaggle dataset for severity classification, and a custom repair cost dataset.

## Models

1. **Fake Image Detection Model (VGG16)**
    - ...

2. **Damage Severity Classification Model (MobileNetV2)**
    - ...

3. **Damaged Parts Detection Model (YOLOv8)**
    - Trained on a dataset of damaged car images.

4. **Repair Cost Prediction Model (Linear Regression)**
    - ...

## Usage

1. Install dependencies.
2. Train and save models.
3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Acknowledgments

- Stable diffusion for AI-generated images.
- Car_DD dataset for real damaged car images.
- Kaggle for severity classification dataset.

## License

This project is licensed under the [MIT License](LICENSE).
