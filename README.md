# Car Insurance Project

## Overview

The Car Insurance Project aims to provide users with a comprehensive set of features related to car damage assessment and insurance prediction. The project utilizes various machine learning models to detect fake images, measure the severity of damage, identify damaged parts within images, and predict the repair cost.

## Dataset

The project uses AI-generated images, Car_DD dataset for real damaged cars, Kaggle dataset for severity classification, and a custom repair cost dataset.

## Models

1. **Fake Image Detection**
    - Uses a VGG16-based model to distinguish AI-generated and real damaged car images.

2. **Damage Severity Classification**
    - Employs a fine-tuned MobileNetV2 model to classify damage severity as minor, moderate, or severe.

3. **Damaged Parts Detection**
    - Utilizes YOLOv8 for detecting damaged parts in car images with bounding boxes.

4. **Repair Cost Prediction**
    - Implements linear regression for predicting repair costs.

## Usage

1. Install dependencies:

    ```bash
    pip install requirements.txt
    ```
2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Acknowledgments

- Stable diffusion for AI-generated images.
- Car_DD dataset for real damaged car images.
- Kaggle for severity classification dataset.

## License

This project is licensed under the [MIT License](LICENSE).
