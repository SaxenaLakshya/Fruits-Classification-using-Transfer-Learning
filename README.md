# Fruits Classification using Transfer Learning

This repository contains a project focused on classifying fruits using Transfer Learning, built with TensorFlow, Streamlit, and other machine learning libraries. The model achieves 65% accuracy and is designed for real-time predictions through an interactive web application.

## Features
- Classification of fruit images using a pre-trained model and Transfer Learning.
- Interactive web interface built with Streamlit for real-time predictions.
- Demonstrates the use of TensorFlow and key ML/DL libraries.

## Dataset
The dataset used for this project consists of fruit images, which were preprocessed and augmented to improve training performance. The dataset was sourced from [Kaggle](https://www.kaggle.com/) and modified using Excel for readiness.

## Model
- **Model Type**: Transfer Learning
- **Framework**: TensorFlow
- **Accuracy**: 65%

## Libraries Used
- TensorFlow
- Streamlit
- NumPy
- Matplotlib

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SaxenaLakshya/Fruits-Classification-using-Transfer-Learning.git
   cd Fruits-Classification-using-Transfer-Learning
   ```

2. Install the required libraries:
   ```bash
   pip install tensorflow, streamlit, numpy, matplotlib
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload an image of a fruit using the Streamlit interface.
2. The model will process the image and display the predicted fruit class.

## Project Structure
- `Fruit_Classifier_1o.h5`: Saved model
- `app.py`: Main file to run the Streamlit app.
- `README.md`: Project documentation.

## Future Improvements
- Enhance model accuracy with additional data and fine-tuning.
- Improve the user interface for better usability.
- Add support for multiple language outputs.

## Acknowledgments
- Dataset from [Kaggle](https://www.kaggle.com/).
- Inspiration from Transfer Learning tutorials, TensorFlow documentation and Deep Learning with Keras and Tensorflow by IBM.

---
**Note**: This project is for learning purposes and can be extended further for real-world applications.
