# Marketing Response Optimization Project

This project implements a machine learning model to predict customer responses to marketing campaigns using the Bank Marketing Dataset from Kaggle.

## Dataset

The dataset is downloaded from [Kaggle Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up Kaggle API (optional, for automatic download):
   - Follow instructions at https://www.kaggle.com/docs/api

## Usage

### Training the Model

Run the Jupyter notebook `marketing_response_model.ipynb` to train and evaluate the model.

### Running the Web App

To launch the Streamlit app:
```
streamlit run app.py
```

The app allows you to input customer data and get predictions on whether they will subscribe to a term deposit.

## Model

- Uses Random Forest Classifier for prediction.
- Features include customer demographics, contact history, and campaign details.
- Achieves good performance on test data (metrics in notebook).

## Front-End Decision

A front-end is included using Streamlit for easy demonstration and practical use. It allows non-technical users to interact with the model. If you prefer a more robust solution, consider Flask or Django for production deployment.