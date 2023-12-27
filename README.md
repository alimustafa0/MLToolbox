# MLToolbox

MLToolbox is a Python package designed to streamline the process of training machine learning models by providing reusable functions for data preprocessing, model training, and exploratory data analysis (EDA).

## Features

- **Automated Model Selection**: MLToolbox automatically selects a suitable machine learning model based on the characteristics of your data.

- **Flexible Preprocessing**: The package includes flexible data preprocessing functions, allowing you to handle numerical and categorical features seamlessly.

- **Exploratory Data Analysis (EDA)**: Perform EDA on your dataset with a single function call to gain insights into data distributions, correlations, and more.

## Installation

You can install MLToolbox using pip:

```bash
pip install mltoolbox
```


# Usage

## `test/test.py` Example

The `test.py` script serves as a comprehensive example showcasing the usage of MLToolbox in a typical machine learning workflow. Here's an overview of the steps it covers:

1. **Data Loading:** Load your dataset.

2. **Exploratory Data Analysis (EDA):** Use the `perform_eda` function to perform exploratory data analysis on your dataset. This includes displaying summary statistics, data info, distribution plots, correlation matrix, pair plots, box plots, and count plots.

3. **Data Preprocessing:** Utilize the `preprocess_data` function to preprocess the data before training the model. This function handles tasks such as handling missing values, scaling numerical features, and one-hot encoding categorical features.

4. **Model Training:** Train a machine learning model using the `train_model` function. The script demonstrates how to use automated model selection or specify a model type explicitly.

5. **Model Evaluation:** Evaluate the trained model using appropriate metrics based on the task type (classification or regression).

## Example Usage:

```bash
python test/test.py
```
