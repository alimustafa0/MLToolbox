import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def perform_eda(data, target_column):
    """
    Perform Exploratory Data Analysis (EDA) on the input data.
    
    Parameters:
    - data: Pandas DataFrame, input data
    """
    # Display basic summary statistics
    print("Summary Statistics:")
    print(data.describe() if isinstance(data, pd.DataFrame) else "Not applicable for non-numeric data")
    
    # Display information about data types and missing values
    print("\nData Info:")
    print(data.info())

    categories = data[target_column].nunique()
    if categories <= 20 or y.dtype == 'O':
        data[target_column] =  data[target_column].astype('category')
    
    # Visualize the distribution of the target variable (assuming it's numeric)
    if data[target_column].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[target_column], bins=30, kde=True)
        plt.title(f'Distribution of {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Frequency')
        plt.show()
    else:
        print(f"Target variable '{target_column}' is not numeric, skipping distribution plot.")
    
    # Visualize correlation matrix for numeric columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Visualize pair plots for a subset of numeric columns
    subset_columns = numeric_columns[:5]  # Adjust the number of columns to display
    sns.pairplot(data[subset_columns])
    plt.suptitle('Pair Plot', y=1.02)
    plt.show()
    
    # Visualize box plots for numeric columns
    plt.figure(figsize=(15, 8))
    num_plots = len(numeric_columns)
    num_rows = (num_plots // 2) + (num_plots % 2)  # Calculate the number of rows needed
    for i, column in enumerate(numeric_columns):
        plt.subplot(num_rows, 2, i+1)
        sns.boxplot(x=data[column])
        plt.title(f'Box Plot - {column}')
    plt.tight_layout()
    plt.show()

    # Visualize count plots for categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        unique_values = data[column].nunique()
        if unique_values <= 12:  # Show count plot only if unique values are 12 or fewer
            plt.figure(figsize=(10, 6))
            sns.countplot(data[column])
            plt.title(f'Count Plot - {column}')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f'Skipping count plot for {column} as it has more than many unique values ({unique_values}).')


    # If the target variable is categorical, include classification metrics
    if data[target_column].dtype == 'object':
        target_values = data[target_column]
        unique_values = target_values.unique()
        if len(unique_values) <= 2:
            # Binary classification metrics
            binary_target = (target_values == unique_values[0]).astype(int)
            print(f'\nClassification Metrics for {unique_values[0]}:')
            print(f'Accuracy: {accuracy_score(binary_target, (binary_target == 0).astype(int))}')
            print(f'Precision: {precision_score(binary_target, (binary_target == 0).astype(int))}')
            print(f'Recall: {recall_score(binary_target, (binary_target == 0).astype(int))}')
            print(f'F1-score: {f1_score(binary_target, (binary_target == 0).astype(int))}')
        else:
            print(f'Target variable has more than two unique values. Skipping binary classification metrics.')