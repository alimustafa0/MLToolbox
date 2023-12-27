from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(data, target_column):
    """
    Preprocess the input data before training the model.
    
    Parameters:
    - data: Pandas DataFrame, input data
    - target_column: str, the target column for regression or classification
    
    Returns:
    - X_preprocessed: Processed features
    - y: Target variable
    - preprocessor: Scikit-learn transformer for later use
    """
    # Separate features (X) and target variable (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    categories = y.nunique()
    if categories <= 20 or y.dtype == 'O':
        y = data[target_column].astype('category')
        task_type = 'classification'
    else:
        task_type = 'regression'
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(exclude=['object']).columns
    
    # Define transformers for preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values with mean imputation
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values with most frequent imputation
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Apply the transformations to the data
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y, task_type



def train_model(data, target_column, model_type='auto'):
    """
    Train a machine learning model on the input data.
    
    Parameters:
    - data: Pandas DataFrame, input data
    - target_column: str, the target column for regression or classification
    - model_type: str, 'auto' for automatic selection, or specific model name
    
    Returns:
    - trained_model: the trained machine learning model
    """
    # X = data.drop(columns=[target_column])
    # y = data[target_column]

    X, y, task_type = preprocess_data(data, target_column)

    # Determine task type based on the data type of the target variable
    #task_type = 'classification' if y.dtype == 'O' else 'regression'

    if model_type == 'auto':
        model = automatic_model_selection(X, y, task=task_type)
    else:
        model = get_model_instance(model_type)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    if task_type == 'regression':
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
    else:
        # For classification tasks, use precision and recall as metrics
        y_pred = model.predict(X_test)
        classification_report_result = classification_report(y_test, y_pred)
        print(f'Classification Report:\n{classification_report_result}')

    return model


def automatic_model_selection(X, y, task='classification'):
    """
    Automatically select a suitable model based on the data characteristics.
    
    Parameters:
    - X: Features
    - y: Target variable
    - task: str, 'classification' or 'regression'
    
    Returns:
    - best_model: The selected machine learning model
    """
    if task == 'classification':
        models = [
            ('Logistic Regression', LogisticRegression()),
            ('Support Vector Machine', SVC()),
            ('Random Forest', RandomForestClassifier()),
            ('Decision Tree', DecisionTreeClassifier()),
            ('k-Nearest Neighbors', KNeighborsClassifier())
        ]
    elif task == 'regression':
        models = [
            ('Linear Regression', LinearRegression()),
            ('Support Vector Machine', SVR()),
            ('Random Forest', RandomForestRegressor()),
            ('Decision Tree', DecisionTreeRegressor()),
            ('k-Nearest Neighbors', KNeighborsRegressor())
        ]
    else:
        raise ValueError(f'Invalid task type: {task}')
    
    best_model = None
    best_score = float('inf')  # Initialize with a large value
    
    for model_name, model in models:
        if task == 'classification':
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            mean_score = scores.mean()
        elif task == 'regression':
            scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            mean_score = scores.mean()
        else:
            raise ValueError(f'Invalid task type: {task}')
        
        print(f'{model_name} - Mean Score: {mean_score}')
        print('------------------------------------------------------------')
        
        if mean_score < best_score:
            best_score = mean_score
            best_model = model
    
    print('\n\n\n************************************************************')
    print(f'Best Model: {type(best_model).__name__} with Mean Score: {best_score}')
    
    return best_model


def get_model_instance(model_type):
    """
    Return an instance of the specified machine learning model.
    """
    if model_type == 'linear_regression':
        return LinearRegression()
    elif model_type == 'svm':
        return SVR()
    elif model_type == 'random_forest':
        return RandomForestRegressor()
    elif model_type == 'decision_tree':
        return DecisionTreeRegressor()
    elif model_type == 'knn':
        return KNeighborsRegressor()
    elif model_type == 'logistic_regression':
        return LogisticRegression()
    elif model_type == 'svc':
        return SVC()
    elif model_type == 'random_forest_classifier':
        return RandomForestClassifier()
    elif model_type == 'decision_tree_classifier':
        return DecisionTreeClassifier()
    elif model_type == 'knn_classifier':
        return KNeighborsClassifier()
    else:
        raise ValueError(f'Invalid model type: {model_type}')