from mltoolbox.model_training import train_model
from mltoolbox.eda import perform_eda
from mltoolbox.data_loading import load_data


# Load your data into a Pandas DataFrame
file_path = r''
target_column = ''

my_data = load_data(file_path)

# Perform EDA on the data
perform_eda(my_data, target_column=target_column)

# train the model
trained_model = train_model(my_data, target_column=target_column, model_type='auto')
