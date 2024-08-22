import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

# Paths to datasets
DATA_PATH = os.path.join(PACKAGE_ROOT, "data")
SMALL_DATASET = 'dataset_input_small.csv'
TINY_DATASET = 'dataset_input_tiny.csv'
TRAIN_FILE = 'original_dataset.csv'
TEST_FILE_ONE = 'database_input.csv'  
TEST_FILE_TWO = 'database_input2.csv'  
TEST_FILE_THREE = 'database_input3.csv'  
TEST_FILE_FOUR = 'dataset_input4.csv'  
TEST_FILE_FIVE = 'dataset_input5.csv'  
ORIG_DATA_PATH = os.path.join(DATA_PATH, TRAIN_FILE)

# DATABASES
DB_SUFFIX = '.csv'


# Model configuration
MODEL_NAME = 'churn_model.pickle'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'models')

TARGET = 'Churn'
LEADING_COLUMN = 'customerID'

FEATURES = ['TotalCharges', 'Month-to-month', 'One year', 'Two year', 'PhoneService', 'tenure']
FEATURES_AFTER_CONTRACT = ['TotalCharges', 'Contract', 'PhoneService', 'tenure']
NUM_FEATURES = ['TotalCharges', 'tenure']
CAT_FEATURES = ['Month-to-month', 'One year', 'Two year', 'PhoneService']

# Assuming you have the correct column names from the dataset
COLUMN_NAMES = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                'MonthlyCharges', 'TotalCharges']

FEATURES_TO_ENCODE = ['Month-to-month', 'One year', 'Two year', 'PhoneService']
FEATURE_TO_MODIFY = 'TotalCharges'
FEATURE_TO_FILL = {'TotalCharges': 2279}

# Columns to drop (if any)
DROP_FEATURES = []

# Python and library versions
PYTHON_VERSION = "3.10.12"
PANDAS_VERSION = "2.0.3"
SKLEARN_VERSION = "1.2.2"
PICKLE_VERSION = "4.0"