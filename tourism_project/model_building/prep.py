# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/vinodcwanted/Tourism-Package-Prediction/tourism.csv"
bank_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ---------------------------
# 🔹 DATA CLEANING SECTION
# ---------------------------

# 1. Remove duplicate rows
bank_dataset = bank_dataset.drop_duplicates()

# 2. Standardize Gender column — fix inconsistent values
# Convert to lowercase → strip spaces → map to Male/Female
bank_dataset['Gender'] = (
    bank_dataset['Gender']
    .astype(str)
    .str.lower()
    .str.replace(" ", "")        # remove all internal spaces → "fem ale" → "female"
    .replace({
        'male': 'Male',
        'm': 'Male',
        'female': 'Female',
        'f': 'Female'
    })
)

# 3. Drop CustomerID (not useful for modelling)
if 'CustomerID' in bank_dataset.columns:
    bank_dataset = bank_dataset.drop(columns=['CustomerID'])

# Reset index after cleaning
bank_dataset.reset_index(drop=True, inplace=True)

print("Data cleaned successfully.")

# ---------------------------
# TARGET & FEATURE SELECTION
# ---------------------------

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Define predictor matrix (X)
X = bank_dataset[numeric_features + categorical_features]

# Define target variable
y = bank_dataset[target]

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save split datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload data to HuggingFace Dataset repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="vinodcwanted/Tourism-Package-Prediction",
        repo_type="dataset",
    )
