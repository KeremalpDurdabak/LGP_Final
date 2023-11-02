import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Dataset:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    @staticmethod
    def label_encode(df):
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = label_encoder.fit_transform(df[col])
        return df

    @staticmethod
    def set_real_estate():
        # Read CSV and drop 'No' column
        df = pd.read_csv('datasets/real_estate.csv')
        df.drop('No', axis=1, inplace=True)
        df = df.sample(frac=1).reset_index(drop=True)

        # Label encode if necessary
        df = Dataset.label_encode(df)

        # Convert to NumPy array
        data = df.to_numpy()

        # Separate features and labels
        X, y = data[:, :-1], data[:, -1]

        # Split the data into training and testing sets
        Dataset.X_train, Dataset.X_test, Dataset.y_train, Dataset.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def set_medical_cost():
        # Read CSV
        df = pd.read_csv('datasets/medical_cost.csv')
        df = df.sample(frac=1).reset_index(drop=True)

        # Label encode if necessary
        df = Dataset.label_encode(df)

        # Convert to NumPy array
        data = df.to_numpy()

        # Separate features and labels
        X, y = data[:, :-1], data[:, -1]

        # Split the data into training and testing sets
        Dataset.X_train, Dataset.X_test, Dataset.y_train, Dataset.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
