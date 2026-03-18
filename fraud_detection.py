import os
os.system("pip install xgboost lightgbm -q")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, f1_score,
                            precision_score, recall_score, accuracy_score)

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')


# 1. LOAD DATA 

def load_data(filepath, sample_size=10000, random_state=42):
    df = pd.read_csv(filepath)
    df = df.sample(n=sample_size, random_state=random_state)
    print(f"Dataset loaded: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df


# 2. PREPROCESS 

def preprocess(df, test_size=0.3, random_state=42):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


# 3. TRAIN & EVALUATE 

def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall':    recall_score(y_test, y_pred),
            'F1 Score':  f1_score(y_test, y_pred),
            'ROC AUC':   roc_auc_score(y_test, y_prob),
        }

        print(f"\n{name}")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")

    return results


# 4. MAIN

if __name__ == "__main__":
    df = load_data("data/creditcard_sample.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    summary = pd.DataFrame(results).T
    print("\n\nFinal Model Comparison:")
    print(summary)