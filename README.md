# Credit Card Fraud Detection

Detects fraudulent credit card transactions using machine learning.
Handles severe class imbalance (~0.17% fraud) using SMOTE.

## Project Structure
```
credit-card-fraud-detection/
├── data/                        # Sample dataset (10,000 rows)
├── src/                         # Python script
│   └── fraud_detection.py
├── Credit_Card_fraud_detection.ipynb
└── README.md
```

## Models Used
**Notebook:**
| Model               |
|---------------------|
| Logistic Regression |
| Random Forest       |
| XGBoost             |
| LightGBM            |

**Script (`src/fraud_detection.py`):**
| Model               |
|---------------------|
| Logistic Regression |
| Random Forest       |
| Gradient Boosting   |

## Highlights
- SMOTE oversampling to handle class imbalance
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Interactive Plotly visualizations (in notebook)
- Feature importance analysis

## Tech Stack
Python, Scikit-learn, XGBoost, LightGBM, imbalanced-learn, Pandas, NumPy, Plotly, Seaborn

## Dataset
Full dataset: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Sample (10,000 rows) available in `data/` folder.

## How to Run
**Notebook:** Open in Google Colab and run all cells.

**Script:**
```bash
pip install -r requirements.txt
python src/fraud_detection.py
```