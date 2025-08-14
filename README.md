# ANZ Customer Transaction Prediction

## Overview
This project predicts **customer annual revenue** from ANZ’s transaction data using **Exploratory Data Analysis (EDA)**, **feature engineering**, and **machine learning models**.  
The dataset simulates 3 months of transaction activity for 100 hypothetical customers, including purchases, recurring transactions, and payroll transactions.  

We extended the base project to also include:
- **Time-series revenue forecasting** (Smart Prediction)
- **Feature importance analysis** (Analyse Outcomes)
- **Advanced model (XGBoost)** (Build Models)
- **Customer segmentation with KMeans** (Drive Innovation)

---

## Business Understanding

**Problem Statements:**
- How can we predict customer annual revenue from ANZ transaction data?
- Which machine learning model is most accurate for predicting transaction amounts?
- How can transaction data be used to identify customer segments and trends?

**Goals:**
1. Predict customer annual revenue using historical transaction patterns.
2. Identify features that most influence spending.
3. Compare multiple algorithms to select the best performer.
4. Provide insights for business innovation, such as customer segmentation.

**Approach:**
- Perform **EDA** to understand patterns, anomalies, and correlations.
- Preprocess data: outlier removal, one-hot encoding, and scaling.
- Train and compare **KNN**, **Random Forest**, **AdaBoost**, and **XGBoost** models.
- Extend analysis with **time-series forecasting** and **customer segmentation**.

---

## Data Understanding

**Dataset:**  
[Synthesized ANZ Transaction Dataset (Kaggle)](https://www.kaggle.com/datasets/ashraf1997/anz-synthesised-transaction-dataset)

**Key Variables:**
- **amount** — Transaction amount
- **movement** — Debit or credit
- **balance** — Account balance
- **age** — Customer age
- **gender** — Customer gender
- **merchant_id**, **merchant_state**, **txn_description** — Merchant details
- **date** — Transaction date

The dataset contains **categorical** (e.g., status, account, gender) and **numerical** (e.g., amount, age, balance) features.

---

## Data Preparation

1. **Outlier Removal:**  
   Used IQR method to remove extreme values in `balance`, `age`, and `amount`.

2. **Encoding:**  
   Applied **One-Hot Encoding** to categorical features.

3. **Train-Test Split:**  
   90% train, 10% test.

4. **Scaling:**  
   Applied **StandardScaler** to numerical features (`balance`, `age`).

---

## Modeling

**Algorithms Used:**
1. **K-Nearest Neighbors (KNN)** — Simple distance-based regressor.
2. **Random Forest (RF)** — Ensemble of decision trees for regression.
3. **AdaBoost** — Boosting algorithm to improve weak learners.
4. **XGBoost** *(added)* — Gradient boosting framework optimized for performance.

---

## Evaluation

Metric: **Mean Squared Error (MSE)** (scaled by 1e3 for readability).

| Model     | Train MSE | Test MSE |
|-----------|-----------|----------|
| KNN       | 0.2608    | 0.4769   |
| RandomForest | 0.2395 | 0.3524   |
| Boosting  | 0.3546    | 0.3695   |
| XGBoost *(new)* | — | — *(depends on tuning)* |

---

## Extended Features (Aligned with OneBanc Keywords)


**1. Analyse Outcomes:**  
- Generated **feature importance plots** from RandomForest to identify top revenue drivers.

**2. Build Models:**  
- Added **XGBoost** for better accuracy and efficiency.

**3. Drive Innovation:**  
- Applied **KMeans clustering** to segment customers into groups based on balance and age.
- Provided average transaction amount per cluster for targeted offers.

---

## How to Run

### Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet
```

### Run the Notebook
```bash
jupyter notebook predictive_analysis_enhanced.ipynb
```

---

## Conclusion
- **Random Forest** performed best in test MSE, showing strong generalization.
- **Feature importance** reveals key variables driving transaction amounts.
- **Time-series forecasting** provides insights into expected revenue trends.
- **Customer segmentation** supports targeted marketing and product personalization.

---

## References
- ANZ Synthesised Dataset: https://www.kaggle.com/datasets/ashraf1997/anz-synthesised-transaction-dataset
- M. Sathye, “Internet Banking in Australia,” SSRN Electron. J., 2005.
- J. Hirst, M. J. Taylor, “The internationalisation of Australian banking,” Aust. Geogr., 1985.
