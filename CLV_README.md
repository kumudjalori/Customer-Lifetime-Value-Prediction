# Customer Lifetime Value (CLV) Prediction using Machine Learning

A beginner-to-intermediate, end-to-end data science project that predicts Customer Lifetime Value (CLV) using Linear Regression on features engineered from raw e-commerce transaction data.

## 📌 Project Overview

Customer Lifetime Value (CLV) estimates how much revenue a business can expect from a customer over time. This project builds an interpretable regression pipeline that predicts a customer's **future spend** based on their **past purchasing behavior**, using a real-world online retail transaction dataset.

The goal is not to use the most advanced modeling techniques, but to demonstrate a **clean, complete, and correctly-framed predictive workflow** — from business understanding through feature engineering, modeling, evaluation, and business recommendations.

## 🎯 Business Problem

- **What is CLV?** An estimate of total revenue a business can expect from a customer over their relationship with the company.
- **Why predict it?** It helps businesses prioritize marketing spend, identify high-value customers, and forecast revenue.
- **How does it help retention and marketing?** CLV predictions allow businesses to segment customers, personalize offers, and proactively retain valuable customers who show signs of disengagement (e.g., rising recency).

## 📊 Dataset

- **Source:** [Kaggle — E-Commerce Data (Online Retail Dataset)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Size:** ~541,909 transaction line items from a UK-based online retailer, spanning **01 Dec 2010 – 09 Dec 2011**
- **Columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

### Why this dataset?
- Widely used for CLV and RFM (Recency-Frequency-Monetary) analysis, with strong community support and tutorials.
- Real, raw transaction-level data — ideal for practicing feature engineering from scratch.
- Sufficiently large (500K+ rows, thousands of customers) without requiring heavy infrastructure.
- Simple, flat structure suited for a beginner-level project — no complex joins required.
- Spans roughly a year, allowing a natural **calibration period / holdout period** split — enabling genuine CLV *prediction* (not just historical description) without needing time-series forecasting models.

> **Note:** The dataset is not included in this repository. The notebook downloads it automatically via `kagglehub`, or it can be downloaded manually — see "How to Run" below.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, kagglehub
- **Environment:** Jupyter Notebook
- **Model:** Linear Regression

## 🧭 Project Workflow

1. **Business Understanding** — Framing CLV and why it matters for retention and marketing.
2. **Dataset Selection & Justification** — Explaining why this dataset fits a beginner-level CLV regression task.
3. **Data Loading** — Downloading and reading the dataset with Pandas.
4. **Data Cleaning** — Handling missing `CustomerID`s, duplicates, cancelled orders, and invalid transactions; converting date types.
5. **Exploratory Data Analysis (EDA)**
   - Distribution of customer spending
   - Purchase frequency
   - Revenue distribution by country
   - Correlation heatmap
   - Histograms of key features
   - Boxplots for outlier detection
   - Scatter plots between purchases and spend
6. **Feature Engineering** — Building a calibration/holdout time split, then engineering:
   - Total Spend
   - Number of Purchases
   - Average Purchase Value
   - Customer Tenure
   - Recency
   - Purchase Frequency
7. **Preprocessing** — Encoding `Country` as a simple binary feature, scaling numerical features with `StandardScaler`, and train/test splitting.
8. **Modeling** — Training a Linear Regression model with scikit-learn to predict future customer spend (CLV).
9. **Evaluation** — Assessing performance with MAE, MSE, RMSE, and R² Score.
10. **Feature Importance** — Interpreting standardized coefficients to identify the most influential drivers of CLV.
11. **Business Recommendations** — Practical guidance on customer segmentation, retention, and personalized marketing.

---

## 📊 Model Performance

The Linear Regression model was evaluated using standard regression metrics.

| Metric | Value |
|:-------------------------------|-------:|
| Mean Absolute Error (MAE) | **770.23** |
| Mean Squared Error (MSE) | **12,116,033.92** |
| Root Mean Squared Error (RMSE) | **3480.81** |
| R² Score | **0.6693** |

### Key Observations

- The model explains approximately **66.9%** of the variation in customers' future spending.
- Most customers' future spending is predicted reasonably well, particularly for low- and medium-value customers.
- Prediction errors increase for customers with exceptionally high future spending, which is expected when using a simple Linear Regression model.
- The engineered behavioral features (Recency, Frequency, Monetary Value, Purchase Frequency, Customer Tenure, and Average Purchase Value) provide meaningful predictive power for estimating Customer Lifetime Value.

---

## 📈 How CLV Is Framed as a Predictive (Not Just Descriptive) Problem

Rather than simply summarizing historical spend, this project splits the transaction timeline into:
- A **calibration period** (first ~9 months) — used to compute behavioral features.
- A **holdout period** (last ~3 months) — the actual amount spent here becomes the prediction **target**.

This creates a genuine forward-looking prediction task using only a simple regression model — no time-series forecasting is required.

## 🚀 How to Run This Project

1. Clone or download this repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Kaggle API credentials (one-time step):
   - Create a free Kaggle account at https://www.kaggle.com
   - Go to **Kaggle → Settings → API → Create New Token** to download `kaggle.json`
   - Place it at `~/.kaggle/kaggle.json` (Mac/Linux) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows)
   - Alternatively, set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Customer_Lifetime_Value_Prediction.ipynb
   ```
5. Run the cells from top to bottom — the dataset downloads automatically via `kagglehub`.

   *(Prefer not to set up API access? Manually download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data), place it in the project folder, and load it with `pd.read_csv("data.csv", encoding="ISO-8859-1")` as shown in the notebook's fallback note.)*

## 📁 Project Structure

```
clv-prediction/
│
├── Customer_Lifetime_Value_Prediction.ipynb   # Main notebook (EDA, feature engineering, modeling, evaluation)
├── README.md                                   # Project documentation (this file)
└── requirements.txt                            # Python dependencies
```

The dataset itself is not stored in this repo — it's downloaded automatically at runtime via `kagglehub` (or manually, as a fallback — see "How to Run" above).

## 🔑 Key Learnings

- Built an end-to-end regression pipeline for Customer Lifetime Value prediction using real-world transaction data.
- Engineered RFM-based behavioral features from raw transaction logs using a time-based calibration and holdout approach.
- Applied Exploratory Data Analysis (EDA) to uncover customer purchasing patterns and spending behavior.
- Evaluated model performance using MAE, MSE, RMSE, and R² Score, and interpreted results in a business context.
- Demonstrated how predictive analytics can support customer segmentation, personalized marketing, and retention strategies.

## 🔮 Possible Future Improvements

- Experiment with different calibration/holdout period lengths to test model robustness.
- Add customer segmentation visualizations (e.g., CLV tiers plotted against Recency and Frequency).
- Explore log-transforming the skewed target variable to see if it improves model fit.
- Build a simple dashboard to let marketing teams explore predicted CLV by segment.

## 👤 Author

**Kumud Jalori**

B.Tech, Engineering Physics  
Indian Institute of Technology Hyderabad

## 📄 License

This project is for educational purposes. Dataset license belongs to the original dataset contributors (UCI Machine Learning Repository / Kaggle user carrie1).
