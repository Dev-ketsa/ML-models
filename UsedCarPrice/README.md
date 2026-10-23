# Used Cars Price Prediction

A beginner machine learning project exploring used-car resale prices and building a price prediction model with **Linear Regression** and **scikit-learn**.

## Project Overview

This project covers exploratory data analysis, categorical-feature encoding, train-test splitting, model training, and evaluation. The aim is to understand how vehicle attributes relate to resale prices and establish a baseline regression model.

## Dataset

The notebook was developed using `CAR DETAILS FROM CAR DEKHO.csv`.

The original dataset contains **4,340 rows and 8 columns**.

| Feature | Description |
|---|---|
| `name` | Car model/name |
| `year` | Manufacturing year |
| `selling_price` | Selling price (target variable) |
| `km_driven` | Distance driven in kilometres |
| `fuel` | Fuel type |
| `seller_type` | Seller category |
| `transmission` | Transmission type |
| `owner` | Previous-owner category |

The model's selected input features should be checked against the final notebook before treating this table as the definitive feature list.

## Tools and Libraries

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Jupyter Notebook

## Workflow

1. **Exploratory Data Analysis (EDA):** inspect the dataset, summary statistics, distributions, and relationships between features and selling price.
2. **Preprocessing:** split input features and target, create training and test sets, and apply one-hot encoding to categorical columns with `OneHotEncoder` and `ColumnTransformer`.
3. **Training:** fit a Linear Regression model on the transformed training data.
4. **Evaluation:** use the test-set R² score and actual-versus-predicted visualizations to inspect model performance.
5. **Data checks:** inspect duplicates and potential outliers before deciding how to handle them.

## Results

The latest score reported during development was approximately **R² = 0.30** after removing duplicate rows. Re-run the notebook from top to bottom and update this value to the exact final score before publishing.

R² measures the proportion of variation in the target explained by the model on the evaluated data. A score around 0.30 indicates that this baseline model has limited predictive performance; it is a starting point for further analysis, not a production-ready pricing system.

## Run Locally

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Put `CAR DETAILS FROM CAR DEKHO.csv` in the project root, or update the file path in the notebook.
4. Open `CAR.ipynb` in Jupyter Notebook or VS Code.
5. Run all cells from top to bottom.

## Repository Structure

```text
used-cars-price-prediction/
├── CAR.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## Limitations

- This project currently focuses on a Linear Regression baseline.
- Prediction quality is limited based on the reported R² score.
- Results depend on the dataset, preprocessing decisions, and train-test split.
- Duplicate records should only be removed when they are genuinely redundant; repeated-looking listings may represent distinct observations.
- The notebook should be rerun end-to-end after the final data-cleaning decisions so that reported metrics match the committed version.

## Possible Next Steps

- Complete duplicate and outlier investigation.
- Add MAE and RMSE to the evaluation.
- Inspect residuals and prediction errors.
- Document the final feature set and exact final test metrics.
- Consider a simple interactive demo after the core notebook is finalized.

---

**Status:** Initial Linear Regression model implemented; evaluation and data-cleaning decisions should be finalized before presenting the results as definitive.
