# BlaBlaCar Ride-Sharing Analysis

Exploratory data analysis, business KPI design, and machine learning applied to a ~97,600-row BlaBlaCar-style ride-sharing dataset, using Python (pandas, scikit-learn, statsmodels).

## Project overview

Starting from raw trip-level data (driver attributes, trip attributes, comfort/behavior flags), this project:

- Cleans and imputes missing values (mode for categorical columns, group-wise mean for numeric columns)
- Answers business questions through data manipulation (top drivers by distance/detour time, driver lookup by ID range)
- Builds a composite **BlaBlaCar Index** to rank drivers on business value vs. passenger comfort
- Visualizes key patterns (satisfaction vs. price, evaluations by driver status, driver age distribution)
- Predicts whether a ride will be **recommended** using a Logistic Regression (scikit-learn + statsmodels for interpretability)
- Segments drivers with **K-Means clustering** on trip distance and price, with feature scaling and both the elbow and silhouette methods used to select k

## Repository structure

```
.
├── BlaBlaCar - Analysis.ipynb   # Main analysis notebook
├── BlaBla-Car_Data.xlsx         # Source dataset
├── requirements.txt             # Python dependencies
└── README.md
```

## Key results

- **Data quality**: after cleaning, the dataset has zero missing values across all 31 columns.
- **Logistic Regression**: predicts ride recommendation from `music`, `talk`, `pet`, `smoking`, `detour_time` — `talk` is by far the strongest, statistically significant predictor (coefficient ≈ 1.38, p < 0.001), while the others are not statistically significant.
- **K-Means clustering**: after min-max scaling (distance and price are on very different scales), the silhouette method points to **k = 2** as the best-separated driver segmentation on trip distance vs. price.

## Tech stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `statsmodels`

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook "BlaBlaCar - Analysis.ipynb"
```

## Possible next steps

- Cross-validate the classification model and benchmark against Random Forest / Gradient Boosting
- Test interaction terms and additional features in the logistic regression
- Validate cluster stability using more features (ratings, evaluations, comfort index)

## Author

Thibault Verger — [GitHub](https://github.com/thibaultverger)
