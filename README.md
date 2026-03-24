#  Movie Rating Prediction using Regression

## Project Overview
This project predicts IMDb movie ratings using machine learning regression models. It compares **Linear Regression** and **Random Forest Regressor** to determine which model best predicts ratings based on features like genre, director, and cast.

## Dataset Description
The **IMDb Movies India** dataset contains information about Indian movies with the following key features:

| Feature | Description |
|---------|-------------|
| Name | Movie title |
| Year | Release year |
| Duration | Runtime |
| Genre | Movie genre |
| Rating | **Target** — IMDb rating (1.0–10.0) |
| Votes | Number of votes |
| Director | Director name |
| Actor 1/2/3 | Lead actors |

## Technologies Used
- **Python 3**
- **pandas** — Data manipulation & cleaning
- **NumPy** — Numerical operations
- **seaborn & matplotlib** — Data visualization
- **scikit-learn** — Model training & evaluation (LinearRegression, RandomForestRegressor, LabelEncoder)

## Workflow

1. **Load Dataset** — Read `IMDb Movies India.csv` with latin1 encoding.
2. **Data Cleaning**:
   - Remove duplicate rows.
   - Drop non-predictive columns (`Name`, `Year`, `Duration`, `Votes`).
   - Drop rows with missing values.
   - Convert `Rating` to numeric.
3. **Feature Engineering** — Label-encode categorical columns (`Genre`, `Director`, `Actor 1/2/3`).
4. **Visualizations**:
   - Rating distribution histogram with KDE
   - Top 10 genres by average rating
   - Feature correlation heatmap
5. **Train-Test Split** — 80:20 ratio with `random_state=42`.
6. **Model Training**:
   - Linear Regression
   - Random Forest Regressor (`n_estimators=100`)
7. **Evaluation** — MSE, RMSE, and R² Score for both models.
8. **Evaluation Visualizations**:
   - Actual vs Predicted scatter plots
   - Feature importance (Random Forest)
   - Model comparison bar charts

## Results
- **Models Compared**: Linear Regression vs Random Forest Regressor
- **Metrics**: MSE, RMSE, R² Score
- The best model is automatically determined based on R² score.

## Visualizations Generated
| Plot | File |
|------|------|
| Rating Distribution | `plot_rating_distribution.png` |
| Top Genres by Rating | `plot_genre_ratings.png` |
| Correlation Heatmap | `plot_correlation_heatmap.png` |
| Actual vs Predicted | `plot_actual_vs_predicted.png` |
| Feature Importance | `plot_feature_importance.png` |
| Model Comparison | `plot_model_comparison.png` |

## How to Run

```bash
# Ensure IMDb Movies India.csv is in the same directory
python movie_rating_prediction.py
```

Required libraries are auto-installed by the script. Alternatively:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Author

**Mayur Patil**  

[![GitHub](https://img.shields.io/badge/GitHub-Mayur2004566-black?logo=github)](https://github.com/Mayur2004566)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayur%20Patil-blue?logo=linkedin)](https://linkedin.com/in/mayur-patil45)

---

> If you found this project helpful, please give it a star on GitHub!
