import subprocess
import sys

# ── Auto-install required libraries ──────────────────────────────────────────
required = ['pandas', 'numpy', 'seaborn', 'matplotlib', 'scikit-learn']
for package in required:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 55)
print(" Movie Rating Prediction using Regression")
print("=" * 55)
print("All libraries installed and imported successfully!\n")

# ══════════════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ══════════════════════════════════════════════════════════
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

print(f"Dataset Shape   : {df.shape}")
print(f"Columns         : {list(df.columns)}\n")
print("First 5 rows:")
print(df.head())

# ══════════════════════════════════════════════════════════
# STEP 2 — Data Cleaning & Preprocessing
# ══════════════════════════════════════════════════════════
print("\n── Data Cleaning ──────────────────────────────────────")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed : {before - len(df)}")

# Drop unnecessary columns
drop_cols = [col for col in ['Name', 'Year', 'Duration', 'Votes'] if col in df.columns]
df = df.drop(columns=drop_cols)
print(f"Columns dropped    : {drop_cols}")

# Drop rows with missing values
before = len(df)
df = df.dropna()
print(f"Rows after dropna  : {len(df)}  (removed {before - len(df)} rows)")

# Clean Rating column — ensure numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

print(f"\nFinal dataset shape: {df.shape}")
print(f"Rating range       : {df['Rating'].min():.1f} – {df['Rating'].max():.1f}")

# ══════════════════════════════════════════════════════════
# STEP 3 — Feature Engineering
# ══════════════════════════════════════════════════════════
print("\n── Feature Engineering ────────────────────────────────")

le = LabelEncoder()
cat_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Label-encoded : {col}")

# ══════════════════════════════════════════════════════════
# STEP 4 — Data Visualisation
# ══════════════════════════════════════════════════════════
print("\n── Generating Visualisations ──────────────────────────")
sns.set_theme(style='whitegrid', palette='Set2')

# --- Plot 1: Rating Distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(df['Rating'], bins=30, kde=True, color='steelblue')
plt.title("Distribution of Movie Ratings", fontsize=14, fontweight='bold')
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_rating_distribution.png", dpi=150)
plt.show()
print("  Saved: plot_rating_distribution.png")

# --- Plot 2: Average Rating by Genre (top 10) ---
if 'Genre' in df.columns:
    # Reload original genres for readable labels
    df_orig = pd.read_csv("IMDb Movies India.csv", encoding='latin1').dropna(subset=['Genre', 'Rating'])
    df_orig['Rating'] = pd.to_numeric(df_orig['Rating'], errors='coerce')
    genre_avg = df_orig.groupby('Genre')['Rating'].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=genre_avg.values, y=genre_avg.index, palette='viridis')
    plt.title("Top 10 Genres by Average Rating", fontsize=14, fontweight='bold')
    plt.xlabel("Average Rating")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig("plot_genre_ratings.png", dpi=150)
    plt.show()
    print("  Saved: plot_genre_ratings.png")

# --- Plot 3: Correlation Heatmap ---
plt.figure(figsize=(9, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, annot_kws={'size': 8})
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", dpi=150)
plt.show()
print("  Saved: plot_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════
# STEP 5 — Train / Test Split
# ══════════════════════════════════════════════════════════
X = df.drop('Rating', axis=1)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ══════════════════════════════════════════════════════════
# STEP 6 — Model Training
# ══════════════════════════════════════════════════════════
print("\n── Training Models ────────────────────────────────────")

# Model 1 — Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print(" Linear Regression trained")

# Model 2 — Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Regressor trained")

# ══════════════════════════════════════════════════════════
# STEP 7 — Model Evaluation
# ══════════════════════════════════════════════════════════
print("\n── Model Evaluation ───────────────────────────────────")

def evaluate(name, y_true, y_pred_vals):
    mse  = mean_squared_error(y_true, y_pred_vals)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred_vals)
    print(f"\n   {name}")
    print(f"     MSE  : {mse:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     R²   : {r2:.4f}")
    return mse, rmse, r2

lr_mse,  lr_rmse,  lr_r2  = evaluate("Linear Regression",      y_test, lr_pred)
rf_mse,  rf_rmse,  rf_r2  = evaluate("Random Forest Regressor", y_test, rf_pred)

# Summary table
print("\n" + "=" * 52)
print(f"  {'Model':<28} {'RMSE':>8}  {'R²':>8}")
print("=" * 52)
print(f"  {'Linear Regression':<28} {lr_rmse:>8.4f}  {lr_r2:>8.4f}")
print(f"  {'Random Forest Regressor':<28} {rf_rmse:>8.4f}  {rf_r2:>8.4f}")
print("=" * 52)
winner = "Random Forest Regressor" if rf_r2 >= lr_r2 else "Linear Regression"
print(f"\n Best Model : {winner}")

# ══════════════════════════════════════════════════════════
# STEP 8 — Evaluation Visualisations
# ══════════════════════════════════════════════════════════

# --- Plot 4: Actual vs Predicted (both models) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Actual vs Predicted Ratings", fontsize=15, fontweight='bold')

for ax, pred, title, color in zip(
    axes,
    [lr_pred, rf_pred],
    ["Linear Regression", "Random Forest Regressor"],
    ["#5B9BD5", "#70AD47"]
):
    ax.scatter(y_test, pred, alpha=0.4, color=color, edgecolors='white', s=30)
    mn, mx = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Fit')
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(title, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig("plot_actual_vs_predicted.png", dpi=150)
plt.show()
print("  Saved: plot_actual_vs_predicted.png")

# --- Plot 5: Feature Importance (Random Forest) ---
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='#70AD47', edgecolor='white')
plt.title("Feature Importance — Random Forest", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.show()
print("  Saved: plot_feature_importance.png")

# --- Plot 6: Model Comparison Bar Chart ---
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'R² Score': [lr_r2, rf_r2]
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=metrics_df, x='Model', y='RMSE', palette=['#5B9BD5', '#70AD47'], ax=axes[0])
axes[0].set_title("RMSE Comparison (Lower = Better)", fontweight='bold')
axes[0].set_ylabel("RMSE")

sns.barplot(data=metrics_df, x='Model', y='R² Score', palette=['#5B9BD5', '#70AD47'], ax=axes[1])
axes[1].set_title("R² Score Comparison (Higher = Better)", fontweight='bold')
axes[1].set_ylabel("R² Score")

plt.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("plot_model_comparison.png", dpi=150)
plt.show()
print("  Saved: plot_model_comparison.png")

print("\n" + "=" * 55)
print(" Movie Rating Prediction Completed Successfully!")
print("=" * 55)
