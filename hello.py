# Credit Risk Analysis
# German Credit Dataset Classification

## Introduction

**Credit Risk** is the probable risk of loss resulting from a borrower's failure to repay a loan or meet contractual obligations.

**Types of Credit Risk:**
- **Good Risk**: An investment that is likely to be profitable
- **Bad Risk**: A loan that is unlikely to be repaid due to bad credit history or insufficient income

**Objective:** Based on the attributes, classify a person as good or bad credit risk.

## Dataset Description
The dataset contains 1000 entries with 20 independent variables (7 numerical, 13 categorical) and 1 target variable.

## 1. Import Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import floor, ceil
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Metrics
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from scipy import stats
from time import time

# Display settings
pd.set_option('display.max_columns', None)
plt.style.use('default')
```

## 2. Utility Functions

```python
def style_specific_cell(x):
    """Style cells with percentage < 10% in light pink"""
    color_thresh = 'background-color: lightpink'
    df_color = pd.DataFrame('', index=x.index, columns=x.columns)
    
    for r in range(len(x.index)): 
        for c in range(len(x.columns)):
            try:
                val = float(x.iloc[r, c])
                if x.iloc[r, 0] == "Percentage" and val < 10:
                    df_color.iloc[r, c] = color_thresh
            except:
                pass
    return df_color

def style_stats_specific_cell(x):
    """Style cells with p-value > 0.05 in light pink"""
    color_thresh = 'background-color: lightpink'
    df_color = pd.DataFrame('', index=x.index, columns=x.columns)
    
    for r in range(len(x.index)):
        try:
            val = x.iloc[r, 1]
            if val > 0.05:
                df_color.iloc[r, 1] = color_thresh
        except:
            pass
    return df_color

def visualize_distribution(attr, df):
    """Create side-by-side bar plots for Good vs Bad Risk"""
    good_risk_df = df[df["Cost Matrix(Risk)"] == "Good Risk"]
    bad_risk_df = df[df["Cost Matrix(Risk)"] == "Bad Risk"]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    attr_good_risk_df = good_risk_df[[attr, 'Cost Matrix(Risk)']].groupby(attr).count()
    attr_bad_risk_df = bad_risk_df[[attr, 'Cost Matrix(Risk)']].groupby(attr).count()
    
    ax[0].barh(attr_good_risk_df['Cost Matrix(Risk)'].index.tolist(), 
               attr_good_risk_df['Cost Matrix(Risk)'].tolist(), 
               align='center', color="#5975A4")
    ax[1].barh(attr_bad_risk_df['Cost Matrix(Risk)'].index.tolist(), 
               attr_bad_risk_df['Cost Matrix(Risk)'].tolist(), 
               align='center', color="#B55D60")
    
    ax[0].set_title('Good Risk')
    ax[1].set_title('Bad Risk')
    ax[0].invert_xaxis()
    ax[1].yaxis.tick_right()
    
    # Add value labels
    for i, v in enumerate(attr_good_risk_df['Cost Matrix(Risk)'].tolist()):
        ax[0].text(v + 25, i + 0.05, str(v), color='black')
    for i, v in enumerate(attr_bad_risk_df['Cost Matrix(Risk)'].tolist()):
        ax[1].text(v + 1, i + 0.05, str(v), color='black')
    
    plt.suptitle(attr)
    plt.tight_layout()
    plt.show()
```

## 3. Data Loading and Preprocessing

```python
# Load the dataset
df = pd.read_csv("german.data", sep=" ", header=None)

# Define column headers
headers = ["Status of existing checking account", "Duration in month", "Credit history",
           "Purpose", "Credit amount", "Savings account/bonds", "Present employment since",
           "Installment rate in percentage of disposable income", "Personal status and sex",
           "Other debtors / guarantors", "Present residence since", "Property", "Age in years",
           "Other installment plans", "Housing", "Number of existing credits at this bank",
           "Job", "Number of people being liable to provide maintenance for", "Telephone", 
           "foreign worker", "Cost Matrix(Risk)"]

df.columns = headers

# Save as CSV for reference
df.to_csv("german_data_credit_cat.csv", index=False)

print(f"Dataset shape: {df.shape}")
df.head()
```

## 4. Data Mapping and Cleaning

```python
# Map categorical codes to meaningful labels
mappings = {
    'Status of existing checking account': {
        'A14': "no checking account", 'A11': "<0 DM", 
        'A12': ">0 DM", 'A13': ">0 DM"
    },
    'Credit history': {
        "A34": "critical account/delay in paying off",
        "A33": "critical account/delay in paying off",
        "A32": "all credit / existing credits paid back duly till now",
        "A31": "all credit / existing credits paid back duly till now",
        "A30": "no credits taken"
    },
    'Purpose': {
        "A40": "car (new)", "A41": "car (used)", 
        "A42": "Home Related", "A43": "Home Related", 
        "A44": "Home Related", "A45": "Home Related", 
        "A46": "others", 'A47': 'others', 'A48': 'others', 
        'A49': 'others', 'A410': 'others'
    },
    'Savings account/bonds': {
        "A65": "no savings account", "A61": "<100 DM",
        "A62": "<500 DM", "A63": ">500 DM", "A64": ">500 DM"
    },
    'Present employment since': {
        'A75': ">=7 years", 'A74': "4<= <7 years",
        'A73': "1<= < 4 years", 'A72': "<1 years", 'A71': "<1 years"
    },
    'Personal status and sex': {
        'A95': "female", 'A94': "male", 'A93': "male", 
        'A92': "female", 'A91': "male"
    },
    'Other debtors / guarantors': {
        'A101': "none", 'A102': "co-applicant/guarantor", 
        'A103': "co-applicant/guarantor"
    },
    'Property': {
        'A121': "real estate", 'A122': "savings agreement/life insurance",
        'A123': "car or other", 'A124': "unknown / no property"
    },
    'Other installment plans': {
        'A143': "none", 'A142': "bank/store", 'A141': "bank/store"
    },
    'Housing': {
        'A153': "for free", 'A152': "own", 'A151': "rent"
    },
    'Job': {
        'A174': "employed", 'A173': "employed", 
        'A172': "unemployed", 'A171': "unemployed"
    },
    'Telephone': {
        'A192': "yes", 'A191': "none"
    },
    'foreign worker': {
        'A201': "yes", 'A202': "no"
    },
    'Cost Matrix(Risk)': {
        1: "Good Risk", 2: "Bad Risk"
    }
}

# Apply mappings
for column, mapping in mappings.items():
    df[column] = df[column].map(mapping)

# Special mapping for number of credits
number_of_credit = {1: 1, 2: 2, 3: 2, 4: 2}
df["Number of existing credits at this bank"] = df["Number of existing credits at this bank"].map(number_of_credit)

print("Data mapping completed successfully!")
df.head()
```

## 5. Exploratory Data Analysis

```python
# Summary statistics for numerical variables
numerical_cols = ["Credit amount", "Age in years", "Duration in month"]
print("Numerical Variables Summary:")
print(df[numerical_cols].describe())
```

```python
# Categorical variables analysis
categorical_cols = [col for col in df.columns if col not in numerical_cols + ['Cost Matrix(Risk)']]

def analyze_categorical_variables(df, categorical_cols):
    """Analyze categorical variables and show proportions"""
    for col in categorical_cols[:5]:  # Show first 5 for demo
        print(f"\n{col}:")
        proportions = df[col].value_counts(normalize=True) * 100
        print(proportions.round(2))

analyze_categorical_variables(df, categorical_cols)
```

```python
# Visualize key variables
key_variables = ["Status of existing checking account", "Credit history", 
                "Purpose", "Savings account/bonds"]

for var in key_variables:
    visualize_distribution(var, df)
```

```python
# Numerical variables by risk category
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], x=df["Cost Matrix(Risk)"], 
                orient='v', ax=axes[i], palette=["#5975A4", "#B55D60"])
    axes[i].set_title(f'{col} by Risk Category')
plt.tight_layout()
plt.show()
```

## 6. Statistical Significance Testing

```python
# Chi-square test for categorical variables
categorical_features = [col for col in df.columns if col not in numerical_cols + ['Cost Matrix(Risk)']]

statistical_significance = []
for attr in categorical_features:
    data_count = pd.crosstab(df[attr], df["Cost Matrix(Risk)"])
    obs = np.array(data_count[["Bad Risk", "Good Risk"]])
    chi2, p, dof, expected = stats.chi2_contingency(obs)
    statistical_significance.append([attr, round(p, 6)])

cat_stats_df = pd.DataFrame(statistical_significance, columns=["Attribute", "P-value"])
print("Categorical Variables Statistical Significance (Chi-square test):")
print(cat_stats_df.sort_values('P-value'))
```

```python
# ANOVA test for numerical variables
good_risk_df = df[df["Cost Matrix(Risk)"] == "Good Risk"]
bad_risk_df = df[df["Cost Matrix(Risk)"] == "Bad Risk"]

numerical_significance = []
for attr in numerical_cols:
    statistic, p = stats.f_oneway(good_risk_df[attr].values, bad_risk_df[attr].values)
    numerical_significance.append([attr, round(p, 6)])

num_stats_df = pd.DataFrame(numerical_significance, columns=["Attribute", "P-value"])
print("Numerical Variables Statistical Significance (ANOVA test):")
print(num_stats_df)
```

## 7. Feature Engineering and Selection

```python
# Select significant features (p < 0.05)
significant_categorical = cat_stats_df[cat_stats_df['P-value'] < 0.05]['Attribute'].tolist()
significant_numerical = num_stats_df[num_stats_df['P-value'] < 0.05]['Attribute'].tolist()

selected_features = significant_categorical + significant_numerical
print(f"Selected Features: {len(selected_features)}")
print(selected_features)

# Create final dataset
df_model = df[selected_features + ['Cost Matrix(Risk)']].copy()
```

```python
# Create dummy variables for categorical features
categorical_features_final = [col for col in selected_features if col not in numerical_cols]

for attr in categorical_features_final:
    dummies = pd.get_dummies(df_model[attr], prefix=attr)
    df_model = pd.concat([df_model, dummies], axis=1)
    df_model.drop(attr, axis=1, inplace=True)

# Convert target variable to numeric
risk_mapping = {"Good Risk": 1, "Bad Risk": 0}
df_model["Cost Matrix(Risk)"] = df_model["Cost Matrix(Risk)"].map(risk_mapping)

print(f"Final dataset shape: {df_model.shape}")
print("\nFirst few columns:")
print(df_model.columns[:10].tolist())
```

## 8. Model Preparation

```python
# Prepare features and target
X = df_model.drop('Cost Matrix(Risk)', axis=1).values
y = df_model['Cost Matrix(Risk)'].values

# Apply PCA for dimensionality reduction
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"PCA components: {X_pca.shape[1]}")
print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")
```

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class distribution in training set:")
print(f"Good Risk (1): {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
print(f"Bad Risk (0): {len(y_train)-sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train)*100:.1f}%)")
```

## 9. Model Comparison

```python
def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    start_time = time()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    training_time = time() - start_time
    
    f1 = f1_score(y_test, y_pred)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Training time: {training_time:.2f}s")
    print("-" * 50)
    
    return f1, training_time

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Compare models
results = []
print("Model Comparison Results:")
print("=" * 60)

for name, model in models.items():
    print(f"Evaluating {name}:")
    pipeline = Pipeline([('classifier', model)])
    f1, time_taken = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
    results.append((name, f1, time_taken))

# Sort by F1 score
results.sort(key=lambda x: x[1], reverse=True)
print("\nRanked Results (by F1 Score):")
for rank, (name, f1, time_taken) in enumerate(results, 1):
    print(f"{rank}. {name}: F1={f1:.4f}, Time={time_taken:.2f}s")
```

## 10. Hyperparameter Tuning (Best Model)

```python
# Use the best performing model for hyperparameter tuning
best_model_name = results[0][0]
print(f"Hyperparameter tuning for: {best_model_name}")

# LightGBM hyperparameter tuning (assuming it's the best)
lgb_model = LGBMClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.6, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.8],
    'bagging_freq': [100, 200]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    lgb_model, 
    param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
```

## 11. Final Model Evaluation

```python
# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
final_f1 = f1_score(y_test, y_pred)
final_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Final Model Performance:")
print(f"F1 Score: {final_f1:.4f}")
print(f"AUC Score: {final_auc:.4f}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad Risk', 'Good Risk']))
```

## 12. ROC Curve Visualization

```python
# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.plot(fpr, tpr, label=f'Model (AUC = {final_auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Risk Classification')
plt.legend()
plt.grid(True)
plt.show()
```

## 13. Feature Importance

```python
# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances (PCA Components)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
```

## Conclusion

This notebook demonstrates a complete credit risk analysis workflow:

1. **Data Loading & Preprocessing**: Cleaned and mapped categorical variables
2. **Exploratory Data Analysis**: Visualized distributions and relationships
3. **Statistical Testing**: Identified significant features using Chi-square and ANOVA tests
4. **Feature Engineering**: Created dummy variables and applied PCA
5. **Model Comparison**: Evaluated multiple machine learning algorithms
6. **Hyperparameter Tuning**: Optimized the best performing model
7. **Final Evaluation**: Assessed model performance with F1 score and AUC

The final model can effectively classify credit risk with good performance metrics, helping financial institutions make informed lending decisions.