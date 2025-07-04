
# 🏠 Pune Housing Price Prediction - ML Regression Project

This project aims to predict the price of residential properties in Pune using Machine Learning regression models. The dataset contains various features such as total square footage, number of bedrooms, bathrooms, balconies, area type, and location.

---

## 📁 Project Structure

-  Data Cleaning and Feature Engineering  
- Exploratory Data Analysis (EDA)  
-  Correlation Analysis & Model Building  

---

## 📊 Dataset Description

The dataset includes the following features:
- `area_type`: Type of property (Built-up Area, Carpet Area, etc.)
- `total_sqft`: Total area in square feet
- `bathroom`: Number of bathrooms
- `balcony`: Number of balconies
- `BHK`: Number of bedrooms
- `site_location`: Location of the property
- `price`: Price in lakhs (target variable)

---

## Data Cleaning & Feature Engineering

- Cleaned inconsistent `total_sqft` values (e.g., `'900 - 1200'`).
- Removed outliers and missing values.
- Handled non-numeric data.
- Created new features (`BHK`, one-hot encoded `area_type` and `site_location`).
- Saved cleaned dataset to `Cleaned_data.csv`.

---

##  Exploratory Data Analysis (EDA)

- Visualized price distribution and feature relationships.
- Examined price trends with respect to BHK, area type, and site location.
- Used `seaborn` plots for barplots and scatter plots:
  - BHK vs Price
  - Bathrooms vs BHK
  - Balconies vs BHK
- Analyzed average prices by location and area type using groupby and pivot tables.

---

##  Correlation Analysis & Model Building

### ✅ Correlation Insights
- `total_sqft` has the highest correlation with price (0.84).
- `balcony` shows the weakest correlation (0.21).
- Pearson p-values confirmed statistical significance.

### 🔢 Categorical Feature Encoding
- Used `pd.get_dummies()` for `area_type` and `site_location`.

### 🤖 Models Implemented
| Model | R² (Test) | Cross-Val Accuracy | RMSE |
|-------|-----------|--------------------|------|
| **Linear Regression** | 0.79 | ~0.74 | ~31.57 |
| Decision Tree Regressor | 0.79 | ~0.80 | ~31.31 |
| Random Forest Regressor | 0.79 | ~0.81 | ~29.37 |
| **LazyPredict Best Model: LGBMRegressor** | 0.80 | – | 28.14 |

### 🧪 Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)
- RMSE
- Cross-validation scores (KFold)

### 📈 Visualization
- Distribution plot comparing actual vs predicted prices.
- Scatter plot of actual vs predicted using `regplot`.

---

## 💻 Deployment & Testing

A utility function is defined for testing predictions on new property data:
```python
predict_price(area_type, location, sqft, balcony, bathroom, BHK)
````

Example:

```python
predict_price('Carpet  Area', 'Baner', 1250, 2, 2, 3) → 88.1
```

---

## 🧠 Final Observations & Learnings

### 💡 Key Takeaways

* Strong features: `total_sqft`, `BHK`, and `bathroom`.
* Weakest correlation: `balcony`, yet potentially important contextually.
* Data preprocessing plays a crucial role in model performance.
* Linear models are interpretable; ensemble models like LGBM offer better accuracy.

### 🎯 Challenges & How They Were Overcome

| Challenge                                 | Solution                                         |
| ----------------------------------------- | ------------------------------------------------ |
| Non-numeric `total_sqft` values           | Used regex parsing and mean of ranges            |
| Sparse one-hot encoded location data      | Switched to tree-based models                    |
| Negative CV scores with Linear Regression | Tried better regularization and used LazyPredict |

---

## 📦 Files Included

* `Housing_Price_Prediction.ipynb`: Jupyter notebook with complete code
* `Cleaned_data.csv`: Cleaned dataset
* `bangalore_home_prices_model.pickle`: Serialized Linear Regression model
* `Columns.json`: JSON of all model input columns
* `README.md`: This file

---

## 🔧 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lazypredict lightgbm xgboost
```

---

## 📬 Author

**Jitendra Kumar Gupta**
*ML Engineer | AI Reasearcher | Data Scientist*
