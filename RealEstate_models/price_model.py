import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform
import numpy as np
import joblib

data = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')

X = data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF',
          'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
          'KitchenQual', 'Neighborhood', 'SaleCondition']]

X = pd.get_dummies(X)
y = data['SalePrice']

import json
with open('model_columns.json', 'w') as f:
    json.dump(X.columns.tolist(), f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(5, 30),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.5, 0.5),
    "bootstrap": [True, False],
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Najlepsze parametry:", search.best_params_)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE na zbiorze testowym: {rmse:.2f} zł")

joblib.dump(best_model, 'random_forest_real_estate.joblib')
