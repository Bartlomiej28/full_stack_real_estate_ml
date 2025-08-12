import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
            '1stFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
            'KitchenQual', 'Neighborhood']

X = df[features]
y_price = df['SalePrice']

num_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                '1stFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']
cat_features = ['KitchenQual', 'Neighborhood']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

regressor = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

regressor.fit(X, y_price)

predicted_price = regressor.predict(X)
df['price_error'] = predicted_price - y_price

threshold = -20000
df['is_deal'] = (df['price_error'] < threshold).astype(int)


y_deal = df['is_deal']

classifier = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

classifier.fit(X, y_deal)

joblib.dump(classifier, 'deal_classifier.joblib')

def predict_if_deal(user_input_dict):
    deal_classifier = joblib.load('deal_classifier.joblib')
    user_df = pd.DataFrame([user_input_dict])
    is_deal = deal_classifier.predict(user_df)[0]
    deal_prob = deal_classifier.predict_proba(user_df)[0, 1]
    return {
        'is_deal': bool(is_deal),
        'deal_probability': deal_prob
    }

user_input = {
    'OverallQual': 7,
    'GrLivArea': 1600,
    'GarageCars': 2,
    'TotalBsmtSF': 900,
    '1stFlrSF': 850,
    'YearBuilt': 1999,
    'FullBath': 2,
    'TotRmsAbvGrd': 7,
    'Fireplaces': 1,
    'KitchenQual': 'Gd',
    'Neighborhood': 'CollgCr'
}

print(predict_if_deal(user_input))
