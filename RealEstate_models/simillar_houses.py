import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
from supabase import create_client, Client


def train_and_save_model():
    SUPABASE_URL = "https://qnumgnuvmwkfulwdbtos.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudW1nbnV2bXdrZnVsd2RidG9zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NzM4MTYsImV4cCI6MjA2ODI0OTgxNn0.CkmQFR-v-HrL1BGY0Q2djXh5TVb7uQibyvOtGGa24hw"
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Pobieranie danych z Supabase...")
    response = supabase.table('Houses').select('*').execute()
    df = pd.DataFrame(response.data)

    features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                'FstFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
                'KitchenQual', 'Neighborhood', 'SaleCondition']

    df_features = df[features]

    num_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                    'FstFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces']
    cat_features = ['KitchenQual', 'Neighborhood', 'SaleCondition']

    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    cat_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    print("Trenowanie modelu...")
    X = preprocessor.fit_transform(df_features)

    knn = NearestNeighbors(n_neighbors=8, algorithm='kd_tree', metric='euclidean', n_jobs=-1)
    knn.fit(X)

    print("Zapisywanie modelu...")
    joblib.dump(knn, 'knn_model.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(df[['ID']], 'house_ids.joblib')

    print("Model został pomyślnie wytrenowany i zapisany!")


if __name__ == '__main__':
    train_and_save_model()
