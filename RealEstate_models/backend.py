from flask import Flask, jsonify, request
from supabase import create_client, Client
from flask_cors import CORS
import pandas as pd
import joblib
import json
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

SUPABASE_URL = "https://qnumgnuvmwkfulwdbtos.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudW1nbnV2bXdrZnVsd2RidG9zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NzM4MTYsImV4cCI6MjA2ODI0OTgxNn0.CkmQFR-v-HrL1BGY0Q2djXh5TVb7uQibyvOtGGa24hw"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

with open('model_columns.json', 'r') as f:
    TRAIN_COLUMNS = json.load(f)

model = joblib.load('random_forest_real_estate.joblib')
knn = joblib.load('knn_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')
house_ids = joblib.load('house_ids.joblib')


@app.route('/api/get-featured', methods=['GET'])
def get_properties():
    try:
        response = supabase.table('Houses').select("*").limit(3).execute()
        data = response.data
        print(data)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recently-added", methods=["GET"])
def get_featured():
    try:
        response = (
            supabase
            .table("Houses")
            .select("*")
            .order("ID", desc=True)
            .limit(24)
            .execute()
        )
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def get_search():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 24))

    query = supabase.table("Houses").select("*", count='exact')

    filters = {
        "Street": request.args.get("Street"),
        "City": request.args.get("City"),
        "TotRmsAbvGrd": request.args.get("TotRmsAbvGrd"),
        "FullBath": request.args.get("FullBath"),
        "GrLivArea": request.args.get("GrLivArea"),
        "SalePrice": request.args.get("SalePrice"),
    }

    for field, value in filters.items():
        if value:
            if field in ["TotRmsAbvGrd", "FullBath", "GrLivArea", "SalePrice"]:
                try:
                    numeric_value = int(value)
                    query = query.eq(field, numeric_value)
                except ValueError:
                    continue
            else:
                query = query.ilike(field, f"%{value}%")

    query = query.range((page - 1) * per_page, page * per_page - 1)

    response = query.execute()

    return jsonify({
        "properties": response.data,
        "total_count": response.count
    }), 200

@app.route("/api/property-details/<string:property_id>", methods=["GET"])
def get_property_by_id(property_id):
    try:
        response = (
            supabase
            .table("Houses")
            .select("*")
            .eq("ID", property_id)
            .single()
            .execute()
        )
        return jsonify(response.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/add-property", methods=["POST"])
def add_property():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = [
            "City", "Street", "OverallQual", "GrLivArea", "GarageCars",
            "TotalBsmtSF", "FirstFlrSF", "YearBuilt", "FullBath",
            "TotRmsAbvGrd", "Fireplaces", "KitchenQual", "Neighborhood",
            "SaleCondition", "Description", "Title", "Image"
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        supabase_data = {
            "City": data["City"],
            "Street": data["Street"],
            "OverallQual": int(data["OverallQual"]),
            "GrLivArea": int(data["GrLivArea"]),
            "GarageCars": int(data["GarageCars"]),
            "TotalBsmtSF": int(data["TotalBsmtSF"]),
            "1stFlrSF": int(data["FirstFlrSF"]),
            "YearBuilt": int(data["YearBuilt"]),
            "FullBath": int(data["FullBath"]),
            "TotRmsAbvGrd": int(data["TotRmsAbvGrd"]),
            "Fireplaces": int(data["Fireplaces"]),
            "KitchenQual": data["KitchenQual"],
            "Neighborhood": data["Neighborhood"],
            "SaleCondition": data["SaleCondition"],
            "Description": data["Description"],
            "Title": data["Title"],
            "Image": data["Image"]
        }

        response = supabase.table("Houses").insert(supabase_data).execute()

        if hasattr(response, 'error') and response.error:
            return jsonify({"error": "Database error", "details": str(response.error)}), 500

        return jsonify({
            "message": "Property added successfully",
            "data": response.data[0] if response.data else None
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    try:
        data = request.json

        input_data = pd.DataFrame([{
            'OverallQual': int(data['OverallQual']),
            'GrLivArea': int(data['GrLivArea']),
            'GarageCars': int(data['GarageCars']),
            'TotalBsmtSF': int(data['TotalBsmtSF']),
            '1stFlrSF': int(data['FirstFlrSF']),
            'YearBuilt': int(data['YearBuilt']),
            'FullBath': int(data['FullBath']),
            'TotRmsAbvGrd': int(data['TotRmsAbvGrd']),
            'Fireplaces': int(data['Fireplaces']),
            'KitchenQual': data['KitchenQual'],
            'Neighborhood': data['Neighborhood'],
            'SaleCondition': data['SaleCondition']
        }])

        input_encoded = pd.get_dummies(input_data)

        for col in TRAIN_COLUMNS:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[TRAIN_COLUMNS]

        predicted_price = model.predict(input_encoded)

        return jsonify({
            "success": True,
            "predicted_price": round(float(predicted_price[0]), 2),
            "currency": "USD"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


try:
    knn = joblib.load('knn_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    house_ids = joblib.load('house_ids.joblib')
    print("Model został pomyślnie załadowany")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    raise

@app.route('/api/similar-houses', methods=['POST'])
def similar_houses():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = [
            'SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
            'FstFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
            'KitchenQual', 'Neighborhood', 'SaleCondition'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        input_data = {
            'SalePrice': float(data['SalePrice']),
            'OverallQual': int(data['OverallQual']),
            'GrLivArea': int(data['GrLivArea']),
            'GarageCars': int(data['GarageCars']),
            'TotalBsmtSF': int(data['TotalBsmtSF']),
            'FstFlrSF': int(data.get('FstFlrSF')),
            'YearBuilt': int(data['YearBuilt']),
            'FullBath': int(data['FullBath']),
            'TotRmsAbvGrd': int(data['TotRmsAbvGrd']),
            'Fireplaces': int(data['Fireplaces']),
            'KitchenQual': data['KitchenQual'],
            'Neighborhood': data['Neighborhood'],
            'SaleCondition': data['SaleCondition']
        }

        user_df = pd.DataFrame([input_data])
        user_X = preprocessor.transform(user_df)

        distances, indices = knn.kneighbors(user_X)
        similar_ids = house_ids.iloc[indices[0]]['ID'].tolist()

        current_house_id = data.get('currentHouseId')
        if current_house_id:
            similar_ids = [id for id in similar_ids if id != current_house_id]

        response = supabase.table('Houses').select('*').in_('ID', similar_ids).execute()
        similar_houses = response.data

        id_to_distance = {similar_ids[i]: float(distances[0][i]) for i in range(len(similar_ids))}
        for house in similar_houses:
            house['Distance'] = id_to_distance.get(house['ID'], 0.0)

        similar_houses = sorted(similar_houses, key=lambda x: x['Distance'])
        return jsonify(similar_houses)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

CORS(app, origins=["http://localhost:3000"])