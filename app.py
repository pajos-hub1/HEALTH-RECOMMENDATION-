from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import ast
import logging
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import NearestNeighbors

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained KNN model and scaler
with open('knn_model.pkl', 'rb') as model_file:
    knn = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load nutritional data
nutritional_data = pd.read_excel('nutritional_data_1.xlsx')

# Clean the nutritional data
nutritional_data.columns = [
    'Food and Serving', 'Calories', 'Total Fat', 'Sodium', 'Potassium',
    'Total Carbo-hydrate', 'Dietary Fiber', 'Sugars', 'Protein',
    'Vitamin A', 'Vitamin C', 'Calcium', 'Iron'
]

# Ensure all relevant columns are numeric, and handle non-numeric values
numeric_columns = [
    'Calories', 'Total Fat', 'Sodium', 'Potassium', 'Total Carbo-hydrate',
    'Dietary Fiber', 'Sugars', 'Protein', 'Vitamin A', 'Vitamin C',
    'Calcium', 'Iron'
]

for column in numeric_columns:
    nutritional_data[column] = pd.to_numeric(nutritional_data[column], errors='coerce')

# Drop rows with NaN values in numeric columns
nutritional_data = nutritional_data.dropna(subset=numeric_columns)

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]
)


# Define the request body model
class UserProfile(BaseModel):
    Disliked_Foods: str
    Health_Objectives: str


def get_preferred_foods(user_profile, main_df):
    try:
        disliked_foods = ast.literal_eval(user_profile.Disliked_Foods)  # Access directly with dot notation
    except (ValueError, SyntaxError) as e:
        logger.error("Error parsing Disliked_Foods: %s", str(e))
        return main_df  # Fallback to returning the full DataFrame

    # Normalize the strings for comparison
    disliked_foods = [food.lower().strip() for food in disliked_foods]
    main_df['Food and Serving'] = main_df['Food and Serving'].str.lower().str.strip()

    preferred_foods = main_df[~main_df['Food and Serving'].isin(disliked_foods)]

    # Accessing the health objectives directly from the profile
    health_objective = user_profile.Health_Objectives
    if health_objective == 'Digestive Issues':
        preferred_foods = preferred_foods[preferred_foods['Dietary Fiber'] > 3]
    elif health_objective == 'High blood pressure':
        preferred_foods = preferred_foods[preferred_foods['Sodium'] > 50]
    elif health_objective == 'Heart Health':
        preferred_foods = preferred_foods[(preferred_foods['Potassium'] > 300) & (preferred_foods['Total Fat'] > 1)]
    elif health_objective == 'Weight Loss':
        preferred_foods = preferred_foods[(preferred_foods['Calories'] > 1) & (preferred_foods['Total Fat'] > 0)]
    elif health_objective == 'Skin Health':
        preferred_foods = preferred_foods[(preferred_foods['Vitamin A'] > 50) & (preferred_foods['Vitamin C'] > 50)]
    elif health_objective == 'Immune system support':
        preferred_foods = preferred_foods[(preferred_foods['Vitamin A'] > 50) & (preferred_foods['Vitamin C'] > 50)]
    elif health_objective == 'Bone Health':
        preferred_foods = preferred_foods[preferred_foods['Calcium'] > 4]
    elif health_objective == 'Eye Health':
        preferred_foods = preferred_foods[preferred_foods['Vitamin A'] > 150]
    elif health_objective == 'Joint Health':
        preferred_foods = preferred_foods[preferred_foods['Vitamin C'] > 100]
    elif health_objective == 'Brain Health':
        preferred_foods = preferred_foods[preferred_foods['Vitamin C'] > 200]
    elif health_objective == 'Muscle Gain':
        preferred_foods = preferred_foods[preferred_foods['Protein'] > 2]

    return preferred_foods




def recommend_foods(user_profile, main_df, scaler):
    # Get the preferred foods based on the user's profile
    preferred_foods = get_preferred_foods(user_profile, main_df)
    if preferred_foods.empty:
        return pd.DataFrame()  # Return an empty DataFrame if there are no preferred foods

    # Keep the original indices of the preferred foods
    original_indices = preferred_foods.index
    print("Original Indices:", original_indices)
    
    try:
        # Check the number of samples
        n_samples = len(preferred_foods)
        n_neighbors = min(4, n_samples)  # Adjust n_neighbors to be less than or equal to the number of samples
        
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        
        # Drop non-numeric columns and scale the features
        preferred_features = preferred_foods.drop(columns=['Food and Serving'])
        scaled_preferred_features = scaler.transform(preferred_features)
        
        # Use the mean of the preferred features as the query point
        query_point = scaled_preferred_features.mean(axis=0).reshape(1, -1)
        print("Query Point:", query_point)
        
        knn.fit(scaled_preferred_features)
        
        # Get the nearest neighbors
        distances, indices = knn.kneighbors(query_point)
        print("KNN Returned Indices:", indices)

        # Convert KNN indices (relative) to original DataFrame indices
        recommended_indices = original_indices[indices[0]]
        print("Mapped Recommended Indices:", recommended_indices)

        # Retrieve the recommended foods from the original DataFrame using these indices
        recommended_foods = main_df.loc[recommended_indices]
    except IndexError as e:
        print("IndexError:", str(e))
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

    return recommended_foods

@app.post("/recommended")
def recommend(user_profile: UserProfile):
    try:
        logger.info("Received request: %s", user_profile)
        recommendations = recommend_foods(user_profile, nutritional_data, scaler)
        logger.info("Recommendations: %s", recommendations)
        return {"recommended_foods": recommendations['Food and Serving'].tolist()}
    except Exception as e:
        logger.error("Error in recommendation: %s", str(e))
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
