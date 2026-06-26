import job sklearndoc
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from data_loader import load_wine_data

def train_and_save_model():
    # 1. Load Data
    df = load_wine_data()
    
    # Features (X) and Target (y)
    X = df[['description', 'country', 'variety', 'price']]
    y = df['points']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building preprocessing pipeline...")
    # 2. Define Preprocessing for Text, Categorical, and Numerical Columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=2500, stop_words='english'), 'description'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['country', 'variety']),
            ('num', StandardScaler(), ['price'])
        ]
    )
    
    # 3. Create full training Pipeline with a Ridge Regressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # 4. Train the Model
    print("Training the model (this might take a minute)...")
    model_pipeline.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    predictions = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Training Metrics:\n- Mean Absolute Error: {mae:.2f} points\n- R2 Score: {r2:.2f}")
    
    # 6. Save the Pipeline locally
    model_filename = 'wine_pipeline.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model_pipeline, f)
    print(f"Model pipeline successfully saved to {model_filename}")

if __name__ == "__main__":
    train_and_save_model()