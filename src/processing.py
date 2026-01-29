import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

# Professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess(file_path: str) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Load the heart disease dataset and encode categorical variables.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        
        encoders = {}
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
        return df, encoders
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def train_heart_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the heart dataset.
    """
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Model training complete.")
    return model