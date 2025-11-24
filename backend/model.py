"""
Advanced multi-model prediction system with ensemble approach.
Combines 5 different ML models for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Technical Analysis
import ta


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive feature set from historical price data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Price-based features (returns)
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)
    
    # Moving Averages
    df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # Price relative to moving averages
    df['price_to_sma10'] = df['Close'] / df['sma_10'] - 1
    df['price_to_sma20'] = df['Close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['Close'] / df['sma_50'] - 1
    
    # RSI (Relative Strength Index)
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # Volume indicators
    df['volume_change'] = df['Volume'].pct_change(1)
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # On-Balance Volume
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['obv_change'] = df['obv'].pct_change(5)
    
    # Volatility
    df['volatility'] = df['return_1d'].rolling(window=20).std()
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Momentum
    df['stochastic'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['roc'] = ta.momentum.roc(df['Close'], window=10)
    
    # Target: 1 if next day's return is positive, 0 otherwise
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df


def prepare_data(df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with features
        lookback: Number of days for LSTM lookback
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Select feature columns (exclude target and non-feature columns)
    exclude_cols = ['target', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                   'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                   'bb_high', 'bb_low', 'bb_mid', 'obv', 'volume_sma_20']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Drop rows with NaN values
    df_clean = df[feature_cols + ['target']].dropna()
    
    X = df_clean[feature_cols].values
    y = df_clean['target'].values
    
    return X, y, feature_cols


def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
    """Train Logistic Regression model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Cross-validation score
    cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    
    return (model, scaler), cv_score


def train_random_forest(X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    return model, cv_score


def train_xgboost(X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
    """Train XGBoost model."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X, y)
    
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    return model, cv_score


def train_lightgbm(X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
    """Train LightGBM model."""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X, y)
    
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    return model, cv_score


def train_lstm(X: np.ndarray, y: np.ndarray, lookback: int = 60) -> Tuple[Any, float]:
    """
    Train LSTM neural network.
    
    Args:
        X: Feature array
        y: Target array
        lookback: Number of timesteps to look back
    
    Returns:
        Tuple of (model, scaler, accuracy)
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for LSTM
    X_lstm = []
    y_lstm = []
    
    for i in range(lookback, len(X_scaled)):
        X_lstm.append(X_scaled[i-lookback:i])
        y_lstm.append(y[i])
    
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, X.shape[1])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with validation split
    history = model.fit(
        X_lstm, y_lstm,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Get final validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]
    
    return (model, scaler, lookback), val_accuracy


def train_all_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train all models and return them with their scores.
    
    Args:
        df: DataFrame with historical data
    
    Returns:
        Dictionary containing all trained models and their scores
    """
    # Build features
    df_features = build_features(df)
    
    # Prepare data
    X, y, feature_names = prepare_data(df_features)
    
    if len(X) < 100:
        raise ValueError("Insufficient data for training. Need at least 100 data points.")
    
    print(f"Training models with {len(X)} samples and {len(feature_names)} features...")
    
    # Train all models
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    models['logistic'], models['logistic_score'] = train_logistic_regression(X, y)
    
    # Random Forest
    print("Training Random Forest...")
    models['random_forest'], models['random_forest_score'] = train_random_forest(X, y)
    
    # XGBoost
    print("Training XGBoost...")
    models['xgboost'], models['xgboost_score'] = train_xgboost(X, y)
    
    # LightGBM
    print("Training LightGBM...")
    models['lightgbm'], models['lightgbm_score'] = train_lightgbm(X, y)
    
    # LSTM
    print("Training LSTM...")
    models['lstm'], models['lstm_score'] = train_lstm(X, y)
    
    # Store feature names for later use
    models['feature_names'] = feature_names
    models['df_features'] = df_features
    
    print("All models trained successfully!")
    
    return models


def ensemble_prediction(df: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make ensemble prediction using all trained models.
    
    Args:
        df: DataFrame with historical data
        models: Dictionary containing all trained models
    
    Returns:
        Dictionary with ensemble prediction and individual model predictions
    """
    # Build features for the latest data
    df_features = build_features(df)
    
    # Get the latest row (most recent data)
    X, y, _ = prepare_data(df_features)
    
    if len(X) == 0:
        raise ValueError("No valid data for prediction")
    
    # Get latest features
    X_latest = X[-1:]
    
    # Individual model predictions
    predictions = {}
    
    # Logistic Regression
    lr_model, lr_scaler = models['logistic']
    X_scaled = lr_scaler.transform(X_latest)
    predictions['logistic'] = float(lr_model.predict_proba(X_scaled)[0, 1])
    
    # Random Forest
    predictions['random_forest'] = float(models['random_forest'].predict_proba(X_latest)[0, 1])
    
    # XGBoost
    predictions['xgboost'] = float(models['xgboost'].predict_proba(X_latest)[0, 1])
    
    # LightGBM
    predictions['lightgbm'] = float(models['lightgbm'].predict_proba(X_latest)[0, 1])
    
    # LSTM
    lstm_model, lstm_scaler, lookback = models['lstm']
    X_scaled_all = lstm_scaler.transform(X)
    
    if len(X_scaled_all) >= lookback:
        X_lstm = X_scaled_all[-lookback:].reshape(1, lookback, -1)
        predictions['lstm'] = float(lstm_model.predict(X_lstm, verbose=0)[0, 0])
    else:
        # If not enough data for LSTM, use average of other models
        predictions['lstm'] = np.mean([predictions['logistic'], predictions['random_forest'],
                                       predictions['xgboost'], predictions['lightgbm']])
    
    # Weighted ensemble
    weights = {
        'lstm': 0.30,
        'xgboost': 0.25,
        'lightgbm': 0.20,
        'random_forest': 0.15,
        'logistic': 0.10
    }
    
    ensemble_prob = sum(predictions[model] * weights[model] for model in weights.keys())
    
    # Calculate confidence based on model agreement
    pred_values = list(predictions.values())
    confidence = 1.0 - np.std(pred_values)  # Lower std = higher confidence
    
    # Get technical signals
    latest_features = df_features.iloc[-1]
    technical_signals = {
        'rsi': float(latest_features.get('rsi', 0)),
        'macd': float(latest_features.get('macd', 0)),
        'macd_signal': float(latest_features.get('macd_signal', 0)),
        'bb_position': float(latest_features.get('bb_position', 0.5)),
        'volume_ratio': float(latest_features.get('volume_ratio', 1.0)),
    }
    
    # Determine sentiment
    if ensemble_prob > 0.65:
        sentiment = "Bullish"
    elif ensemble_prob < 0.35:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    
    return {
        'prob_up': ensemble_prob,
        'confidence': confidence,
        'sentiment': sentiment,
        'model_predictions': predictions,
        'model_scores': {
            'logistic': models['logistic_score'],
            'random_forest': models['random_forest_score'],
            'xgboost': models['xgboost_score'],
            'lightgbm': models['lightgbm_score'],
            'lstm': models['lstm_score']
        },
        'technical_signals': technical_signals
    }


def save_models(models: Dict[str, Any], directory: str = "models"):
    """
    Save trained models to disk.
    
    Args:
        models: Dictionary containing trained models
        directory: Directory to save models
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save sklearn/xgboost/lightgbm models
    joblib.dump(models['logistic'], os.path.join(directory, 'logistic.joblib'))
    joblib.dump(models['random_forest'], os.path.join(directory, 'random_forest.joblib'))
    joblib.dump(models['xgboost'], os.path.join(directory, 'xgboost.joblib'))
    joblib.dump(models['lightgbm'], os.path.join(directory, 'lightgbm.joblib'))
    
    # Save LSTM model
    lstm_model, lstm_scaler, lookback = models['lstm']
    lstm_model.save(os.path.join(directory, 'lstm.keras'))
    joblib.dump(lstm_scaler, os.path.join(directory, 'lstm_scaler.joblib'))
    
    # Save metadata
    metadata = {
        'feature_names': models['feature_names'],
        'scores': {
            'logistic': models['logistic_score'],
            'random_forest': models['random_forest_score'],
            'xgboost': models['xgboost_score'],
            'lightgbm': models['lightgbm_score'],
            'lstm': models['lstm_score']
        },
        'lookback': lookback
    }
    joblib.dump(metadata, os.path.join(directory, 'metadata.joblib'))
    print(f"Models saved to {directory}")


def load_models(directory: str = "models") -> Dict[str, Any]:
    """
    Load trained models from disk.
    
    Args:
        directory: Directory to load models from
    
    Returns:
        Dictionary containing loaded models
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Model directory {directory} not found")
        
    models = {}
    
    # Load sklearn/xgboost/lightgbm models
    models['logistic'] = joblib.load(os.path.join(directory, 'logistic.joblib'))
    models['random_forest'] = joblib.load(os.path.join(directory, 'random_forest.joblib'))
    models['xgboost'] = joblib.load(os.path.join(directory, 'xgboost.joblib'))
    models['lightgbm'] = joblib.load(os.path.join(directory, 'lightgbm.joblib'))
    
    # Load LSTM model
    lstm_model = keras.models.load_model(os.path.join(directory, 'lstm.keras'))
    lstm_scaler = joblib.load(os.path.join(directory, 'lstm_scaler.joblib'))
    
    # Load metadata
    metadata = joblib.load(os.path.join(directory, 'metadata.joblib'))
    
    models['lstm'] = (lstm_model, lstm_scaler, metadata['lookback'])
    models['feature_names'] = metadata['feature_names']
    
    # Restore scores
    models['logistic_score'] = metadata['scores']['logistic']
    models['random_forest_score'] = metadata['scores']['random_forest']
    models['xgboost_score'] = metadata['scores']['xgboost']
    models['lightgbm_score'] = metadata['scores']['lightgbm']
    models['lstm_score'] = metadata['scores']['lstm']
    
    return models


def get_latest_prediction(df: pd.DataFrame, models: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get prediction for the latest data using pre-trained models.
    
    Args:
        df: DataFrame with historical OHLCV data
        models: Optional dictionary with pre-trained models. If None, trains new ones.
    
    Returns:
        Dictionary with prediction results
    """
    if models is None:
        # Fallback to training on the fly (slow)
        print("No models provided, training from scratch...")
        models = train_all_models(df)
    
    # Get ensemble prediction
    prediction = ensemble_prediction(df, models)
    
    return prediction
