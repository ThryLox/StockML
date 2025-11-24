"""
FastAPI backend for Antigravity Markets stock analyzer.
Provides endpoints for historical data, predictions, and fundamental analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import sys
import os

# Add current directory to sys.path to allow imports when running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_pipeline import get_history, get_fundamentals, get_stock_info
from model import get_latest_prediction, load_models
from fundamental_analyzer import analyze_fundamentals
from cache import cache

# Initialize FastAPI app
app = FastAPI(
    title="Antigravity Markets API",
    description="Stock prediction and fundamental analysis API",
    version="1.0.0"
)

# Global variable to store loaded models
models = None

@app.on_event("startup")
async def startup_event():
    global models
    try:
        print("Loading pre-trained models...")
        models = load_models()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Server will fall back to training on-the-fly (slower).")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",  # All Vercel deployments
        "https://vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class HistoryResponse(BaseModel):
    symbol: str
    data: list


class PredictionResponse(BaseModel):
    symbol: str
    prob_up: float
    confidence: float
    sentiment: str
    model_predictions: Dict[str, float]
    model_scores: Dict[str, float]
    technical_signals: Dict[str, float]


class FundamentalsResponse(BaseModel):
    symbol: str
    company_info: Dict[str, Any]
    categories: Dict[str, Any]
    growth_metrics: Dict[str, Any]
    summary: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Antigravity Markets API",
        "version": "1.0.0",
        "endpoints": {
            "/history": "Get historical price data",
            "/predict": "Get price prediction",
            "/fundamentals": "Get fundamental analysis",
            "/analysis-summary": "Get analysis summary",
        },
        "disclaimer": "This is for educational purposes only. Not financial advice."
    }


@app.get("/history")
async def get_stock_history(symbol: str, period: str = "5y"):
    """
    Get historical OHLCV data for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        Historical price data
    """
    try:
        symbol = symbol.upper().strip()
        data = get_history(symbol, period)
        return data
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict")
async def predict_stock(symbol: str):
    """
    Get ensemble prediction for stock price movement.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
    
    Returns:
        Prediction with probability, confidence, and model breakdown
    """
    try:
        symbol = symbol.upper().strip()
        
        # Get historical data
        history_data = get_history(symbol, period="5y")
        
        # Convert to DataFrame
        df = pd.DataFrame(history_data['data'])
        df['Date'] = pd.to_datetime(df['date'])
        df = df.set_index('Date')
        
        # Capitalize column names to match model expectations
        df.columns = [col.capitalize() for col in df.columns]
        
        # Get prediction using loaded models
        prediction = get_latest_prediction(df, models)
        
        return {
            "symbol": symbol,
            **prediction
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/fundamentals")
async def get_stock_fundamentals(symbol: str):
    """
    Get fundamental analysis for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
    
    Returns:
        Comprehensive fundamental analysis with ratios and insights
    """
    try:
        symbol = symbol.upper().strip()
        
        # Get fundamental data
        fundamentals = get_fundamentals(symbol)
        
        # Analyze fundamentals
        analysis = analyze_fundamentals(fundamentals)
        
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analysis-summary")
async def get_analysis_summary(symbol: str):
    """
    Get a quick summary of both technical and fundamental analysis.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
    
    Returns:
        Combined analysis summary
    """
    try:
        symbol = symbol.upper().strip()
        
        # Get stock info
        info = get_stock_info(symbol)
        
        # Get fundamental analysis
        fundamentals = get_fundamentals(symbol)
        analysis = analyze_fundamentals(fundamentals)
        
        # Get prediction (lightweight version)
        try:
            history_data = get_history(symbol, period="1y")  # Use 1 year for faster processing
            df = pd.DataFrame(history_data['data'])
            df['Date'] = pd.to_datetime(df['date'])
            df = df.set_index('Date')
            df.columns = [col.capitalize() for col in df.columns]
            
            prediction = get_latest_prediction(df, models)
            technical_summary = {
                "prediction": prediction['sentiment'],
                "probability": prediction['prob_up'],
                "confidence": prediction['confidence']
            }
        except Exception as pred_error:
            technical_summary = {
                "prediction": "N/A",
                "probability": None,
                "confidence": None,
                "error": str(pred_error)
            }
        
        return {
            "symbol": symbol,
            "company_name": info.get('name'),
            "current_price": info.get('current_price'),
            "technical_analysis": technical_summary,
            "fundamental_analysis": analysis['summary'],
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return cache.get_stats()


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached data."""
    cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Antigravity Markets API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
