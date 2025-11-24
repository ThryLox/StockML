import sys
import os

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import get_history
from model import train_all_models, save_models
import pandas as pd

def train_and_save(symbol: str = "SPY"):
    """
    Train models on a representative stock (e.g., SPY) and save them.
    Using a general market index like SPY allows us to have a pre-trained model
    that can provide 'general' market sentiment predictions instantly,
    avoiding the 10s+ latency of training per-request.
    """
    print(f"Fetching data for {symbol}...")
    try:
        # Fetch 5 years of data
        data = get_history(symbol, period="5y")
        df = pd.DataFrame(data['data'])
        df['Date'] = pd.to_datetime(df['date'])
        df = df.set_index('Date')
        df.columns = [col.capitalize() for col in df.columns]
        
        print(f"Training models on {len(df)} data points...")
        models = train_all_models(df)
        
        print("Saving models...")
        save_models(models)
        print("Done!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_and_save()
