# 🚀 Antigravity Markets

> **Advanced AI-Powered Stock Analysis & Prediction Platform**

Antigravity Markets is a modern web application that combines real-time financial data with advanced machine learning to provide actionable stock market insights. It features interactive charts, fundamental health checks, and ensemble AI predictions.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Antigravity+Markets+Dashboard)

## ✨ Features

- **📈 Interactive Price Charts**: Visualize 5 years of historical OHLCV data with zoom and pan capabilities.
- **🤖 AI-Powered Predictions**: 
  - **Ensemble Architecture**: Combines **Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM**, and **LSTM** (Deep Learning).
  - **Instant Inference**: Pre-trained models (on SPY) provide sub-second predictions for any stock.
  - **Sentiment Analysis**: Bullish/Bearish signals with confidence scores.
- **mj Fundamental Analysis**:
  - Automated health checks for Profitability, Liquidity, Leverage, and Efficiency.
  - "Strengths" vs "Concerns" categorization.
- **⚡ High Performance**:
  - **Model Persistence**: Models are loaded into memory on startup for instant scoring.
  - **Caching**: Intelligent caching of API responses to handle rate limits.

## 🛠️ Tech Stack

### **Frontend**
- **Framework**: [Next.js 14](https://nextjs.org/) (React)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Charts**: [Recharts](https://recharts.org/)
- **Icons**: [Lucide React](https://lucide.dev/)

### **Backend**
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **ML Libraries**: `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow` (Keras)
- **Data Source**: `yfinance` (Yahoo Finance API)
- **Data Processing**: `pandas`, `numpy`, `ta` (Technical Analysis Library)

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18+)
- **Python** (v3.9+) or **Anaconda** (Recommended)

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**Important**: If you are on Windows, we recommend using **Anaconda** to avoid dependency issues with libraries like `numpy` and `tensorflow`.

Run the server:
```bash
# The server will start on http://localhost:8000
python -m uvicorn main:app --reload
```
*On startup, the backend will load the pre-trained models from the `models/` directory.*

### 2. Frontend Setup

Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📂 Project Structure

```
stock/
├── backend/
│   ├── main.py              # FastAPI entry point & API endpoints
│   ├── model.py             # ML model definitions (LSTM, XGBoost, etc.)
│   ├── train_models.py      # Script to train and save models
│   ├── data_pipeline.py     # Data fetching & processing (yfinance)
│   ├── fundamental_analyzer.py # Financial ratio analysis logic
│   ├── cache.py             # In-memory caching system
│   └── models/              # Directory containing saved .joblib/.keras models
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main dashboard page
│   │   └── components/      # React components (PriceChart, PredictionCard, etc.)
│   └── public/
└── README.md
```

## 🧠 Model Architecture

The application uses a **Transfer Learning** approach for stock prediction:
1.  **Training**: Models are trained on **SPY** (S&P 500 ETF) data to learn general market dynamics.
2.  **Persistence**: The trained ensemble (Logistic, RF, XGB, LGBM, LSTM) is saved to disk.
3.  **Inference**: When a user requests a stock (e.g., AAPL), the backend calculates technical indicators for AAPL and feeds them into the pre-loaded SPY models.
4.  **Benefit**: This allows for **instant predictions** (<1s) without the need to retrain models for every single request (~10s latency).

## ☁️ Deployment

### Backend (Render)
1. Create a new **Web Service** on Render.
2. Connect your repository.
3. Use the following settings:
   - **Runtime**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables (if any).

*Note: The `build.sh` script installs dependencies and trains the models so they are available when the server starts.*

### Frontend (Vercel)
1. Import the project into Vercel.
2. Set the **Root Directory** to `frontend`.
3. Deploy!

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. The predictions and analysis provided by the AI models are probabilistic and should **not** be taken as financial advice. Always do your own due diligence before making investment decisions.
