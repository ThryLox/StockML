"""
Data pipeline for fetching stock data using yfinance.
Provides both historical price data and fundamental analysis data.
"""

import yfinance as yf
import pandas as pd
import time
import math
from typing import Dict, Any, List
from cache import cache


def clean_numeric_data(data: Any) -> Any:
    """
    Recursively clean NaN and Inf values from data structures.
    Replaces NaN/Inf with None for JSON compatibility.
    """
    if isinstance(data, dict):
        return {k: clean_numeric_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_numeric_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


def get_history(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """Fetch historical OHLCV data for a given stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        period: Time period for historical data (default: '5y')
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    Returns:
        Dictionary containing symbol and historical data as list of records.
    """
    cache_key = f"history:{symbol.upper()}:{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    max_retries = 3
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)  # small delay before Yahoo call
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            hist = hist.reset_index()
            data = []
            for _, row in hist.iterrows():
                data.append({
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
            result = {"symbol": symbol.upper(), "data": data}
            cache.set(cache_key, result, 300)  # 5 minutes TTL
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Error fetching history for {symbol}: {str(e)}")


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Fetch fundamental data for a given stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Dictionary containing company info and financial data.
    """
    cache_key = f"fundamentals:{symbol.upper()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    max_retries = 3
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow

            fundamentals = {
                "symbol": symbol.upper(),
                "company_info": {
                    "name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "market_cap": info.get("marketCap"),
                    "employees": info.get("fullTimeEmployees"),
                },
                "valuation": {
                    "current_price": info.get("currentPrice"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                },
                "profitability": {
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "gross_margin": info.get("grossMargins"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                },
                "liquidity": {
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    "cash": info.get("totalCash"),
                    "cash_per_share": info.get("totalCashPerShare"),
                },
                "leverage": {
                    "debt_to_equity": info.get("debtToEquity"),
                    "total_debt": info.get("totalDebt"),
                    "total_cash": info.get("totalCash"),
                    "interest_coverage": None,
                },
                "efficiency": {
                    "asset_turnover": None,
                    "inventory_turnover": None,
                    "receivables_turnover": None,
                },
                "growth": {
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue": info.get("totalRevenue"),
                    "earnings": info.get("netIncomeToCommon"),
                },
                "financial_statements": {
                    "has_financials": not financials.empty if financials is not None else False,
                    "has_balance_sheet": not balance_sheet.empty if balance_sheet is not None else False,
                    "has_cashflow": not cashflow.empty if cashflow is not None else False,
                },
            }

            # Additional ratio calculations
            if not financials.empty and not balance_sheet.empty:
                try:
                    latest_financials = financials.iloc[:, 0]
                    latest_balance = balance_sheet.iloc[:, 0]
                    if "EBIT" in latest_financials.index and "Interest Expense" in latest_financials.index:
                        ebit = latest_financials.get("EBIT", 0)
                        interest_expense = latest_financials.get("Interest Expense", 0)
                        # Interest expense can be reported as negative or positive
                        # We want absolute value: EBIT / |Interest Expense|
                        if interest_expense != 0 and ebit != 0:
                            fundamentals["leverage"]["interest_coverage"] = float(ebit / abs(interest_expense))
                    if "Total Revenue" in latest_financials.index and "Total Assets" in latest_balance.index:
                        revenue = latest_financials.get("Total Revenue", 0)
                        assets = latest_balance.get("Total Assets", 1)
                        if assets > 0:
                            fundamentals["efficiency"]["asset_turnover"] = float(revenue / assets)
                    if "Cost Of Revenue" in latest_financials.index and "Inventory" in latest_balance.index:
                        cogs = latest_financials.get("Cost Of Revenue", 0)
                        inventory = latest_balance.get("Inventory", 1)
                        if inventory > 0:
                            fundamentals["efficiency"]["inventory_turnover"] = float(cogs / inventory)
                except Exception as calc_error:
                    print(f"Error calculating additional ratios: {calc_error}")

            # Clean NaN/Inf values before returning
            fundamentals = clean_numeric_data(fundamentals)
            
            cache.set(cache_key, fundamentals, 900)  # 15 minutes TTL
            return fundamentals
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Error fetching fundamentals for {symbol}: {str(e)}")


def get_stock_info(symbol: str) -> Dict[str, Any]:
    """Get basic stock information.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with basic stock info.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName", "N/A"),
            "current_price": info.get("currentPrice"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception as e:
        raise Exception(f"Error fetching info for {symbol}: {str(e)}")
