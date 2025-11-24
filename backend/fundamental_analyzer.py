"""
Fundamental analysis module for calculating and analyzing financial ratios.
Provides comprehensive analysis across 5 categories: Profitability, Liquidity, 
Leverage, Efficiency, and Valuation.
"""

from typing import Dict, List, Any, Optional


def analyze_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze fundamental data and calculate all financial ratios.
    
    Args:
        fundamentals: Dictionary containing fundamental data from data_pipeline
    
    Returns:
        Dictionary with categorized ratios and analysis summary
    """
    
    # Extract data for easier access
    valuation = fundamentals.get("valuation", {})
    profitability = fundamentals.get("profitability", {})
    liquidity = fundamentals.get("liquidity", {})
    leverage = fundamentals.get("leverage", {})
    efficiency = fundamentals.get("efficiency", {})
    growth = fundamentals.get("growth", {})
    
    # Organize ratios by category
    analysis = {
        "symbol": fundamentals.get("symbol"),
        "company_info": fundamentals.get("company_info", {}),
        "categories": {
            "profitability": _analyze_profitability(profitability),
            "liquidity": _analyze_liquidity(liquidity),
            "leverage": _analyze_leverage(leverage),
            "efficiency": _analyze_efficiency(efficiency),
            "valuation": _analyze_valuation(valuation),
        },
        "growth_metrics": _analyze_growth(growth),
        "summary": _generate_summary(profitability, liquidity, leverage, efficiency, valuation, growth)
    }
    
    return analysis


def _analyze_profitability(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze profitability ratios."""
    
    roe = data.get("roe")
    roa = data.get("roa")
    profit_margin = data.get("profit_margin")
    operating_margin = data.get("operating_margin")
    gross_margin = data.get("gross_margin")
    
    ratios = []
    
    if roe is not None:
        ratios.append({
            "name": "Return on Equity (ROE)",
            "value": roe,
            "formatted": f"{roe * 100:.2f}%",
            "status": _get_status(roe, 0.15, 0.10),  # Good > 15%, Moderate > 10%
            "description": "Measures profitability relative to shareholders' equity"
        })
    
    if roa is not None:
        ratios.append({
            "name": "Return on Assets (ROA)",
            "value": roa,
            "formatted": f"{roa * 100:.2f}%",
            "status": _get_status(roa, 0.05, 0.02),  # Good > 5%, Moderate > 2%
            "description": "Measures how efficiently assets generate profit"
        })
    
    if profit_margin is not None:
        ratios.append({
            "name": "Profit Margin",
            "value": profit_margin,
            "formatted": f"{profit_margin * 100:.2f}%",
            "status": _get_status(profit_margin, 0.15, 0.08),  # Good > 15%, Moderate > 8%
            "description": "Net income as percentage of revenue"
        })
    
    if operating_margin is not None:
        ratios.append({
            "name": "Operating Margin",
            "value": operating_margin,
            "formatted": f"{operating_margin * 100:.2f}%",
            "status": _get_status(operating_margin, 0.15, 0.10),  # Good > 15%, Moderate > 10%
            "description": "Operating income as percentage of revenue"
        })
    
    if gross_margin is not None:
        ratios.append({
            "name": "Gross Margin",
            "value": gross_margin,
            "formatted": f"{gross_margin * 100:.2f}%",
            "status": _get_status(gross_margin, 0.40, 0.25),  # Good > 40%, Moderate > 25%
            "description": "Revenue minus cost of goods sold, as percentage"
        })
    
    return {
        "category": "Profitability",
        "ratios": ratios,
        "overall_health": _calculate_category_health(ratios)
    }


def _analyze_liquidity(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze liquidity ratios."""
    
    current_ratio = data.get("current_ratio")
    quick_ratio = data.get("quick_ratio")
    cash = data.get("cash")
    cash_per_share = data.get("cash_per_share")
    
    ratios = []
    
    if current_ratio is not None:
        ratios.append({
            "name": "Current Ratio",
            "value": current_ratio,
            "formatted": f"{current_ratio:.2f}",
            "status": _get_status(current_ratio, 2.0, 1.0),  # Good > 2.0, Moderate > 1.0
            "description": "Current assets divided by current liabilities"
        })
    
    if quick_ratio is not None:
        ratios.append({
            "name": "Quick Ratio",
            "value": quick_ratio,
            "formatted": f"{quick_ratio:.2f}",
            "status": _get_status(quick_ratio, 1.5, 1.0),  # Good > 1.5, Moderate > 1.0
            "description": "Liquid assets divided by current liabilities"
        })
    
    if cash is not None:
        ratios.append({
            "name": "Total Cash",
            "value": cash,
            "formatted": _format_large_number(cash),
            "status": "info",
            "description": "Total cash and cash equivalents"
        })
    
    if cash_per_share is not None:
        ratios.append({
            "name": "Cash Per Share",
            "value": cash_per_share,
            "formatted": f"${cash_per_share:.2f}",
            "status": "info",
            "description": "Cash divided by shares outstanding"
        })
    
    return {
        "category": "Liquidity",
        "ratios": ratios,
        "overall_health": _calculate_category_health(ratios)
    }


def _analyze_leverage(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze leverage/debt ratios."""
    
    debt_to_equity = data.get("debt_to_equity")
    total_debt = data.get("total_debt")
    total_cash = data.get("total_cash")
    interest_coverage = data.get("interest_coverage")
    
    ratios = []
    
    if debt_to_equity is not None:
        # Lower is better for debt ratios
        ratios.append({
            "name": "Debt-to-Equity Ratio",
            "value": debt_to_equity,
            "formatted": f"{debt_to_equity:.2f}",
            "status": _get_status_inverse(debt_to_equity, 0.5, 1.5),  # Good < 0.5, Moderate < 1.5
            "description": "Total debt divided by shareholders' equity"
        })
    
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash
        ratios.append({
            "name": "Net Debt",
            "value": net_debt,
            "formatted": _format_large_number(net_debt),
            "status": _get_status_inverse(net_debt, 0, total_debt * 0.5) if total_debt > 0 else "good",
            "description": "Total debt minus cash"
        })
    
    if interest_coverage is not None:
        ratios.append({
            "name": "Interest Coverage",
            "value": interest_coverage,
            "formatted": f"{interest_coverage:.2f}x",
            "status": _get_status(interest_coverage, 5.0, 2.0),  # Good > 5x, Moderate > 2x
            "description": "EBIT divided by interest expense"
        })
    
    return {
        "category": "Leverage",
        "ratios": ratios,
        "overall_health": _calculate_category_health(ratios)
    }


def _analyze_efficiency(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze efficiency ratios."""
    
    asset_turnover = data.get("asset_turnover")
    inventory_turnover = data.get("inventory_turnover")
    receivables_turnover = data.get("receivables_turnover")
    
    ratios = []
    
    if asset_turnover is not None:
        ratios.append({
            "name": "Asset Turnover",
            "value": asset_turnover,
            "formatted": f"{asset_turnover:.2f}",
            "status": _get_status(asset_turnover, 1.0, 0.5),  # Good > 1.0, Moderate > 0.5
            "description": "Revenue divided by total assets"
        })
    
    if inventory_turnover is not None:
        ratios.append({
            "name": "Inventory Turnover",
            "value": inventory_turnover,
            "formatted": f"{inventory_turnover:.2f}",
            "status": _get_status(inventory_turnover, 8.0, 4.0),  # Good > 8, Moderate > 4
            "description": "Cost of goods sold divided by inventory"
        })
    
    if receivables_turnover is not None:
        ratios.append({
            "name": "Receivables Turnover",
            "value": receivables_turnover,
            "formatted": f"{receivables_turnover:.2f}",
            "status": _get_status(receivables_turnover, 10.0, 5.0),  # Good > 10, Moderate > 5
            "description": "Revenue divided by accounts receivable"
        })
    
    return {
        "category": "Efficiency",
        "ratios": ratios,
        "overall_health": _calculate_category_health(ratios)
    }


def _analyze_valuation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze valuation ratios."""
    
    pe_ratio = data.get("pe_ratio")
    forward_pe = data.get("forward_pe")
    peg_ratio = data.get("peg_ratio")
    price_to_book = data.get("price_to_book")
    price_to_sales = data.get("price_to_sales")
    ev_to_ebitda = data.get("ev_to_ebitda")
    current_price = data.get("current_price")
    
    ratios = []
    
    if current_price is not None:
        ratios.append({
            "name": "Current Price",
            "value": current_price,
            "formatted": f"${current_price:.2f}",
            "status": "info",
            "description": "Current stock price"
        })
    
    if pe_ratio is not None:
        ratios.append({
            "name": "P/E Ratio (Trailing)",
            "value": pe_ratio,
            "formatted": f"{pe_ratio:.2f}",
            "status": _get_valuation_status(pe_ratio, 15, 25),  # Good < 15, Moderate < 25
            "description": "Price divided by earnings per share"
        })
    
    if forward_pe is not None:
        ratios.append({
            "name": "Forward P/E",
            "value": forward_pe,
            "formatted": f"{forward_pe:.2f}",
            "status": _get_valuation_status(forward_pe, 15, 25),
            "description": "Price divided by estimated future earnings"
        })
    
    if peg_ratio is not None:
        ratios.append({
            "name": "PEG Ratio",
            "value": peg_ratio,
            "formatted": f"{peg_ratio:.2f}",
            "status": _get_valuation_status(peg_ratio, 1.0, 2.0),  # Good < 1.0, Moderate < 2.0
            "description": "P/E ratio divided by earnings growth rate"
        })
    
    if price_to_book is not None:
        ratios.append({
            "name": "Price-to-Book",
            "value": price_to_book,
            "formatted": f"{price_to_book:.2f}",
            "status": _get_valuation_status(price_to_book, 3.0, 5.0),
            "description": "Price divided by book value per share"
        })
    
    if price_to_sales is not None:
        ratios.append({
            "name": "Price-to-Sales",
            "value": price_to_sales,
            "formatted": f"{price_to_sales:.2f}",
            "status": _get_valuation_status(price_to_sales, 2.0, 4.0),
            "description": "Market cap divided by total revenue"
        })
    
    if ev_to_ebitda is not None:
        ratios.append({
            "name": "EV/EBITDA",
            "value": ev_to_ebitda,
            "formatted": f"{ev_to_ebitda:.2f}",
            "status": _get_valuation_status(ev_to_ebitda, 10, 15),
            "description": "Enterprise value divided by EBITDA"
        })
    
    return {
        "category": "Valuation",
        "ratios": ratios,
        "overall_health": _calculate_category_health(ratios)
    }


def _analyze_growth(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze growth metrics."""
    
    revenue_growth = data.get("revenue_growth")
    earnings_growth = data.get("earnings_growth")
    
    metrics = []
    
    if revenue_growth is not None:
        metrics.append({
            "name": "Revenue Growth",
            "value": revenue_growth,
            "formatted": f"{revenue_growth * 100:.2f}%",
            "status": _get_status(revenue_growth, 0.15, 0.05),  # Good > 15%, Moderate > 5%
            "description": "Year-over-year revenue growth rate"
        })
    
    if earnings_growth is not None:
        metrics.append({
            "name": "Earnings Growth",
            "value": earnings_growth,
            "formatted": f"{earnings_growth * 100:.2f}%",
            "status": _get_status(earnings_growth, 0.15, 0.05),  # Good > 15%, Moderate > 5%
            "description": "Year-over-year earnings growth rate"
        })
    
    return {
        "metrics": metrics,
        "overall_health": _calculate_category_health(metrics)
    }


def _generate_summary(profitability: Dict, liquidity: Dict, leverage: Dict, 
                     efficiency: Dict, valuation: Dict, growth: Dict) -> Dict[str, Any]:
    """Generate AI-powered analysis summary."""
    
    strengths = []
    concerns = []
    insights = []
    
    # Analyze profitability
    roe = profitability.get("roe")
    profit_margin = profitability.get("profit_margin")
    
    if roe and roe > 0.15:
        strengths.append(f"Strong return on equity ({roe*100:.1f}%)")
    elif roe and roe < 0.05:
        concerns.append(f"Low return on equity ({roe*100:.1f}%)")
    
    if profit_margin and profit_margin > 0.15:
        strengths.append(f"Healthy profit margin ({profit_margin*100:.1f}%)")
    elif profit_margin and profit_margin < 0.05:
        concerns.append(f"Thin profit margin ({profit_margin*100:.1f}%)")
    
    # Analyze liquidity
    current_ratio = liquidity.get("current_ratio")
    
    if current_ratio and current_ratio > 2.0:
        strengths.append(f"Excellent liquidity (current ratio: {current_ratio:.2f})")
    elif current_ratio and current_ratio < 1.0:
        concerns.append(f"Liquidity concerns (current ratio: {current_ratio:.2f})")
    
    # Analyze leverage
    debt_to_equity = leverage.get("debt_to_equity")
    
    if debt_to_equity and debt_to_equity < 0.5:
        strengths.append(f"Low debt levels (D/E: {debt_to_equity:.2f})")
    elif debt_to_equity and debt_to_equity > 2.0:
        concerns.append(f"High debt burden (D/E: {debt_to_equity:.2f})")
    
    # Analyze valuation
    pe_ratio = valuation.get("pe_ratio")
    peg_ratio = valuation.get("peg_ratio")
    
    if peg_ratio and peg_ratio < 1.0:
        insights.append(f"Potentially undervalued based on growth (PEG: {peg_ratio:.2f})")
    elif peg_ratio and peg_ratio > 2.0:
        insights.append(f"May be overvalued relative to growth (PEG: {peg_ratio:.2f})")
    
    if pe_ratio:
        if pe_ratio < 15:
            insights.append(f"Trading at reasonable valuation (P/E: {pe_ratio:.1f})")
        elif pe_ratio > 30:
            insights.append(f"Premium valuation (P/E: {pe_ratio:.1f})")
    
    # Analyze growth
    revenue_growth = growth.get("revenue_growth")
    
    if revenue_growth and revenue_growth > 0.20:
        strengths.append(f"Strong revenue growth ({revenue_growth*100:.1f}%)")
    elif revenue_growth and revenue_growth < 0:
        concerns.append(f"Declining revenue ({revenue_growth*100:.1f}%)")
    
    # Determine overall assessment
    strength_count = len(strengths)
    concern_count = len(concerns)
    
    if strength_count > concern_count * 2:
        overall = "Strong fundamentals with multiple positive indicators"
        health_score = "strong"
    elif concern_count > strength_count * 2:
        overall = "Weak fundamentals with several areas of concern"
        health_score = "weak"
    else:
        overall = "Mixed fundamentals with both strengths and weaknesses"
        health_score = "moderate"
    
    return {
        "overall_assessment": overall,
        "health_score": health_score,
        "strengths": strengths[:5],  # Top 5 strengths
        "concerns": concerns[:5],  # Top 5 concerns
        "insights": insights[:3],  # Top 3 insights
        "recommendation": _generate_recommendation(health_score, strengths, concerns)
    }


def _generate_recommendation(health_score: str, strengths: List[str], concerns: List[str]) -> str:
    """Generate investment recommendation based on analysis."""
    
    if health_score == "strong":
        return "The company shows strong fundamentals across multiple metrics. Consider for long-term investment, but always do additional research."
    elif health_score == "weak":
        return "The company shows concerning fundamentals. Exercise caution and consider waiting for improvement in key metrics."
    else:
        return "The company shows mixed fundamentals. Suitable for investors with moderate risk tolerance. Monitor key metrics closely."


def _get_status(value: float, good_threshold: float, moderate_threshold: float) -> str:
    """Determine status based on value (higher is better)."""
    if value >= good_threshold:
        return "good"
    elif value >= moderate_threshold:
        return "moderate"
    else:
        return "poor"


def _get_status_inverse(value: float, good_threshold: float, moderate_threshold: float) -> str:
    """Determine status based on value (lower is better)."""
    if value <= good_threshold:
        return "good"
    elif value <= moderate_threshold:
        return "moderate"
    else:
        return "poor"


def _get_valuation_status(value: float, good_threshold: float, moderate_threshold: float) -> str:
    """Determine valuation status (lower is generally better for valuation ratios)."""
    if value <= good_threshold:
        return "good"
    elif value <= moderate_threshold:
        return "moderate"
    else:
        return "expensive"


def _calculate_category_health(ratios: List[Dict]) -> str:
    """Calculate overall health for a category based on ratio statuses."""
    if not ratios:
        return "unknown"
    
    status_counts = {"good": 0, "moderate": 0, "poor": 0, "expensive": 0, "info": 0}
    
    for ratio in ratios:
        status = ratio.get("status", "info")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Exclude info items from calculation
    total = len(ratios) - status_counts["info"]
    
    if total == 0:
        return "unknown"
    
    good_ratio = status_counts["good"] / total
    poor_ratio = (status_counts["poor"] + status_counts["expensive"]) / total
    
    if good_ratio >= 0.6:
        return "strong"
    elif poor_ratio >= 0.6:
        return "weak"
    else:
        return "moderate"


def _format_large_number(num: float) -> str:
    """Format large numbers with B/M/K suffixes."""
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"
