"use client";

import { useState } from "react";
import { Search, Loader2, AlertCircle } from "lucide-react";
import axios from "axios";
import PriceChart from "./PriceChart";
import PredictionCard from "./PredictionCard";
import FundamentalsCard from "./FundamentalsCard";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function StockDashboard() {
    const [symbol, setSymbol] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [data, setData] = useState<any>(null);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!symbol) return;

        setLoading(true);
        setError("");
        setData(null);

        try {
            // Fetch all data in parallel
            const [historyRes, predictRes, fundamentalsRes] = await Promise.all([
                axios.get(`${API_URL}/history?symbol=${symbol}`),
                axios.get(`${API_URL}/predict?symbol=${symbol}`),
                axios.get(`${API_URL}/fundamentals?symbol=${symbol}`),
            ]);

            setData({
                history: historyRes.data.data,
                prediction: predictRes.data,
                fundamentals: fundamentalsRes.data,
            });
        } catch (err: any) {
            console.error("Error fetching data:", err);
            setError(
                err.response?.data?.detail ||
                err.message ||
                "Failed to fetch stock data. Please check the symbol and try again."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-zinc-50 px-4 py-8 dark:bg-black sm:px-6 lg:px-8">
            <div className="mx-auto max-w-7xl">
                <div className="mb-8 text-center">
                    <h1 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 sm:text-4xl">
                        Antigravity Markets
                    </h1>
                    <p className="mt-2 text-lg text-zinc-600 dark:text-zinc-400">
                        AI-Powered Stock Analysis & Prediction
                    </p>
                </div>

                <div className="mx-auto mb-12 max-w-xl">
                    <form onSubmit={handleSearch} className="relative flex items-center">
                        <Search className="absolute left-4 h-5 w-5 text-zinc-400" />
                        <input
                            type="text"
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value)}
                            placeholder="Enter stock symbol (e.g., AAPL, TSLA)"
                            className="h-12 w-full rounded-full border border-zinc-200 bg-white pl-12 pr-4 text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100"
                        />
                        <button
                            type="submit"
                            disabled={loading || !symbol}
                            className="absolute right-2 rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
                        >
                            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Analyze"}
                        </button>
                    </form>
                    {error && (
                        <div className="mt-4 flex items-center justify-center gap-2 text-sm text-red-500">
                            <AlertCircle className="h-4 w-4" />
                            {error}
                        </div>
                    )}
                </div>

                {data && (
                    <div className="space-y-6">
                        <div className="grid gap-6 lg:grid-cols-3">
                            <div className="lg:col-span-2">
                                <PriceChart data={data.history} symbol={symbol.toUpperCase()} />
                            </div>
                            <div>
                                <PredictionCard
                                    prediction={data.prediction}
                                    symbol={symbol.toUpperCase()}
                                />
                            </div>
                        </div>

                        <FundamentalsCard data={data.fundamentals} />
                    </div>
                )}
            </div>
        </div>
    );
}
