"use client";

import { ArrowDown, ArrowUp, Minus } from "lucide-react";

interface PredictionData {
    prob_up: number;
    confidence: number;
    sentiment: string;
    model_predictions: Record<string, number>;
}

interface PredictionCardProps {
    prediction: PredictionData;
    symbol: string;
}

export default function PredictionCard({ prediction, symbol }: PredictionCardProps) {
    if (!prediction) return null;

    const isBullish = prediction.sentiment === "Bullish";
    const isBearish = prediction.sentiment === "Bearish";

    const sentimentColor = isBullish
        ? "text-green-500"
        : isBearish
            ? "text-red-500"
            : "text-yellow-500";

    const sentimentBg = isBullish
        ? "bg-green-500/10"
        : isBearish
            ? "bg-red-500/10"
            : "bg-yellow-500/10";

    const Icon = isBullish ? ArrowUp : isBearish ? ArrowDown : Minus;

    return (
        <div className="rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-6 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                    AI Prediction
                </h3>
                <div className={`flex items-center gap-2 rounded-full px-3 py-1 ${sentimentBg}`}>
                    <Icon className={`h-4 w-4 ${sentimentColor}`} />
                    <span className={`text-sm font-medium ${sentimentColor}`}>
                        {prediction.sentiment}
                    </span>
                </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                    <p className="text-sm text-zinc-500">Probability Up</p>
                    <div className="flex items-end gap-2">
                        <span className="text-3xl font-bold text-zinc-900 dark:text-zinc-100">
                            {(prediction.prob_up * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
                        <div
                            className={`h-full rounded-full ${isBullish ? "bg-green-500" : isBearish ? "bg-red-500" : "bg-yellow-500"
                                }`}
                            style={{ width: `${prediction.prob_up * 100}%` }}
                        />
                    </div>
                </div>

                <div className="space-y-2">
                    <p className="text-sm text-zinc-500">Model Confidence</p>
                    <div className="flex items-end gap-2">
                        <span className="text-3xl font-bold text-zinc-900 dark:text-zinc-100">
                            {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
                        <div
                            className="h-full rounded-full bg-blue-500"
                            style={{ width: `${prediction.confidence * 100}%` }}
                        />
                    </div>
                </div>
            </div>

            <div className="mt-6">
                <h4 className="mb-3 text-sm font-medium text-zinc-900 dark:text-zinc-100">
                    Model Consensus
                </h4>
                <div className="space-y-3">
                    {Object.entries(prediction.model_predictions).map(([model, prob]) => (
                        <div key={model} className="flex items-center justify-between text-sm">
                            <span className="capitalize text-zinc-500">
                                {model.replace("_", " ")}
                            </span>
                            <div className="flex items-center gap-3">
                                <div className="h-1.5 w-24 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
                                    <div
                                        className={`h-full rounded-full ${prob > 0.5 ? "bg-green-500" : "bg-red-500"
                                            }`}
                                        style={{ width: `${prob * 100}%` }}
                                    />
                                </div>
                                <span className="w-12 text-right font-medium text-zinc-700 dark:text-zinc-300">
                                    {(prob * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
