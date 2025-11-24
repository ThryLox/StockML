"use client";

interface Ratio {
    name: string;
    value: number;
    formatted: string;
    status: string;
    description: string;
}

interface CategoryAnalysis {
    category: string;
    ratios: Ratio[];
    overall_health: string;
}

interface FundamentalsData {
    symbol: string;
    company_info: any;
    categories: {
        valuation: CategoryAnalysis;
        profitability: CategoryAnalysis;
        liquidity: CategoryAnalysis;
        leverage: CategoryAnalysis;
        efficiency: CategoryAnalysis;
    };
    growth_metrics: {
        metrics: Ratio[];
        overall_health: string;
    };
    summary: {
        overall_assessment: string;
        health_score: string;
        strengths: string[];
        concerns: string[];
        insights: string[];
        recommendation: string;
    };
}

interface FundamentalsCardProps {
    data: FundamentalsData;
}

export default function FundamentalsCard({ data }: FundamentalsCardProps) {
    if (!data) return null;

    const { summary, categories, growth_metrics } = data;

    const healthColor =
        summary.health_score === "strong" ? "text-green-500" :
            summary.health_score === "weak" ? "text-red-500" : "text-yellow-500";

    // Helper to find a metric value by name
    const getMetricValue = (ratios: Ratio[], name: string): string => {
        const ratio = ratios?.find(r => r.name.includes(name));
        return ratio ? ratio.formatted : "N/A";
    };

    return (
        <div className="rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="mb-6">
                <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                    Fundamental Analysis
                </h3>
                <p className={`mt-1 text-sm font-medium ${healthColor}`}>
                    {summary.overall_assessment}
                </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <div>
                    <h4 className="mb-3 text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        Key Strengths
                    </h4>
                    <ul className="space-y-2">
                        {summary.strengths.map((strength, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-zinc-600 dark:text-zinc-400">
                                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-green-500" />
                                {strength}
                            </li>
                        ))}
                    </ul>
                </div>

                <div>
                    <h4 className="mb-3 text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        Concerns
                    </h4>
                    <ul className="space-y-2">
                        {summary.concerns.length > 0 ? (
                            summary.concerns.map((concern, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm text-zinc-600 dark:text-zinc-400">
                                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-500" />
                                    {concern}
                                </li>
                            ))
                        ) : (
                            <li className="text-sm text-zinc-500">No major concerns identified</li>
                        )}
                    </ul>
                </div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-4 border-t border-zinc-100 pt-6 dark:border-zinc-800 sm:grid-cols-4">
                <div>
                    <p className="text-xs text-zinc-500">P/E Ratio</p>
                    <p className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                        {getMetricValue(categories.valuation.ratios, "P/E Ratio")}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-zinc-500">Profit Margin</p>
                    <p className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                        {getMetricValue(categories.profitability.ratios, "Profit Margin")}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-zinc-500">ROE</p>
                    <p className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                        {getMetricValue(categories.profitability.ratios, "Return on Equity")}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-zinc-500">Rev Growth</p>
                    <p className="mt-1 font-medium text-zinc-900 dark:text-zinc-100">
                        {getMetricValue(growth_metrics.metrics, "Revenue Growth")}
                    </p>
                </div>
            </div>
        </div>
    );
}
