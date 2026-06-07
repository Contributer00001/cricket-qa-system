import { useState, useCallback } from 'react';
import { HeroSection } from './vision/HeroSection';
import { ConversationalInterface } from './vision/ConversationalInterface';
import { CapabilityCards } from './vision/CapabilityCards';
import { AIArchitecture } from './vision/AIArchitecture';
import { AnalyticsModules } from './vision/AnalyticsModules';
import { QueryResults } from './vision/QueryResults';
import {
  infer,
  parseContextStats,
  generateInsights,
  INTENT_LABELS,
} from '../api/client';

export interface QueryResult {
  answer: string;
  intent: string;
  intentLabel: string;
  confidence: number;      // 0–100
  context_used: string;
  llm_used: boolean;
  response_time_ms: number;
  insights: string[];
  stats: Record<string, string>;
  // DuckDB list or record response (null for per-match queries)
  data?: any[] | Record<string, any> | number | null;
}

export function VisionProCricketPlatform() {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showArchitecture, setShowArchitecture] = useState(false);

  const handleQuery = useCallback(async (userQuery: string) => {
    if (!userQuery.trim() || isProcessing) return;

    setIsProcessing(true);
    setResult(null);
    setError(null);

    const t0 = performance.now();

    try {
      const response = await infer({ question: userQuery.trim() });
      const elapsed = Math.round(performance.now() - t0);
      const stats = parseContextStats(response.context_used);

      setResult({
        answer: response.answer,
        intent: response.intent,
        intentLabel: INTENT_LABELS[response.intent] ?? response.intent,
        confidence: Math.round(response.confidence * 100),
        context_used: response.context_used,
        llm_used: response.llm_used,
        response_time_ms: elapsed,
        insights: generateInsights(stats, response.intent, response.answer),
        stats,
        data: response.data ?? null,
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not reach the backend. Make sure the service is running on port 8000.'
      );
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  const handleCapabilitySelect = useCallback((prompt: string) => {
    setQuery(prompt);
    handleQuery(prompt);
  }, [handleQuery]);

  const handleReset = () => {
    setResult(null);
    setError(null);
    setQuery('');
  };

  return (
    <div className="relative min-h-screen bg-[#0a0a0f] overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-[#0a0a0f] via-[#1a1a2e] to-[#16213e]" />

      {/* Ambient Orbs */}
      <div className="fixed top-0 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="fixed bottom-0 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      <div className="fixed top-1/2 left-1/2 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />

      <div className="relative z-10">
        <HeroSection onShowArchitecture={() => setShowArchitecture(!showArchitecture)} />

        <div className="container mx-auto px-4 pb-20">
          <ConversationalInterface
            query={query}
            setQuery={setQuery}
            onSubmit={handleQuery}
            isProcessing={isProcessing}
          />

          {/* Error state */}
          {error && (
            <div className="max-w-4xl mx-auto mb-8">
              <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6 backdrop-blur-xl">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 mt-2 rounded-full bg-red-400 flex-shrink-0" />
                  <div>
                    <p className="text-red-300 font-semibold mb-1">Request Failed</p>
                    <p className="text-red-400/80 text-sm">{error}</p>
                    <button
                      onClick={handleReset}
                      className="mt-3 text-xs text-red-400 underline hover:text-red-300 transition-colors"
                    >
                      Clear and try again
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Idle state: show capability cards + architecture */}
          {!result && !isProcessing && !error && (
            <>
              <CapabilityCards onSelectPrompt={handleCapabilitySelect} />
              {showArchitecture && <AIArchitecture />}
            </>
          )}

          {/* Result state */}
          {result && (
            <div className="space-y-6">
              <QueryResults result={result} onReset={handleReset} />
              {showArchitecture && <AIArchitecture />}
            </div>
          )}

          {!result && !isProcessing && !error && <AnalyticsModules />}
        </div>
      </div>
    </div>
  );
}
