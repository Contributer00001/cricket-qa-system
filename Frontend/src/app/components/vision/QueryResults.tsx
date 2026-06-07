import {
  CheckCircle2, TrendingUp, Sparkles, Target,
  BarChart3, Zap, RotateCcw, Tag, Clock, Cpu,
} from 'lucide-react';
import { QueryResult } from '../VisionProCricketPlatform';

interface QueryResultsProps {
  result: QueryResult;
  onReset: () => void;
}

// ── Intents whose DuckDB response should render as a ranked table ─────────

const TABLE_INTENTS = new Set([
  'top_run_scorers',
  'top_wicket_takers',
  'points_table',
  'powerplay_stats',
  'most_sixes',
  'most_fours',
  'best_economy_rates',
  'best_batting_average',
  'death_overs_stats',
]);

// ── Helpers ───────────────────────────────────────────────────────────────

function numericRows(stats: Record<string, string>) {
  return Object.entries(stats)
    .filter(([, v]) => !isNaN(parseFloat(v)))
    .map(([label, value]) => ({ label, value: parseFloat(value), raw: value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);
}

function primaryMetricKey(intent: string): string {
  const map: Record<string, string> = {
    top_run_scorers:    'runs',
    top_wicket_takers:  'wickets',
    most_sixes:         'sixes',
    most_fours:         'fours',
    best_economy_rates: 'economy',
    best_batting_average: 'average',
    points_table:       'points',
    powerplay_stats:    'avg_runs',
    death_overs_stats:  'avg_runs',
  };
  return map[intent] ?? 'value';
}

function nameKey(row: Record<string, any>): string {
  return row.player ?? row.team ?? row.abbr ?? '—';
}

const BAR_GRADIENTS = [
  'from-cyan-500 to-blue-500',
  'from-blue-500 to-purple-500',
  'from-purple-500 to-pink-500',
  'from-pink-500 to-rose-500',
  'from-rose-500 to-orange-500',
  'from-orange-500 to-amber-500',
];

// ── Ranked table component ────────────────────────────────────────────────

function RankedTable({ data, intent }: { data: any[]; intent: string }) {
  const metricKey = primaryMetricKey(intent);
  const maxVal = Math.max(...data.map(r => Number(r[metricKey]) || 0), 1);

  return (
    <div className="relative group">
      <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl blur opacity-20 group-hover:opacity-40 transition-opacity" />
      <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h4 className="text-white font-semibold">Rankings</h4>
          <BarChart3 className="w-5 h-5 text-cyan-400" />
        </div>
        <div className="space-y-3">
          {data.slice(0, 5).map((row, i) => {
            const name = nameKey(row);
            const metric = Number(row[metricKey]) || 0;
            const team = row.team ?? row.abbr ?? '';
            return (
              <div key={i}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-[10px] font-mono text-slate-500 w-4 flex-shrink-0">
                      #{i + 1}
                    </span>
                    <span className="text-xs text-slate-200 truncate max-w-[140px]">{name}</span>
                    {team && name !== team && (
                      <span className="text-[10px] text-slate-500 flex-shrink-0">{team}</span>
                    )}
                  </div>
                  <span className="text-xs font-bold text-white ml-2 flex-shrink-0">
                    {metric} <span className="text-slate-500 font-normal">{metricKey}</span>
                  </span>
                </div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-gradient-to-r ${BAR_GRADIENTS[i % BAR_GRADIENTS.length]} rounded-full transition-all duration-700`}
                    style={{ width: `${Math.max((metric / maxVal) * 100, 3)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────

export function QueryResults({ result, onReset }: QueryResultsProps) {
  const rows = numericRows(result.stats);
  const maxVal = rows.length > 0 ? Math.max(...rows.map(r => r.value)) : 1;

  const isAvailable = !result.answer.toLowerCase().includes('not available');
  const speedLabel = result.response_time_ms < 5
    ? `${result.response_time_ms}ms ⚡`
    : result.response_time_ms < 100
    ? `${result.response_time_ms}ms`
    : `${(result.response_time_ms / 1000).toFixed(2)}s`;

  // Ranked-table mode: DuckDB array response for known list intents
  const isTableMode =
    TABLE_INTENTS.has(result.intent) &&
    Array.isArray(result.data) &&
    (result.data as any[]).length > 0;

  return (
    <div className="max-w-6xl mx-auto space-y-6 mb-16">

      {/* ── Main Answer Card ──────────────────────────────────────────── */}
      <div className="relative group">
        <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-3xl blur-xl opacity-30 group-hover:opacity-50 transition-opacity" />
        <div className="relative bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 shadow-2xl">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              <div className="p-3 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl">
                <Sparkles className="w-7 h-7 text-white" />
              </div>
            </div>
            <div className="flex-1 min-w-0">

              {/* Intent Badge */}
              <div className="flex items-center gap-2 mb-3">
                <Tag className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                  {result.intentLabel}
                </span>
              </div>

              {/* Answer */}
              <div className={`text-5xl font-black mb-4 tracking-tight ${
                isAvailable
                  ? 'bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent'
                  : 'text-slate-400'
              }`}>
                {isAvailable ? result.answer : '—'}
              </div>

              {!isAvailable && (
                <p className="text-slate-400 text-sm mb-4">
                  The requested statistic is not available in the current data.
                </p>
              )}

              {/* Metric Bar */}
              <div className="flex flex-wrap items-center gap-6 pt-4 border-t border-white/10">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs text-slate-400">Confidence</span>
                  <span className="text-sm font-bold text-white">{result.confidence}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-blue-400" />
                  <span className="text-xs text-slate-400">Response</span>
                  <span className="text-sm font-bold text-white">{speedLabel}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  <span className="text-xs text-slate-400">Mode</span>
                  <span className="text-sm font-bold text-white">
                    {result.llm_used ? 'LLM + Grounding' : 'Deterministic'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  <span className="text-xs text-slate-400">Grounded</span>
                </div>
              </div>
            </div>

            {/* Reset Button */}
            <button
              onClick={onReset}
              className="flex-shrink-0 p-2 hover:bg-white/10 rounded-xl transition-colors group/btn"
              title="New query"
            >
              <RotateCcw className="w-5 h-5 text-slate-400 group-hover/btn:text-white transition-colors" />
            </button>
          </div>
        </div>
      </div>

      {/* ── Ranked Table (DuckDB list responses) ──────────────────────── */}
      {isTableMode && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <RankedTable data={result.data as any[]} intent={result.intent} />

          {/* Insights */}
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-5">
                <h4 className="text-white font-semibold">Insights</h4>
                <TrendingUp className="w-5 h-5 text-purple-400" />
              </div>
              <div className="space-y-3">
                {result.insights.map((insight, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-1 w-1.5 h-1.5 rounded-full bg-purple-400" />
                    <p className="text-slate-300 text-sm leading-relaxed">{insight}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Stats Visualization (per-match numeric context) ───────────── */}
      {!isTableMode && rows.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* Bar Chart */}
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl blur opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-5">
                <h4 className="text-white font-semibold">Computed Statistics</h4>
                <BarChart3 className="w-5 h-5 text-cyan-400" />
              </div>
              <div className="space-y-3">
                {rows.map(({ label, value, raw }, i) => (
                  <div key={label}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400 truncate max-w-[160px]">{label}</span>
                      <span className="text-xs font-bold text-white ml-2">{raw}</span>
                    </div>
                    <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r ${BAR_GRADIENTS[i % BAR_GRADIENTS.length]} rounded-full transition-all duration-700`}
                        style={{ width: `${Math.max((value / maxVal) * 100, 3)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Insights Grid */}
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-20 group-hover:opacity-40 transition-opacity" />
            <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-5">
                <h4 className="text-white font-semibold">Insights</h4>
                <TrendingUp className="w-5 h-5 text-purple-400" />
              </div>
              <div className="space-y-3">
                {result.insights.map((insight, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-1 w-1.5 h-1.5 rounded-full bg-purple-400" />
                    <p className="text-slate-300 text-sm leading-relaxed">{insight}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Pipeline Trace ────────────────────────────────────────────── */}
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-slate-600 to-slate-700 rounded-2xl blur opacity-20" />
        <div className="relative bg-white/3 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-slate-400" />
              <h4 className="text-slate-300 font-semibold text-sm">Pipeline Output</h4>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-500">
              <span>QueryRouter → StatsEngine/DuckDB → Formatter</span>
            </div>
          </div>
          <pre className="text-xs font-mono text-slate-400 whitespace-pre-wrap leading-relaxed overflow-x-auto">
{result.context_used}

{`Intent   : ${result.intentLabel}
Confidence: ${result.confidence}%
Latency  : ${speedLabel}
LLM used : ${result.llm_used}`}
          </pre>
        </div>
      </div>

      <div className="text-center">
        <p className="text-xs text-slate-600">
          Answers computed deterministically from DuckDB / ball-by-ball commentary • No hallucinations
        </p>
      </div>
    </div>
  );
}
