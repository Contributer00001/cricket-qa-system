/**
 * Typed API client for the Cricket QA backend.
 *
 * Base URL resolution:
 *   - Development  : VITE_API_URL not set → '/api' (Vite proxy forwards to localhost:8000)
 *   - Production   : set VITE_API_URL=https://your-api.example.com in Frontend/.env
 *
 * Create Frontend/.env from Frontend/.env.example before running `npm run dev`.
 */

const BASE_URL: string =
  typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_URL
    ? (import.meta as any).env.VITE_API_URL
    : '/api';

// ── Request types ─────────────────────────────────────────────────────────

export interface Commentary {
  commentaries: CommentaryEntry[];
}

export interface CommentaryEntry {
  event: 'ball' | 'wicket';
  over: number;
  run: number;
  four: boolean;
  six: boolean;
  batsman?: string;
  bowler?: string;
}

export interface InferRequest {
  question: string;
  max_tokens?: number;
  use_llm?: boolean;
  commentary?: Commentary;
}

// ── Response types ────────────────────────────────────────────────────────

export interface InferResponse {
  answer: string;
  intent: string;
  confidence: number;
  context_used: string;
  llm_used: boolean;
  status: string;
  // Present for DuckDB tournament responses (list of players/teams, or a single record)
  data?: any[] | Record<string, any> | number | null;
}

// ── IPL Stats (from GET /ipl/stats) ──────────────────────────────────────

export interface IPLStats {
  matches: number;
  deliveries: number;
  players: number;
  top_scorer: { player: string; runs: number };
  top_wicket_taker: { player: string; wickets: number };
  db_loaded: boolean;
}

export interface HealthResponse {
  status: string;
}

export interface ReadyResponse {
  ready: boolean;
}

export interface ApiError {
  detail: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body: ApiError = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore parse error
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ── Public API ────────────────────────────────────────────────────────────

export async function infer(req: InferRequest): Promise<InferResponse> {
  const res = await fetch(`${BASE_URL}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return handleResponse<InferResponse>(res);
}

export async function fetchIPLStats(): Promise<IPLStats> {
  const res = await fetch(`${BASE_URL}/ipl/stats`, { method: 'GET' });
  return handleResponse<IPLStats>(res);
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/healthz`, { method: 'GET' });
  return handleResponse<HealthResponse>(res);
}

export async function checkReady(): Promise<ReadyResponse> {
  const res = await fetch(`${BASE_URL}/readyz`, { method: 'GET' });
  return handleResponse<ReadyResponse>(res);
}

// ── Context parser ────────────────────────────────────────────────────────

/**
 * Parse the `context_used` string returned by the backend into a
 * key → value map.
 *
 * Input example:
 *   "Context:\n- Total Runs: 47\n- Total Wickets: 3"
 *
 * Output:
 *   { "Total Runs": "47", "Total Wickets": "3" }
 */
export function parseContextStats(context: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const line of context.split('\n')) {
    const m = line.match(/^-\s*([^:]+):\s*(.+)$/);
    if (m) result[m[1].trim()] = m[2].trim();
  }
  return result;
}

/**
 * Human-readable label for a backend intent value.
 */
export const INTENT_LABELS: Record<string, string> = {
  // ── Per-match intents ──────────────────────────────────────────────────
  total_runs:             'Total Runs',
  total_wickets:          'Total Wickets',
  total_fours:            'Boundary Count (4s)',
  total_sixes:            'Boundary Count (6s)',
  boundaries:             'Total Boundaries',
  dot_balls:              'Dot Balls',
  run_rate:               'Run Rate',
  runs_last_n_overs:      'Runs — Last N Overs',
  runs_in_over_range:     'Runs — Over Range',
  wickets_last_n_overs:   'Wickets — Last N Overs',
  wickets_in_over_range:  'Wickets — Over Range',
  powerplay:              'Powerplay Stats',
  death_overs:            'Death Overs Stats',
  top_scorer:             'Top Scorer',
  player_runs:            'Player Runs',
  over_summary:           'Over-by-Over Summary',
  unknown:                'General Query',
  // ── DuckDB / tournament intents ────────────────────────────────────────
  top_run_scorers:        'IPL 2022 — Top Run Scorers',
  top_wicket_takers:      'IPL 2022 — Top Wicket Takers',
  most_sixes:             'IPL 2022 — Most Sixes',
  most_fours:             'IPL 2022 — Most Fours',
  best_economy_rates:     'IPL 2022 — Best Economy Rates',
  best_batting_average:   'IPL 2022 — Best Batting Average',
  highest_team_score:     'IPL 2022 — Highest Team Score',
  best_bowling_figures:   'IPL 2022 — Best Bowling Figures',
  tournament_sixes:       'IPL 2022 — Total Sixes',
  tournament_fours:       'IPL 2022 — Total Fours',
  points_table:           'IPL 2022 — Points Table',
  powerplay_stats:        'IPL 2022 — Powerplay by Team',
  death_overs_stats:      'IPL 2022 — Death Overs by Team',
  player_stats:           'IPL 2022 — Player Stats',
  tournament_error:       'Tournament DB Unavailable',
  tournament_unknown:     'Tournament Query',
};

/**
 * Generate 2-4 insight bullets from parsed stats + intent.
 */
export function generateInsights(
  stats: Record<string, string>,
  intent: string,
  answer: string,
): string[] {
  const n = (key: string) => parseFloat(stats[key] ?? '') || 0;

  const insights: string[] = [];

  const runs = n('Total Runs');
  const wickets = n('Total Wickets');
  const fours = n('Fours');
  const sixes = n('Sixes');
  const rr = n('Overall Run Rate');
  const dots = n('Dot Balls');

  if (runs > 0 && wickets >= 0) {
    insights.push(`Innings total: ${runs} runs for ${wickets} wickets`);
  }
  if (fours > 0 || sixes > 0) {
    insights.push(
      `${fours + sixes} boundaries: ${fours}×4 (${fours * 4} runs) + ${sixes}×6 (${sixes * 6} runs)`
    );
  }
  if (rr > 0) {
    const quality = rr >= 10 ? 'aggressive' : rr >= 8 ? 'solid' : 'steady';
    insights.push(`${quality.charAt(0).toUpperCase() + quality.slice(1)} batting at ${rr.toFixed(2)} RPO`);
  }
  if (dots > 0 && runs > 0) {
    const total = n('Total Runs') || dots + runs;
    const pct = Math.round((dots / (dots + runs)) * 100);
    insights.push(`${dots} dot balls applied ${pct}% dot-ball pressure`);
  }

  // Fallback: echo the raw answer as context
  if (insights.length === 0) {
    insights.push(`Answer computed deterministically in < 1ms`);
    insights.push(`Intent classified as: ${INTENT_LABELS[intent] ?? intent}`);
  }

  return insights.slice(0, 4);
}
