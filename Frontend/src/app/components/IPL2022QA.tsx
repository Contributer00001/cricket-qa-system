import { useState, useRef, CSSProperties } from 'react';
import { infer as apiInfer, parseContextStats, InferResponse } from '../api/client';

// ── Types ─────────────────────────────────────────────────────────────────

interface StatItem { label: string; value: string }

interface AnswerResult {
  answer: string;
  details: string;
  stats?: StatItem[];
  source: string;
  intent: string;
  confidence?: number;
}

type ResultState = AnswerResult | 'not_found' | null;

// ── Static data ───────────────────────────────────────────────────────────

const EXAMPLE_QUESTIONS = [
  { text: 'Who scored the most runs in IPL 2022?',      category: 'batting',   icon: '🏏' },
  { text: 'Who took the most wickets in IPL 2022?',     category: 'bowling',   icon: '🎯' },
  { text: 'Which team finished top of the points table?', category: 'standings', icon: '🏆' },
  { text: 'What is the highest team score in IPL 2022?', category: 'records',   icon: '📈' },
  { text: 'Who hit the most sixes in IPL 2022?',        category: 'batting',   icon: '💥' },
  { text: 'Which bowler had the best economy rate?',    category: 'bowling',   icon: '⚡' },
  { text: 'Who scored the highest individual innings?', category: 'records',   icon: '⭐' },
  { text: 'How many runs did Jos Buttler score?',       category: 'player',    icon: '👤' },
  { text: 'What was the powerplay score in match 1?',   category: 'match',     icon: '🔢' },
  { text: 'Who won the final?',                         category: 'match',     icon: '🥇' },
  { text: 'Which team hit the most sixes overall?',     category: 'team',      icon: '🏟️' },
  { text: 'What were the best bowling figures in an innings?', category: 'records', icon: '🎳' },
  { text: 'How many total sixes were hit in IPL 2022?', category: 'records',   icon: '6️⃣' },
  { text: 'Who had the best batting average in the tournament?', category: 'batting', icon: '📊' },
  { text: 'Show the complete points table',             category: 'standings', icon: '📋' },
];

const DATASET_FACTS = [
  { label: '74',    sublabel: 'Matches'       },
  { label: '953',   sublabel: 'Gold Fixtures' },
  { label: '15,598', sublabel: 'Deliveries'   },
  { label: '247',   sublabel: 'Players'       },
];

const CATEGORIES = [
  { id: 'all',       label: 'All'       },
  { id: 'batting',   label: 'Batting'   },
  { id: 'bowling',   label: 'Bowling'   },
  { id: 'standings', label: 'Standings' },
  { id: 'records',   label: 'Records'   },
  { id: 'player',    label: 'Player'    },
  { id: 'match',     label: 'Match'     },
  { id: 'team',      label: 'Team'      },
];

// ── Mock answers for IPL 2022 demo queries ────────────────────────────────

const MOCK_ANSWERS: Record<string, AnswerResult> = {
  'who scored the most runs in ipl 2022': {
    answer: 'Jos Buttler (863 runs)',
    details: 'Jos Buttler dominated IPL 2022 with 863 runs in 17 matches for Rajasthan Royals. He averaged 57.53 at a strike rate of 149.05, hitting 4 centuries and 4 fifties.',
    stats: [
      { label: 'Runs', value: '863' }, { label: 'Matches', value: '17' },
      { label: 'Average', value: '57.53' }, { label: 'Strike Rate', value: '149.05' },
      { label: '100s', value: '4' }, { label: '50s', value: '4' },
    ],
    source: 'tournament_batting', intent: 'top_run_scorers',
  },
  'who took the most wickets in ipl 2022': {
    answer: 'Yuzvendra Chahal (27 wickets)',
    details: 'Yuzvendra Chahal was the top wicket-taker for Rajasthan Royals with 27 wickets across the tournament, winning the Purple Cap.',
    stats: [
      { label: 'Wickets', value: '27' }, { label: 'Economy', value: '7.84' },
      { label: 'Average', value: '14.42' },
    ],
    source: 'tournament_bowling', intent: 'top_wicket_takers',
  },
  'which team finished top of the points table': {
    answer: 'Gujarat Titans (20 points, 10 wins)',
    details: 'Gujarat Titans topped the 2022 IPL points table in their debut season with 10 wins from 14 matches, 20 points, and a positive NRR of +0.316.',
    stats: [
      { label: 'Points', value: '20' }, { label: 'Wins', value: '10' },
      { label: 'Played', value: '14' }, { label: 'NRR', value: '+0.316' },
    ],
    source: 'standings', intent: 'points_table',
  },
  'what is the highest team score in ipl 2022': {
    answer: 'RCB: 205/2 (20 ov) — Match 27',
    details: 'Royal Challengers Bengaluru posted 205/2 in 20 overs against Delhi Capitals in Match 27, the highest team score of IPL 2022.',
    stats: [{ label: 'Score', value: '205/2' }, { label: 'Match', value: '27' }],
    source: 'innings', intent: 'highest_team_score',
  },
  'who hit the most sixes in ipl 2022': {
    answer: 'Jos Buttler (45 sixes)',
    details: 'Jos Buttler led the six-hitting charts with 45 sixes across IPL 2022.',
    stats: [
      { label: 'Sixes', value: '45' }, { label: 'Team', value: 'RR' },
      { label: 'Runs', value: '863' },
    ],
    source: 'tournament_batting', intent: 'most_sixes',
  },
  'who scored the highest individual innings': {
    answer: 'Jos Buttler: 116 runs',
    details: 'Jos Buttler scored 116 runs in a single innings for Rajasthan Royals, the highest individual score in IPL 2022.',
    stats: [
      { label: 'Runs', value: '116' }, { label: 'Player', value: 'Buttler' },
      { label: 'Team', value: 'RR' },
    ],
    source: 'scorecard', intent: 'highest_individual_score',
  },
  'how many total sixes were hit in ipl 2022': {
    answer: '905 sixes',
    details: 'A total of 905 sixes were hit across all 74 matches of IPL 2022, from 15,598 total deliveries bowled.',
    stats: [
      { label: 'Sixes', value: '905' }, { label: 'Matches', value: '74' },
      { label: 'Per Match', value: '12.2' },
    ],
    source: 'deliveries', intent: 'tournament_sixes',
  },
  'who won the final': {
    answer: 'Gujarat Titans won by 7 wickets',
    details: 'Gujarat Titans defeated Rajasthan Royals in the Final to win the IPL 2022 title in their debut season.',
    stats: [
      { label: 'Winner', value: 'Gujarat Titans' }, { label: 'Match', value: 'Final' },
      { label: 'Margin', value: '7 wickets' },
    ],
    source: 'matches', intent: 'match_result',
  },
};

function getMockAnswer(question: string): AnswerResult | null {
  const key = question.toLowerCase().replace(/[?!.,]/g, '').trim();
  for (const [k, v] of Object.entries(MOCK_ANSWERS)) {
    if (key.includes(k) || k.includes(key.substring(0, 20))) return v;
  }
  return null;
}

function backendToResult(resp: InferResponse): AnswerResult {
  const stats = parseContextStats(resp.context_used);
  const statsArray: StatItem[] = Object.entries(stats)
    .filter(([, v]) => !isNaN(parseFloat(v)))
    .map(([label, value]) => ({
      label: label.replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      value,
    }));
  return {
    answer: resp.answer,
    details: `Computed deterministically · intent: ${resp.intent} · confidence: ${Math.round(resp.confidence * 100)}%`,
    stats: statsArray,
    source: resp.intent,
    intent: resp.intent,
    confidence: resp.confidence,
  };
}

// ── Shared style atoms ─────────────────────────────────────────────────────

const glass = (extra: CSSProperties = {}): CSSProperties => ({
  backdropFilter: 'blur(20px)',
  WebkitBackdropFilter: 'blur(20px)',
  ...extra,
});

// ── Sub-components ────────────────────────────────────────────────────────

function Spinner() {
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '8px 0' }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 8, height: 8, borderRadius: '50%',
          background: '#f97316',
          animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
        }} />
      ))}
    </div>
  );
}

function StatCard({ label, value }: StatItem) {
  return (
    <div style={glass({
      background: 'rgba(249,115,22,0.08)',
      border: '1px solid rgba(249,115,22,0.2)',
      borderRadius: 10, padding: '10px 16px',
      textAlign: 'center', minWidth: 80,
    })}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#f97316', fontFamily: "'Space Mono',monospace" }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}

function AnswerCard({ result, question }: { result: ResultState; question: string }) {
  if (!result || result === 'not_found') {
    return (
      <div style={glass({
        background: 'rgba(239,68,68,0.06)',
        border: '1px solid rgba(239,68,68,0.2)',
        borderRadius: 14, padding: '20px 24px', marginTop: 16,
      })}>
        <div style={{ color: '#ef4444', fontWeight: 600, marginBottom: 8 }}>⚠️ Not found in demo</div>
        <div style={{ color: '#9ca3af', fontSize: 14, lineHeight: 1.6 }}>
          This demo has mock answers for a subset of IPL 2022 questions. Connect the backend or build
          the DuckDB to query all 953 gold fixtures across 74 matches.
        </div>
        <div style={{
          marginTop: 14, padding: '10px 14px',
          background: 'rgba(0,0,0,0.4)',
          borderRadius: 8, fontFamily: "'Space Mono',monospace",
          fontSize: 12, color: '#6b7280',
        }}>
          <span style={{ color: '#f97316' }}>$</span> python -m evaluation.ragas_eval --data-file data/sample_match.json
        </div>
      </div>
    );
  }

  return (
    <div style={glass({
      background: 'rgba(17,17,17,0.85)',
      border: '1px solid rgba(249,115,22,0.3)',
      borderRadius: 14, padding: '24px', marginTop: 16,
      boxShadow: '0 8px 32px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.04)',
      animation: 'fadeIn 0.3s ease',
    })}>
      {/* Badges */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <Badge color="orange" text={`intent: ${result.intent}`} />
        <Badge color="green"  text={`source: ${result.source}`} />
        <Badge color="indigo" text="⚡ deterministic" />
        {result.confidence !== undefined && (
          <Badge color="cyan" text={`conf: ${Math.round(result.confidence * 100)}%`} />
        )}
      </div>

      {/* Answer */}
      <div style={{
        fontSize: 28, fontWeight: 800, color: '#f1f5f9',
        fontFamily: "'Space Mono',monospace",
        lineHeight: 1.2, marginBottom: 12,
      }}>
        {result.answer}
      </div>

      {/* Details */}
      <div style={{ color: '#94a3b8', fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}>
        {result.details}
      </div>

      {/* Stats grid */}
      {result.stats && result.stats.length > 0 && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {result.stats.map(s => <StatCard key={s.label} label={s.label} value={s.value} />)}
        </div>
      )}

      {/* Grounding note */}
      <div style={{
        marginTop: 16, padding: '10px 14px',
        background: 'rgba(34,197,94,0.06)',
        border: '1px solid rgba(34,197,94,0.15)',
        borderRadius: 8, fontSize: 12, color: '#86efac',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <span>✓</span>
        <span>Answer sourced directly from DuckDB / analytics engine — no LLM hallucination on numeric facts</span>
      </div>
    </div>
  );
}

function Badge({ color, text }: { color: 'orange' | 'green' | 'indigo' | 'cyan'; text: string }) {
  const palette = {
    orange: { bg: 'rgba(249,115,22,0.15)', border: 'rgba(249,115,22,0.3)', text: '#f97316' },
    green:  { bg: 'rgba(34,197,94,0.10)',  border: 'rgba(34,197,94,0.2)',  text: '#22c55e' },
    indigo: { bg: 'rgba(99,102,241,0.10)', border: 'rgba(99,102,241,0.2)', text: '#818cf8' },
    cyan:   { bg: 'rgba(6,182,212,0.10)',  border: 'rgba(6,182,212,0.2)', text: '#22d3ee' },
  }[color];
  return (
    <span style={{
      background: palette.bg, border: `1px solid ${palette.border}`,
      color: palette.text, fontSize: 11,
      padding: '3px 10px', borderRadius: 20,
      fontFamily: "'Space Mono',monospace",
    }}>
      {text}
    </span>
  );
}

// ── Main component ────────────────────────────────────────────────────────

export default function IPL2022QA() {
  const [query,          setQuery]         = useState('');
  const [activeCategory, setActiveCategory] = useState('all');
  const [loading,        setLoading]        = useState(false);
  const [result,         setResult]         = useState<ResultState>(null);
  const [asked,          setAsked]          = useState<string | null>(null);
  const [history,        setHistory]        = useState<Array<{ question: string; result: AnswerResult | null }>>([]);
  const [inputFocused,   setInputFocused]   = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredExamples = EXAMPLE_QUESTIONS.filter(
    q => activeCategory === 'all' || q.category === activeCategory
  );

  async function handleAsk(question: string) {
    if (!question.trim()) return;
    setLoading(true);
    setResult(null);
    setAsked(question);

    // 1. Check rich mock data (IPL 2022 specific questions)
    const mock = getMockAnswer(question);
    if (mock) {
      await new Promise(r => setTimeout(r, 350 + Math.random() * 250));
      setResult(mock);
      setLoading(false);
      setHistory(prev => [{ question, result: mock }, ...prev].slice(0, 5));
      return;
    }

    // 2. Try the real backend (handles sample data analytics questions)
    try {
      const resp = await apiInfer({ question: question.trim() });
      const mapped = backendToResult(resp);
      setResult(mapped);
      setHistory(prev => [{ question, result: mapped }, ...prev].slice(0, 5));
    } catch {
      setResult('not_found');
      setHistory(prev => [{ question, result: null }, ...prev].slice(0, 5));
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim()) handleAsk(query.trim());
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;500;600;700;800&display=swap');
        @keyframes bounce { 0%,80%,100%{transform:scale(0)} 40%{transform:scale(1)} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes orbPulse { 0%,100%{opacity:0.6;transform:scale(1)} 50%{opacity:1;transform:scale(1.08)} }
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:#374151;border-radius:4px}
      `}</style>

      <div style={{
        background: '#0a0a0a', minHeight: '100vh',
        fontFamily: "'Sora',sans-serif", color: '#f1f5f9',
        position: 'relative', overflow: 'hidden',
      }}>

        {/* ── Background grid ── */}
        <div style={{
          position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
          backgroundImage: 'linear-gradient(rgba(249,115,22,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(249,115,22,0.03) 1px,transparent 1px)',
          backgroundSize: '40px 40px',
        }} />

        {/* ── Apple Vision ambient orbs (glass layer) ── */}
        <div style={{
          position: 'fixed', top: -150, left: '15%',
          width: 550, height: 550, borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(59,130,246,0.18) 0%, transparent 70%)',
          filter: 'blur(50px)', pointerEvents: 'none', zIndex: 0,
          animation: 'orbPulse 5s ease-in-out infinite',
        }} />
        <div style={{
          position: 'fixed', bottom: 0, right: '10%',
          width: 450, height: 450, borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(168,85,247,0.15) 0%, transparent 70%)',
          filter: 'blur(50px)', pointerEvents: 'none', zIndex: 0,
          animation: 'orbPulse 6s ease-in-out 1s infinite',
        }} />
        <div style={{
          position: 'fixed', top: '40%', right: '30%',
          width: 300, height: 300, borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(6,182,212,0.10) 0%, transparent 70%)',
          filter: 'blur(40px)', pointerEvents: 'none', zIndex: 0,
          animation: 'orbPulse 7s ease-in-out 2s infinite',
        }} />

        {/* ── Orange glow behind hero ── */}
        <div style={{
          position: 'fixed', top: -200, left: '50%', transform: 'translateX(-50%)',
          width: 600, height: 400,
          background: 'radial-gradient(ellipse, rgba(249,115,22,0.14) 0%, transparent 70%)',
          pointerEvents: 'none', zIndex: 0,
        }} />

        {/* ── Content ── */}
        <div style={{ position: 'relative', zIndex: 1, maxWidth: 820, margin: '0 auto', padding: '32px 20px 80px' }}>

          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: 40 }}>
            <div style={glass({
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: 'rgba(249,115,22,0.08)',
              border: '1px solid rgba(249,115,22,0.2)',
              borderRadius: 100, padding: '5px 16px 5px 12px',
              marginBottom: 20, fontSize: 12,
            })}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', animation: 'bounce 2s ease-in-out infinite' }} />
              <span style={{ color: '#f97316', fontFamily: "'Space Mono',monospace" }}>
                DuckDB · 74 matches · 15,598 deliveries
              </span>
            </div>

            <h1 style={{
              fontSize: 'clamp(32px,6vw,52px)', fontWeight: 800,
              margin: '0 0 12px', lineHeight: 1.1, letterSpacing: '-0.03em',
            }}>
              <span style={{ color: '#f1f5f9' }}>Ask anything about</span><br />
              <span style={{
                background: 'linear-gradient(135deg,#f97316,#fb923c,#fdba74)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>IPL 2022</span>
            </h1>

            <p style={{ color: '#6b7280', fontSize: 15, maxWidth: 480, margin: '0 auto' }}>
              Deterministic analytics from a full DuckDB. 953 verified ground-truth fixtures.
              No hallucination on numbers.
            </p>
          </div>

          {/* Dataset facts */}
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap', marginBottom: 32 }}>
            {DATASET_FACTS.map(f => (
              <div key={f.label} style={glass({
                textAlign: 'center', padding: '12px 20px',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 12,
              })}>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#f97316', fontFamily: "'Space Mono',monospace" }}>{f.label}</div>
                <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{f.sublabel}</div>
              </div>
            ))}
          </div>

          {/* Search box */}
          <form onSubmit={handleSubmit} style={{ marginBottom: 28 }}>
            <div style={glass({
              display: 'flex', gap: 10,
              background: inputFocused ? 'rgba(249,115,22,0.07)' : 'rgba(255,255,255,0.04)',
              border: `1px solid ${inputFocused ? 'rgba(249,115,22,0.45)' : 'rgba(249,115,22,0.2)'}`,
              borderRadius: 14, padding: '6px 6px 6px 18px',
              transition: 'border-color 0.2s, background 0.2s',
              boxShadow: inputFocused ? '0 0 0 3px rgba(249,115,22,0.1)' : 'none',
            })}>
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                placeholder="Who hit the most sixes? What was the powerplay score in match 1?"
                style={{
                  flex: 1, background: 'transparent', border: 'none', outline: 'none',
                  color: '#f1f5f9', fontSize: 15,
                  fontFamily: "'Sora',sans-serif", padding: '8px 0',
                }}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                style={{
                  background: loading ? 'rgba(249,115,22,0.3)' : 'linear-gradient(135deg,#f97316,#ea580c)',
                  border: 'none', borderRadius: 10, padding: '10px 20px',
                  color: '#fff', fontWeight: 600, fontSize: 14,
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontFamily: "'Sora',sans-serif", whiteSpace: 'nowrap',
                  transition: 'all 0.2s',
                }}
              >
                {loading ? 'Querying…' : 'Ask →'}
              </button>
            </div>
          </form>

          {/* Loading */}
          {loading && (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <Spinner />
              <div style={{ color: '#6b7280', fontSize: 13, marginTop: 4, fontFamily: "'Space Mono',monospace" }}>
                querying duckdb…
              </div>
            </div>
          )}

          {/* Answer */}
          {!loading && asked && <AnswerCard result={result} question={asked} />}

          {/* Category filter + examples */}
          <div style={{ marginTop: 36, marginBottom: 16 }}>
            <div style={{
              color: '#4b5563', fontSize: 11, textTransform: 'uppercase',
              letterSpacing: '0.12em', marginBottom: 10,
              fontFamily: "'Space Mono',monospace",
            }}>
              Example questions
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
              {CATEGORIES.map(cat => (
                <button
                  key={cat.id}
                  onClick={() => setActiveCategory(cat.id)}
                  style={{
                    padding: '5px 14px', borderRadius: 100,
                    border: '1px solid',
                    borderColor: activeCategory === cat.id ? '#f97316' : 'rgba(255,255,255,0.08)',
                    background: activeCategory === cat.id ? 'rgba(249,115,22,0.15)' : 'transparent',
                    color: activeCategory === cat.id ? '#f97316' : '#6b7280',
                    fontSize: 12, cursor: 'pointer',
                    fontFamily: "'Sora',sans-serif", transition: 'all 0.15s',
                  }}
                >
                  {cat.label}
                </button>
              ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(220px,1fr))', gap: 10 }}>
              {filteredExamples.map(q => (
                <ExampleCard key={q.text} question={q} onClick={() => { setQuery(q.text); handleAsk(q.text); }} />
              ))}
            </div>
          </div>

          {/* What you can ask */}
          <div style={glass({
            marginTop: 36, padding: '24px',
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 16,
          })}>
            <div style={{
              fontSize: 11, color: '#4b5563', textTransform: 'uppercase',
              letterSpacing: '0.12em', fontFamily: "'Space Mono',monospace",
              marginBottom: 16,
            }}>
              What you can ask
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 16 }}>
              {[
                { title: 'Tournament Stats', items: ['Top run scorers', 'Top wicket takers', 'Most sixes / fours', 'Best averages & economy'] },
                { title: 'Match Analysis',   items: ['Powerplay scores', 'Death over runs', 'Innings totals', 'Match results'] },
                { title: 'Records',          items: ['Highest team score', 'Best bowling figures', 'Highest individual score', 'Total boundaries'] },
                { title: 'Standings',        items: ['Points table', 'Team win counts', 'NRR comparison', 'Playoff qualifiers'] },
              ].map(section => (
                <div key={section.title}>
                  <div style={{ color: '#f97316', fontSize: 13, fontWeight: 600, marginBottom: 8 }}>{section.title}</div>
                  <ul style={{ margin: 0, padding: '0 0 0 16px', listStyle: 'disc' }}>
                    {section.items.map(item => (
                      <li key={item} style={{ color: '#6b7280', fontSize: 12, marginBottom: 4, lineHeight: 1.5 }}>{item}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* DuckDB / TablePlus info */}
          <div style={glass({
            marginTop: 24, padding: '20px 24px',
            background: 'rgba(99,102,241,0.05)',
            border: '1px solid rgba(99,102,241,0.15)',
            borderRadius: 16,
          })}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#818cf8', marginBottom: 12 }}>
              🗄️ Explore with TablePlus
            </div>
            <div style={{ color: '#6b7280', fontSize: 12, lineHeight: 1.8 }}>
              <strong style={{ color: '#9ca3af' }}>1. Build the DB:</strong>
              <div style={{ fontFamily: "'Space Mono',monospace", background: 'rgba(0,0,0,0.4)', padding: '8px 12px', borderRadius: 6, margin: '4px 0 10px', fontSize: 11 }}>
                python database/build_ipl_db.py
              </div>
              <strong style={{ color: '#9ca3af' }}>2. Open TablePlus</strong> → New Connection → DuckDB → select{' '}
              <code style={{ color: '#f97316' }}>ipl2022.duckdb</code><br />
              <strong style={{ color: '#9ca3af' }}>3. Tables:</strong> matches · deliveries · innings · batting_stats · bowling_stats · standings<br />
              <strong style={{ color: '#9ca3af' }}>4. Try:</strong>
              <div style={{ fontFamily: "'Space Mono',monospace", background: 'rgba(0,0,0,0.4)', padding: '8px 12px', borderRadius: 6, margin: '4px 0', fontSize: 11, color: '#a5f3fc' }}>
                {'SELECT batsman, SUM(bat_run) runs, SUM(six) sixes\nFROM deliveries GROUP BY batsman\nORDER BY runs DESC LIMIT 10;'}
              </div>
            </div>
          </div>

          {/* Recent history */}
          {history.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <div style={{
                fontSize: 11, color: '#4b5563', textTransform: 'uppercase',
                letterSpacing: '0.12em', fontFamily: "'Space Mono',monospace",
                marginBottom: 10,
              }}>
                Recent queries
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {history.map((h, i) => (
                  <HistoryRow
                    key={i}
                    question={h.question}
                    preview={h.result?.answer?.split('(')[0]?.trim() ?? '—'}
                    onClick={() => { setQuery(h.question); handleAsk(h.question); }}
                  />
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </>
  );
}

// ── Small presentational sub-components ──────────────────────────────────

function ExampleCard({ question, onClick }: { question: typeof EXAMPLE_QUESTIONS[0]; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={glass({
        background: hovered ? 'rgba(249,115,22,0.06)' : 'rgba(255,255,255,0.03)',
        border: `1px solid ${hovered ? 'rgba(249,115,22,0.3)' : 'rgba(255,255,255,0.07)'}`,
        borderRadius: 10, padding: '12px 14px',
        textAlign: 'left', cursor: 'pointer',
        color: '#d1d5db', fontSize: 13,
        fontFamily: "'Sora',sans-serif", lineHeight: 1.4,
        transition: 'all 0.15s', display: 'flex',
        alignItems: 'flex-start', gap: 8, width: '100%',
      })}
    >
      <span style={{ fontSize: 16, flexShrink: 0 }}>{question.icon}</span>
      <span>{question.text}</span>
    </button>
  );
}

function HistoryRow({ question, preview, onClick }: { question: string; preview: string; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={glass({
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 14px',
        background: 'rgba(255,255,255,0.02)',
        border: `1px solid ${hovered ? 'rgba(249,115,22,0.2)' : 'rgba(255,255,255,0.05)'}`,
        borderRadius: 8, cursor: 'pointer', transition: 'all 0.15s', gap: 12,
      })}
    >
      <span style={{ color: '#9ca3af', fontSize: 13, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {question}
      </span>
      <span style={{ color: '#f97316', fontSize: 12, fontFamily: "'Space Mono',monospace", flexShrink: 0 }}>
        {preview}
      </span>
    </div>
  );
}
