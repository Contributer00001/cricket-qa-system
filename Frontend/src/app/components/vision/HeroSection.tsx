import { useEffect, useState } from 'react';
import { Sparkles, Network, Loader2 } from 'lucide-react';
import { checkHealth, fetchIPLStats, IPLStats } from '../../api/client';

interface HeroSectionProps {
  onShowArchitecture: () => void;
}

type BackendStatus = 'checking' | 'online' | 'offline';

export function HeroSection({ onShowArchitecture }: HeroSectionProps) {
  const [status, setStatus] = useState<BackendStatus>('checking');
  const [iplStats, setIplStats] = useState<IPLStats | null>(null);

  useEffect(() => {
    let mounted = true;

    const poll = async () => {
      try {
        await checkHealth();
        if (mounted) setStatus('online');
      } catch {
        if (mounted) setStatus('offline');
      }
    };

    poll();
    const id = setInterval(poll, 15_000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    fetchIPLStats()
      .then(stats => setIplStats(stats))
      .catch(() => setIplStats(null));
  }, []);

  const statusDot = {
    checking: <Loader2 className="w-3 h-3 text-slate-400 animate-spin" />,
    online:   <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse block" />,
    offline:  <span className="w-2 h-2 bg-red-400 rounded-full block" />,
  }[status];

  const statusText = {
    checking: 'Connecting…',
    online:   'Backend Online',
    offline:  'Backend Offline — start the service on port 8000',
  }[status];

  // DuckDB badge content
  const dbBadgeContent = iplStats === null
    ? { text: 'Loading…', color: 'text-slate-400', dot: null }
    : iplStats.db_loaded
    ? {
        text: `${iplStats.deliveries.toLocaleString()} Deliveries · ${iplStats.matches} Matches`,
        color: 'text-slate-300',
        dot: <span className="w-2 h-2 bg-green-400 rounded-full block" />,
      }
    : {
        text: 'DuckDB not loaded',
        color: 'text-amber-400',
        dot: <span className="w-2 h-2 bg-amber-400 rounded-full block" />,
      };

  return (
    <div className="relative py-12 mb-8">
      <div className="container mx-auto px-4">
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center gap-3 mb-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 blur-xl opacity-50" />
              <div className="relative bg-gradient-to-br from-blue-500/20 to-purple-500/20 backdrop-blur-xl border border-white/10 rounded-2xl p-4">
                <Sparkles className="w-10 h-10 text-cyan-400" />
              </div>
            </div>
          </div>

          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent tracking-tight">
            Cricket Intelligence
          </h1>

          <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-6">
            Deterministic analytics powered by a query-understanding pipeline and grounded AI
          </p>

          {/* Status Badges */}
          <div className="flex items-center justify-center gap-4 flex-wrap">

            {/* Live backend status */}
            <div className="px-4 py-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-full">
              <div className="flex items-center gap-2">
                {statusDot}
                <span className={`text-sm ${
                  status === 'online'
                    ? 'text-slate-300'
                    : status === 'offline'
                    ? 'text-red-400'
                    : 'text-slate-500'
                }`}>
                  {statusText}
                </span>
              </div>
            </div>

            {/* DuckDB / deliveries badge */}
            <div className={`px-4 py-2 bg-white/5 backdrop-blur-xl border rounded-full ${
              iplStats?.db_loaded === false
                ? 'border-amber-500/30'
                : 'border-white/10'
            }`}>
              <div className="flex items-center gap-2">
                {dbBadgeContent.dot}
                <span className={`text-sm ${dbBadgeContent.color}`}>
                  {dbBadgeContent.text}
                </span>
              </div>
            </div>

            <div className="px-4 py-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-full">
              <span className="text-sm text-slate-300">100% Benchmark Accuracy</span>
            </div>

            <button
              onClick={onShowArchitecture}
              className="px-4 py-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-full hover:bg-white/10 transition-all group"
            >
              <div className="flex items-center gap-2">
                <Network className="w-4 h-4 text-cyan-400 group-hover:rotate-180 transition-transform duration-500" />
                <span className="text-sm text-slate-300">View AI Architecture</span>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
