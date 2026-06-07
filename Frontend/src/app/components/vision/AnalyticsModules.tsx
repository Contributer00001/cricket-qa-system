import {
  BarChart3, Tag, Zap, CheckCircle2,
  Trophy, Target, MessageSquare, Shield,
} from 'lucide-react';

export function AnalyticsModules() {
  const modules = [
    {
      icon: Tag,
      title: 'Query Router',
      description: 'Rule-based intent classification',
      metric: '15',
      metricLabel: 'Intents',
      gradient: 'from-cyan-500 to-blue-500',
    },
    {
      icon: Zap,
      title: 'Response Speed',
      description: 'Deterministic no-LLM path',
      metric: '< 1ms',
      metricLabel: 'Avg Latency',
      gradient: 'from-blue-500 to-purple-500',
    },
    {
      icon: CheckCircle2,
      title: 'Benchmark',
      description: 'Exact-match on ground-truth dataset',
      metric: '100%',
      metricLabel: 'Exact Match',
      gradient: 'from-purple-500 to-pink-500',
    },
    {
      icon: Shield,
      title: 'Hallucination Rate',
      description: 'Regex-grounded answer extraction',
      metric: '0%',
      metricLabel: 'Hallucination',
      gradient: 'from-pink-500 to-rose-500',
    },
    {
      icon: BarChart3,
      title: 'Stats Engine',
      description: 'Pandas-based analytics',
      metric: '12+',
      metricLabel: 'Stat Methods',
      gradient: 'from-rose-500 to-orange-500',
    },
    {
      icon: Target,
      title: 'Intent Accuracy',
      description: 'Correct routing on benchmark',
      metric: '100%',
      metricLabel: 'Accuracy',
      gradient: 'from-orange-500 to-amber-500',
    },
    {
      icon: MessageSquare,
      title: 'Commentary',
      description: 'Ball-by-ball data events',
      metric: '30',
      metricLabel: 'Deliveries',
      gradient: 'from-amber-500 to-yellow-500',
    },
    {
      icon: Trophy,
      title: 'Test Coverage',
      description: 'Unit + integration tests',
      metric: '62',
      metricLabel: 'Tests Passing',
      gradient: 'from-yellow-500 to-green-500',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-bold text-white mb-2">System Metrics</h2>
        <p className="text-slate-400">Measured on the deterministic pipeline — no mocked numbers</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {modules.map((module, index) => {
          const Icon = module.icon;
          return (
            <div key={index} className="group relative">
              {/* Hover Glow */}
              <div
                className={`absolute -inset-0.5 bg-gradient-to-r ${module.gradient} rounded-2xl blur opacity-0 group-hover:opacity-40 transition-opacity`}
              />

              <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all">
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${module.gradient} mb-4`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>

                <h3 className="text-white font-semibold mb-1 text-sm">{module.title}</h3>
                <p className="text-slate-400 text-xs mb-4">{module.description}</p>

                <div className="border-t border-white/10 pt-3">
                  <div
                    className={`text-2xl font-bold bg-gradient-to-r ${module.gradient} bg-clip-text text-transparent`}
                  >
                    {module.metric}
                  </div>
                  <div className="text-xs text-slate-500">{module.metricLabel}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
