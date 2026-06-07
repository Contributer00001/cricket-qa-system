import {
  TrendingUp, Target, Layers, BarChart3,
  Zap, Activity, Award, Circle,
} from 'lucide-react';

interface CapabilityCardsProps {
  onSelectPrompt: (prompt: string) => void;
}

export function CapabilityCards({ onSelectPrompt }: CapabilityCardsProps) {
  const capabilities = [
    {
      icon: TrendingUp,
      title: 'Top Run Scorers',
      description: 'Tournament batting leaders ranked by total runs',
      prompt: 'Who scored the most runs in IPL 2022?',
      gradient: 'from-cyan-500 to-blue-500',
      tag: 'top_run_scorers',
    },
    {
      icon: Target,
      title: 'Top Wicket Takers',
      description: 'Purple Cap race — leading wicket-takers',
      prompt: 'Who took the most wickets in IPL 2022?',
      gradient: 'from-blue-500 to-purple-500',
      tag: 'top_wicket_takers',
    },
    {
      icon: Layers,
      title: 'Points Table',
      description: 'Final IPL 2022 standings with wins and NRR',
      prompt: 'Show me the IPL 2022 points table',
      gradient: 'from-purple-500 to-pink-500',
      tag: 'points_table',
    },
    {
      icon: BarChart3,
      title: 'Highest Team Score',
      description: 'The biggest innings total of the tournament',
      prompt: 'What is the highest team score in IPL 2022?',
      gradient: 'from-pink-500 to-rose-500',
      tag: 'highest_team_score',
    },
    {
      icon: Zap,
      title: 'Most Sixes',
      description: 'Six-hitting leaders across all 74 matches',
      prompt: 'Who hit the most sixes in IPL 2022?',
      gradient: 'from-rose-500 to-orange-500',
      tag: 'most_sixes',
    },
    {
      icon: Activity,
      title: 'Best Economy',
      description: 'Bowlers who conceded the fewest runs per over',
      prompt: 'Which bowler had the best economy rate?',
      gradient: 'from-orange-500 to-amber-500',
      tag: 'best_economy_rates',
    },
    {
      icon: Award,
      title: 'Highest Innings',
      description: 'The greatest individual batting performance',
      prompt: 'Who scored the highest individual innings?',
      gradient: 'from-amber-500 to-yellow-500',
      tag: 'highest_individual_score',
    },
    {
      icon: Circle,
      title: 'Total Sixes',
      description: 'Tournament-wide six count across all deliveries',
      prompt: 'How many total sixes were hit in IPL 2022?',
      gradient: 'from-yellow-500 to-green-500',
      tag: 'tournament_sixes',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto mb-16">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">What would you like to analyse?</h2>
        <p className="text-slate-400">
          Select a category or type your own question below
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {capabilities.map((cap, index) => {
          const Icon = cap.icon;
          return (
            <button
              key={index}
              onClick={() => onSelectPrompt(cap.prompt)}
              className="group relative text-left"
            >
              {/* Hover Glow */}
              <div
                className={`absolute -inset-0.5 bg-gradient-to-r ${cap.gradient} rounded-2xl blur opacity-0 group-hover:opacity-30 transition-opacity`}
              />

              {/* Card */}
              <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all h-full">
                {/* Icon */}
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${cap.gradient} mb-4`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>

                {/* Title */}
                <h3 className="text-base font-semibold text-white mb-1">{cap.title}</h3>
                <p className="text-xs text-slate-400 mb-4 leading-relaxed">{cap.description}</p>

                {/* Prompt preview */}
                <div className="bg-white/5 border border-white/5 rounded-lg p-3">
                  <p className="text-xs text-slate-300 italic">"{cap.prompt}"</p>
                </div>

                {/* Intent tag */}
                <div className="mt-3">
                  <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">
                    {cap.tag}
                  </span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
