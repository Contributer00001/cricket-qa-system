import { Brain, Database, FileSearch, GitBranch, Sparkles, ArrowRight } from 'lucide-react';

export function AIArchitecture() {
  const stages = [
    {
      icon: Brain,
      title: 'Query Understanding',
      description: 'Natural language processing and intent detection',
      color: 'from-cyan-500 to-blue-500'
    },
    {
      icon: GitBranch,
      title: 'Agent Planning',
      description: 'Multi-step reasoning and tool selection',
      color: 'from-blue-500 to-purple-500'
    },
    {
      icon: Database,
      title: 'Statistical Analysis',
      description: 'SQL query generation and execution',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: FileSearch,
      title: 'Retrieval Systems',
      description: 'RAG-based commentary and context retrieval',
      color: 'from-pink-500 to-rose-500'
    },
    {
      icon: Sparkles,
      title: 'Answer Generation',
      description: 'Synthesis and natural language response',
      color: 'from-rose-500 to-orange-500'
    }
  ];

  return (
    <div className="max-w-7xl mx-auto mb-16">
      {/* Header */}
      <div className="text-center mb-10">
        <h2 className="text-3xl font-bold text-white mb-2">AI Architecture</h2>
        <p className="text-slate-400">How our intelligent system processes your queries</p>
      </div>

      {/* Architecture Flow */}
      <div className="relative">
        {/* Glass Container */}
        <div className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-8">
          {/* Stages */}
          <div className="flex items-center justify-between gap-4 overflow-x-auto">
            {stages.map((stage, index) => {
              const Icon = stage.icon;
              return (
                <div key={index} className="flex items-center gap-4">
                  {/* Stage Card */}
                  <div className="relative group">
                    {/* Glow */}
                    <div className={`absolute -inset-1 bg-gradient-to-r ${stage.color} rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity`} />

                    {/* Content */}
                    <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 min-w-[200px]">
                      <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${stage.color} mb-3`}>
                        <Icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-white font-semibold mb-1 text-sm">
                        {stage.title}
                      </h3>
                      <p className="text-slate-400 text-xs">
                        {stage.description}
                      </p>
                    </div>
                  </div>

                  {/* Arrow */}
                  {index < stages.length - 1 && (
                    <ArrowRight className="w-6 h-6 text-slate-600 flex-shrink-0" />
                  )}
                </div>
              );
            })}
          </div>

          {/* Technical Details */}
          <div className="mt-8 pt-8 border-t border-white/10">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <h4 className="text-white font-semibold mb-2 text-sm">Hybrid Retrieval</h4>
                <p className="text-slate-400 text-xs">
                  Combines SQL-based structured queries with semantic RAG retrieval for comprehensive results
                </p>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-2 text-sm">Multi-Hop Reasoning</h4>
                <p className="text-slate-400 text-xs">
                  Agent chains multiple queries and synthesizes information across data sources
                </p>
              </div>
              <div>
                <h4 className="text-white font-semibold mb-2 text-sm">Grounded Insights</h4>
                <p className="text-slate-400 text-xs">
                  All answers are backed by actual data and commentary from 100K+ IPL records
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
