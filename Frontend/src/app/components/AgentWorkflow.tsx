import { CheckCircle2, Loader2, Brain, Database, FileSearch, GitBranch, Sparkles } from 'lucide-react';
import { AgentStep } from './AgenticCricketPlatform';

interface AgentWorkflowProps {
  steps: AgentStep[];
  isProcessing: boolean;
}

export function AgentWorkflow({ steps, isProcessing }: AgentWorkflowProps) {
  const getIcon = (type: string) => {
    switch (type) {
      case 'planning': return Brain;
      case 'sql': return Database;
      case 'rag': return FileSearch;
      case 'reasoning': return GitBranch;
      case 'result': return Sparkles;
      default: return CheckCircle2;
    }
  };

  const getColor = (type: string) => {
    switch (type) {
      case 'planning': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'sql': return 'text-green-600 bg-green-50 border-green-200';
      case 'rag': return 'text-purple-600 bg-purple-50 border-purple-200';
      case 'reasoning': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'result': return 'text-pink-600 bg-pink-50 border-pink-200';
      default: return 'text-slate-600 bg-slate-50 border-slate-200';
    }
  };

  return (
    <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
      <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
        <Brain className="w-5 h-5 text-purple-600" />
        Agent Workflow
        {isProcessing && (
          <span className="text-sm font-normal text-slate-600 flex items-center gap-1">
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing...
          </span>
        )}
      </h3>

      <div className="space-y-3">
        {steps.map((step, index) => {
          const Icon = getIcon(step.type);
          const colorClass = getColor(step.type);
          const isLast = index === steps.length - 1;
          const isRunning = isLast && isProcessing;

          return (
            <div key={index} className="relative">
              {/* Connector Line */}
              {!isLast && (
                <div className="absolute left-6 top-12 bottom-0 w-0.5 bg-slate-300" />
              )}

              {/* Step Card */}
              <div className={`relative flex gap-4 p-4 rounded-lg border ${colorClass} transition-all`}>
                <div className="flex-shrink-0">
                  <div className={`p-2 rounded-lg ${colorClass}`}>
                    {isRunning ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-medium text-slate-900">{step.title}</h4>
                    {step.duration > 0 && (
                      <span className="text-xs font-medium text-slate-600 bg-white px-2 py-1 rounded">
                        {step.duration}s
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    {step.content}
                  </p>
                  {step.status === 'complete' && (
                    <div className="mt-2 flex items-center gap-1 text-xs text-slate-600">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>Complete</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Total Execution Time */}
      {steps.length > 0 && !isProcessing && (
        <div className="mt-4 pt-4 border-t border-slate-300">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-slate-700">Total Execution Time</span>
            <span className="font-bold text-purple-700">
              {steps.reduce((acc, step) => acc + step.duration, 0).toFixed(2)}s
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
