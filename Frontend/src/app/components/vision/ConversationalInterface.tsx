import { Sparkles, Send, Loader2, Mic } from 'lucide-react';

interface ConversationalInterfaceProps {
  query: string;
  setQuery: (value: string) => void;
  onSubmit: (query: string) => void;
  isProcessing: boolean;
}

export function ConversationalInterface({
  query,
  setQuery,
  onSubmit,
  isProcessing
}: ConversationalInterfaceProps) {
  const handleSubmit = () => {
    if (query.trim() && !isProcessing) {
      onSubmit(query);
    }
  };

  return (
    <div className="max-w-4xl mx-auto mb-12">
      {/* Main Search Interface */}
      <div className="relative group">
        {/* Glow Effect */}
        <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-3xl blur-xl opacity-30 group-hover:opacity-50 transition-opacity" />

        {/* Glass Container */}
        <div className="relative bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-2 shadow-2xl">
          <div className="flex items-center gap-3">
            {/* Icon */}
            <div className="pl-4">
              <Sparkles className="w-6 h-6 text-cyan-400" />
            </div>

            {/* Input */}
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="Ask anything about IPL cricket..."
              disabled={isProcessing}
              className="flex-1 bg-transparent text-white text-lg placeholder-slate-400 focus:outline-none py-4"
            />

            {/* Voice Input */}
            <button className="p-3 hover:bg-white/5 rounded-xl transition-colors">
              <Mic className="w-5 h-5 text-slate-400 hover:text-cyan-400 transition-colors" />
            </button>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={isProcessing || !query.trim()}
              className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed rounded-2xl font-semibold text-white transition-all shadow-lg hover:shadow-cyan-500/50 disabled:shadow-none flex items-center gap-2"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Thinking...</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Ask</span>
                </>
              )}
            </button>
          </div>

          {/* Processing Indicator */}
          {isProcessing && (
            <div className="px-6 py-3 border-t border-white/5">
              <div className="flex items-center gap-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
                <span className="text-sm text-slate-400">AI is analyzing your query...</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Helper Text */}
      <div className="text-center mt-4">
        <p className="text-sm text-slate-500">
          Powered by advanced AI reasoning • Natural language understanding • Real-time analytics
        </p>
      </div>
    </div>
  );
}
