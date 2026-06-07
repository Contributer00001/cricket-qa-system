import { Send, Loader2, Sparkles } from 'lucide-react';

interface QueryInputProps {
  query: string;
  setQuery: (value: string) => void;
  onSubmit: (query: string) => void;
  isProcessing: boolean;
}

export function QueryInput({ query, setQuery, onSubmit, isProcessing }: QueryInputProps) {
  const sampleQueries = [
    "Who scored the most runs in IPL 2022?",
    "Show me top wicket-takers with economy rate under 7",
    "Which batsman has the highest strike rate in powerplay?",
    "Compare Virat Kohli vs Rohit Sharma performance this season",
    "What was the highest team score in IPL 2023?",
    "Show match-winning performances by all-rounders"
  ];

  const handleSubmit = () => {
    if (query.trim() && !isProcessing) {
      onSubmit(query);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block font-semibold text-slate-800 mb-3">
          Ask Anything About IPL Cricket
        </label>
        <div className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="e.g., Who scored the most runs in IPL 2022?"
            className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-lg"
            disabled={isProcessing}
          />
          <button
            onClick={handleSubmit}
            disabled={isProcessing || !query.trim()}
            className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all font-semibold shadow-lg"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Analyze
              </>
            )}
          </button>
        </div>
      </div>

      {/* Sample Queries */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-5 border border-purple-100">
        <h3 className="font-medium text-slate-800 mb-3 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-purple-600" />
          Sample Queries to Try
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {sampleQueries.map((sample, index) => (
            <button
              key={index}
              onClick={() => setQuery(sample)}
              disabled={isProcessing}
              className="text-left text-sm text-slate-700 hover:text-purple-700 bg-white hover:bg-purple-50 px-4 py-3 rounded-lg border border-slate-200 hover:border-purple-300 transition-all disabled:opacity-50"
            >
              {sample}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
