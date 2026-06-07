import { Send, Loader2, Sparkles } from 'lucide-react';

interface QueryInterfaceProps {
  query: string;
  setQuery: (value: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  hasCommentary: boolean;
}

export function QueryInterface({ query, setQuery, onAnalyze, isAnalyzing, hasCommentary }: QueryInterfaceProps) {
  const sampleQueries = [
    'How many runs did the batsman score in the powerplay?',
    'What was the strike rate in the first 10 overs?',
    'How many wickets fell in the death overs?',
    'What was the economy rate of the opening bowler?',
    'How many boundaries were hit in the match?',
    'Which over had the most runs?'
  ];

  return (
    <div className="space-y-4">
      <div>
        <label className="block font-medium text-slate-700 mb-2">
          Ask a Question About the Match
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !isAnalyzing && hasCommentary && onAnalyze()}
            placeholder="e.g., How many runs were scored in the powerplay overs?"
            className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            disabled={!hasCommentary}
          />
          <button
            onClick={onAnalyze}
            disabled={isAnalyzing || !query.trim() || !hasCommentary}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors font-medium"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Analyze
              </>
            )}
          </button>
        </div>
        {!hasCommentary && (
          <p className="mt-2 text-sm text-amber-600">
            Please load commentary data first before querying
          </p>
        )}
      </div>

      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
        <h3 className="font-medium text-slate-700 mb-3 flex items-center gap-2">
          <Sparkles className="w-4 h-4" />
          Sample Questions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {sampleQueries.map((sample, index) => (
            <button
              key={index}
              onClick={() => setQuery(sample)}
              className="text-left text-sm text-slate-600 hover:text-green-700 hover:bg-green-50 px-3 py-2 rounded border border-slate-200 hover:border-green-300 transition-colors"
              disabled={!hasCommentary}
            >
              {sample}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
