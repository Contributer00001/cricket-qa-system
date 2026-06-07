import { CheckCircle2, TrendingUp, Target, Activity, Award } from 'lucide-react';

interface CricketAnalysis {
  answer: string;
  statistics: {
    runs: number;
    wickets: number;
    boundaries: { fours: number; sixes: number };
    strikeRate: number;
    economy: number;
  };
  insights: string[];
  ballByBallBreakdown: Array<{
    over: number;
    ball: number;
    runs: number;
    isWicket: boolean;
    commentary: string;
  }>;
  confidence: number;
}

interface AnalysisResultsProps {
  analysis: CricketAnalysis;
}

export function AnalysisResults({ analysis }: AnalysisResultsProps) {
  return (
    <div className="space-y-6 mt-6">
      {/* Main Answer */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold text-green-900 mb-2">Answer</h3>
            <p className="text-green-800 leading-relaxed">{analysis.answer}</p>
            <div className="mt-3 flex items-center gap-2">
              <span className="text-sm text-green-700">Confidence:</span>
              <div className="flex-1 max-w-xs bg-green-200 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full transition-all"
                  style={{ width: `${analysis.confidence}%` }}
                />
              </div>
              <span className="text-sm font-medium text-green-700">
                {analysis.confidence}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics Grid */}
      <div>
        <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Key Statistics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-white rounded-lg p-4 border border-slate-200 text-center">
            <div className="text-3xl font-bold text-green-700 mb-1">
              {analysis.statistics.runs}
            </div>
            <div className="text-sm text-slate-600">Runs</div>
          </div>
          <div className="bg-white rounded-lg p-4 border border-slate-200 text-center">
            <div className="text-3xl font-bold text-red-600 mb-1">
              {analysis.statistics.wickets}
            </div>
            <div className="text-sm text-slate-600">Wickets</div>
          </div>
          <div className="bg-white rounded-lg p-4 border border-slate-200 text-center">
            <div className="text-3xl font-bold text-blue-600 mb-1">
              {analysis.statistics.boundaries.fours}
            </div>
            <div className="text-sm text-slate-600">Fours</div>
          </div>
          <div className="bg-white rounded-lg p-4 border border-slate-200 text-center">
            <div className="text-3xl font-bold text-purple-600 mb-1">
              {analysis.statistics.boundaries.sixes}
            </div>
            <div className="text-sm text-slate-600">Sixes</div>
          </div>
          <div className="bg-white rounded-lg p-4 border border-slate-200 text-center">
            <div className="text-3xl font-bold text-orange-600 mb-1">
              {analysis.statistics.strikeRate.toFixed(2)}
            </div>
            <div className="text-sm text-slate-600">Strike Rate</div>
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-white rounded-lg p-6 border border-slate-200">
        <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          AI-Generated Insights
        </h3>
        <div className="space-y-3">
          {analysis.insights.map((insight, index) => (
            <div key={index} className="flex items-start gap-3">
              <Award className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <p className="text-slate-700">{insight}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Ball-by-Ball Breakdown */}
      <div className="bg-white rounded-lg p-6 border border-slate-200">
        <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Ball-by-Ball Breakdown
        </h3>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {analysis.ballByBallBreakdown.map((ball, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 p-3 rounded-lg border ${
                ball.isWicket
                  ? 'bg-red-50 border-red-200'
                  : ball.runs >= 4
                  ? 'bg-green-50 border-green-200'
                  : 'bg-slate-50 border-slate-200'
              }`}
            >
              <div className="flex-shrink-0 font-mono text-sm font-medium text-slate-700 min-w-[3rem]">
                {ball.over}.{ball.ball}
              </div>
              <div className="flex-1 text-sm text-slate-700">
                {ball.commentary}
              </div>
              <div
                className={`flex-shrink-0 font-bold min-w-[2rem] text-center ${
                  ball.isWicket
                    ? 'text-red-600'
                    : ball.runs >= 4
                    ? 'text-green-600'
                    : 'text-slate-600'
                }`}
              >
                {ball.isWicket ? 'W' : ball.runs}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* API Response Info */}
      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
        <h3 className="font-medium text-slate-700 mb-2">About This Analysis</h3>
        <p className="text-sm text-slate-600">
          This response was generated by combining statistical computation with fine-tuned language models
          and agent-based processing. The system analyzed the ball-by-ball commentary to extract accurate,
          grounded, and explainable cricket statistics.
        </p>
      </div>
    </div>
  );
}
