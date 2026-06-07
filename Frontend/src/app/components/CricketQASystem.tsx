import { useState } from 'react';
import { Loader2, Send, Activity, Target, TrendingUp, FileText, BarChart2 } from 'lucide-react';
import { CommentaryInput } from './CommentaryInput';
import { QueryInterface } from './QueryInterface';
import { AnalysisResults } from './AnalysisResults';

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

export function CricketQASystem() {
  const [commentary, setCommentary] = useState('');
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<CricketAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<'commentary' | 'query'>('commentary');

  const analyzeQuery = async () => {
    if (!commentary.trim() || !query.trim()) return;

    setIsAnalyzing(true);

    try {
      // Simulate AI processing with delay
      await new Promise(resolve => setTimeout(resolve, 2500));

      // Mock analysis result
      const mockAnalysis: CricketAnalysis = {
        answer: 'Based on the ball-by-ball commentary analysis, the batsman scored 47 runs off 32 balls in the powerplay overs (1-6), which includes 6 fours and 2 sixes. The strike rate during this period was 146.87.',
        statistics: {
          runs: 47,
          wickets: 1,
          boundaries: { fours: 6, sixes: 2 },
          strikeRate: 146.87,
          economy: 7.83
        },
        insights: [
          'Aggressive batting approach with 8 boundaries in powerplay',
          'Strike rate significantly above powerplay average of 120',
          'Lost one wicket attempting a boundary in over 5',
          'Maximum runs scored in overs 2 and 4 (12 runs each)',
          'Bowlers struggled with line and length in first 6 overs'
        ],
        ballByBallBreakdown: [
          { over: 1, ball: 1, runs: 0, isWicket: false, commentary: 'Good length delivery, defended back' },
          { over: 1, ball: 2, runs: 4, isWicket: false, commentary: 'Short and wide, cut away for four' },
          { over: 1, ball: 3, runs: 1, isWicket: false, commentary: 'Pushed to mid-on for a single' },
          { over: 1, ball: 4, runs: 0, isWicket: false, commentary: 'Dot ball, played to covers' },
          { over: 1, ball: 5, runs: 2, isWicket: false, commentary: 'Worked to deep square leg for two' },
          { over: 1, ball: 6, runs: 4, isWicket: false, commentary: 'Driven through covers for a boundary' },
          { over: 2, ball: 1, runs: 6, isWicket: false, commentary: 'SIX! Pulled over mid-wicket' },
          { over: 2, ball: 2, runs: 1, isWicket: false, commentary: 'Single to third man' },
          { over: 2, ball: 3, runs: 0, isWicket: false, commentary: 'Dot ball outside off' },
          { over: 2, ball: 4, runs: 4, isWicket: false, commentary: 'Four through point region' },
          { over: 2, ball: 5, runs: 1, isWicket: false, commentary: 'Nudged to square leg' },
          { over: 2, ball: 6, runs: 0, isWicket: false, commentary: 'Beaten outside off stump' }
        ],
        confidence: 94.5
      };

      setAnalysis(mockAnalysis);
      setActiveTab('query');
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const loadSampleData = () => {
    setCommentary(`Over 1:
1.1 - Good length delivery outside off, defended solidly back to bowler
1.2 - Short and wide, batsman cuts hard through point for FOUR
1.3 - Full on middle, pushed gently to mid-on, quick single taken
1.4 - Dot ball, defended to covers
1.5 - Worked off the pads to deep square leg, two runs
1.6 - Overpitched outside off, driven beautifully through covers for FOUR

Over 2:
2.1 - Short ball, PULLED POWERFULLY over mid-wicket for SIX!
2.2 - Outside edge runs down to third man, single taken
2.3 - Dot ball, beaten outside off stump by pace
2.4 - Width offered, slashed hard through point for FOUR
2.5 - Nudged to square leg for a single
2.6 - Good delivery, beaten outside off

Over 3:
3.1 - Full toss, driven to mid-off, no run
3.2 - Short ball pulled to deep mid-wicket, single
3.3 - WICKET! Caught at mid-off trying to loft over the fielder
3.4 - New batsman defends first ball
3.5 - Driven to covers, no run
3.6 - Flicked to fine leg for two runs`);

    setQuery('How many runs did the batsman score in the powerplay overs with what strike rate?');
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Tab Navigation */}
      <div className="bg-white rounded-t-lg shadow-sm border-b border-slate-200">
        <div className="flex">
          <button
            onClick={() => setActiveTab('commentary')}
            className={`flex-1 px-6 py-4 font-medium transition-colors ${
              activeTab === 'commentary'
                ? 'text-green-700 border-b-2 border-green-700 bg-green-50'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <FileText className="w-5 h-5" />
              Ball-by-Ball Commentary
            </div>
          </button>
          <button
            onClick={() => setActiveTab('query')}
            className={`flex-1 px-6 py-4 font-medium transition-colors ${
              activeTab === 'query'
                ? 'text-green-700 border-b-2 border-green-700 bg-green-50'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <Activity className="w-5 h-5" />
              Query & Analysis
            </div>
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="bg-white rounded-b-lg shadow-sm p-6 border border-slate-200 border-t-0">
        {activeTab === 'commentary' ? (
          <CommentaryInput
            commentary={commentary}
            setCommentary={setCommentary}
            onLoadSample={loadSampleData}
          />
        ) : (
          <div className="space-y-6">
            <QueryInterface
              query={query}
              setQuery={setQuery}
              onAnalyze={analyzeQuery}
              isAnalyzing={isAnalyzing}
              hasCommentary={!!commentary.trim()}
            />
            {analysis && <AnalysisResults analysis={analysis} />}
          </div>
        )}
      </div>

      {/* Quick Stats Bar */}
      {commentary && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg p-4 shadow-sm border border-slate-200">
            <div className="flex items-center gap-2 text-slate-600 mb-1">
              <FileText className="w-4 h-4" />
              <span className="text-sm">Commentary Loaded</span>
            </div>
            <div className="text-2xl font-bold text-green-700">
              {commentary.split('\n').filter(l => l.trim()).length} lines
            </div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-slate-200">
            <div className="flex items-center gap-2 text-slate-600 mb-1">
              <Target className="w-4 h-4" />
              <span className="text-sm">System Status</span>
            </div>
            <div className="text-lg font-bold text-green-600">Ready</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-slate-200">
            <div className="flex items-center gap-2 text-slate-600 mb-1">
              <BarChart2 className="w-4 h-4" />
              <span className="text-sm">AI Model</span>
            </div>
            <div className="text-lg font-bold text-slate-700">Cricket-LLM</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-slate-200">
            <div className="flex items-center gap-2 text-slate-600 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm">Analysis Mode</span>
            </div>
            <div className="text-lg font-bold text-slate-700">Agent-Based</div>
          </div>
        </div>
      )}
    </div>
  );
}
