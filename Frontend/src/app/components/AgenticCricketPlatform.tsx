import { useState } from 'react';
import { QueryInput } from './QueryInput';
import { AgentWorkflow } from './AgentWorkflow';
import { AnalyticsDashboard } from './AnalyticsDashboard';
import { DatabaseSchema } from './DatabaseSchema';
import { Brain, Database, Activity } from 'lucide-react';

export interface AgentStep {
  type: 'planning' | 'sql' | 'rag' | 'reasoning' | 'result';
  title: string;
  content: string;
  duration: number;
  status: 'running' | 'complete';
}

export interface AnalyticsResult {
  answer: string;
  sqlQuery?: string;
  data: any[];
  commentary?: string[];
  confidence: number;
  executionTime: number;
  dataSource: string;
}

export function AgenticCricketPlatform() {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [result, setResult] = useState<AnalyticsResult | null>(null);
  const [activeTab, setActiveTab] = useState<'query' | 'schema'>('query');

  const processQuery = async (userQuery: string) => {
    setIsProcessing(true);
    setAgentSteps([]);
    setResult(null);

    // Step 1: Query Planning
    await new Promise(resolve => setTimeout(resolve, 800));
    setAgentSteps(prev => [...prev, {
      type: 'planning',
      title: 'Query Planning & Intent Analysis',
      content: 'Analyzing query intent... Detected: Statistical aggregation + Player performance analysis. Route: SQL (batting_stats) + RAG (match_innings_commentary)',
      duration: 0.3,
      status: 'complete'
    }]);

    // Step 2: SQL Generation
    await new Promise(resolve => setTimeout(resolve, 600));
    setAgentSteps(prev => [...prev, {
      type: 'sql',
      title: 'SQL Query Generation',
      content: 'SELECT player_name, SUM(runs_scored) as total_runs, AVG(strike_rate) as avg_sr, COUNT(DISTINCT match_id) as matches FROM batting_stats WHERE season = 2022 GROUP BY player_name ORDER BY total_runs DESC LIMIT 10',
      duration: 0.2,
      status: 'complete'
    }]);

    // Step 3: RAG Retrieval
    await new Promise(resolve => setTimeout(resolve, 700));
    setAgentSteps(prev => [...prev, {
      type: 'rag',
      title: 'RAG Commentary Retrieval',
      content: 'Retrieved 156 relevant ball-by-ball commentary entries using hybrid search (semantic + keyword). Top matches: Match #234 (RCB vs MI), Match #189 (CSK vs DC)',
      duration: 0.4,
      status: 'complete'
    }]);

    // Step 4: Multi-hop Reasoning
    await new Promise(resolve => setTimeout(resolve, 500));
    setAgentSteps(prev => [...prev, {
      type: 'reasoning',
      title: 'Multi-Hop Reasoning & Synthesis',
      content: 'Cross-referencing SQL results with commentary context. Identifying key performance patterns. Grounding insights with actual match events.',
      duration: 0.15,
      status: 'complete'
    }]);

    // Step 5: Final Result
    await new Promise(resolve => setTimeout(resolve, 400));
    const mockResult: AnalyticsResult = {
      answer: 'Jos Buttler dominated IPL 2022 with 863 runs across 17 matches at an exceptional strike rate of 149.05. His standout performances included a blistering 106* against Delhi Capitals where he scored at 210 SR in the powerplay. Buttler hit 43 fours and 43 sixes throughout the season, with peak form in matches 5-9 where he averaged 68.4. Commentary analysis reveals his preference for attacking spinners in the middle overs, particularly effective against leg-spin.',
      sqlQuery: 'SELECT player_name, SUM(runs_scored) as total_runs, AVG(strike_rate) as avg_sr, COUNT(DISTINCT match_id) as matches FROM batting_stats WHERE season = 2022 GROUP BY player_name ORDER BY total_runs DESC LIMIT 10',
      data: [
        { player: 'Jos Buttler', runs: 863, average: 57.53, strikeRate: 149.05, matches: 17, fifties: 4, hundreds: 4 },
        { player: 'KL Rahul', runs: 616, average: 51.33, strikeRate: 135.38, matches: 15, fifties: 6, hundreds: 2 },
        { player: 'Quinton de Kock', runs: 508, average: 36.29, strikeRate: 148.54, matches: 15, fifties: 4, hundreds: 0 },
        { player: 'Hardik Pandya', runs: 487, average: 44.27, strikeRate: 131.27, matches: 15, fifties: 4, hundreds: 0 },
        { player: 'Shikhar Dhawan', runs: 460, average: 38.33, strikeRate: 122.22, matches: 14, fifties: 3, hundreds: 0 }
      ],
      commentary: [
        '[Match 234, Over 5.2] FOUR! Buttler dances down to the spinner, gets to the pitch and lofts it over mid-off for a boundary',
        '[Match 189, Over 12.4] SIX! Short ball, Buttler swivels and pulls it magnificently over deep square leg',
        '[Match 201, Over 3.5] FOUR! Width offered, Buttler slashes it past backward point for another boundary',
        '[Match 234, Over 18.3] Maximum! Buttler makes room and carves it over extra cover for six more'
      ],
      confidence: 96.8,
      executionTime: 0.89,
      dataSource: 'IPL 2022-2023 Database (103,847 records)'
    };

    setAgentSteps(prev => [...prev, {
      type: 'result',
      title: 'Final Result Generated',
      content: 'Analysis complete. Generated grounded insights combining statistical data and match commentary.',
      duration: 0.1,
      status: 'complete'
    }]);

    setResult(mockResult);
    setIsProcessing(false);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-md border border-slate-200">
        <div className="flex border-b border-slate-200">
          <button
            onClick={() => setActiveTab('query')}
            className={`flex-1 px-6 py-4 font-medium transition-colors ${
              activeTab === 'query'
                ? 'text-purple-700 border-b-2 border-purple-700 bg-purple-50'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <Brain className="w-5 h-5" />
              Query Analytics
            </div>
          </button>
          <button
            onClick={() => setActiveTab('schema')}
            className={`flex-1 px-6 py-4 font-medium transition-colors ${
              activeTab === 'schema'
                ? 'text-purple-700 border-b-2 border-purple-700 bg-purple-50'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <Database className="w-5 h-5" />
              Database Schema
            </div>
          </button>
        </div>

        <div className="p-6">
          {activeTab === 'query' ? (
            <div className="space-y-6">
              <QueryInput
                query={query}
                setQuery={setQuery}
                onSubmit={processQuery}
                isProcessing={isProcessing}
              />

              {agentSteps.length > 0 && (
                <AgentWorkflow steps={agentSteps} isProcessing={isProcessing} />
              )}

              {result && (
                <AnalyticsDashboard result={result} />
              )}
            </div>
          ) : (
            <DatabaseSchema />
          )}
        </div>
      </div>

      {/* System Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg p-5 shadow-sm border border-slate-200">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-100 rounded-lg">
              <Database className="w-6 h-6 text-purple-700" />
            </div>
            <div>
              <div className="text-sm text-slate-600">Database Records</div>
              <div className="text-2xl font-bold text-slate-900">103,847</div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg p-5 shadow-sm border border-slate-200">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-100 rounded-lg">
              <Activity className="w-6 h-6 text-green-700" />
            </div>
            <div>
              <div className="text-sm text-slate-600">Avg Query Time</div>
              <div className="text-2xl font-bold text-slate-900">0.89s</div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg p-5 shadow-sm border border-slate-200">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-100 rounded-lg">
              <Brain className="w-6 h-6 text-blue-700" />
            </div>
            <div>
              <div className="text-sm text-slate-600">Agent Type</div>
              <div className="text-xl font-bold text-slate-900">Hybrid SQL+RAG</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
