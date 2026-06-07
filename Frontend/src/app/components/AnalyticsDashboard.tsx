import { CheckCircle2, Code, TrendingUp, MessageSquare, Clock, Database, Target } from 'lucide-react';
import { AnalyticsResult } from './AgenticCricketPlatform';

interface AnalyticsDashboardProps {
  result: AnalyticsResult;
}

export function AnalyticsDashboard({ result }: AnalyticsDashboardProps) {
  return (
    <div className="space-y-6">
      {/* Main Answer */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-200 shadow-lg">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="w-7 h-7 text-purple-600 flex-shrink-0 mt-1" />
          <div className="flex-1">
            <h3 className="font-semibold text-purple-900 mb-3 text-lg">AI-Generated Answer</h3>
            <p className="text-slate-800 leading-relaxed text-lg">{result.answer}</p>
            <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-purple-600" />
                <span className="text-slate-700">Confidence:</span>
                <span className="font-bold text-purple-700">{result.confidence}%</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-purple-600" />
                <span className="text-slate-700">Execution:</span>
                <span className="font-bold text-purple-700">{result.executionTime}s</span>
              </div>
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-purple-600" />
                <span className="text-slate-700">{result.dataSource}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Data Table */}
      <div className="bg-white rounded-lg p-6 border border-slate-200 shadow-sm">
        <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-green-600" />
          Statistical Results
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-700">Player</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-700">Runs</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-700">Average</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-700">Strike Rate</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-700">Matches</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-700">50s/100s</th>
              </tr>
            </thead>
            <tbody>
              {result.data.map((row, index) => (
                <tr key={index} className={`border-b border-slate-100 ${index % 2 === 0 ? 'bg-slate-50' : 'bg-white'} hover:bg-purple-50 transition-colors`}>
                  <td className="py-3 px-4 font-medium text-slate-900">{row.player}</td>
                  <td className="py-3 px-4 text-right font-bold text-green-700">{row.runs}</td>
                  <td className="py-3 px-4 text-right text-slate-700">{row.average}</td>
                  <td className="py-3 px-4 text-right text-slate-700">{row.strikeRate}</td>
                  <td className="py-3 px-4 text-right text-slate-700">{row.matches}</td>
                  <td className="py-3 px-4 text-right text-slate-700">{row.fifties}/{row.hundreds}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* SQL Query */}
      {result.sqlQuery && (
        <div className="bg-slate-900 rounded-lg p-6 border border-slate-700 shadow-sm">
          <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
            <Code className="w-5 h-5 text-green-400" />
            Generated SQL Query
          </h3>
          <pre className="text-sm text-green-400 font-mono overflow-x-auto">
            {result.sqlQuery}
          </pre>
        </div>
      )}

      {/* Commentary Insights */}
      {result.commentary && result.commentary.length > 0 && (
        <div className="bg-white rounded-lg p-6 border border-slate-200 shadow-sm">
          <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-purple-600" />
            RAG-Retrieved Commentary
          </h3>
          <div className="space-y-3">
            {result.commentary.map((comment, index) => (
              <div key={index} className="bg-purple-50 border border-purple-100 rounded-lg p-4">
                <p className="text-slate-800 text-sm leading-relaxed font-mono">
                  {comment}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Technical Details */}
      <div className="bg-slate-50 rounded-lg p-5 border border-slate-200">
        <h3 className="font-medium text-slate-800 mb-3">How This Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <div className="font-semibold text-slate-700 mb-1">1. Query Planning</div>
            <p className="text-slate-600">Agent analyzes intent and routes to appropriate tools (SQL/RAG)</p>
          </div>
          <div>
            <div className="font-semibold text-slate-700 mb-1">2. Hybrid Retrieval</div>
            <p className="text-slate-600">Combines structured SQL queries with semantic commentary search</p>
          </div>
          <div>
            <div className="font-semibold text-slate-700 mb-1">3. Multi-Hop Reasoning</div>
            <p className="text-slate-600">Synthesizes data with context to generate grounded insights</p>
          </div>
        </div>
      </div>
    </div>
  );
}
