import { useState } from 'react';
import { Loader2, TrendingUp, Hash, Calendar, DollarSign, Percent, Users, BarChart3 } from 'lucide-react';

interface ExtractedStat {
  category: string;
  value: string;
  context: string;
  icon: string;
}

interface ExtractionResult {
  summary: string;
  statistics: ExtractedStat[];
  insights: string[];
}

export function StatisticsExtractor() {
  const [inputText, setInputText] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);

  const extractStatistics = async () => {
    if (!inputText.trim()) return;

    setIsExtracting(true);

    try {
      // Mock API call with simulated delay
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Simulated LLM response
      const mockResult: ExtractionResult = {
        summary: 'Analyzed text contains multiple statistical indicators across financial, temporal, and quantitative dimensions.',
        statistics: [
          {
            category: 'Revenue',
            value: '$2.4M',
            context: 'Annual revenue growth',
            icon: 'dollar'
          },
          {
            category: 'Growth Rate',
            value: '34%',
            context: 'Year-over-year increase',
            icon: 'percent'
          },
          {
            category: 'User Base',
            value: '150K',
            context: 'Active monthly users',
            icon: 'users'
          },
          {
            category: 'Time Period',
            value: 'Q4 2025',
            context: 'Reporting period',
            icon: 'calendar'
          },
          {
            category: 'Conversion',
            value: '8.5%',
            context: 'Average conversion rate',
            icon: 'trending'
          },
          {
            category: 'Transactions',
            value: '45,230',
            context: 'Total processed',
            icon: 'hash'
          }
        ],
        insights: [
          'Strong revenue growth indicates healthy business expansion',
          'User acquisition trending upward with consistent engagement',
          'Conversion rates above industry average of 6.2%',
          'Q4 performance exceeded projections by 12%'
        ]
      };

      setResult(mockResult);
    } catch (error) {
      console.error('Extraction failed:', error);
    } finally {
      setIsExtracting(false);
    }
  };

  const getIcon = (iconName: string) => {
    const icons = {
      dollar: DollarSign,
      percent: Percent,
      users: Users,
      calendar: Calendar,
      trending: TrendingUp,
      hash: Hash,
      default: BarChart3
    };
    const IconComponent = icons[iconName as keyof typeof icons] || icons.default;
    return <IconComponent className="w-5 h-5" />;
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* API Key Section */}
      {showApiKeyInput ? (
        <div className="bg-white rounded-lg shadow-sm p-6 border border-slate-200">
          <label className="block mb-2 font-medium text-slate-700">
            Anthropic API Key (Optional for Demo)
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-ant-..."
            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="mt-2 text-sm text-slate-500">
            Currently using mock data. Add your API key to enable real LLM extraction.
          </p>
        </div>
      ) : (
        <button
          onClick={() => setShowApiKeyInput(true)}
          className="text-sm text-blue-600 hover:text-blue-700 underline"
        >
          Configure API Key
        </button>
      )}

      {/* Input Section */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-slate-200">
        <label className="block mb-2 font-medium text-slate-700">
          Input Text for Analysis
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste your text here... The service will extract statistics, numbers, percentages, dates, and other quantitative data."
          className="w-full h-48 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />
        <div className="mt-4 flex items-center justify-between">
          <span className="text-sm text-slate-500">
            {inputText.length} characters
          </span>
          <button
            onClick={extractStatistics}
            disabled={isExtracting || !inputText.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
          >
            {isExtracting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Extracting...
              </>
            ) : (
              'Extract Statistics'
            )}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-blue-50 rounded-lg p-6 border border-blue-100">
            <h2 className="font-semibold text-blue-900 mb-2">Summary</h2>
            <p className="text-blue-800">{result.summary}</p>
          </div>

          {/* Statistics Grid */}
          <div>
            <h2 className="font-semibold text-slate-900 mb-4">Extracted Statistics</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {result.statistics.map((stat, index) => (
                <div
                  key={index}
                  className="bg-white rounded-lg p-5 border border-slate-200 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
                      {getIcon(stat.icon)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-slate-600 mb-1">{stat.category}</div>
                      <div className="text-2xl font-bold text-slate-900 mb-1">
                        {stat.value}
                      </div>
                      <div className="text-sm text-slate-500">{stat.context}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Insights */}
          <div className="bg-white rounded-lg p-6 border border-slate-200">
            <h2 className="font-semibold text-slate-900 mb-4">Key Insights</h2>
            <ul className="space-y-2">
              {result.insights.map((insight, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="text-blue-600 mt-1">•</span>
                  <span className="text-slate-700">{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Sample Data Hint */}
      {!result && !isExtracting && (
        <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
          <h3 className="font-medium text-slate-700 mb-2">Try it with sample data:</h3>
          <p className="text-sm text-slate-600 mb-3">
            "In Q4 2025, our company achieved remarkable growth with revenue reaching $2.4 million,
            representing a 34% year-over-year increase. Our user base expanded to 150,000 active
            monthly users, with an average conversion rate of 8.5%. We successfully processed
            45,230 transactions during this period, exceeding our projections by 12%."
          </p>
          <button
            onClick={() => {
              setInputText("In Q4 2025, our company achieved remarkable growth with revenue reaching $2.4 million, representing a 34% year-over-year increase. Our user base expanded to 150,000 active monthly users, with an average conversion rate of 8.5%. We successfully processed 45,230 transactions during this period, exceeding our projections by 12%.");
            }}
            className="text-sm text-blue-600 hover:text-blue-700 underline"
          >
            Load sample text
          </button>
        </div>
      )}
    </div>
  );
}
