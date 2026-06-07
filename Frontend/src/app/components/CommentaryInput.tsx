import { Upload, FileText } from 'lucide-react';

interface CommentaryInputProps {
  commentary: string;
  setCommentary: (value: string) => void;
  onLoadSample: () => void;
}

export function CommentaryInput({ commentary, setCommentary, onLoadSample }: CommentaryInputProps) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="font-medium text-slate-700">
            Match Commentary Input
          </label>
          <button
            onClick={onLoadSample}
            className="text-sm text-green-600 hover:text-green-700 underline flex items-center gap-1"
          >
            <FileText className="w-4 h-4" />
            Load Sample Commentary
          </button>
        </div>
        <textarea
          value={commentary}
          onChange={(e) => setCommentary(e.target.value)}
          placeholder="Paste ball-by-ball commentary here...&#10;&#10;Example format:&#10;Over 1:&#10;1.1 - Good length delivery, defended&#10;1.2 - Short and wide, cut away for FOUR&#10;1.3 - Full toss, SIX over mid-wicket..."
          className="w-full h-96 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 resize-none font-mono text-sm"
        />
        <div className="mt-2 flex items-center justify-between text-sm text-slate-500">
          <span>{commentary.length} characters</span>
          <span>{commentary.split('\n').filter(l => l.trim()).length} lines</span>
        </div>
      </div>

      <div className="bg-green-50 rounded-lg p-4 border border-green-100">
        <h3 className="font-medium text-green-900 mb-2 flex items-center gap-2">
          <Upload className="w-4 h-4" />
          Commentary Format Guidelines
        </h3>
        <ul className="text-sm text-green-800 space-y-1">
          <li>• Include over and ball numbers (e.g., "1.1", "2.3")</li>
          <li>• Describe each delivery with outcome (runs, wicket, dot)</li>
          <li>• Mark boundaries as "FOUR" or "SIX" for easy detection</li>
          <li>• Include wicket information with "WICKET" or "OUT"</li>
          <li>• Add context like shot type, fielding positions</li>
        </ul>
      </div>
    </div>
  );
}
