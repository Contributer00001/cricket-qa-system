import { Database, Table, Columns3 } from 'lucide-react';
import schemaImage from '../../imports/Screenshot_2026-06-07_at_3.20.35_PM.png';

export function DatabaseSchema() {
  const tables = [
    {
      name: 'batting_stats',
      description: 'Individual batting performance statistics',
      columns: ['match_id', 'player_name', 'runs_scored', 'balls_faced', 'strike_rate', 'fours', 'sixes', 'dismissal_type']
    },
    {
      name: 'bowling_stats',
      description: 'Individual bowling performance statistics',
      columns: ['match_id', 'player_name', 'overs', 'runs_conceded', 'wickets', 'economy', 'maiden_overs', 'dots']
    },
    {
      name: 'match_info',
      description: 'Match metadata and information',
      columns: ['match_id', 'season', 'date', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']
    },
    {
      name: 'match_innings_commentary',
      description: 'Ball-by-ball commentary for all matches',
      columns: ['match_id', 'innings', 'over', 'ball', 'batsman', 'bowler', 'runs', 'commentary_text', 'wicket']
    },
    {
      name: 'match_live_details',
      description: 'Live match details and updates',
      columns: ['match_id', 'current_score', 'current_rr', 'required_rr', 'target', 'wickets_fallen']
    },
    {
      name: 'match_wagon_wheel',
      description: 'Shot placement and wagon wheel data',
      columns: ['match_id', 'player_name', 'shot_type', 'runs', 'x_coord', 'y_coord', 'boundary']
    },
    {
      name: 'matches',
      description: 'Core match records',
      columns: ['match_id', 'season', 'city', 'date', 'team1', 'team2', 'result', 'dl_applied', 'winner', 'player_of_match']
    },
    {
      name: 'player_career_stats',
      description: 'Career statistics for all players',
      columns: ['player_name', 'total_matches', 'total_runs', 'total_wickets', 'highest_score', 'best_bowling']
    },
    {
      name: 'scorecards',
      description: 'Detailed scorecards for each match',
      columns: ['match_id', 'innings', 'team', 'total_runs', 'total_wickets', 'extras', 'run_rate']
    },
    {
      name: 'squads',
      description: 'Team squad compositions',
      columns: ['team_name', 'season', 'player_name', 'role', 'country']
    },
    {
      name: 'standings',
      description: 'Team standings and points table',
      columns: ['season', 'team', 'matches', 'won', 'lost', 'points', 'nrr', 'position']
    },
    {
      name: 'team_stats',
      description: 'Team-level performance statistics',
      columns: ['team_name', 'season', 'total_runs', 'total_wickets', 'avg_score', 'highest_team_score']
    },
    {
      name: 'teams',
      description: 'Team information and metadata',
      columns: ['team_id', 'team_name', 'home_ground', 'coach', 'captain', 'founded_year']
    }
  ];

  return (
    <div className="space-y-6">
      {/* Schema Image */}
      <div className="bg-slate-900 rounded-lg p-6 border border-slate-700">
        <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-400" />
          Database Schema: Indian_Premier_League_2022-03-26
        </h3>
        <img
          src={schemaImage}
          alt="Database Schema"
          className="w-full max-w-md mx-auto rounded-lg border border-slate-600"
        />
      </div>

      {/* Tables Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tables.map((table, index) => (
          <div key={index} className="bg-white rounded-lg p-5 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-start gap-3 mb-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Table className="w-5 h-5 text-purple-700" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-semibold text-slate-900 mb-1">{table.name}</h4>
                <p className="text-sm text-slate-600">{table.description}</p>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-slate-100">
              <div className="flex items-center gap-2 mb-2">
                <Columns3 className="w-4 h-4 text-slate-500" />
                <span className="text-xs font-medium text-slate-600">Key Columns:</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {table.columns.slice(0, 4).map((col, idx) => (
                  <span key={idx} className="text-xs bg-purple-50 text-purple-700 px-2 py-1 rounded border border-purple-200">
                    {col}
                  </span>
                ))}
                {table.columns.length > 4 && (
                  <span className="text-xs text-slate-500 px-2 py-1">
                    +{table.columns.length - 4} more
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Dataset Info */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border border-purple-200">
        <h3 className="font-semibold text-slate-900 mb-4">Dataset Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <div className="font-medium text-slate-700 mb-1">Total Records</div>
            <div className="text-2xl font-bold text-purple-700">103,847+</div>
          </div>
          <div>
            <div className="font-medium text-slate-700 mb-1">Coverage Period</div>
            <div className="text-2xl font-bold text-purple-700">2008-2023</div>
          </div>
          <div>
            <div className="font-medium text-slate-700 mb-1">Tables</div>
            <div className="text-2xl font-bold text-purple-700">13</div>
          </div>
        </div>
      </div>
    </div>
  );
}
