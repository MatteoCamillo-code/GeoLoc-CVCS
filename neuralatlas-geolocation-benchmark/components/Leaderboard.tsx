
import React from 'react';
import { ScoreEntry, GameState } from '../types';
import { formatDistance } from '../utils';

interface LeaderboardProps {
  scores: ScoreEntry[];
  gameState: GameState;
}

const Leaderboard: React.FC<LeaderboardProps> = ({ scores, gameState }) => {
  const sortedScores = [...scores].sort((a, b) => a.totalError - b.totalError);

  return (
    <div className="p-4 bg-slate-950 overflow-y-auto flex-1 custom-scrollbar">
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-800">
        <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-500">Metric Dashboard</h3>
        <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
            <span className="text-[9px] font-mono font-bold text-cyan-500">REAL-TIME DATA</span>
        </div>
      </div>

      <div className="space-y-2">
        {sortedScores.map((entry, index) => (
          <div
            key={entry.name}
            className={`flex flex-col p-3 rounded-xl border transition-all duration-300 ${
              entry.type === 'human'
                ? 'bg-sky-950/40 border-sky-500/30'
                : 'bg-slate-900/50 border-slate-800'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono font-bold text-slate-600 w-3">
                  {index + 1}
                </span>
                <div
                  className="w-2.5 h-2.5 rounded-full ring-2 ring-slate-950 shadow-[0_0_8px_rgba(0,0,0,0.5)]"
                  style={{ backgroundColor: entry.color }}
                ></div>
                <span className={`text-xs font-bold ${entry.type === 'human' ? 'text-sky-300' : 'text-slate-200'}`}>
                  {entry.name}
                </span>
              </div>
              <div className="text-right">
                <div className="text-[11px] font-mono font-bold text-white">
                  {formatDistance(entry.totalError)}
                </div>
              </div>
            </div>

            {gameState === 'revealed' && (
              <div className="mt-2 pt-2 border-t border-slate-800/50 flex justify-between items-center animate-fade-in">
                <span className="text-[9px] font-mono font-bold text-slate-500 tracking-tighter uppercase">Round Error</span>
                <span className={`text-[10px] font-mono font-bold ${entry.currentError < 100 ? 'text-emerald-400' : 'text-orange-400'}`}>
                  +{formatDistance(entry.currentError)}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Leaderboard;
