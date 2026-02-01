
import React from 'react';
import { RoundData, GameState } from '../types';

interface ImagePanelProps {
  round: RoundData;
  gameState: GameState;
  currentRoundIndex: number;
  totalRounds: number;
}

const ImagePanel: React.FC<ImagePanelProps> = ({ round, gameState, currentRoundIndex, totalRounds }) => {
  return (
    <div className="flex flex-col h-full bg-slate-950 border-b border-slate-800">
      <div className="p-4 bg-slate-900 flex justify-between items-center border-b border-slate-800">
        <div>
          <h1 className="text-xl font-black tracking-tighter text-white uppercase italic">NeuralAtlas</h1>
          <p className="text-[10px] text-cyan-500 font-mono font-bold uppercase tracking-widest">Benchmark Sequence</p>
        </div>
        <div className="text-right">
          <span className="text-[10px] font-mono font-bold text-slate-500">PHASE</span>
          <div className="text-xl font-black text-white">
            {currentRoundIndex + 1}<span className="text-slate-700">/</span>{totalRounds}
          </div>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden group">
        <img
          src={round.imageUrl}
          alt="Target Location"
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent"></div>
        {gameState === 'revealed' && (
          <div className="absolute inset-x-0 bottom-0 bg-slate-900/95 backdrop-blur-md p-4 border-t border-emerald-500/50 animate-fade-in">
             <p className="text-[10px] font-mono font-bold text-emerald-400 mb-1 tracking-widest uppercase">Target Resolved</p>
             <h2 className="text-lg font-black text-white">{round.locationName}</h2>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImagePanel;
