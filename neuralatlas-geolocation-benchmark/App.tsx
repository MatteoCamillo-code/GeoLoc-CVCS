
import React, { useState } from 'react';
import { GameState, ScoreEntry, Coordinate } from './types';
import { DATASET, COLORS, generateRandomDataset, TARGET_STATES } from './constants';
import { calculateDistance } from './utils';
import ImagePanel from './components/ImagePanel';
import Leaderboard from './components/Leaderboard';
import MapComponent from './components/MapComponent';

const App: React.FC = () => {
  const [sessionDataset, setSessionDataset] = useState(DATASET);
  const [currentRoundIndex, setCurrentRoundIndex] = useState(0);
  const [gameState, setGameState] = useState<GameState>('guessing');
  const [userGuess, setUserGuess] = useState<Coordinate | null>(null);
  const [showIntro, setShowIntro] = useState(true);
  
  const [scores, setScores] = useState<ScoreEntry[]>(() => {
    const initialScores: ScoreEntry[] = [
      { name: 'Human Subject', currentError: 0, totalError: 0, color: COLORS.HUMAN, type: 'human' }
    ];
    Object.keys(sessionDataset[0].predictions).forEach((modelName, index) => {
      const colors = [COLORS.MODEL_A, COLORS.MODEL_B, COLORS.MODEL_C, COLORS.MODEL_D];
      initialScores.push({
        name: modelName,
        currentError: 0,
        totalError: 0,
        color: colors[index % colors.length],
        type: 'model'
      });
    });
    return initialScores;
  });

  const currentRound = sessionDataset[currentRoundIndex];
  const isLastRound = currentRoundIndex === sessionDataset.length - 1;

  const handleGuess = (coord: Coordinate) => {
    if (gameState === 'guessing') setUserGuess(coord);
  };

  const handleReveal = () => {
    if (!userGuess) return;
    setGameState('revealed');
    setScores(prevScores => {
      return prevScores.map(entry => {
        let error = 0;
        if (entry.type === 'human') {
          error = calculateDistance(userGuess, currentRound.truth);
        } else {
          const prediction = currentRound.predictions[entry.name];
          error = calculateDistance(prediction, currentRound.truth);
        }
        return {
          ...entry,
          currentError: error,
          totalError: entry.totalError + error
        };
      });
    });
  };

  const handleNextRound = () => {
    if (!isLastRound) {
      setCurrentRoundIndex(prev => prev + 1);
      setGameState('guessing');
      setUserGuess(null);
    } else {
      setGameState('finished');
    }
  };

  const handleRestart = () => {
      const newData = generateRandomDataset(5);
      setSessionDataset(newData);
      setCurrentRoundIndex(0);
      setGameState('guessing');
      setUserGuess(null);
      setScores(prev => prev.map(s => ({ ...s, currentError: 0, totalError: 0 })));
  };

  if (showIntro) {
    return (
      <div className="fixed inset-0 z-[5000] flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4">
        <div className="bg-slate-900 w-full max-w-lg rounded-3xl shadow-2xl overflow-hidden border border-slate-800">
           <div className="bg-slate-950 p-8 text-center border-b border-slate-800">
              <h1 className="text-3xl font-black italic uppercase tracking-tighter text-white mb-2">NeuralAtlas</h1>
              <div className="h-1.5 w-12 bg-emerald-500 mx-auto rounded-full"></div>
           </div>
           <div className="p-8 space-y-6">
              <div className="space-y-4 text-slate-400 text-sm leading-relaxed">
                 <p className="font-bold text-cyan-500 uppercase tracking-widest text-[10px]">Benchmark Objective:</p>
                 <p>Compare your spatial intuition against 4 distinct <b>Computer Vision models</b> in 5 geographic challenges. The dataset is restricted to the following states:</p>
                 <div className="grid grid-cols-2 gap-2 pt-1">
                    {TARGET_STATES.map(s => (
                      <div key={s} className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                        <span className="text-[11px] font-bold text-slate-200">{s}</span>
                      </div>
                    ))}
                 </div>
              </div>
              <div className="space-y-3 py-2 border-t border-slate-800">
                 <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-sky-900/50 flex items-center justify-center text-sky-400 font-bold text-xs">1</div>
                    <p className="text-xs font-semibold text-slate-300">Analyze the high-res image probe</p>
                 </div>
                 <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-sky-900/50 flex items-center justify-center text-sky-400 font-bold text-xs">2</div>
                    <p className="text-xs font-semibold text-slate-300">Click the map inside highlighted states to guess</p>
                 </div>
                 <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-sky-900/50 flex items-center justify-center text-sky-400 font-bold text-xs">3</div>
                    <p className="text-xs font-semibold text-slate-300">Submit and analyze spatial error vs. AI</p>
                 </div>
              </div>
              <button 
                onClick={() => setShowIntro(false)}
                className="w-full py-4 bg-white hover:bg-slate-200 text-slate-950 font-black text-sm rounded-2xl shadow-xl transition-all uppercase tracking-widest active:scale-95"
              >
                Accept Mission
              </button>
           </div>
        </div>
      </div>
    );
  }

  if (gameState === 'finished') {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-slate-950 p-4 overflow-hidden relative">
        <div className="max-w-md w-full bg-slate-900 border border-slate-800 p-8 rounded-3xl shadow-2xl text-center space-y-6 relative z-10">
            <div>
              <h2 className="text-2xl font-black tracking-tighter text-white uppercase italic">Final Report</h2>
              <div className="h-1 w-16 bg-emerald-500 mx-auto mt-2 rounded-full"></div>
            </div>
            <p className="text-slate-500 font-mono text-[10px] font-bold uppercase tracking-widest">Benchmark Sequence Complete</p>
            <div className="py-2 max-h-[50vh] overflow-y-auto custom-scrollbar">
               <Leaderboard scores={scores} gameState="finished" />
            </div>
            <button onClick={handleRestart} className="w-full py-4 bg-white hover:bg-slate-200 text-slate-950 font-black text-sm rounded-2xl transition-all shadow-xl uppercase tracking-widest">
                Initiate New Sequence
            </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row h-screen w-screen overflow-hidden bg-slate-950">
      <div className="w-full md:w-1/3 lg:w-1/4 h-1/2 md:h-full flex flex-col border-r border-slate-800 z-20 shadow-xl shadow-black/50">
        <div className="flex-1">
          <ImagePanel round={currentRound} gameState={gameState} currentRoundIndex={currentRoundIndex} totalRounds={sessionDataset.length} />
        </div>
        <div className="h-1/2 md:h-2/5 overflow-hidden flex flex-col">
           <Leaderboard scores={scores} gameState={gameState} />
        </div>
      </div>

      <div className="flex-1 h-1/2 md:h-full relative z-10 bg-slate-50">
        <MapComponent round={currentRound} gameState={gameState} userGuess={userGuess} onGuess={handleGuess} />

        {gameState === 'revealed' && (
             <div className="absolute inset-0 bg-transparent cursor-not-allowed z-[2500]"></div>
        )}

        {gameState === 'guessing' && (
             <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-slate-900/95 backdrop-blur-sm px-4 py-2 rounded-full border border-slate-700 text-white text-[10px] font-black font-mono tracking-widest shadow-xl pointer-events-none z-[2000] flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                {userGuess ? 'LOCATION LOCKED' : 'CLICK HIGHLIGHTED STATES'}
             </div>
        )}

        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4 pointer-events-none w-full max-w-[200px] z-[3000]">
          {gameState === 'guessing' ? (
            <button
              onClick={handleReveal}
              disabled={!userGuess}
              className={`w-full py-3.5 rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-2xl transition-all border flex items-center justify-center gap-2 pointer-events-auto ${
                  userGuess 
                  ? 'bg-sky-600 hover:bg-sky-500 text-white border-sky-400 scale-105 active:scale-95' 
                  : 'bg-slate-800 text-slate-600 border-slate-700 cursor-not-allowed opacity-50'
              }`}
            >
              Confirm Guess
            </button>
          ) : (
            <button
              onClick={handleNextRound}
              className={`w-full py-3.5 ${isLastRound ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-white hover:bg-slate-200 text-slate-950'} rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-2xl transition-all border flex items-center justify-center gap-2 pointer-events-auto scale-105 active:scale-95 animate-fade-in`}
            >
              {isLastRound ? 'Results' : 'Next Round'}
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
