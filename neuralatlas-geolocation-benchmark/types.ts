
export interface Coordinate {
  lat: number;
  lng: number;
}

export interface RoundData {
  id: string;
  imageUrl: string;
  truth: Coordinate;
  locationName: string;
  predictions: Record<string, Coordinate>;
}

export type GameState = 'guessing' | 'revealed' | 'finished';

export interface ScoreEntry {
  name: string;
  currentError: number; // in km
  totalError: number;   // in km
  color: string;
  type: 'human' | 'model';
}
