
import { RoundData } from './types';

export const COLORS = {
  TRUTH: '#10B981', // Emerald Green
  HUMAN: '#0EA5E9', // Ocean Blue
  MODEL_A: '#D946EF', // Fuchsia
  MODEL_B: '#F59E0B', // Amber
  MODEL_C: '#6366F1', // Indigo
  MODEL_D: '#EF4444', // Rose Red
};

export const TARGET_STATES = [
  'Texas', 'Kansas', 'Oregon', 'Nevada', 'Oklahoma', 
  'Montana', 'Idaho', 'Nebraska', 'New Mexico', 'Wyoming'
];

const STATE_BOUNDS: Record<string, [number, number, number, number]> = {
  'Texas': [25.8, 36.5, -106.6, -93.5],
  'Kansas': [37.0, 40.0, -102.0, -94.6],
  'Oregon': [42.0, 46.2, -124.5, -116.5],
  'Nevada': [35.0, 42.0, -120.0, -114.0],
  'Oklahoma': [33.6, 37.0, -103.0, -94.4],
  'Montana': [44.3, 49.0, -116.0, -104.0],
  'Idaho': [42.0, 49.0, -117.2, -111.0],
  'Nebraska': [40.0, 43.0, -104.0, -95.3],
  'New Mexico': [31.3, 37.0, -109.0, -103.0],
  'Wyoming': [41.0, 45.0, -111.0, -104.0]
};

const getRandomCoordInTargetStates = () => {
  const stateName = TARGET_STATES[Math.floor(Math.random() * TARGET_STATES.length)];
  const bounds = STATE_BOUNDS[stateName];
  return {
    lat: Math.random() * (bounds[1] - bounds[0]) + bounds[0],
    lng: Math.random() * (bounds[3] - bounds[2]) + bounds[2],
    state: stateName
  };
};

export const generateRandomDataset = (count: number): RoundData[] => {
  const placeNames = ["Sector A", "Quadrant B", "Node C", "Site D", "Point E"];
  return Array.from({ length: count }, (_, i) => {
    const { lat, lng, state } = getRandomCoordInTargetStates();
    const truth = { lat, lng };
    return {
      id: `IMG-88${i + 1}`,
      imageUrl: `https://picsum.photos/seed/${Math.random()}/1200/800`,
      locationName: `${placeNames[i % placeNames.length]} in ${state}`,
      truth,
      predictions: {
        'NeuralNet-Alpha': { lat: truth.lat + (Math.random() - 0.5) * 1.5, lng: truth.lng + (Math.random() - 0.5) * 1.5 },
        'VisualTrans-9': { lat: truth.lat + (Math.random() - 0.5) * 0.8, lng: truth.lng + (Math.random() - 0.5) * 0.8 },
        'Legacy-Vision': { lat: truth.lat + (Math.random() - 0.5) * 4.0, lng: truth.lng + (Math.random() - 0.5) * 4.0 },
        'SpatialAI-v2': { lat: truth.lat + (Math.random() - 0.5) * 1.2, lng: truth.lng + (Math.random() - 0.5) * 1.2 }
      }
    };
  });
};

export const DATASET: RoundData[] = generateRandomDataset(5);
