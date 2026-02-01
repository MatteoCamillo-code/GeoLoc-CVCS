
import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Coordinate, RoundData, GameState } from '../types';
import { COLORS, TARGET_STATES } from '../constants';

interface MapComponentProps {
  round: RoundData;
  gameState: GameState;
  userGuess: Coordinate | null;
  onGuess: (coord: Coordinate) => void;
}

const MapComponent: React.FC<MapComponentProps> = ({ round, gameState, userGuess, onGuess }) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);
  const geojsonRef = useRef<L.GeoJSON | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    mapRef.current = L.map(mapContainerRef.current, {
      center: [39, -100], 
      zoom: 4,
      minZoom: 3,
      maxZoom: 12,
      zoomControl: false,
      attributionControl: false,
    });

    // Switched to 'light_nolabels' as per user request to remove names
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(mapRef.current);

    markersRef.current = L.layerGroup().addTo(mapRef.current);

    // Style States GeoJSON
    fetch('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
      .then(res => res.json())
      .then(data => {
        if (!mapRef.current) return;
        
        geojsonRef.current = L.geoJSON(data, {
          style: (feature) => {
            const isTarget = TARGET_STATES.includes(feature?.properties?.name);
            return {
              fillColor: isTarget ? '#10B981' : '#F1F5F9',
              weight: isTarget ? 3 : 1, // Increased weight for target states
              opacity: 1,
              color: isTarget ? '#059669' : '#CBD5E1', // Darker green for bold borders
              fillOpacity: isTarget ? 0.4 : 0.6,
            };
          }
        }).addTo(mapRef.current);
        
        // CRITICAL: Bring all target layers to front to avoid shared borders being covered by grey ones
        geojsonRef.current.eachLayer((layer: any) => {
          if (layer.feature && TARGET_STATES.includes(layer.feature.properties.name)) {
            layer.bringToFront();
          }
        });
        
        mapRef.current.invalidateSize();
      });

    L.control.zoom({ position: 'bottomright' }).addTo(mapRef.current);

    mapRef.current.on('click', (e: L.LeafletMouseEvent) => {
      if (gameState === 'guessing') {
        onGuess({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
    });

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !markersRef.current) return;

    const markers = markersRef.current;
    markers.clearLayers();

    const points: L.LatLngExpression[] = [];

    if (userGuess) {
      const userIcon = L.divIcon({
        className: 'custom-div-icon',
        html: `<div style="background-color: ${COLORS.HUMAN}; width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 4px 6px rgba(0,0,0,0.2)"></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      });
      L.marker([userGuess.lat, userGuess.lng], { icon: userIcon }).addTo(markers);
      points.push([userGuess.lat, userGuess.lng]);
    }

    if (gameState === 'revealed') {
      const truthIcon = L.divIcon({
        className: 'custom-div-icon',
        html: `<div style="background-color: ${COLORS.TRUTH}; width: 16px; height: 16px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 10px ${COLORS.TRUTH}"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8]
      });
      L.marker([round.truth.lat, round.truth.lng], { icon: truthIcon }).addTo(markers);
      points.push([round.truth.lat, round.truth.lng]);

      const modelColors = [COLORS.MODEL_A, COLORS.MODEL_B, COLORS.MODEL_C, COLORS.MODEL_D];
      Object.entries(round.predictions).forEach(([name, coord], idx) => {
        const color = modelColors[idx % modelColors.length];
        const aiIcon = L.divIcon({
          className: 'custom-div-icon',
          html: `<div style="background-color: ${color}; width: 12px; height: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1)"></div>`,
          iconSize: [12, 12],
          iconAnchor: [6, 6]
        });
        L.marker([coord.lat, coord.lng], { icon: aiIcon }).addTo(markers);
        points.push([coord.lat, coord.lng]);
        L.polyline([[coord.lat, coord.lng], [round.truth.lat, round.truth.lng]], {
          color: color, weight: 1.5, dashArray: '4, 4', opacity: 0.5
        }).addTo(markers);
      });

      if (userGuess) {
        L.polyline([[userGuess.lat, userGuess.lng], [round.truth.lat, round.truth.lng]], {
          color: COLORS.HUMAN, weight: 2, dashArray: '2, 4', opacity: 0.7
        }).addTo(markers);
      }

      if (points.length > 1 && mapRef.current) {
        const bounds = L.latLngBounds(points);
        mapRef.current.fitBounds(bounds, { padding: [100, 100], animate: true, duration: 1.2 });
      }
    }
  }, [gameState, userGuess, round]);

  return <div ref={mapContainerRef} className="w-full h-full bg-[#f8fafc]" />;
};

export default MapComponent;
