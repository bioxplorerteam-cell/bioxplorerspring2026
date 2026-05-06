import React from 'react';
import PubMedArticles from './PubMedArticles';

/**
 * HeatmapView component displays abstract text with word-level attention weights
 * Shows attention weights below each word with color highlighting
 * Color gradient: white (0) -> dark red (high attention)
 */
const HeatmapView = ({ abstract, meshTerms, meshCategories, selectedMeshTerms, attentionWeights, onMeshSelect }) => {
  const OVERLAP_THRESHOLD = 0.15;
  const getWordColor = (weight) => {
    if (!weight || weight === 0) return 'rgb(255, 255, 255)'; // White for zero/missing
    
    // Clamp weight between 0 and 1
    const clampedWeight = Math.max(0, Math.min(1, weight));
    
    // Color gradient: white -> light pink -> red -> dark red
    if (clampedWeight < 0.1) {
      // Very low attention: white to very light pink
      const red = 255;
      const green = Math.floor(255 - (clampedWeight * 10) * 20);
      const blue = Math.floor(255 - (clampedWeight * 10) * 20);
      return `rgb(${red}, ${green}, ${blue})`;
    } else if (clampedWeight < 0.3) {
      // Low attention: light pink to pink
      const scaledWeight = (clampedWeight - 0.1) / 0.2;
      const red = 255;
      const green = Math.floor(235 - scaledWeight * 85);
      const blue = Math.floor(235 - scaledWeight * 85);
      return `rgb(${red}, ${green}, ${blue})`;
    } else if (clampedWeight < 0.6) {
      // Medium attention: pink to red
      const scaledWeight = (clampedWeight - 0.3) / 0.3;
      const red = 255;
      const green = Math.floor(150 - scaledWeight * 100);
      const blue = Math.floor(150 - scaledWeight * 100);
      return `rgb(${red}, ${green}, ${blue})`;
    } else {
      // High attention: red to dark red
      const scaledWeight = (clampedWeight - 0.6) / 0.4;
      const red = Math.floor(255 - scaledWeight * 55);
      const green = Math.floor(50 - scaledWeight * 50);
      const blue = Math.floor(50 - scaledWeight * 50);
      return `rgb(${red}, ${green}, ${blue})`;
    }
  };

  const renderAbstractWithAttention = () => {
    if (!abstract || !selectedMeshTerms || selectedMeshTerms.length === 0) {
      return <p style={{ padding: '20px', color: '#666', fontSize: '14px' }}>
        {abstract || 'Select a MeSH term to view attention heatmap'}
      </p>;
    }

    const words = abstract.split(/\s+/);

    return (
      <div style={{ 
        padding: '20px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '4px 2px',
        alignItems: 'flex-start',
        fontSize: '14px',
        fontFamily: 'Georgia, serif',
        lineHeight: '1.0'
      }}>
        {words.map((word, index) => {
          const termWeights = selectedMeshTerms.map(
            term => attentionWeights?.[term]?.[index] ?? 0
          );
          const maxWeight = termWeights.length ? Math.max(...termWeights) : 0;
          const overlapCount = termWeights.filter(w => w > OVERLAP_THRESHOLD).length;

          let backgroundColor;
          let textColor = '#333';
          let fontWeight = '400';

          if (overlapCount >= 2) {
            // Blue for words attended to by multiple selected terms
            const intensity = Math.min(1, maxWeight * 1.5);
            const rg = Math.floor(220 - intensity * 180);
            backgroundColor = `rgb(${rg}, ${rg}, 255)`;
            textColor = intensity > 0.5 ? 'white' : '#333';
            fontWeight = '600';
          } else {
            backgroundColor = getWordColor(maxWeight);
            textColor = maxWeight > 0.6 ? 'white' : '#333';
            fontWeight = maxWeight > 0.5 ? '600' : '400';
          }

          return (
            <span
              key={index}
              style={{
                display: 'inline-flex',
                flexDirection: 'column',
                alignItems: 'center',
                minWidth: 'fit-content',
                marginBottom: '28px',
                position: 'relative'
              }}
            >
              <span style={{ 
                padding: '2px 4px',
                backgroundColor,
                borderRadius: '3px',
                fontWeight,
                color: textColor,
                fontSize: '14px',
                whiteSpace: 'nowrap',
                display: 'inline-block'
              }}>
                {word}
              </span>
              <span
                style={{
                  fontSize: '15px',
                  fontWeight: '700',
                  color: overlapCount >= 2 ? '#1565c0' : (maxWeight > 0.3 ? '#d32f2f' : '#666'),
                  fontFamily: 'monospace',
                  marginTop: '4px',
                  whiteSpace: 'nowrap'
                }}
              >
                {maxWeight.toFixed(3)}
              </span>
            </span>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{ 
      width: '100%',
      backgroundColor: '#fff',
      borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      overflow: 'hidden'
    }}>
      {/* Abstract with Word-Overlap Highlighting */}
      <div style={{
        maxHeight: '600px',
        overflowY: 'auto',
        backgroundColor: '#fafafa'
      }}>
        {/* Heuristic disclaimer — always visible */}
        <div style={{
          padding: '8px 20px',
          backgroundColor: '#e8eaf6',
          borderBottom: '1px solid #c5cae9',
          fontSize: '12px',
          color: '#37474f'
        }}>
          <strong>Note:</strong> Highlighting shows <em>word-overlap</em> between the abstract and each predicted MeSH term — it is a text-matching heuristic, not transformer attention.
        </div>
        {selectedMeshTerms && selectedMeshTerms.length > 0 && (
          <div style={{
            padding: '15px 20px',
            backgroundColor: '#fff3cd',
            borderBottom: '1px solid #ffeaa7',
            fontWeight: '600',
            color: '#856404',
            fontSize: '14px'
          }}>
            Showing word overlap for:{' '}
            {selectedMeshTerms.map((term, i) => (
              <span key={term} style={{ color: '#d32f2f' }}>
                {term}{i < selectedMeshTerms.length - 1 ? ', ' : ''}
              </span>
            ))}
            {selectedMeshTerms.length > 1 && (
              <span style={{ color: '#1565c0', marginLeft: '12px', fontWeight: '500' }}>
                ● Blue = shared overlap across multiple terms
              </span>
            )}
          </div>
        )}
        {renderAbstractWithAttention()}
      </div>

      {/* Legend */}
      <div style={{
        padding: '15px 20px',
        borderTop: '2px solid #e0e0e0',
        backgroundColor: '#f8f9fa',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '15px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: '600', color: '#333', fontSize: '14px' }}>Word Overlap Score:</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ 
              width: '40px', 
              height: '20px', 
              backgroundColor: getWordColor(0.0),
              border: '1px solid #ccc',
              borderRadius: '3px'
            }} />
            <span style={{ fontSize: '12px', color: '#666' }}>Low (0.0)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ 
              width: '40px', 
              height: '20px', 
              backgroundColor: getWordColor(0.3),
              border: '1px solid #ccc',
              borderRadius: '3px'
            }} />
            <span style={{ fontSize: '12px', color: '#666' }}>Medium (0.3)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ 
              width: '40px', 
              height: '20px', 
              backgroundColor: getWordColor(0.6),
              border: '1px solid #ccc',
              borderRadius: '3px'
            }} />
            <span style={{ fontSize: '12px', color: '#666' }}>High (0.6)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ 
              width: '40px', 
              height: '20px', 
              backgroundColor: getWordColor(0.9),
              border: '1px solid #ccc',
              borderRadius: '3px'
            }} />
            <span style={{ fontSize: '12px', color: '#666' }}>Very High (0.9)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ 
              width: '40px', 
              height: '20px', 
              backgroundColor: 'rgb(40, 40, 255)',
              border: '1px solid #ccc',
              borderRadius: '3px'
            }} />
            <span style={{ fontSize: '12px', color: '#1565c0', fontWeight: '600' }}>Overlap</span>
          </div>
        </div>
      </div>

      {/* PubMed Related Articles */}
      {selectedMeshTerms && selectedMeshTerms.length > 0 && (
        <PubMedArticles meshTerm={selectedMeshTerms[0]} />
      )}
    </div>
  );
};

export default HeatmapView;