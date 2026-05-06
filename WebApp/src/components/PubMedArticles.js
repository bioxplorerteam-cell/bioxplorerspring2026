import React, { useState, useEffect } from 'react';
import { searchPubMedByMesh } from '../services/pubmedService';

/**
 * PubMedArticles Component
 * Displays related PubMed articles for a selected MeSH term
 */
const PubMedArticles = ({ meshTerm }) => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const fetchArticles = async () => {
      if (!meshTerm) {
        setArticles([]);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const results = await searchPubMedByMesh([meshTerm], 5);
        setArticles(results);
      } catch (err) {
        setError('Failed to fetch articles from PubMed. Please try again.');
        console.error('PubMed fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchArticles();
  }, [meshTerm]);

  if (!meshTerm) {
    return null;
  }

  return (
    <div style={{
      marginTop: '20px',
      padding: '20px',
      backgroundColor: '#fff',
      borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
    }}>
      {/* Header with expand/collapse */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px',
        paddingBottom: '10px',
        borderBottom: '2px solid #e0e0e0'
      }}>
        <h3 style={{
          margin: 0,
          color: '#333',
          fontSize: '18px',
          fontWeight: '600'
        }}>
          Related PubMed Articles
          <span style={{
            marginLeft: '8px',
            fontSize: '14px',
            color: '#666',
            fontWeight: '400'
          }}>
            ({meshTerm})
          </span>
        </h3>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            padding: '6px 12px',
            backgroundColor: '#f0f0f0',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            color: '#333',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.backgroundColor = '#e0e0e0';
          }}
          onMouseLeave={(e) => {
            e.target.style.backgroundColor = '#f0f0f0';
          }}
        >
          {isExpanded ? 'Collapse ▲' : 'Expand ▼'}
        </button>
      </div>

      {/* Content */}
      {isExpanded && (
        <div>
          {loading && (
            <div style={{
              textAlign: 'center',
              padding: '40px',
              color: '#666'
            }}>
              <div style={{
                display: 'inline-block',
                width: '40px',
                height: '40px',
                border: '4px solid #f0f0f0',
                borderTopColor: '#4CAF50',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }} />
              <p style={{ marginTop: '15px', fontSize: '14px' }}>
                Searching PubMed...
              </p>
              <style>
                {`
                  @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                  }
                `}
              </style>
            </div>
          )}

          {error && (
            <div style={{
              padding: '15px',
              backgroundColor: '#ffebee',
              border: '1px solid #ef5350',
              borderRadius: '4px',
              color: '#c62828',
              fontSize: '14px'
            }}>
              {error}
            </div>
          )}

          {!loading && !error && articles.length === 0 && (
            <div style={{
              padding: '20px',
              textAlign: 'center',
              color: '#666',
              fontSize: '14px'
            }}>
              No related articles found for this MeSH term.
            </div>
          )}

          {!loading && !error && articles.length > 0 && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '15px'
            }}>
              {articles.map((article, index) => (
                <div
                  key={article.pmid}
                  style={{
                    padding: '15px',
                    backgroundColor: '#f8f9fa',
                    borderLeft: '4px solid #4CAF50',
                    borderRadius: '4px',
                    transition: 'all 0.2s ease',
                    cursor: 'pointer'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#e8f5e9';
                    e.currentTarget.style.transform = 'translateX(5px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#f8f9fa';
                    e.currentTarget.style.transform = 'translateX(0)';
                  }}
                  onClick={() => window.open(article.pubmedUrl, '_blank')}
                >
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: '10px'
                  }}>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        fontSize: '12px',
                        color: '#666',
                        marginBottom: '5px',
                        fontWeight: '600'
                      }}>
                        #{index + 1} • PMID: {article.pmid}
                      </div>
                      <h4 style={{
                        margin: '0 0 8px 0',
                        fontSize: '16px',
                        fontWeight: '600',
                        color: '#1976d2',
                        lineHeight: '1.4'
                      }}>
                        {article.title}
                      </h4>
                      <div style={{
                        fontSize: '13px',
                        color: '#555',
                        marginBottom: '5px'
                      }}>
                        <strong>Authors:</strong> {article.authors}
                        {article.authors.includes(',') && ' et al.'}
                      </div>
                      <div style={{
                        fontSize: '13px',
                        color: '#555',
                        marginBottom: '5px'
                      }}>
                        <strong>Journal:</strong> {article.journal}
                      </div>
                      <div style={{
                        fontSize: '13px',
                        color: '#555'
                      }}>
                        <strong>Published:</strong> {article.pubdate}
                        {article.doi && (
                          <span style={{ marginLeft: '10px' }}>
                            <strong>DOI:</strong> {article.doi}
                          </span>
                        )}
                      </div>
                    </div>
                    <div style={{
                      fontSize: '12px',
                      color: '#1976d2',
                      fontWeight: '600',
                      whiteSpace: 'nowrap',
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      View on PubMed →
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PubMedArticles;
