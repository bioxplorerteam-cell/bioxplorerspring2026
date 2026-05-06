import React, { useState } from 'react';
import axios from 'axios';
import HeatmapView from './HeatmapView';
import { FaUpload, FaChevronDown, FaChevronRight } from 'react-icons/fa';
import API_URL from '../config';

const CATEGORY_COLORS = {
    'Diseases & Pathology':    { bg: '#ffebee', border: '#ef9a9a', check: '#c62828' },
    'Chemicals & Drugs':       { bg: '#e8f5e9', border: '#a5d6a7', check: '#2e7d32' },
    'Anatomy':                 { bg: '#e3f2fd', border: '#90caf9', check: '#1565c0' },
    'Organisms':               { bg: '#f3e5f5', border: '#ce93d8', check: '#6a1b9a' },
    'Procedures & Techniques': { bg: '#fff8e1', border: '#ffe082', check: '#f57f17' },
    'Demographics':            { bg: '#fce4ec', border: '#f48fb1', check: '#880e4f' },
    'Phenomena & Processes':   { bg: '#e0f7fa', border: '#80deea', check: '#00695c' },
    'Other':                   { bg: '#f5f5f5', border: '#bdbdbd', check: '#424242' },
};

const FileUpload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');
    const [abstractText, setAbstractText] = useState('');
    const [title, setTitle] = useState('');
    const [meshTerms, setMeshTerms] = useState([]);
    const [meshCategories, setMeshCategories] = useState({});
    const [selectedMeshTerms, setSelectedMeshTerms] = useState([]);
    const [attentionWeights, setAttentionWeights] = useState({});
    const [loading, setLoading] = useState(false);
    const [openGroups, setOpenGroups] = useState({});

    const toggleGroup = (cat) =>
        setOpenGroups(prev => ({ ...prev, [cat]: !prev[cat] }));

    const toggleTerm = (term) =>
        setSelectedMeshTerms(prev =>
            prev.includes(term) ? prev.filter(t => t !== term) : [...prev, term]
        );

    // Build grouped structure
    const groupedTerms = meshTerms.reduce((acc, term) => {
        const cat = (meshCategories && meshCategories[term]) || 'Other';
        if (!acc[cat]) acc[cat] = [];
        acc[cat].push(term);
        return acc;
    }, {});

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setAbstractText('');
        setTitle('');
        setMeshTerms([]);
        setMeshCategories({});
        setAttentionWeights({});
        setSelectedMeshTerms([]);
        setOpenGroups({});
    }

    const handleUpload = async () => {
        if (!selectedFile) {
            setUploadStatus('No file selected');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);
        setLoading(true);

        try {
            // Use the NEW /upload endpoint that returns word-level attention
            const response = await axios.post(`${API_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });
            
            setUploadStatus('File uploaded successfully');
            setAbstractText(response.data.abstract || '');
            setTitle(response.data.title || 'Therapeutic targeting of oncogenic transcription factors');
            setMeshTerms(response.data.mesh_terms || []);
            setMeshCategories(response.data.mesh_categories || {});
            setAttentionWeights(response.data.attention_weights || {});
            setSelectedMeshTerms([]);
            setSelectedFile(null);
        } catch (error) {
            console.error('Upload error:', error);
            setUploadStatus('File upload failed: ' + (error.response?.data?.detail || error.message));
        } finally {
            setLoading(false);
        }
    }

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            width: '100%',
            padding: '20px',
            boxSizing: 'border-box',
            fontFamily: 'Arial, sans-serif',
        }}>
            <div style={{
                backgroundColor: '#4a90d9',
                width: '100%',
                height: '70px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: '10px',
                marginBottom: '20px',
            }}>
                <h2 style={{ color: 'black', margin: 0 }}>Personalized Medical Literature Curation</h2>
            </div>

            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'flex-start',
                height: '80vh',
                flexDirection: 'row',
                gap: '20px',
                width: '100%',
            }}>
                {/* Left Box */}
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    backgroundColor: '#b4d3b2',
                    width: '30%',
                    height: '100%',
                    padding: '20px',
                    borderRadius: '10px',
                    boxSizing: 'border-box',
                    overflowY: 'auto',
                    gap: '14px',
                }}>
                    {uploadStatus && (
                        <p style={{ 
                            fontSize: '14px', 
                            margin: 0,
                            color: uploadStatus.includes('failed') ? '#e74c3c' : '#27ae60'
                        }}>
                            {uploadStatus}
                        </p>
                    )}

                    {/* File input row */}
                    <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <input 
                            className="button-27" 
                            style={{ flex: 1 }} 
                            type='file' 
                            accept=".pdf"
                            onChange={handleFileChange} 
                        />
                        <FaUpload
                            onClick={handleUpload}
                            style={{ 
                                marginLeft: '10px', 
                                cursor: selectedFile ? 'pointer' : 'not-allowed', 
                                color: selectedFile ? 'white' : '#ccc', 
                                fontSize: '20px' 
                            }}
                        />
                    </div>

                    {/* Extract button */}
                    <button 
                        className="button-27" 
                        onClick={handleUpload} 
                        disabled={!selectedFile || loading}
                        style={{
                            opacity: (!selectedFile || loading) ? 0.5 : 1,
                            cursor: (!selectedFile || loading) ? 'not-allowed' : 'pointer',
                            width: '100%'
                        }}
                    >
                        {loading ? 'Extracting MeSH Terms...' : 'Extract MeSH Terms'}
                    </button>

                    {/* MeSH accordion — shown after extraction */}
                    {meshTerms.length > 0 && (
                        <div>
                            <p style={{ margin: '0 0 8px 0', fontSize: '13px', fontWeight: '600', color: '#2c5f2d' }}>
                                MeSH Terms — click a group to expand:
                            </p>
                            {Object.entries(groupedTerms).map(([cat, terms]) => {
                                const colors = CATEGORY_COLORS[cat] || CATEGORY_COLORS['Other'];
                                const isOpen = !!openGroups[cat];
                                const selectedCount = terms.filter(t => selectedMeshTerms.includes(t)).length;
                                return (
                                    <div key={cat} style={{ marginBottom: '6px', borderRadius: '8px', overflow: 'hidden', border: `1px solid ${colors.border}` }}>
                                        {/* Accordion header */}
                                        <div
                                            onClick={() => toggleGroup(cat)}
                                            style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'space-between',
                                                padding: '8px 12px',
                                                backgroundColor: colors.bg,
                                                cursor: 'pointer',
                                                userSelect: 'none',
                                            }}
                                        >
                                            <span style={{ fontSize: '12px', fontWeight: '700', color: colors.check, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                                                {cat}
                                            </span>
                                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                {selectedCount > 0 && (
                                                    <span style={{ fontSize: '11px', backgroundColor: colors.check, color: 'white', borderRadius: '10px', padding: '1px 7px' }}>
                                                        {selectedCount}
                                                    </span>
                                                )}
                                                {isOpen
                                                    ? <FaChevronDown size={11} color={colors.check} />
                                                    : <FaChevronRight size={11} color={colors.check} />
                                                }
                                            </span>
                                        </div>

                                        {/* Accordion body — checkboxes */}
                                        {isOpen && (
                                            <div style={{ backgroundColor: '#fff', padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                                {terms.map(term => (
                                                    <label key={term} style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px', color: '#333' }}>
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedMeshTerms.includes(term)}
                                                            onChange={() => toggleTerm(term)}
                                                            style={{ accentColor: colors.check, width: '15px', height: '15px', cursor: 'pointer', flexShrink: 0 }}
                                                        />
                                                        {term}
                                                    </label>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Right Box */}
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    backgroundColor: '#b4d3b2',
                    width: '70%',
                    height: '100%',
                    padding: '20px',
                    borderRadius: '10px',
                    boxSizing: 'border-box',
                    overflowY: 'auto',
                }}>
                    {title && (
                        <h3 style={{
                            fontSize: '24px',
                            fontWeight: 'bold',
                            marginBottom: '20px',
                            color: '#2c5f2d',
                        }}>
                            {title}
                        </h3>
                    )}
                    
                    {loading && (
                        <div style={{
                            textAlign: 'center',
                            padding: '40px',
                            color: '#2c5f2d',
                            fontSize: '18px'
                        }}>
                            <p>Analyzing abstract with LLaMA...</p>
                            <p style={{ fontSize: '14px', marginTop: '10px' }}>This may take 30-60 seconds</p>
                        </div>
                    )}
                    
                    {/* Use the NEW HeatmapView with word-level attention */}
                    {!loading && meshTerms.length > 0 && (
                        <HeatmapView
                            abstract={abstractText}
                            meshTerms={meshTerms}
                            meshCategories={meshCategories}
                            selectedMeshTerms={selectedMeshTerms}
                            attentionWeights={attentionWeights}
                            onMeshSelect={(term) => setSelectedMeshTerms(prev =>
                                prev.includes(term) ? prev.filter(t => t !== term) : [...prev, term]
                            )}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}

export default FileUpload;