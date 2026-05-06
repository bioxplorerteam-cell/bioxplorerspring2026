/**
 * PubMed API Service
 * Provides functions to search PubMed articles using E-utilities API
 */

const API_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils';

/**
 * Search PubMed articles by MeSH terms
 * @param {Array<string>} meshTerms - Array of MeSH terms to search
 * @param {number} maxResults - Maximum number of results to return (default: 5)
 * @returns {Promise<Array>} Array of article objects
 */
export const searchPubMedByMesh = async (meshTerms, maxResults = 5) => {
  if (!meshTerms || meshTerms.length === 0) {
    return [];
  }

  try {
    // Convert MeSH terms to search query
    // Search for articles tagged with these MeSH terms
    const searchTerm = Array.isArray(meshTerms) 
      ? meshTerms.map(term => `"${term}"[MeSH Terms]`).join(' OR ')
      : `"${meshTerms}"[MeSH Terms]`;

    // Step 1: Search for PMIDs
    const searchUrl = `${API_BASE}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(searchTerm)}&retmax=${maxResults}&retmode=json&sort=relevance`;
    
    const searchResponse = await fetch(searchUrl);
    const searchData = await searchResponse.json();
    
    const pmids = searchData.esearchresult?.idlist || [];
    
    if (pmids.length === 0) {
      return [];
    }

    // Step 2: Fetch article details
    const summaryUrl = `${API_BASE}/esummary.fcgi?db=pubmed&id=${pmids.join(',')}&retmode=json`;
    
    const summaryResponse = await fetch(summaryUrl);
    const summaryData = await summaryResponse.json();
    
    // Parse article data
    const articles = pmids.map(pmid => {
      const article = summaryData.result?.[pmid];
      if (!article) return null;

      return {
        pmid: pmid,
        title: article.title || 'No title available',
        authors: article.authors?.slice(0, 3).map(a => a.name).join(', ') || 'Unknown authors',
        journal: article.fulljournalname || article.source || 'Unknown journal',
        pubdate: article.pubdate || 'Unknown date',
        doi: article.elocationid || article.articleids?.find(id => id.idtype === 'doi')?.value || '',
        pubmedUrl: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`
      };
    }).filter(article => article !== null);

    return articles;
  } catch (error) {
    console.error('Error fetching PubMed articles:', error);
    throw error;
  }
};
