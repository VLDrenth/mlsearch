// MLSearch Web Interface JavaScript

class MLSearchApp {
    constructor() {
        this.currentSearchId = null;
        this.websocket = null;
        this.searchInProgress = false;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Search form submission
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => this.handleSearchSubmit(e));
        }
        
        // Search input enter key
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !this.searchInProgress) {
                    this.handleSearchSubmit(e);
                }
            });
        }
    }
    
    async handleSearchSubmit(event) {
        event.preventDefault();
        
        if (this.searchInProgress) {
            return;
        }
        
        const searchInput = document.getElementById('searchInput');
        const query = searchInput.value.trim();
        
        if (!query) {
            this.showError('Please enter a search query');
            return;
        }
        
        try {
            this.setSearchInProgress(true);
            await this.startSearch(query);
        } catch (error) {
            console.error('Search failed:', error);
            this.showError('Search failed: ' + error.message);
            this.setSearchInProgress(false);
        }
    }
    
    async startSearch(query) {
        const maxResults = document.getElementById('maxResults')?.value || 100;
        
        const requestBody = {
            query: query,
            max_results: parseInt(maxResults)
        };
        
        console.log('Starting search with:', requestBody);
        
        const response = await fetch('/api/search/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Search initiated:', data);
        
        this.currentSearchId = data.search_id;
        this.showSearchProgress();
        this.connectWebSocket(data.search_id);
        this.startProgressPolling();
    }
    
    connectWebSocket(searchId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${searchId}`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            console.log('WebSocket message:', event.data);
            try {
                const data = JSON.parse(event.data);
                this.updateProgress(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    async startProgressPolling() {
        if (!this.currentSearchId) return;
        
        const pollProgress = async () => {
            try {
                const response = await fetch(`/api/search/${this.currentSearchId}/status`);
                if (response.ok) {
                    const progress = await response.json();
                    this.updateProgress(progress);
                    
                    if (progress.status === 'completed' || progress.status === 'failed') {
                        this.setSearchInProgress(false);
                        if (progress.status === 'completed') {
                            await this.loadResults();
                        }
                        return; // Stop polling
                    }
                }
            } catch (error) {
                console.error('Error polling progress:', error);
            }
            
            // Continue polling if search is still in progress
            if (this.searchInProgress) {
                setTimeout(pollProgress, 2000); // Poll every 2 seconds
            }
        };
        
        setTimeout(pollProgress, 1000); // Start polling after 1 second
    }
    
    updateProgress(progress) {
        console.log('Updating progress:', progress);
        
        // Update overall progress
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = `${progress.progress_percentage || 0}%`;
        }
        
        const progressText = document.getElementById('progressText');
        if (progressText) {
            progressText.textContent = `${progress.status} - ${progress.total_papers_found || 0} papers found`;
        }
        
        // Update agent progress
        const agentProgress = document.getElementById('agentProgress');
        if (agentProgress && progress.agents) {
            agentProgress.innerHTML = progress.agents.map(agent => `
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-name">${agent.agent_id}</span>
                        <span class="agent-status status-${agent.status}">${agent.status}</span>
                    </div>
                    <div class="agent-details">
                        <div>Strategy: ${agent.search_strategy}</div>
                        <div>Papers: ${agent.papers_found} (${agent.relevant_papers} relevant)</div>
                        <div>Progress: ${agent.progress_percentage.toFixed(1)}%</div>
                    </div>
                </div>
            `).join('');
        }
    }
    
    async loadResults() {
        if (!this.currentSearchId) return;
        
        try {
            const response = await fetch(`/api/search/${this.currentSearchId}/results`);
            if (response.ok) {
                const results = await response.json();
                this.displayResults(results);
            } else {
                throw new Error(`Failed to load results: ${response.status}`);
            }
        } catch (error) {
            console.error('Error loading results:', error);
            this.showError('Failed to load results: ' + error.message);
        }
    }
    
    displayResults(results) {
        console.log('Displaying results:', results);
        
        // Hide progress section
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
        
        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        // Update results header
        const resultsTitle = document.getElementById('resultsTitle');
        if (resultsTitle) {
            resultsTitle.textContent = `Search Results for "${results.query}"`;
        }
        
        const resultsStats = document.getElementById('resultsStats');
        if (resultsStats) {
            resultsStats.textContent = `Found ${results.total_papers} papers in ${results.duration_seconds?.toFixed(1) || 'N/A'} seconds`;
        }
        
        // Display papers
        const papersContainer = document.getElementById('papersContainer');
        if (papersContainer && results.papers) {
            papersContainer.innerHTML = results.papers.map((paper, index) => this.renderPaperCard(paper, index)).join('');
        }
    }
    
    renderPaperCard(paper, index) {
        const publishedDate = new Date(paper.published).toLocaleDateString();
        const authorsText = paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '');
        
        // Extract relevance explanation from abstract if present
        let abstract = paper.abstract || '';
        let relevanceExplanation = '';
        
        if (abstract.includes('**Why relevant:**')) {
            const parts = abstract.split('**Why relevant:**');
            abstract = parts[0].trim();
            relevanceExplanation = parts[1].trim();
        }
        
        return `
            <div class="paper-card">
                <div class="paper-header">
                    <span class="paper-rank">${index + 1}</span>
                    <a href="${paper.url}" target="_blank" class="paper-title">
                        ${this.escapeHtml(paper.title)}
                    </a>
                </div>
                <div class="paper-meta">
                    <span class="paper-authors">${this.escapeHtml(authorsText)}</span>
                    <span class="paper-date">${publishedDate}</span>
                    ${paper.relevance_score ? `<span class="relevance-score">Relevance: ${(paper.relevance_score * 100).toFixed(0)}%</span>` : ''}
                    ${paper.found_by_agent ? `<span class="category-tag">Found by: ${paper.found_by_agent}</span>` : ''}
                </div>
                <div class="paper-abstract">
                    ${this.escapeHtml(this.truncateText(abstract, 250))}
                    ${relevanceExplanation ? `
                        <div class="relevance-explanation">
                            <strong>Why this paper is relevant:</strong> ${this.escapeHtml(relevanceExplanation)}
                        </div>
                    ` : ''}
                </div>
                <div class="paper-footer">
                    <div class="paper-categories">
                        ${paper.categories.map(cat => `<span class="category-tag">${cat}</span>`).join('')}
                        ${paper.search_strategy ? `<span class="category-tag">Strategy: ${paper.search_strategy}</span>` : ''}
                    </div>
                    <div class="paper-actions">
                        <a href="${paper.url}" target="_blank" class="action-button">View Paper</a>
                        <a href="${paper.pdf_url}" target="_blank" class="action-button primary">Download PDF</a>
                    </div>
                </div>
            </div>
        `;
    }
    
    showSearchProgress() {
        // Hide search form and results
        const searchSection = document.getElementById('searchSection');
        if (searchSection) {
            searchSection.style.display = 'none';
        }
        
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
        // Show progress section
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
    }
    
    setSearchInProgress(inProgress) {
        this.searchInProgress = inProgress;
        
        const searchButton = document.getElementById('searchButton');
        if (searchButton) {
            searchButton.disabled = inProgress;
            searchButton.textContent = inProgress ? 'Searching...' : 'Search Papers';
        }
        
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.disabled = inProgress;
        }
    }
    
    showError(message) {
        // Simple error display - could be enhanced with a proper notification system
        alert(message);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }
    
    // Method to start a new search (for search again functionality)
    newSearch() {
        this.currentSearchId = null;
        this.searchInProgress = false;
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        // Show search section
        const searchSection = document.getElementById('searchSection');
        if (searchSection) {
            searchSection.style.display = 'block';
        }
        
        // Hide other sections
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
        
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
        // Re-enable search controls
        this.setSearchInProgress(false);
        
        // Clear search input
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.value = '';
            searchInput.focus();
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mlSearchApp = new MLSearchApp();
    console.log('MLSearch app initialized');
});