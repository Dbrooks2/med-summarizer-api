// Enhanced MedAI Summarizer - Full AI Integration
class EnhancedMedAISummarizer {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentAnalysis = null;
        this.settings = this.loadSettings();
        this.aiCapabilities = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.checkAPIHealth();
        this.checkAICapabilities();
        this.updateSettingsUI();
    }

    setupEventListeners() {
        // File input change
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Text input change
        document.getElementById('textInput').addEventListener('input', (e) => {
            this.validateInput();
        });

        // Analysis type changes
        document.getElementById('analysisType').addEventListener('change', (e) => {
            this.updateAnalysisOptions(e.target.value);
        });
    }

    setupFileUpload() {
        const dropZone = document.querySelector('.upload-area');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    async checkAPIHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateAPIStatus('success', 'API Ready');
                this.updateAICapabilities(data.ai_service);
            } else {
                this.updateAPIStatus('error', 'API Error');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            this.updateAPIStatus('error', 'API Unavailable');
        }
    }

    async checkAICapabilities() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/models`);
            const data = await response.json();
            this.aiCapabilities = data;
            this.updateAICapabilities(data);
        } catch (error) {
            console.error('Failed to check AI capabilities:', error);
        }
    }

    updateAICapabilities(capabilities) {
        const capabilitiesElement = document.getElementById('aiCapabilities');
        if (!capabilitiesElement) return;

        let capabilitiesHtml = '<div class="flex flex-wrap gap-2">';
        
        if (capabilities.sentence_bert_loaded) {
            capabilitiesHtml += '<span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">sentence-BERT</span>';
        }
        
        if (capabilities.llama_loaded) {
            capabilitiesHtml += '<span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">LLaMA-2</span>';
        }
        
        if (capabilities.faiss_index_built) {
            capabilitiesHtml += `<span class="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">FAISS (${capabilities.corpus_size})</span>`;
        }
        
        capabilitiesHtml += '</div>';
        capabilitiesElement.innerHTML = capabilitiesHtml;
    }

    updateAPIStatus(status, message) {
        const statusElement = document.getElementById('apiStatus');
        const icon = statusElement.querySelector('i');
        
        statusElement.className = `text-sm px-2 py-1 rounded-full ${
            status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`;
        
        icon.className = `fas fa-circle text-xs mr-1 ${
            status === 'success' ? 'text-green-600' : 'text-red-600'
        }`;
        
        statusElement.innerHTML = `<i class="${icon.className}"></i>${message}`;
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const allowedTypes = ['.txt', '.pdf', '.doc', '.docx'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            this.showError('Please select a valid file type (.txt, .pdf, .doc, .docx)');
            return;
        }

        // Read file content
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            document.getElementById('textInput').value = content;
            this.validateInput();
        };
        reader.readAsText(file);
    }

    validateInput() {
        const textInput = document.getElementById('textInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const text = textInput.value.trim();
        
        analyzeBtn.disabled = text.length < 50;
        analyzeBtn.classList.toggle('opacity-50', text.length < 50);
    }

    updateAnalysisOptions(analysisType) {
        const optionsContainer = document.getElementById('analysisOptions');
        
        switch (analysisType) {
            case 'enhanced':
                optionsContainer.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <label class="flex items-center">
                            <input type="checkbox" id="includeEntities" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Extract Medical Entities</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="includeFindings" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Identify Key Findings</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="includeRecommendations" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Generate Recommendations</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="useLLaMA" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Use LLaMA-2 (if available)</span>
                        </label>
                    </div>
                `;
                break;
                
            case 'rag':
                optionsContainer.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Retrieval Count (k)</label>
                            <input type="number" id="retrievalCount" value="3" min="1" max="10" class="w-full p-2 border border-gray-300 rounded-lg">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Summary Length</label>
                            <input type="number" id="ragSummaryLength" value="150" min="50" max="500" class="w-full p-2 border border-gray-300 rounded-lg">
                        </div>
                        <label class="flex items-center">
                            <input type="checkbox" id="ragUseLLaMA" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Use LLaMA-2 for RAG</span>
                        </label>
                    </div>
                `;
                break;
                
            case 'semantic':
                optionsContainer.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Top Results (k)</label>
                            <input type="number" id="topK" value="5" min="1" max="20" class="w-full p-2 border border-gray-300 rounded-lg">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Similarity Threshold</label>
                            <input type="number" id="similarityThreshold" value="0.3" min="0.0" max="1.0" step="0.1" class="w-full p-2 border border-gray-300 rounded-lg">
                        </div>
                    </div>
                `;
                break;
                
            default:
                optionsContainer.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <label class="flex items-center">
                            <input type="checkbox" id="summary" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Generate Summary</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="keyFindings" checked class="mr-2 text-purple-600">
                            <span class="text-sm">Extract Key Findings</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="recommendations" class="mr-2 text-purple-600">
                            <span class="text-sm">Generate Recommendations</span>
                        </label>
                    </div>
                `;
        }
    }

    async analyzeReport() {
        const textInput = document.getElementById('textInput');
        const text = textInput.value.trim();
        
        if (text.length < 50) {
            this.showError('Please enter at least 50 characters of medical text');
            return;
        }

        const analysisType = document.getElementById('analysisType').value;
        
        // Show results section and loading
        this.showResultsSection();
        this.showLoading(true);

        try {
            let results;
            
            switch (analysisType) {
                case 'enhanced':
                    results = await this.performEnhancedAnalysis(text);
                    break;
                case 'rag':
                    results = await this.performRAGAnalysis(text);
                    break;
                case 'semantic':
                    results = await this.performSemanticSearch(text);
                    break;
                default:
                    results = await this.performBasicAnalysis(text);
            }
            
            // Display results
            this.displayResults(results, analysisType);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Analysis failed. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    async performEnhancedAnalysis(text) {
        const options = {
            include_entities: document.getElementById('includeEntities')?.checked ?? true,
            include_findings: document.getElementById('includeFindings')?.checked ?? true,
            include_recommendations: document.getElementById('includeRecommendations')?.checked ?? true
        };

        const response = await fetch(`${this.apiBaseUrl}/enhanced_summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text, 
                max_words: parseInt(this.settings.defaultLength),
                ...options
            })
        });
        
        if (!response.ok) {
            throw new Error(`Enhanced analysis failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async performRAGAnalysis(text) {
        const k = parseInt(document.getElementById('retrievalCount')?.value || 3);
        const maxWords = parseInt(document.getElementById('ragSummaryLength')?.value || 150);
        const useLLaMA = document.getElementById('ragUseLLaMA')?.checked ?? true;

        const response = await fetch(`${this.apiBaseUrl}/rag_summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: text,
                k,
                max_words: maxWords,
                use_llama: useLLaMA
            })
        });
        
        if (!response.ok) {
            throw new Error(`RAG analysis failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async performSemanticSearch(text) {
        const topK = parseInt(document.getElementById('topK')?.value || 5);
        const threshold = parseFloat(document.getElementById('similarityThreshold')?.value || 0.3);

        const response = await fetch(`${this.apiBaseUrl}/semantic_search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: text,
                top_k: topK,
                similarity_threshold: threshold
            })
        });
        
        if (!response.ok) {
            throw new Error(`Semantic search failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async performBasicAnalysis(text) {
        const options = {
            summary: document.getElementById('summary')?.checked ?? true,
            keyFindings: document.getElementById('keyFindings')?.checked ?? true,
            recommendations: document.getElementById('recommendations')?.checked ?? false
        };

        const results = {};

        // Generate summary
        if (options.summary) {
            try {
                const summaryResponse = await fetch(`${this.apiBaseUrl}/summarize`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text, 
                        max_words: parseInt(this.settings.defaultLength),
                        use_llama: true,
                        extract_entities: true
                    })
                });
                
                if (summaryResponse.ok) {
                    const summaryData = await summaryResponse.json();
                    results.summary = summaryData.summary;
                    if (summaryData.entities) {
                        results.entities = summaryData.entities;
                    }
                }
            } catch (error) {
                console.error('Summary generation failed:', error);
            }
        }

        // Extract key findings and recommendations
        if (options.keyFindings || options.recommendations) {
            const enhancedResponse = await fetch(`${this.apiBaseUrl}/enhanced_summarize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text, 
                    max_words: parseInt(this.settings.defaultLength),
                    include_entities: false,
                    include_findings: options.keyFindings,
                    include_recommendations: options.recommendations
                })
            });
            
            if (enhancedResponse.ok) {
                const enhancedData = await enhancedResponse.json();
                if (options.keyFindings) {
                    results.keyFindings = enhancedData.key_findings;
                }
                if (options.recommendations) {
                    results.recommendations = enhancedData.recommendations;
                }
            }
        }

        return results;
    }

    displayResults(results, analysisType) {
        // Clear previous results
        this.clearResults();
        
        // Display based on analysis type
        switch (analysisType) {
            case 'enhanced':
                this.displayEnhancedResults(results);
                break;
            case 'rag':
                this.displayRAGResults(results);
                break;
            case 'semantic':
                this.displaySemanticResults(results);
                break;
            default:
                this.displayBasicResults(results);
        }
        
        // Show results content
        document.getElementById('resultsContent').classList.remove('hidden');
        
        // Store current analysis for download/copy
        this.currentAnalysis = { results, analysisType };
    }

    displayEnhancedResults(results) {
        // Display summary
        if (results.summary) {
            document.getElementById('summaryContent').innerHTML = `
                <p class="text-lg">${results.summary}</p>
            `;
        }

        // Display entities
        if (results.entities) {
            const entitiesHtml = this.formatEntities(results.entities);
            document.getElementById('entitiesContent').innerHTML = entitiesHtml;
            document.getElementById('entitiesSection').classList.remove('hidden');
        }

        // Display key findings
        if (results.key_findings) {
            const findingsHtml = results.key_findings.map(finding => 
                `<li class="mb-2"><i class="fas fa-check-circle text-green-500 mr-2"></i>${finding}</li>`
            ).join('');
            
            document.getElementById('findingsContent').innerHTML = `
                <ul class="space-y-2">${findingsHtml}</ul>
            `;
        }

        // Display recommendations
        if (results.recommendations) {
            const recommendationsHtml = results.recommendations.map(rec => 
                `<li class="mb-2"><i class="fas fa-arrow-right text-blue-500 mr-2"></i>${rec}</li>`
            ).join('');
            
            document.getElementById('recommendationsContent').innerHTML = `
                <ul class="space-y-2">${recommendationsHtml}</ul>
            `;
        }

        // Display metadata
        if (results.metadata) {
            document.getElementById('metadataContent').innerHTML = `
                <div class="text-sm text-gray-600">
                    <p><strong>Model:</strong> ${results.metadata.model_used}</p>
                    <p><strong>Generated:</strong> ${new Date(results.metadata.generated_at).toLocaleString()}</p>
                    <p><strong>Entities found:</strong> ${results.metadata.entities_found}</p>
                </div>
            `;
            document.getElementById('metadataSection').classList.remove('hidden');
        }
    }

    displayRAGResults(results) {
        // Display summary
        if (results.summary) {
            document.getElementById('summaryContent').innerHTML = `
                <p class="text-lg">${results.summary}</p>
            `;
        }

        // Display search results
        if (results.search_results) {
            const searchHtml = results.search_results.map((result, index) => `
                <div class="border-l-4 border-blue-500 pl-4 mb-3">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm font-medium text-blue-600">Result ${index + 1}</span>
                        <span class="text-xs text-gray-500">Similarity: ${(result.similarity * 100).toFixed(1)}%</span>
                    </div>
                    <p class="text-sm text-gray-700">${result.text}</p>
                </div>
            `).join('');
            
            document.getElementById('searchResultsContent').innerHTML = searchHtml;
            document.getElementById('searchResultsSection').classList.remove('hidden');
        }

        // Display metadata
        document.getElementById('metadataContent').innerHTML = `
            <div class="text-sm text-gray-600">
                <p><strong>Documents retrieved:</strong> ${results.documents_retrieved}</p>
                <p><strong>Model used:</strong> ${results.model_used}</p>
                <p><strong>Query:</strong> ${results.query}</p>
            </div>
        `;
        document.getElementById('metadataSection').classList.remove('hidden');
    }

    displaySemanticResults(results) {
        // Display search results
        if (results.results) {
            const searchHtml = results.results.map((result, index) => `
                <div class="border-l-4 border-green-500 pl-4 mb-3">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm font-medium text-green-600">Rank ${result.rank}</span>
                        <span class="text-xs text-gray-500">Similarity: ${(result.similarity * 100).toFixed(1)}%</span>
                    </div>
                    <p class="text-sm text-gray-700">${result.text}</p>
                </div>
            `).join('');
            
            document.getElementById('searchResultsContent').innerHTML = searchHtml;
            document.getElementById('searchResultsSection').classList.remove('hidden');
        }

        // Display metadata
        document.getElementById('metadataContent').innerHTML = `
            <div class="text-sm text-gray-600">
                <p><strong>Query:</strong> ${results.query}</p>
                <p><strong>Total found:</strong> ${results.total_found}</p>
            </div>
        `;
        document.getElementById('metadataSection').classList.remove('hidden');
    }

    displayBasicResults(results) {
        // Display summary
        if (results.summary) {
            document.getElementById('summaryContent').innerHTML = `
                <p class="text-lg">${results.summary}</p>
            `;
        }

        // Display key findings
        if (results.keyFindings) {
            const findingsHtml = results.keyFindings.map(finding => 
                `<li class="mb-2"><i class="fas fa-check-circle text-green-500 mr-2"></i>${finding}</li>`
            ).join('');
            
            document.getElementById('findingsContent').innerHTML = `
                <ul class="space-y-2">${findingsHtml}</ul>
            `;
        }

        // Display recommendations
        if (results.recommendations) {
            const recommendationsHtml = results.recommendations.map(rec => 
                `<li class="mb-2"><i class="fas fa-arrow-right text-blue-500 mr-2"></i>${rec}</li>`
            ).join('');
            
            document.getElementById('recommendationsContent').innerHTML = `
                <ul class="space-y-2">${recommendationsHtml}</ul>
            `;
        }

        // Display entities if available
        if (results.entities) {
            const entitiesHtml = this.formatEntities(results.entities);
            document.getElementById('entitiesContent').innerHTML = entitiesHtml;
            document.getElementById('entitiesSection').classList.remove('hidden');
        }
    }

    formatEntities(entities) {
        let html = '';
        
        for (const [category, items] of Object.entries(entities)) {
            if (items && items.length > 0) {
                html += `
                    <div class="mb-4">
                        <h4 class="font-medium text-gray-900 mb-2 capitalize">${category.replace('_', ' ')}</h4>
                        <div class="flex flex-wrap gap-2">
                            ${items.map(item => `<span class="px-2 py-1 bg-gray-100 text-gray-700 text-sm rounded">${item}</span>`).join('')}
                        </div>
                    </div>
                `;
            }
        }
        
        return html;
    }

    clearResults() {
        // Hide all optional sections
        const optionalSections = [
            'entitiesSection',
            'searchResultsSection', 
            'metadataSection'
        ];
        
        optionalSections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.classList.add('hidden');
            }
        });
    }

    showResultsSection() {
        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    showLoading(show) {
        const loading = document.getElementById('loadingState');
        const resultsContent = document.getElementById('resultsContent');
        
        if (show) {
            loading.classList.remove('hidden');
            resultsContent.classList.add('hidden');
        } else {
            loading.classList.add('hidden');
        }
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        errorDiv.textContent = message;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    downloadResults() {
        if (!this.currentAnalysis) return;
        
        const content = this.formatResultsForDownload();
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `medical-analysis-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    copyResults() {
        if (!this.currentAnalysis) return;
        
        const content = this.formatResultsForDownload();
        navigator.clipboard.writeText(content).then(() => {
            this.showError('Results copied to clipboard!');
        }).catch(() => {
            this.showError('Failed to copy to clipboard');
        });
    }

    formatResultsForDownload() {
        const { results, analysisType } = this.currentAnalysis;
        
        let content = `MEDICAL REPORT ANALYSIS (${analysisType.toUpperCase()})\n`;
        content += '='.repeat(60) + '\n\n';
        
        if (results.summary) {
            content += 'EXECUTIVE SUMMARY\n';
            content += '-'.repeat(20) + '\n';
            content += results.summary + '\n\n';
        }
        
        if (results.key_findings || results.keyFindings) {
            content += 'KEY FINDINGS\n';
            content += '-'.repeat(20) + '\n';
            const findings = results.key_findings || results.keyFindings || [];
            findings.forEach((finding, index) => {
                content += `${index + 1}. ${finding}\n`;
            });
            content += '\n';
        }
        
        if (results.recommendations) {
            content += 'RECOMMENDATIONS\n';
            content += '-'.repeat(20) + '\n';
            results.recommendations.forEach((rec, index) => {
                content += `${index + 1}. ${rec}\n`;
            });
            content += '\n';
        }
        
        if (results.entities) {
            content += 'MEDICAL ENTITIES\n';
            content += '-'.repeat(20) + '\n';
            for (const [category, items] of Object.entries(results.entities)) {
                if (items && items.length > 0) {
                    content += `${category.toUpperCase()}:\n`;
                    items.forEach(item => content += `  - ${item}\n`);
                    content += '\n';
                }
            }
        }
        
        if (results.search_results) {
            content += 'SEARCH RESULTS\n';
            content += '-'.repeat(20) + '\n';
            results.search_results.forEach((result, index) => {
                content += `Result ${index + 1} (Similarity: ${(result.similarity * 100).toFixed(1)}%):\n`;
                content += `${result.text}\n\n`;
            });
        }
        
        content += `Generated on: ${new Date().toLocaleString()}\n`;
        content += `Analysis Type: ${analysisType}\n`;
        content += 'Powered by Enhanced MedAI Summarizer';
        
        return content;
    }

    resetAnalysis() {
        document.getElementById('textInput').value = '';
        document.getElementById('fileInput').value = '';
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('analysisType').value = 'basic';
        this.updateAnalysisOptions('basic');
        this.currentAnalysis = null;
        this.validateInput();
        
        // Scroll back to upload section
        document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth' });
    }

    // Settings management
    loadSettings() {
        const defaultSettings = {
            apiEndpoint: 'http://localhost:8000',
            defaultLength: 150
        };
        
        const saved = localStorage.getItem('medai-enhanced-settings');
        return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    }

    saveSettings() {
        const settings = {
            apiEndpoint: document.getElementById('apiEndpoint').value,
            defaultLength: parseInt(document.getElementById('defaultLength').value)
        };
        
        localStorage.setItem('medai-enhanced-settings', JSON.stringify(settings));
        this.settings = settings;
        this.apiBaseUrl = settings.apiEndpoint;
        
        // Recheck API health with new endpoint
        this.checkAPIHealth();
    }

    updateSettingsUI() {
        document.getElementById('apiEndpoint').value = this.settings.apiEndpoint;
        document.getElementById('defaultLength').value = this.settings.defaultLength;
    }
}

// Global functions for HTML onclick handlers
function analyzeReport() {
    if (window.enhancedMedAI) {
        window.enhancedMedAI.analyzeReport();
    }
}

function downloadResults() {
    if (window.enhancedMedAI) {
        window.enhancedMedAI.downloadResults();
    }
}

function copyResults() {
    if (window.enhancedMedAI) {
        window.enhancedMedAI.copyResults();
    }
}

function resetAnalysis() {
    if (window.enhancedMedAI) {
        window.enhancedMedAI.resetAnalysis();
    }
}

function showSettings() {
    document.getElementById('settingsModal').classList.remove('hidden');
}

function hideSettings() {
    document.getElementById('settingsModal').classList.add('hidden');
}

function saveSettings() {
    if (window.enhancedMedAI) {
        window.enhancedMedAI.saveSettings();
    }
    hideSettings();
}

// Initialize the enhanced application
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedMedAI = new EnhancedMedAISummarizer();
}); 