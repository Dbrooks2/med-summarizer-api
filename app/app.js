// MedAI Summarizer - Functional Application
class MedAISummarizer {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentAnalysis = null;
        this.settings = this.loadSettings();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.checkAPIHealth();
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
            } else {
                this.updateAPIStatus('error', 'API Error');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            this.updateAPIStatus('error', 'API Unavailable');
        }
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

    async analyzeReport() {
        const textInput = document.getElementById('textInput');
        const text = textInput.value.trim();
        
        if (text.length < 50) {
            this.showError('Please enter at least 50 characters of medical text');
            return;
        }

        // Show results section and loading
        this.showResultsSection();
        this.showLoading(true);

        try {
            // Get analysis options
            const options = {
                summary: document.getElementById('summary').checked,
                keyFindings: document.getElementById('keyFindings').checked,
                recommendations: document.getElementById('recommendations').checked
            };

            // Call API endpoints based on options
            const results = await this.performAnalysis(text, options);
            
            // Display results
            this.displayResults(results);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Analysis failed. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    async performAnalysis(text, options) {
        const results = {};

        // Generate summary
        if (options.summary) {
            try {
                const summaryResponse = await fetch(`${this.apiBaseUrl}/summarize`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text, 
                        max_words: parseInt(this.settings.defaultLength) 
                    })
                });
                
                if (summaryResponse.ok) {
                    const summaryData = await summaryResponse.json();
                    results.summary = summaryData.summary;
                }
            } catch (error) {
                console.error('Summary generation failed:', error);
            }
        }

        // Extract key findings (simplified version)
        if (options.keyFindings) {
            results.keyFindings = this.extractKeyFindings(text);
        }

        // Generate recommendations
        if (options.recommendations) {
            results.recommendations = this.generateRecommendations(text);
        }

        return results;
    }

    extractKeyFindings(text) {
        const findings = [];
        
        // Look for diagnostic statements
        const diagnosticPatterns = [
            /diagnosis:\s*([^.]+)/gi,
            /diagnosed with\s+([^.]+)/gi,
            /findings:\s*([^.]+)/gi
        ];
        
        diagnosticPatterns.forEach(pattern => {
            const matches = text.match(pattern);
            if (matches) {
                findings.push(...matches.map(match => 
                    match.replace(/^(diagnosis|diagnosed with|findings):\s*/i, '').trim()
                ));
            }
        });

        // Look for vital signs and lab values
        const vitalPatterns = [
            /blood pressure\s+(\d+\/\d+)/gi,
            /heart rate\s+(\d+)/gi,
            /temperature\s+(\d+\.?\d*)/gi,
            /(\d+\.?\d*)\s*(mg\/dL|mmHg|bpm)/gi
        ];

        vitalPatterns.forEach(pattern => {
            const matches = text.match(pattern);
            if (matches) {
                findings.push(...matches);
            }
        });

        // If no specific findings, create general ones
        if (findings.length === 0) {
            findings.push(
                'Patient presents with medical symptoms requiring evaluation',
                'Clinical assessment and monitoring recommended',
                'Further diagnostic testing may be indicated'
            );
        }

        return findings.slice(0, 5);
    }

    generateRecommendations(text) {
        const recommendations = [];
        
        // Generate recommendations based on content
        if (text.toLowerCase().includes('chest pain')) {
            recommendations.push(
                'Immediate cardiac evaluation recommended',
                'Consider ECG and cardiac enzyme testing',
                'Monitor for signs of cardiac compromise'
            );
        }
        
        if (text.toLowerCase().includes('hypertension') || text.toLowerCase().includes('high blood pressure')) {
            recommendations.push(
                'Blood pressure monitoring and lifestyle modifications',
                'Consider antihypertensive medication if indicated',
                'Regular follow-up for blood pressure management'
            );
        }
        
        if (text.toLowerCase().includes('diabetes')) {
            recommendations.push(
                'Blood glucose monitoring and dietary management',
                'Regular HbA1c testing and diabetic care',
                'Foot care and eye examination screening'
            );
        }

        // Default recommendations
        if (recommendations.length === 0) {
            recommendations.push(
                'Follow up with primary care provider',
                'Continue monitoring symptoms',
                'Seek immediate care if symptoms worsen'
            );
        }

        return recommendations.slice(0, 4);
    }

    displayResults(results) {
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

        // Show results content
        document.getElementById('resultsContent').classList.remove('hidden');
        
        // Store current analysis for download/copy
        this.currentAnalysis = results;
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
        const { summary, keyFindings, recommendations } = this.currentAnalysis;
        
        let content = 'MEDICAL REPORT ANALYSIS\n';
        content += '='.repeat(50) + '\n\n';
        
        if (summary) {
            content += 'EXECUTIVE SUMMARY\n';
            content += '-'.repeat(20) + '\n';
            content += summary + '\n\n';
        }
        
        if (keyFindings) {
            content += 'KEY FINDINGS\n';
            content += '-'.repeat(20) + '\n';
            keyFindings.forEach((finding, index) => {
                content += `${index + 1}. ${finding}\n`;
            });
            content += '\n';
        }
        
        if (recommendations) {
            content += 'RECOMMENDATIONS\n';
            content += '-'.repeat(20) + '\n';
            recommendations.forEach((rec, index) => {
                content += `${index + 1}. ${rec}\n`;
            });
            content += '\n';
        }
        
        content += `Generated on: ${new Date().toLocaleString()}\n`;
        content += 'Powered by MedAI Summarizer';
        
        return content;
    }

    resetAnalysis() {
        document.getElementById('textInput').value = '';
        document.getElementById('fileInput').value = '';
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('summary').checked = true;
        document.getElementById('keyFindings').checked = true;
        document.getElementById('recommendations').checked = false;
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
        
        const saved = localStorage.getItem('medai-settings');
        return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    }

    saveSettings() {
        const settings = {
            apiEndpoint: document.getElementById('apiEndpoint').value,
            defaultLength: parseInt(document.getElementById('defaultLength').value)
        };
        
        localStorage.setItem('medai-settings', JSON.stringify(settings));
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
    if (window.medAI) {
        window.medAI.analyzeReport();
    }
}

function downloadResults() {
    if (window.medAI) {
        window.medAI.downloadResults();
    }
}

function copyResults() {
    if (window.medAI) {
        window.medAI.copyResults();
    }
}

function resetAnalysis() {
    if (window.medAI) {
        window.medAI.resetAnalysis();
    }
}

function showSettings() {
    document.getElementById('settingsModal').classList.remove('hidden');
}

function hideSettings() {
    document.getElementById('settingsModal').classList.add('hidden');
}

function saveSettings() {
    if (window.medAI) {
        window.medAI.saveSettings();
    }
    hideSettings();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.medAI = new MedAISummarizer();
}); 