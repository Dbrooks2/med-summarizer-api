// MedAI Summarizer - Frontend Application
class MedAISummarizer {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000'; // Change to your deployed API URL
        this.currentAnalysis = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.checkAPIHealth();
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

        // Drag and drop
        this.setupDragAndDrop();
    }

    setupFileUpload() {
        const dropZone = document.querySelector('#upload .border-dashed');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-purple-400', 'bg-purple-50');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-purple-400', 'bg-purple-50');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-purple-400', 'bg-purple-50');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    setupDragAndDrop() {
        const dropZone = document.querySelector('#upload .border-dashed');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-purple-400', 'bg-purple-50');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-purple-400', 'bg-purple-50');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-purple-400', 'bg-purple-50');
            
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
                console.log('API is healthy and ready');
                this.updateUIStatus('ready');
            } else {
                console.warn('API health check failed');
                this.updateUIStatus('error');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            this.updateUIStatus('error');
        }
    }

    updateUIStatus(status) {
        const statusIndicator = document.createElement('div');
        statusIndicator.className = `fixed top-4 right-4 px-4 py-2 rounded-lg text-white text-sm ${
            status === 'ready' ? 'bg-green-500' : 'bg-red-500'
        }`;
        statusIndicator.textContent = status === 'ready' ? 'API Ready' : 'API Error';
        
        document.body.appendChild(statusIndicator);
        
        setTimeout(() => {
            statusIndicator.remove();
        }, 3000);
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
                    body: JSON.stringify({ text, max_words: 150 })
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
        // Simple rule-based extraction - in production, this would use more sophisticated NLP
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
                findings.push(...matches.map(match => match.replace(/^(diagnosis|diagnosed with|findings):\s*/i, '').trim()));
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

        return findings.slice(0, 5); // Limit to 5 findings
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

        return recommendations.slice(0, 4); // Limit to 4 recommendations
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
        
        // Store current analysis for download/share
        this.currentAnalysis = results;
    }

    showResultsSection() {
        document.getElementById('results').classList.remove('hidden');
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
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

    shareResults() {
        if (!this.currentAnalysis) return;
        
        if (navigator.share) {
            const content = this.formatResultsForDownload();
            navigator.share({
                title: 'Medical Report Analysis',
                text: content.substring(0, 200) + '...',
                url: window.location.href
            });
        } else {
            // Fallback: copy to clipboard
            const content = this.formatResultsForDownload();
            navigator.clipboard.writeText(content).then(() => {
                this.showError('Results copied to clipboard!');
            });
        }
    }

    resetForm() {
        document.getElementById('textInput').value = '';
        document.getElementById('fileInput').value = '';
        document.getElementById('results').classList.add('hidden');
        document.getElementById('summary').checked = true;
        document.getElementById('keyFindings').checked = true;
        document.getElementById('recommendations').checked = false;
        this.currentAnalysis = null;
        this.validateInput();
        
        // Scroll back to upload section
        document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
    }
}

// Utility functions
function scrollToUpload() {
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
}

function scrollToDemo() {
    // Scroll to features section as demo
    document.getElementById('features').scrollIntoView({ behavior: 'smooth' });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.medAI = new MedAISummarizer();
});

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

function shareResults() {
    if (window.medAI) {
        window.medAI.shareResults();
    }
}

function resetForm() {
    if (window.medAI) {
        window.medAI.resetForm();
    }
} 