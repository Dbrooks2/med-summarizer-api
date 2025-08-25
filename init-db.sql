-- Initialize database for Medical Summarizer API
CREATE DATABASE IF NOT EXISTS medsummarizer;

-- Create tables for storing medical reports and summaries
CREATE TABLE IF NOT EXISTS medical_reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(255) UNIQUE NOT NULL,
    patient_id VARCHAR(255),
    report_text TEXT NOT NULL,
    report_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    report_id INTEGER REFERENCES medical_reports(id),
    summary_text TEXT NOT NULL,
    summary_type VARCHAR(50) DEFAULT 'extractive',
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_requests (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    request_data JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id SERIAL PRIMARY KEY,
    artifact_name VARCHAR(255) NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_type VARCHAR(50),
    version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_medical_reports_patient_id ON medical_reports(patient_id);
CREATE INDEX IF NOT EXISTS idx_medical_reports_created_at ON medical_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_summaries_report_id ON summaries(report_id);
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON api_requests(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_medical_reports_updated_at 
    BEFORE UPDATE ON medical_reports 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data
INSERT INTO medical_reports (report_id, patient_id, report_text, report_type) VALUES
('RPT001', 'PAT001', 'Patient presents with chest pain and shortness of breath. EKG shows ST elevation in leads II, III, and aVF. Troponin levels are elevated. Diagnosis: STEMI.', 'ECG Report'),
('RPT002', 'PAT002', 'Patient reports headache and blurred vision. Blood pressure 180/110 mmHg. Fundoscopic exam reveals papilledema. Diagnosis: Hypertensive crisis.', 'Neurology Report'),
('RPT003', 'PAT003', 'Patient complains of abdominal pain and nausea. CT scan shows appendiceal inflammation. WBC count elevated. Diagnosis: Acute appendicitis.', 'Radiology Report')
ON CONFLICT (report_id) DO NOTHING; 