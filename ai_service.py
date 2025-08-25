"""
Enhanced AI Service for Medical Summarization
Integrates FAISS, sentence-BERT, and LLaMA-2 for advanced text processing
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime

# Core ML libraries
import faiss
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline,
    GenerationConfig
)

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import textstat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAIService:
    """
    Advanced AI service combining FAISS, sentence-BERT, and LLaMA-2
    for medical text processing and summarization
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or "meta-llama/Llama-2-7b-chat-hf"
        
        # Initialize models
        self.sentence_model = None
        self.llama_model = None
        self.llama_tokenizer = None
        self.faiss_index = None
        self.medical_corpus = None
        
        # Load models
        self._load_models()
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def _load_models(self):
        """Load and initialize all AI models"""
        try:
            logger.info("Loading sentence-BERT model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            logger.info("Loading LLaMA-2 model...")
            self._load_llama_model()
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_llama_model(self):
        """Load LLaMA-2 model with quantization for efficiency"""
        try:
            # Quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load tokenizer
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Load model with quantization
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Set padding token
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            
            logger.info("LLaMA-2 model loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Could not load LLaMA-2 model: {e}")
            logger.info("Falling back to basic summarization methods")
            self.llama_model = None
            self.llama_tokenizer = None
    
    def build_faiss_index(self, texts: List[str], save_path: str = None) -> str:
        """
        Build FAISS index from medical texts for efficient similarity search
        
        Args:
            texts: List of medical text documents
            save_path: Path to save the index
            
        Returns:
            Path to saved index
        """
        try:
            logger.info(f"Building FAISS index for {len(texts)} documents...")
            
            # Generate embeddings using sentence-BERT
            embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy().astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.faiss_index.add(embeddings)
            
            # Store corpus for retrieval
            self.medical_corpus = texts
            
            # Save index if path provided
            if save_path:
                faiss.write_index(self.faiss_index, f"{save_path}.faiss")
                with open(f"{save_path}_corpus.json", 'w') as f:
                    json.dump(texts, f)
                logger.info(f"FAISS index saved to {save_path}")
            
            return save_path or "index"
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
    
    def load_faiss_index(self, index_path: str):
        """Load existing FAISS index"""
        try:
            self.faiss_index = faiss.read_index(f"{index_path}.faiss")
            with open(f"{index_path}_corpus.json", 'r') as f:
                self.medical_corpus = json.load(f)
            logger.info(f"FAISS index loaded from {index_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using FAISS and sentence-BERT
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with similarity scores
        """
        if self.faiss_index is None or self.medical_corpus is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        try:
            # Encode query
            query_embedding = self.sentence_model.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy().astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search index
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1:  # Valid index
                    results.append({
                        'rank': i + 1,
                        'similarity': float(similarity),
                        'text': self.medical_corpus[idx],
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    def generate_llama_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate summary using LLaMA-2 model
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        if self.llama_model is None:
            raise ValueError("LLaMA-2 model not loaded")
        
        try:
            # Create prompt for medical summarization
            prompt = f"""<s>[INST] You are a medical professional. Please provide a concise, accurate summary of the following medical report. Focus on key findings, diagnoses, and recommendations. Keep the summary under {max_length} words.

Medical Report:
{text}

Summary: [/INST]"""
            
            # Tokenize input
            inputs = self.llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llama_tokenizer.eos_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id
                )
            
            # Decode and clean output
            summary = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.replace(prompt, "").strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating LLaMA summary: {e}")
            raise
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using spaCy and custom patterns
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Dictionary of extracted entities by category
        """
        try:
            doc = self.nlp(text)
            
            # Define medical entity patterns
            entities = {
                'symptoms': [],
                'diagnoses': [],
                'medications': [],
                'procedures': [],
                'vital_signs': [],
                'lab_values': []
            }
            
            # Extract entities using spaCy
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'CONDITION']:
                    entities['diagnoses'].append(ent.text)
                elif ent.label_ in ['DRUG', 'MEDICATION']:
                    entities['medications'].append(ent.text)
                elif ent.label_ in ['PROCEDURE']:
                    entities['procedures'].append(ent.text)
            
            # Custom pattern matching for medical terms
            import re
            
            # Vital signs patterns
            bp_pattern = r'blood pressure[:\s]*(\d+/\d+)'
            hr_pattern = r'heart rate[:\s]*(\d+)'
            temp_pattern = r'temperature[:\s]*(\d+\.?\d*)'
            
            bp_matches = re.findall(bp_pattern, text, re.IGNORECASE)
            hr_matches = re.findall(hr_pattern, text, re.IGNORECASE)
            temp_matches = re.findall(temp_pattern, text, re.IGNORECASE)
            
            entities['vital_signs'].extend([f"BP: {bp}" for bp in bp_matches])
            entities['vital_signs'].extend([f"HR: {hr}" for hr in hr_matches])
            entities['vital_signs'].extend([f"Temp: {temp}" for temp in temp_matches])
            
            # Lab values patterns
            lab_pattern = r'(\d+\.?\d*)\s*(mg/dL|mmol/L|mEq/L|ng/mL)'
            lab_matches = re.findall(lab_pattern, text)
            entities['lab_values'].extend([f"{value} {unit}" for value, unit in lab_matches])
            
            # Remove duplicates and empty categories
            for category in entities:
                entities[category] = list(set(entities[category]))
                entities[category] = [item for item in entities[category] if item.strip()]
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting medical entities: {e}")
            return {}
    
    def generate_enhanced_summary(self, text: str, max_words: int = 150) -> Dict[str, Any]:
        """
        Generate comprehensive medical summary using all AI components
        
        Args:
            text: Medical text to summarize
            max_words: Maximum words for summary
            
        Returns:
            Dictionary containing summary and extracted information
        """
        try:
            result = {
                'summary': '',
                'key_findings': [],
                'recommendations': [],
                'entities': {},
                'metadata': {}
            }
            
            # Generate LLaMA-2 summary if available
            if self.llama_model:
                try:
                    result['summary'] = self.generate_llama_summary(text, max_words * 2)
                except Exception as e:
                    logger.warning(f"LLaMA summary failed, using fallback: {e}")
                    result['summary'] = self._fallback_summarization(text, max_words)
            else:
                result['summary'] = self._fallback_summarization(text, max_words)
            
            # Extract medical entities
            result['entities'] = self.extract_medical_entities(text)
            
            # Generate key findings
            result['key_findings'] = self._extract_key_findings(text, result['entities'])
            
            # Generate recommendations
            result['recommendations'] = self._generate_recommendations(text, result['entities'])
            
            # Add metadata
            result['metadata'] = {
                'word_count': len(text.split()),
                'summary_length': len(result['summary'].split()),
                'entities_found': sum(len(v) for v in result['entities'].values()),
                'generated_at': datetime.now().isoformat(),
                'model_used': 'llama-2' if self.llama_model else 'fallback'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating enhanced summary: {e}")
            raise
    
    def _fallback_summarization(self, text: str, max_words: int) -> str:
        """Fallback summarization using sentence scoring"""
        try:
            sentences = sent_tokenize(text)
            
            # Score sentences based on medical keywords
            medical_keywords = [
                'diagnosis', 'symptoms', 'treatment', 'medication', 'procedure',
                'results', 'findings', 'recommendation', 'assessment', 'plan'
            ]
            
            sentence_scores = []
            for sentence in sentences:
                score = sum(1 for keyword in medical_keywords if keyword.lower() in sentence.lower())
                sentence_scores.append((score, sentence))
            
            # Sort by score and select top sentences
            sentence_scores.sort(reverse=True)
            
            summary = []
            word_count = 0
            for _, sentence in sentence_scores:
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= max_words:
                    summary.append(sentence)
                    word_count += sentence_words
                else:
                    break
            
            return ' '.join(summary) if summary else text[:max_words * 5]
            
        except Exception as e:
            logger.error(f"Fallback summarization failed: {e}")
            return text[:max_words * 5]
    
    def _extract_key_findings(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Extract key findings from text and entities"""
        findings = []
        
        # Add diagnoses
        if entities.get('diagnoses'):
            findings.extend([f"Diagnosis: {d}" for d in entities['diagnoses'][:3]])
        
        # Add vital signs
        if entities.get('vital_signs'):
            findings.extend([f"Vital signs: {v}" for v in entities['vital_signs'][:3]])
        
        # Add lab values
        if entities.get('lab_values'):
            findings.extend([f"Lab values: {l}" for l in entities['lab_values'][:3]])
        
        # If no specific findings, create general ones
        if not findings:
            findings = [
                "Patient presents with medical symptoms requiring evaluation",
                "Clinical assessment and monitoring recommended",
                "Further diagnostic testing may be indicated"
            ]
        
        return findings[:5]  # Limit to 5 findings
    
    def _generate_recommendations(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Generate medical recommendations based on text and entities"""
        recommendations = []
        
        # Generate recommendations based on content
        text_lower = text.lower()
        
        if 'chest pain' in text_lower:
            recommendations.extend([
                "Immediate cardiac evaluation recommended",
                "Consider ECG and cardiac enzyme testing",
                "Monitor for signs of cardiac compromise"
            ])
        
        if 'hypertension' in text_lower or 'high blood pressure' in text_lower:
            recommendations.extend([
                "Blood pressure monitoring and lifestyle modifications",
                "Consider antihypertensive medication if indicated",
                "Regular follow-up for blood pressure management"
            ])
        
        if 'diabetes' in text_lower:
            recommendations.extend([
                "Blood glucose monitoring and dietary management",
                "Regular HbA1c testing and diabetic care",
                "Foot care and eye examination screening"
            ])
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Follow up with primary care provider",
                "Continue monitoring symptoms",
                "Seek immediate care if symptoms worsen"
            ]
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'sentence_bert_loaded': self.sentence_model is not None,
            'llama_loaded': self.llama_model is not None,
            'faiss_index_built': self.faiss_index is not None,
            'corpus_size': len(self.medical_corpus) if self.medical_corpus else 0,
            'device': self.device,
            'models': {
                'sentence_transformer': 'all-MiniLM-L6-v2',
                'llama': self.model_path if self.llama_model else None
            }
        }

# Utility functions for easy integration
def create_ai_service(model_path: str = None, device: str = None) -> EnhancedAIService:
    """Factory function to create AI service"""
    return EnhancedAIService(model_path, device)

def load_or_build_index(ai_service: EnhancedAIService, texts: List[str], 
                        index_path: str = "medical_index") -> str:
    """Load existing index or build new one"""
    try:
        ai_service.load_faiss_index(index_path)
        return index_path
    except:
        return ai_service.build_faiss_index(texts, index_path) 