from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
import os
import re
import logging
from datetime import datetime

# Import our enhanced AI service
from ai_service import create_ai_service, load_or_build_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Medical Report Summarization API", version="2.0.0")

# Data models
class SummarizeRequest(BaseModel):
    text: str
    max_words: Optional[int] = 150
    use_llama: Optional[bool] = True
    extract_entities: Optional[bool] = True

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.3

class RAGSummarizeRequest(BaseModel):
    query: str
    k: Optional[int] = 3
    max_words: Optional[int] = 150
    use_llama: Optional[bool] = True

class BuildIndexRequest(BaseModel):
    csv_path: str
    text_column: Optional[str] = "transcription"
    save_path: Optional[str] = "medical_index"

class EnhancedSummarizeRequest(BaseModel):
    text: str
    max_words: Optional[int] = 150
    include_entities: Optional[bool] = True
    include_findings: Optional[bool] = True
    include_recommendations: Optional[bool] = True

# Global variables
ai_service = None
index_path = None

@app.on_event("startup")
async def startup_event():
    """Initialize AI service on startup"""
    global ai_service
    
    try:
        logger.info("Initializing Enhanced AI Service...")
        
        # Create AI service (will load models)
        ai_service = create_ai_service()
        
        # Check if we have existing index
        if os.path.exists("medical_index.faiss"):
            try:
                ai_service.load_faiss_index("medical_index")
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
        
        logger.info("Enhanced AI Service initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI service: {e}")
        logger.info("API will run with limited functionality")

@app.get("/health")
async def health_check():
    """Enhanced health check with AI service status"""
    global ai_service
    
    if ai_service:
        model_info = ai_service.get_model_info()
        return {
            "status": "healthy",
            "ai_service": {
                "sentence_bert_loaded": model_info['sentence_bert_loaded'],
                "llama_loaded": model_info['llama_loaded'],
                "faiss_index_built": model_info['faiss_index_built'],
                "corpus_size": model_info['corpus_size'],
                "device": model_info['device']
            },
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "degraded",
            "ai_service": "not_available",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/models")
async def get_model_info():
    """Get detailed information about loaded AI models"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    return ai_service.get_model_info()

@app.post("/enhanced_summarize")
async def enhanced_summarize(request: EnhancedSummarizeRequest):
    """
    Enhanced summarization using all AI components:
    - LLaMA-2 for intelligent summarization
    - Medical entity extraction
    - Key findings identification
    - Recommendations generation
    """
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        logger.info(f"Processing enhanced summary request for {len(request.text)} characters")
        
        # Generate comprehensive summary
        result = ai_service.generate_enhanced_summary(
            request.text, 
            request.max_words
        )
        
        # Filter results based on request
        if not request.include_entities:
            result.pop('entities', None)
        if not request.include_findings:
            result.pop('key_findings', None)
        if not request.include_recommendations:
            result.pop('recommendations', None)
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced summarization error: {str(e)}")

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """Enhanced summarization with LLaMA-2 fallback"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        if request.use_llama and ai_service.llama_model:
            # Use LLaMA-2 for summarization
            summary = ai_service.generate_llama_summary(request.text, request.max_words * 2)
            
            result = {
                "summary": summary,
                "original_length": len(request.text.split()),
                "summary_length": len(summary.split()),
                "max_words": request.max_words,
                "model_used": "llama-2"
            }
            
            # Extract entities if requested
            if request.extract_entities:
                entities = ai_service.extract_medical_entities(request.text)
                result["entities"] = entities
            
            return result
        else:
            # Fallback to basic summarization
            summary = ai_service._fallback_summarization(request.text, request.max_words)
            
            return {
                "summary": summary,
                "original_length": len(request.text.split()),
                "summary_length": len(summary.split()),
                "max_words": request.max_words,
                "model_used": "fallback"
            }
            
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

@app.post("/retrieve")
async def retrieve_documents(request: RetrieveRequest):
    """Semantic document retrieval using FAISS and sentence-BERT"""
    if not ai_service or not ai_service.faiss_index:
        raise HTTPException(status_code=400, detail="No FAISS index built. Use /build_index first.")
    
    try:
        # Perform semantic search
        results = ai_service.semantic_search(
            request.query, 
            request.top_k
        )
        
        # Filter by similarity threshold
        if request.similarity_threshold:
            results = [r for r in results if r['similarity'] >= request.similarity_threshold]
        
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results),
            "similarity_threshold": request.similarity_threshold
        }
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/rag_summarize")
async def rag_summarize(request: RAGSummarizeRequest):
    """RAG-based summarization using retrieved documents and LLaMA-2"""
    if not ai_service or not ai_service.faiss_index:
        raise HTTPException(status_code=400, detail="No FAISS index built. Use /build_index first.")
    
    try:
        # First retrieve relevant documents
        search_results = ai_service.semantic_search(request.query, request.k)
        
        if not search_results:
            return {
                "summary": "No relevant documents found for the query.",
                "query": request.query,
                "documents_retrieved": 0
            }
        
        # Combine relevant texts
        relevant_texts = [result['text'] for result in search_results]
        combined_text = " ".join(relevant_texts)
        
        # Generate summary
        if request.use_llama and ai_service.llama_model:
            summary = ai_service.generate_llama_summary(combined_text, request.max_words * 2)
            model_used = "llama-2"
        else:
            summary = ai_service._fallback_summarization(combined_text, request.max_words)
            model_used = "fallback"
        
        return {
            "summary": summary,
            "query": request.query,
            "documents_retrieved": len(relevant_texts),
            "search_results": search_results,
            "model_used": model_used,
            "max_words": request.max_words
        }
        
    except Exception as e:
        logger.error(f"RAG summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG summarization error: {str(e)}")

@app.post("/build_index")
async def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """Build FAISS index from CSV data using sentence-BERT embeddings"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        # Check if CSV exists
        if not os.path.exists(request.csv_path):
            raise HTTPException(status_code=400, detail=f"CSV file not found: {request.csv_path}")
        
        # Read CSV data
        df = pd.read_csv(request.csv_path)
        text_column = request.text_column if request.text_column in df.columns else df.columns[0]
        
        # Extract texts
        texts = df[text_column].astype(str).tolist()
        texts = [text for text in texts if text.strip() and text.lower() != 'nan']
        
        if not texts:
            raise HTTPException(status_code=400, detail="No valid text data found in CSV")
        
        # Build index in background to avoid blocking
        def build_index_task():
            try:
                global index_path
                index_path = ai_service.build_faiss_index(texts, request.save_path)
                logger.info(f"FAISS index built successfully: {index_path}")
            except Exception as e:
                logger.error(f"Background index building failed: {e}")
        
        background_tasks.add_task(build_index_task)
        
        return {
            "message": "Index building started in background",
            "csv_path": request.csv_path,
            "text_column": text_column,
            "texts_count": len(texts),
            "save_path": request.save_path,
            "status": "building"
        }
        
    except Exception as e:
        logger.error(f"Index building error: {e}")
        raise HTTPException(status_code=500, detail=f"Index building error: {str(e)}")

@app.get("/index_status")
async def get_index_status():
    """Get current FAISS index status"""
    if not ai_service:
        return {"status": "ai_service_not_available"}
    
    try:
        if ai_service.faiss_index:
            return {
                "status": "ready",
                "corpus_size": len(ai_service.medical_corpus) if ai_service.medical_corpus else 0,
                "index_type": "faiss",
                "dimension": ai_service.faiss_index.d if hasattr(ai_service.faiss_index, 'd') else None
            }
        else:
            return {"status": "no_index_built"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/extract_entities")
async def extract_medical_entities(text: str):
    """Extract medical entities from text using spaCy and custom patterns"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        entities = ai_service.extract_medical_entities(text)
        return {
            "text": text,
            "entities": entities,
            "total_entities": sum(len(v) for v in entities.values())
        }
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction error: {str(e)}")

@app.post("/semantic_search")
async def semantic_search(request: RetrieveRequest):
    """Direct semantic search endpoint"""
    if not ai_service or not ai_service.faiss_index:
        raise HTTPException(status_code=400, detail="No FAISS index built. Use /build_index first.")
    
    try:
        results = ai_service.semantic_search(request.query, request.top_k)
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
