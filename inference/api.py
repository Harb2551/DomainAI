"""
Domain Name Suggestion API

This module provides a FastAPI web server to expose domain name suggestion functionality.
"""

import os
import logging
import time
from typing import Dict, Optional, List, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .domain_inferencer import DomainNameInferencer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Domain Name Suggestion API",
    description="API for generating domain name suggestions for businesses",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global inferencer (will be lazy-loaded)
inferencer = None

# Request and response models
class DomainSuggestionRequest(BaseModel):
    business_description: str = Field(..., description="Description of the business")

class DomainSuggestionSuccessResponse(BaseModel):
    status: str = Field("success", description="Status of the request")
    suggestions: List[str] = Field(..., description="List of domain name suggestions")
    confidence: float = Field(..., description="Confidence score from the model")

class DomainSuggestionBlockedResponse(BaseModel):
    status: str = Field("blocked", description="Status of the request")
    suggestions: List[str] = Field([], description="Empty list for blocked requests")
    message: str = Field("request contains inappropriate or ambiguous content", 
                        description="Message explaining why the request was blocked")

# Union type for response, FastAPI will handle serialization based on actual type
DomainSuggestionResponse = Union[DomainSuggestionSuccessResponse, DomainSuggestionBlockedResponse]

def get_inferencer():
    """Get or create the global inferencer instance"""
    global inferencer
    if inferencer is None:
        logger.info("Initializing domain name inferencer")
        # Use the Mistral model by default (best performing based on evaluation)
        inferencer = DomainNameInferencer()
    return inferencer

@app.get("/")
async def root():
    """Root endpoint to verify the API is running"""
    return {
        "status": "online", 
        "message": "Domain Name Suggestion API is running"
    }

@app.post("/suggest", response_model=DomainSuggestionResponse)
async def suggest_domain(request: DomainSuggestionRequest):
    """Generate a domain name suggestion for a business description"""
    try:
        infer = get_inferencer()
        result = infer.generate_suggestion(
            description=request.business_description,
            max_new_tokens=20
        )
        
        # Process the result based on whether it's an edge case or not
        if result["is_edge_case"]:
            # Return blocked response for edge case
            return DomainSuggestionBlockedResponse(
                status="blocked",
                suggestions=[],
                message="request contains inappropriate or ambiguous content"
            )
        else:
            # For successful generation, parse domains into a list
            domain_list = result["domains"].split() if result["domains"] else []
            
            # Limit to maximum 3 domain suggestions
            if len(domain_list) > 3:
                domain_list = domain_list[:3]
            
            # Return success response with domain suggestions (limited to 3)
            return DomainSuggestionSuccessResponse(
                status="success",
                suggestions=domain_list,
                confidence=result["confidence"]
            )
    
    except Exception as e:
        logger.error(f"Error generating domain suggestion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating domain suggestion: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API and model are running"""
    try:
        # Simple test to ensure the model is loaded
        infer = get_inferencer()
        return {"status": "healthy", "model": infer.model_name}
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app with uvicorn when executed directly
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)