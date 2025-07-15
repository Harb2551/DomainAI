"""
Command Line Interface for Domain Name Suggestions

This module provides a simple script to generate domain name suggestions
by modifying the variables at the top of this file.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.domain_inferencer import DomainNameInferencer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================================================================
# CONFIGURATION - Modify these variables as needed
# ======================================================================

# Business description to generate a domain name for
BUSINESS_DESCRIPTION = "a dating website which matches individuals with common interest"

# Model name to use for inference
MODEL_NAME = "mistralai-Mistral-7B-v0-1-finetuned-domainai"

# Maximum number of tokens to generate
MAX_TOKENS = 10

# Output format: "text", "json", or "pretty"
OUTPUT_FORMAT = "pretty"

# ======================================================================

def format_api_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format the inferencer result to match the API response format"""
    if result["is_edge_case"]:
        return {
            "status": "blocked",
            "suggestions": [],
            "message": "request contains inappropriate or ambiguous content"
        }
    else:
        # Parse domains into a list
        domain_list = result["domains"].split() if result["domains"] else []
        
        # Limit to maximum 3 domain suggestions
        if len(domain_list) > 3:
            domain_list = domain_list[:3]
            
        return {
            "status": "success",
            "suggestions": domain_list,
            "confidence": result["confidence"]
        }

def format_output(result: Dict[str, Any], output_format: str) -> str:
    """Format the inference result based on the specified output format"""
    # First convert to API format
    api_response = format_api_response(result)
    
    if output_format == "json":
        return json.dumps(api_response, indent=2)
    
    elif output_format == "text":
        # Simple text output - if it's blocked, show message, otherwise show suggestions
        if api_response["status"] == "blocked":
            return api_response["message"]
        else:
            return " ".join(api_response["suggestions"])
    
    else:  # pretty
        # Pretty-printed output
        output = [
            f"Business Description: {result['business_description']}",
            f"Raw Model Output: {result['raw_output']}",
        ]
        if api_response["status"] == "blocked":
            output.extend([
                f"Status: {api_response['status']}",
                f"Message: {api_response['message']}"
            ])
        else:
            output.extend([
                f"Status: {api_response['status']}",
                f"Suggestions: {', '.join(api_response['suggestions'])}",
                f"Confidence: {api_response['confidence']:.4f}"
            ])
            
        output.append(f"Model: {result['model']}")
        output.append(f"Inference Time: {result['inference_time']:.4f}s")
        
        if "error" in result and result["error"]:
            output.append(f"Error: {result['error']}")
            
        return "\n".join(output)

def main():
    """Main entry point for CLI tool"""
    try:
        # Initialize the inferencer
        logger.info(f"Loading model: {MODEL_NAME}")
        inferencer = DomainNameInferencer(model_name=MODEL_NAME)
        
        # Generate domain suggestion
        logger.info(f"Generating domain for: {BUSINESS_DESCRIPTION}")
        result = inferencer.generate_suggestion(
            description=BUSINESS_DESCRIPTION,
            max_new_tokens=MAX_TOKENS
        )
        
        # Format and print the result
        output = format_output(result, OUTPUT_FORMAT)
        print(output)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()