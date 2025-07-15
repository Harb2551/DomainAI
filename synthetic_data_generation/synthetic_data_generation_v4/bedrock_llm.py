"""
Bedrock LLM Client using pydantic_ai
Wrapper for AWS Bedrock Claude models with response validation
"""

import asyncio
import json
import re
import ast
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai import Agent

class BusinessDescriptionResponse(BaseModel):
    business_description: str

class DomainNamesResponse(BaseModel):
    portmanteau: str
    metaphorical: str
    tech_modern: str
    industry_creative: str
    short_punchy: str

class BedrockLLM:
    def __init__(self, model_id: str = "us.anthropic.claude-opus-4-20250514-v1:0", provider: str = "anthropic", region: str = "us-east-1"):
        """Initialize Bedrock client with pydantic_ai"""
        self.model_id = model_id
        self.provider = provider
        self.region = region
        
        try:
            if provider:
                self.model = BedrockConverseModel(model_id, provider=provider)
            else:
                self.model = BedrockConverseModel(model_id)
            
            self.agent = Agent(self.model)
            self.available = True
        except Exception as e:
            print(f"Failed to initialize Bedrock client: {e}")
            self.available = False

    def generate_business_description(self, prompt: str) -> Optional[str]:
        """Generate business description using Bedrock Claude"""
        if not self.available:
            return None
            
        try:
            result = asyncio.run(self.agent.run(prompt))
            response = result.output
            
            # Extract business description from response
            # Handle both direct response and formatted response
            if isinstance(response, str):
                cleaned = response.strip()
            else:
                cleaned = str(response).strip()
            
            # Clean up the response to match v1 format
            cleaned = self._clean_business_description(cleaned)
            return cleaned
            
        except Exception as e:
            print(f"Business description generation failed: {e}")
            return None

    def _clean_business_description(self, text: str) -> str:
        """Clean business description to remove unwanted formatting"""
        if not text:
            return text
            
        # Remove triple quotes at start and end
        text = text.strip()
        if text.startswith('"""') and text.endswith('"""'):
            text = text[3:-3].strip()
        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        
        # Remove word count annotations like "(17 words)"
        import re
        text = re.sub(r'\s*\(\d+\s+words?\)\s*$', '', text, flags=re.IGNORECASE)
        
        # Remove any trailing quotes or formatting
        text = text.strip().strip('"').strip("'")
        
        # Remove newlines and normalize whitespace
        text = ' '.join(text.split())
        
        return text

    def generate_domain_names(self, prompt: str) -> Optional[Dict[str, str]]:
        """Generate domain names using Bedrock Claude"""
        if not self.available:
            return None
            
        try:
            result = asyncio.run(self.agent.run(prompt))
            response = result.output
            
            # Parse domain names from response
            return self._parse_domain_response(response)
            
        except Exception as e:
            print(f"Domain names generation failed: {e}")
            return None

    def generate_edge_cases(self, prompt: str, n_samples: int) -> Optional[List[str]]:
        """Generate edge cases using Bedrock Claude"""
        if not self.available:
            return None
            
        try:
            result = asyncio.run(self.agent.run(prompt))
            response = result.output
            
            # Parse edge cases from response
            return self._parse_edge_cases_response(response, n_samples)
            
        except Exception as e:
            print(f"Edge cases generation failed: {e}")
            return None

    def _parse_domain_response(self, response: str) -> Dict[str, str]:
        """Parse domain names from LLM response expecting Python list format"""
        categories = ["portmanteau", "metaphorical", "tech_modern", "industry_creative", "short_punchy"]
        
        try:
            # Try to find Python list in response
            list_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if list_match:
                list_str = '[' + list_match.group(1) + ']'
                domain_list = ast.literal_eval(list_str)
                
                if isinstance(domain_list, list) and len(domain_list) >= 5:
                    domains = {}
                    for i, category in enumerate(categories):
                        if i < len(domain_list):
                            domain = str(domain_list[i]).strip().strip('"').lower()
                            # Clean domain name - keep alphanumeric only
                            domain = ''.join(c for c in domain if c.isalnum())
                            domains[category] = domain if domain else f"domain{i+1}"
                        else:
                            domains[category] = f"domain{i+1}"
                    return domains
        except (ValueError, SyntaxError) as e:
            print(f"Failed to parse Python list from response: {e}")
        
        # Fallback: try to parse line by line (original method)
        domains = {}
        lines = response.strip().split('\n')
        for i, category in enumerate(categories):
            if i < len(lines):
                line = lines[i]
                if ":" in line:
                    domain = line.split(":")[-1].strip().strip('"').lower()
                else:
                    domain = line.strip().strip('"').lower()
                    
                # Clean domain name
                domain = ''.join(c for c in domain if c.isalnum())
                domains[category] = domain if domain else f"domain{i+1}"
            else:
                domains[category] = f"domain{i+1}"
        
        return domains

    def _parse_edge_cases_response(self, response: str, n_samples: int) -> List[str]:
        """Parse edge cases from LLM response"""
        lines = response.strip().split('\n')
        cases = []
        
        for line in lines[:n_samples]:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Remove numbering if present
                if line[0].isdigit() and '.' in line[:5]:
                    line = line.split('.', 1)[1].strip()
                if line.startswith('-'):
                    line = line[1:].strip()
                cases.append(line)
        
        return cases[:n_samples]

    def is_available(self) -> bool:
        """Check if Bedrock client is available"""
        return self.available