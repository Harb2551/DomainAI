"""
LLM-Powered Creative Domain Name Generator
Generates creative business descriptions and brandable domain names using Bedrock Claude
"""

import pandas as pd
import random
import json
import time
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from bedrock_llm import BedrockLLM
from generation_data import GenerationData

class CreativeLLMGenerator:
    def __init__(self, model_id: str = "us.anthropic.claude-opus-4-20250514-v1:0", provider: str = None, region: str = "us-east-1"):
        """Initialize the creative LLM generator"""
        self.llm = BedrockLLM(model_id=model_id, provider=provider, region=region)
        self.industries = GenerationData.INDUSTRIES
        self.differentiators = GenerationData.DIFFERENTIATORS
        self.target_markets = GenerationData.TARGET_MARKETS

    def generate_business_description(self, industry: str = None, differentiator: str = None, target_market: str = None) -> str:
        """Generate a concise 15-20 word business description using LLM"""
        # Use provided inputs or randomly select if not provided
        if industry is None:
            industry = random.choice(self.industries)
        if differentiator is None:
            differentiator = random.choice(self.differentiators)
        if target_market is None:
            target_market = random.choice(self.target_markets)
        
        if not self.llm.is_available():
            # Fallback to rule-based generation if LLM not available
            return f"{differentiator.capitalize()} {industry} specializing in premium services for {target_market}"
        
        prompt = f"""Generate a realistic business description in exactly 15-20 words.

Business Type: {industry}
Key Differentiator: {differentiator}
Target Market: {target_market}

Requirements:
- EXACTLY 15-20 words (very important for training consistency)
- Incorporate the business type: "{industry}"
- Highlight the differentiator: "{differentiator}"
- Target the market: "{target_market}"
- Professional and realistic
- Creative naming potential

Examples:
- "AI-powered fitness app helping busy professionals track workouts and nutrition goals effectively"
- "Sustainable coffee roastery specializing in single-origin beans for environmentally conscious consumers downtown"
- "Premium pet grooming salon offering luxury services for affluent pet owners in suburbs"

Generate ONE business description with exactly 15-20 words:"""

        description = self.llm.generate_business_description(prompt)
        
        if description:
            return description
        
        # Fallback to rule-based generation
        return f"{differentiator.capitalize()} {industry} specializing in premium services for {target_market}"

    def generate_creative_domains(self, business_description: str) -> Dict[str, str]:
        """Generate 5 creative domain options using LLM"""
        if not self.llm.is_available():
            # Fallback to simple generation
            base_words = business_description.lower().split()[:3]
            return {
                "portmanteau": "".join(base_words[:2]) + "hub",
                "metaphorical": base_words[0] + "sphere",
                "tech_modern": base_words[0] + "ly",
                "industry_creative": base_words[0] + "pro",
                "short_punchy": base_words[0][:4] + "co"
            }

        prompt = f"""Generate 5 creative domain names for this business:
Business: "{business_description}"

Generate exactly 5 creative domains with these types:
1. PORTMANTEAU: blend 2 relevant words creatively
2. METAPHORICAL: use a metaphor that fits the business
3. TECH-MODERN: add modern suffix like -ly, -ify, -hub, -lab
4. INDUSTRY-CREATIVE: creative but sector-appropriate
5. SHORT & PUNCHY: 1-2 syllables, memorable

Examples:
- Portmanteau: "Craftopia" (craft + utopia)
- Metaphorical: "IronClad" (strength metaphor)
- Tech-Modern: "Designly"
- Industry-Creative: "VelvetPaws"
- Short & Punchy: "FitCore"

Return ONLY a Python list with exactly 5 domain names in this order:
["portmanteau_domain", "metaphorical_domain", "tech_modern_domain", "industry_creative_domain", "short_punchy_domain"]"""

        domains = self.llm.generate_domain_names(prompt)
        
        if domains:
            return domains
        
        # Fallback to simple generation
        base_words = business_description.lower().split()[:3]
        return {
            "portmanteau": "".join(base_words[:2]) + "hub",
            "metaphorical": base_words[0] + "sphere",
            "tech_modern": base_words[0] + "ly",
            "industry_creative": base_words[0] + "pro",
            "short_punchy": base_words[0][:4] + "co"
        }

    def _generate_single_case(self, combination_data) -> Dict:
        """Generate a single creative normal case with all 5 domains combined - helper for multithreading"""
        import time
        import random
        
        creativity_types = ["portmanteau", "metaphorical", "tech_modern", "industry_creative", "short_punchy"]
        max_retries = 5
        base_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                # Extract combination data
                if isinstance(combination_data, tuple):
                    industry, differentiator, target_market = combination_data
                    # Generate business description with specific inputs
                    business_desc = self.generate_business_description(industry, differentiator, target_market)
                else:
                    # Fallback to random generation for backward compatibility
                    business_desc = self.generate_business_description()
                
                # Generate 5 creative domain options
                domains = self.generate_creative_domains(business_desc)
                
                # Combine all domains into a single string separated by spaces
                all_domains = []
                for domain_type in creativity_types:
                    tld = random.choice([".com", ".net", ".org"])
                    domain_name = domains[domain_type] + tld
                    all_domains.append(domain_name)
                
                # Join all domains with spaces
                combined_domains = " ".join(all_domains)
                
                return {
                    "business_description": business_desc,
                    "ideal_domain": combined_domains,
                    "label": "normal"
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    print(f"Failed to generate case after {max_retries} attempts: {e}")
                    raise e

    def generate_creative_normal_cases(self, n_samples: int = 1000) -> List[Dict]:
        """Generate creative normal cases using systematic combinatorial approach with multithreading"""
        import itertools
        import random
        
        print(f"Generating {n_samples} unique business descriptions using combinatorial approach with multithreading...")
        
        # Get the curated lists: 50 industries × 20 differentiators × 20 target markets = 20,000 combinations
        industries = self.industries
        differentiators = self.differentiators
        target_markets = self.target_markets
        
        print(f"Total possible combinations: {len(industries)} × {len(differentiators)} × {len(target_markets)} = {len(industries) * len(differentiators) * len(target_markets)}")
        
        # Generate all possible combinations
        all_combinations = list(itertools.product(industries, differentiators, target_markets))
        
        # Shuffle to get random sample of combinations (but deterministic with fixed seed)
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(all_combinations)
        
        # Take first n_samples combinations
        selected_combinations = all_combinations[:n_samples]
        
        print(f"Selected {len(selected_combinations)} unique combinations")
        
        cases = []
        max_workers = 8
        print(f"Processing combinations with {max_workers} threads...")
        
        # Use ThreadPoolExecutor with 8 threads for parallel API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks - pass combination tuples to _generate_single_case
            future_to_combination = {executor.submit(self._generate_single_case, combination): combination for combination in selected_combinations}
            
            completed = 0
            for future in as_completed(future_to_combination):
                try:
                    case = future.result()  # This is now a single case with combined domains
                    cases.append(case)  # Add the single case with all 5 domains
                    completed += 1
                    
                    if completed % 100 == 0:
                        print(f"Progress: {completed}/{len(selected_combinations)} combinations processed")
                        
                except Exception as e:
                    combination = future_to_combination[future]
                    print(f"Error generating case for combination {combination}: {e}")
                    continue
        
        print(f"Generated {len(cases)} unique business descriptions using combinatorial approach with multithreading")
        return cases
