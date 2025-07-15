"""
Model Evaluation V2 - Enhanced Criteria-Specific Prompts

This version enhances the evaluation criteria with detailed guidance following SOLID principles:
- Improved Creativity Assessment with concrete examples (Spotify, Netflix, Airbnb)
- Enhanced Relevance Evaluation with semantic connection clarity
- Refined Appropriateness Scoring with business viability focus

Key improvements over V1:
- Detailed scoring rubrics with 1-5 scale examples
- Modular class structure following SOLID principles
- Extensible design for adding new criteria enhancers
- Clear separation of concerns
"""

from .bedrock_llm_judge_v2 import BedrockLLMJudgeV2
from .evaluation_pipeline_v2 import evaluate_model_v2
from .domain_criteria_enhancer import DomainCriteriaEnhancer
from .creativity_enhancer import CreativityEnhancer
from .relevance_enhancer import RelevanceEnhancer
from .appropriateness_enhancer import AppropriatenessEnhancer

__all__ = [
    'BedrockLLMJudgeV2', 
    'evaluate_model_v2', 
    'DomainCriteriaEnhancer',
    'CreativityEnhancer',
    'RelevanceEnhancer', 
    'AppropriatenessEnhancer'
]