from typing import List
from ..model_evaluation_v1.bedrock_llm_judge import BedrockLLMJudge
from .domain_criteria_enhancer import DomainCriteriaEnhancer

class BedrockLLMJudgeV2(BedrockLLMJudge):
    """
    Enhanced LLM Judge with improved criteria-specific prompts.
    """

    def __init__(self, model_id: str, provider: str = None, region: str = None):
        super().__init__(model_id, provider, region)
        self._criteria_enhancer = DomainCriteriaEnhancer()

    @staticmethod
    def construct_prompt(description: str, domain: list, criteria: List[str], label: str = "", predicted_label: str = "") -> str:
        """Enhanced prompt using V1 base structure with SOLID-based criteria enhancement."""
        
        # Use dependency injection for criteria enhancement
        criteria_enhancer = DomainCriteriaEnhancer()
        enhanced_descriptions = criteria_enhancer.get_enhanced_descriptions(criteria)
        
        # Get base prompt from V1
        base_prompt = BedrockLLMJudge.construct_prompt(description, domain, criteria, label, predicted_label)
        
        # Replace basic criteria list with enhanced descriptions
        criteria_str = '\n'.join(enhanced_descriptions.values())
        
        # Replace the simple criteria list in base prompt with enhanced version
        basic_criteria_str = '\n'.join(f'- {c}' for c in criteria)
        enhanced_prompt = base_prompt.replace(basic_criteria_str, criteria_str)
        
        # Add instruction for full scale usage
        enhanced_prompt += "\n\nUse the full 1-5 scale and avoid clustering around middle scores. Be decisive in your scoring."
        
        return enhanced_prompt