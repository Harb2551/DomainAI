import json
import asyncio
import re
from typing import List, Dict
from pydantic import BaseModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai import Agent

class LLMJudgeResponse(BaseModel):
    relevance: float = 0
    appropriateness: float = 0
    creativity: float = 0
    # Add more fields if your criteria expand

class BedrockLLMJudge:
    def __init__(self, model_id: str, provider: str = None, region: str = None):
        if provider:
            self.model = BedrockConverseModel(model_id, provider=provider)
        else:
            self.model = BedrockConverseModel(model_id)
        self.agent = Agent(self.model)
        self.region = region  # region is stored for reference but not used by BedrockConverseModel

    def score(self, description: str, domain: list, criteria: List[str], label: str = "", predicted_label: str = "") -> List[Dict[str, float]]:
        prompt = self.construct_prompt(description, domain, criteria, label, predicted_label)
        result = asyncio.run(self.agent.run(prompt))
        response = result.output
        scores = self._extract_json_scores(response)
        if scores is None:
            print(f"[BedrockLLMJudge] Failed to parse LLM response: {response}")
            scores = {}
        return scores

    @staticmethod
    def _extract_json_scores(response: str):
        """
        Extract the first JSON object from the LLM response, even if it's inside a code block or surrounded by text.
        """
        # Try to find a ```json ... ``` code block
        match = re.search(r"```json(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
        else:
            # Fallback: find the first {...} object
            match = re.search(r"\{[\s\S]*?\}", response)
            if match:
                json_str = match.group(0)
            else:
                return None
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"[BedrockLLMJudge] JSON extraction failed: {e}")
            return None

    @staticmethod
    def construct_prompt(description: str, domain: list, criteria: List[str], label: str = "", predicted_label: str = "") -> str:
        criteria_str = '\n'.join(f'- {c}' for c in criteria)
        domain_str = ", ".join(domain) if domain else "[empty]"
        prompt = (
            "You are an expert evaluator for domain name suggestions. "
            "Given a business description, a domain name, a ground truth label, and a predicted label from the model, your task is as follows:\n"
            "- If the ground truth label is 'normal', evaluate the quality of the domain name for the business description on a scale of 1 to 5 for each criterion.\n"
            "- If the ground truth label is not 'normal', this is an edge case. In this case, compare the predicted label to the ground truth label.\n"
            "    - If the predicted label matches the ground truth label, give a full score (5) for all criteria.\n"
            "    - If the predicted label does not match, give a lower score (e.g., 1) for all criteria.\n"
            f"Business Description: {description}\n"
            f"Domain Names: {domain_str}\n"
            f"Ground Truth Label: {label}\n"
            f"Predicted Label: {predicted_label}\n"
            f"Criteria for evaluation:\n{criteria_str}\n\n"
            "Return your output in JSON format as a single object representing the combined score for both domain names.\n"
            "For example: {\"relevance\": 5, \"appropriateness\": 5, \"creativity\": 4}"
        )
        return prompt
