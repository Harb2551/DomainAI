"""
Model Evaluation V3: Simple Multi-Judge Pipeline
Runs v2 evaluation with 3 different judge models.
"""

from .main import main
from .multijudge_pipeline import MultiJudgeEvaluationPipeline

__all__ = ['main', 'MultiJudgeEvaluationPipeline']