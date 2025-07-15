from typing import Dict, List
from .creativity_enhancer import CreativityEnhancer
from .relevance_enhancer import RelevanceEnhancer
from .appropriateness_enhancer import AppropriatenessEnhancer

class DomainCriteriaEnhancer:
    """
    Composite class that manages all domain evaluation criteria enhancers.
    
    Following Composition principle - combines all individual enhancers
    to provide a unified interface for enhanced criteria descriptions.
    """
    
    def __init__(self):
        self._enhancers = {
            'creativity': CreativityEnhancer(),
            'relevance': RelevanceEnhancer(),
            'appropriateness': AppropriatenessEnhancer()
        }
    
    def get_enhanced_descriptions(self, criteria: List[str]) -> Dict[str, str]:
        """
        Get enhanced descriptions for all provided criteria.
        
        Args:
            criteria: List of criterion names to enhance
            
        Returns:
            Dictionary mapping criterion names to their enhanced descriptions
        """
        descriptions = {}
        for criterion in criteria:
            if criterion in self._enhancers:
                descriptions[criterion] = self._enhancers[criterion].get_enhanced_description()
            else:
                # Fallback for unknown criteria
                descriptions[criterion] = f'- {criterion}'
        return descriptions
    
    def get_enhanced_criteria_string(self, criteria: List[str]) -> str:
        """
        Get enhanced criteria descriptions as a formatted string.
        
        Args:
            criteria: List of criterion names to enhance
            
        Returns:
            Formatted string with all enhanced criteria descriptions
        """
        descriptions = self.get_enhanced_descriptions(criteria)
        return '\n'.join(descriptions.values())
    
    def has_enhancer_for(self, criterion: str) -> bool:
        """Check if an enhancer exists for the given criterion."""
        return criterion in self._enhancers
    
    def get_supported_criteria(self) -> List[str]:
        """Get list of all supported criteria."""
        return list(self._enhancers.keys())