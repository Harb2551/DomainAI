from .base_criteria_enhancer import CriteriaEnhancer

class RelevanceEnhancer(CriteriaEnhancer):
    """Enhanced relevance evaluation with semantic connection clarity."""
    
    def get_criterion_name(self) -> str:
        return 'relevance'
    
    def get_enhanced_description(self) -> str:
        return ('Relevance (1-5): Semantic connection and industry appropriateness\n'
                '  • 5: Perfect connection - direct or clear metaphorical relevance\n'
                '  • 4: Strong business connection, easily understood\n'
                '  • 3: Reasonable connection, may require interpretation\n'
                '  • 2: Weak/tangential connection, unclear relationship\n'
                '  • 1: No clear connection to business description')