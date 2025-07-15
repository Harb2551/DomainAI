from .base_criteria_enhancer import CriteriaEnhancer

class AppropriatenessEnhancer(CriteriaEnhancer):
    """Refined appropriateness scoring with business viability focus."""
    
    def get_criterion_name(self) -> str:
        return 'appropriateness'
    
    def get_enhanced_description(self) -> str:
        return ('Appropriateness (1-5): Professionalism and business viability\n'
                '  • 5: Highly professional, excellent for all business contexts\n'
                '  • 4: Professional with minor considerations\n'
                '  • 3: Generally acceptable with some limitations\n'
                '  • 2: Some concerns about professionalism\n'
                '  • 1: Unprofessional, unsafe, or not business-viable')