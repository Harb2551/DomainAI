from .base_criteria_enhancer import CriteriaEnhancer

class CreativityEnhancer(CriteriaEnhancer):
    """Enhanced creativity assessment with concrete examples and clear level definitions."""
    
    def get_criterion_name(self) -> str:
        return 'creativity'
    
    def get_enhanced_description(self) -> str:
        return ('Creativity (1-5): Originality, memorability, and brandability\n'
                '  • 5: Highly creative (e.g., Spotify, Netflix, Airbnb) - unique wordplay, strong brand potential\n'
                '  • 4: Creative with good memorability (e.g., Dropbox, Snapchat)\n'
                '  • 3: Moderately creative but conventional (e.g., TechSolutions, GreenEats)\n'
                '  • 2: Low creativity, predictable (e.g., BestPizza, QuickRepair)\n'
                '  • 1: Generic, unoriginal (e.g., TheShop, Number1Service)')