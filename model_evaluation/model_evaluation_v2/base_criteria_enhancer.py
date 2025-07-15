from abc import ABC, abstractmethod

class CriteriaEnhancer(ABC):
    """
    Abstract base class for criteria enhancers.
    
    Following Single Responsibility Principle - each enhancer is responsible
    for one specific evaluation criterion.
    """
    
    @abstractmethod
    def get_enhanced_description(self) -> str:
        """Return enhanced description for this criterion."""
        pass
    
    @abstractmethod
    def get_criterion_name(self) -> str:
        """Return the name of the criterion this enhancer handles."""
        pass