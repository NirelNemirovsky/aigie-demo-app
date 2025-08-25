# Initialize the aigie package
from .aigie_node import PolicyNode
from .enhanced_policy_node import EnhancedPolicyNode
from .aigie_state_graph import AigieStateGraph
from .error_taxonomy import TrailTaxonomyClassifier, ErrorAnalysis, ErrorCategory, ErrorSeverity
from .gemini_remediator import GeminiRemediator, GeminiRemediationResult, RemediationSuggestion

__version__ = "0.2.0"
__all__ = [
    "PolicyNode", 
    "EnhancedPolicyNode",
    "AigieStateGraph", 
    "TrailTaxonomyClassifier",
    "ErrorAnalysis", 
    "ErrorCategory", 
    "ErrorSeverity",
    "GeminiRemediator",
    "GeminiRemediationResult",
    "RemediationSuggestion"
]