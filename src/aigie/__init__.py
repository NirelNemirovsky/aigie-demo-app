# Initialize the aigie package
from .enhanced_policy_node import PolicyNode
from .aigie_state_graph import AigieStateGraph
from .error_taxonomy import EnhancedTrailTaxonomyClassifier, ErrorAnalysis, ErrorCategory, ErrorSeverity
from .gemini_remediator import GeminiRemediator, GeminiRemediationResult, RemediationSuggestion

from .advanced_proactive_remediation import (
    AdaptiveRemediationEngine,
    EnhancedRemediationResult,
    FixStrategy,
    FixAttempt
)
from .ai_code_generator import (
    AICodeGenerator,
    AICodeGenerationRequest,
    AICodeGenerationResponse
)

__version__ = "0.5.4"
__all__ = [
    "PolicyNode",
    "AigieStateGraph", 
    "EnhancedTrailTaxonomyClassifier",
    "ErrorAnalysis", 
    "ErrorCategory", 
    "ErrorSeverity",
    "GeminiRemediator",
    "GeminiRemediationResult",
    "RemediationSuggestion",
    # Advanced proactive remediation exports
    "AdaptiveRemediationEngine",
    "EnhancedRemediationResult",
    "FixStrategy",
    "FixAttempt",
    # AI code generator exports
    "AICodeGenerator",
    "AICodeGenerationRequest",
    "AICodeGenerationResponse"
]