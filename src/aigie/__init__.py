# Initialize the aigie package
from .aigie_node import PolicyNode
from .enhanced_policy_node import EnhancedPolicyNode
from .aigie_state_graph import AigieStateGraph
from .error_taxonomy import TrailTaxonomyClassifier, ErrorAnalysis, ErrorCategory, ErrorSeverity
from .gemini_remediator import GeminiRemediator, GeminiRemediationResult, RemediationSuggestion
from .state_adapter import (
    StateAdapter, 
    WorkflowStateAdapter, 
    create_state_adapter, 
    create_workflow_adapter,
    pydantic_to_dict,
    dict_to_pydantic,
    validate_workflow_state
)
from .pydantic_compatible_graph import (
    PydanticCompatibleAigieGraph,
    WorkflowCompatibleAigieGraph,
    create_pydantic_compatible_graph,
    create_workflow_compatible_graph
)

__version__ = "0.5.0"
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
    "RemediationSuggestion",
    # State adapter exports
    "StateAdapter",
    "WorkflowStateAdapter", 
    "create_state_adapter",
    "create_workflow_adapter",
    "pydantic_to_dict",
    "dict_to_pydantic",
    "validate_workflow_state",
    # Pydantic-compatible graph exports
    "PydanticCompatibleAigieGraph",
    "WorkflowCompatibleAigieGraph",
    "create_pydantic_compatible_graph",
    "create_workflow_compatible_graph"
]