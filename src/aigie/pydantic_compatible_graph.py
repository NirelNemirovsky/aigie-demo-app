"""
Pydantic-Compatible AigieStateGraph

This module provides a wrapper around AigieStateGraph that automatically handles
conversion between Pydantic models and Aigie's dictionary-based state system.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union, Callable
from pydantic import BaseModel
from .aigie_state_graph import AigieStateGraph
from .state_adapter import StateAdapter, WorkflowStateAdapter, create_state_adapter

T = TypeVar('T', bound=BaseModel)

class PydanticCompatibleAigieGraph:
    """
    Pydantic-compatible wrapper for AigieStateGraph.
    
    This class automatically handles conversion between Pydantic models and Aigie's
    dictionary-based state system, making it easy to integrate with existing
    LangGraph workflows that use Pydantic models.
    """
    
    def __init__(self, 
                 state_model_class: Type[T],
                 enable_gemini_remediation: bool = True,
                 gemini_project_id: Optional[str] = None,
                 auto_apply_fixes: bool = False,
                 log_remediation: bool = True):
        """
        Initialize the Pydantic-compatible Aigie graph.
        
        Args:
            state_model_class: Pydantic model class for the state
            enable_gemini_remediation: Whether to enable Gemini AI-powered error remediation
            gemini_project_id: GCP project ID for Gemini (if None, will auto-detect)
            auto_apply_fixes: Whether to automatically apply AI-suggested fixes
            log_remediation: Whether to log remediation analysis to GCP
        """
        self.state_model_class = state_model_class
        self.adapter = create_state_adapter(state_model_class)
        
        # Create the underlying AigieStateGraph
        self.graph = AigieStateGraph(
            enable_gemini_remediation=enable_gemini_remediation,
            gemini_project_id=gemini_project_id,
            auto_apply_fixes=auto_apply_fixes,
            log_remediation=log_remediation
        )
        
        # Store node wrappers for conversion
        self.node_wrappers = {}
    
    def add_node(self, node_id: str, node_func: Callable, **node_config):
        """
        Add a node that works with Pydantic models.
        
        Args:
            node_id: The identifier for the node
            node_func: Function that takes/returns Pydantic models
            **node_config: Additional configuration for the node
        """
        # Create a wrapper function that handles conversion
        def wrapped_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Convert dictionary to Pydantic model
            state_model = self.adapter.from_dict(state_dict, self.state_model_class)
            
            # Call the original function with Pydantic model
            if isinstance(state_model, dict):
                # If conversion failed, use the dictionary
                result = node_func(state_model)
            else:
                result = node_func(state_model)
            
            # Convert result back to dictionary
            if isinstance(result, dict):
                return result
            else:
                return self.adapter.to_dict(result)
        
        # Add the wrapped node to the underlying graph
        self.graph.add_node(node_id, wrapped_node, **node_config)
        self.node_wrappers[node_id] = wrapped_node
    
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes."""
        self.graph.add_edge(from_node, to_node)
    
    def set_entry_point(self, node_id: str):
        """Set the entry point for the graph."""
        self.graph.set_entry_point(node_id)
    
    def set_finish_point(self, node_id: str):
        """Set the finish point for the graph."""
        self.graph.set_finish_point(node_id)
    
    def compile(self, **kwargs):
        """Compile the graph."""
        return self.graph.compile(**kwargs)
    
    def invoke(self, initial_state: Union[T, Dict[str, Any]], **kwargs) -> Union[T, Dict[str, Any]]:
        """
        Invoke the graph with a Pydantic model or dictionary.
        
        Args:
            initial_state: Initial state as Pydantic model or dictionary
            **kwargs: Additional arguments for graph invocation
            
        Returns:
            Final state as Pydantic model or dictionary
        """
        # Convert initial state to dictionary
        state_dict = self.adapter.to_dict(initial_state)
        
        # Invoke the underlying graph
        compiled_graph = self.compile()
        result_dict = compiled_graph.invoke(state_dict, **kwargs)
        
        # Convert result back to Pydantic model
        return self.adapter.from_dict(result_dict, self.state_model_class)
    
    def stream(self, initial_state: Union[T, Dict[str, Any]], **kwargs):
        """
        Stream the graph execution with a Pydantic model or dictionary.
        
        Args:
            initial_state: Initial state as Pydantic model or dictionary
            **kwargs: Additional arguments for graph streaming
            
        Yields:
            Stream events with converted state
        """
        # Convert initial state to dictionary
        state_dict = self.adapter.to_dict(initial_state)
        
        # Stream the underlying graph
        compiled_graph = self.compile()
        for event in compiled_graph.stream(state_dict, **kwargs):
            # Convert state in events back to Pydantic model
            if 'state' in event:
                event['state'] = self.adapter.from_dict(event['state'], self.state_model_class)
            yield event
    
    def get_node_analytics(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get error analytics for nodes."""
        return self.graph.get_node_analytics(node_id)
    
    def get_graph_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the entire graph."""
        return self.graph.get_graph_analytics()
    
    def clear_analytics(self):
        """Clear analytics data for all nodes."""
        self.graph.clear_analytics()


class WorkflowCompatibleAigieGraph(PydanticCompatibleAigieGraph):
    """
    Specialized Pydantic-compatible graph for workflow states.
    
    This class provides additional utilities for handling workflow-specific
    state management and validation.
    """
    
    def __init__(self, 
                 state_model_class: Type[T],
                 enable_gemini_remediation: bool = True,
                 gemini_project_id: Optional[str] = None,
                 auto_apply_fixes: bool = False,
                 log_remediation: bool = True):
        super().__init__(state_model_class, enable_gemini_remediation, gemini_project_id, 
                        auto_apply_fixes, log_remediation)
        
        # Use workflow-specific adapter
        self.adapter = WorkflowStateAdapter(state_model_class)
    
    def add_workflow_node(self, node_id: str, node_func: Callable, **node_config):
        """
        Add a workflow node that handles workflow-specific state updates.
        
        Args:
            node_id: The identifier for the node
            node_func: Function that takes/returns workflow state
            **node_config: Additional configuration for the node
        """
        def wrapped_workflow_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Convert dictionary to Pydantic model
            state_model = self.adapter.from_dict(state_dict, self.state_model_class)
            
            # Call the original function with Pydantic model
            if isinstance(state_model, dict):
                result = node_func(state_model)
            else:
                result = node_func(state_model)
            
            # Convert result back to dictionary, ensuring only workflow fields are updated
            if isinstance(result, dict):
                return self.adapter.merge_workflow_state(state_dict, result)
            else:
                result_dict = self.adapter.to_dict(result)
                return self.adapter.merge_workflow_state(state_dict, result_dict)
        
        # Add the wrapped node to the underlying graph
        self.graph.add_node(node_id, wrapped_workflow_node, **node_config)
        self.node_wrappers[node_id] = wrapped_workflow_node
    
    def validate_workflow_state(self, state: Union[T, Dict[str, Any]]) -> bool:
        """
        Validate that a state has the required workflow fields.
        
        Args:
            state: State to validate
            
        Returns:
            True if valid, False otherwise
        """
        return self.adapter.validate_state(state, self.state_model_class)
    
    def extract_workflow_fields(self, state: Union[T, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract only workflow-related fields from a state.
        
        Args:
            state: State object or dictionary
            
        Returns:
            Dictionary containing only workflow fields
        """
        return self.adapter.extract_workflow_fields(state)


# Factory functions for easy creation
def create_pydantic_compatible_graph(state_model_class: Type[T], **kwargs) -> PydanticCompatibleAigieGraph:
    """
    Create a Pydantic-compatible Aigie graph.
    
    Args:
        state_model_class: Pydantic model class for the state
        **kwargs: Additional arguments for graph configuration
        
    Returns:
        PydanticCompatibleAigieGraph instance
    """
    return PydanticCompatibleAigieGraph(state_model_class, **kwargs)


def create_workflow_compatible_graph(state_model_class: Type[T], **kwargs) -> WorkflowCompatibleAigieGraph:
    """
    Create a workflow-compatible Aigie graph.
    
    Args:
        state_model_class: Pydantic model class for the state
        **kwargs: Additional arguments for graph configuration
        
    Returns:
        WorkflowCompatibleAigieGraph instance
    """
    return WorkflowCompatibleAigieGraph(state_model_class, **kwargs)
