"""
State Adapter for Pydantic Model Compatibility

This module provides utilities to convert between Pydantic models and Aigie's dictionary-based state system,
enabling seamless integration with existing LangGraph workflows that use Pydantic models.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union
from pydantic import BaseModel
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from datetime import datetime

T = TypeVar('T', bound=BaseModel)

class StateAdapter:
    """
    Adapter class to convert between Pydantic models and Aigie's dictionary-based state.
    
    This enables seamless integration with existing LangGraph workflows that use Pydantic models
    while maintaining Aigie's dictionary-based approach for flexibility and performance.
    """
    
    def __init__(self, model_class: Optional[Type[BaseModel]] = None):
        """
        Initialize the StateAdapter.
        
        Args:
            model_class: Optional Pydantic model class for type validation
        """
        self.model_class = model_class
    
    def to_dict(self, state: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert a Pydantic model or dictionary to a plain dictionary.
        
        Args:
            state: Pydantic model instance or dictionary
            
        Returns:
            Plain dictionary representation of the state
        """
        if isinstance(state, dict):
            return state.copy()
        elif isinstance(state, BaseModel):
            return self._pydantic_to_dict(state)
        elif is_dataclass(state):
            return asdict(state)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
    
    def from_dict(self, state_dict: Dict[str, Any], model_class: Optional[Type[T]] = None) -> Union[T, Dict[str, Any]]:
        """
        Convert a dictionary back to a Pydantic model or return as dictionary.
        
        Args:
            state_dict: Dictionary representation of the state
            model_class: Pydantic model class to convert to (if None, returns dict)
            
        Returns:
            Pydantic model instance or dictionary
        """
        if model_class is None:
            model_class = self.model_class
            
        if model_class is None:
            return state_dict
        
        try:
            return model_class(**state_dict)
        except Exception as e:
            # If conversion fails, return the dictionary with a warning
            print(f"Warning: Could not convert state to {model_class.__name__}: {e}")
            return state_dict
    
    def _pydantic_to_dict(self, model: BaseModel) -> Dict[str, Any]:
        """
        Convert a Pydantic model to a dictionary, handling special types.
        
        Args:
            model: Pydantic model instance
            
        Returns:
            Dictionary representation
        """
        # Use Pydantic's model_dump() method (v2) or dict() method (v1)
        if hasattr(model, 'model_dump'):
            # Pydantic v2
            return model.model_dump()
        else:
            # Pydantic v1
            return model.dict()
    
    def validate_state(self, state: Union[BaseModel, Dict[str, Any]], 
                      model_class: Optional[Type[T]] = None) -> bool:
        """
        Validate that a state conforms to the expected model schema.
        
        Args:
            state: State to validate
            model_class: Model class to validate against (if None, uses self.model_class)
            
        Returns:
            True if valid, False otherwise
        """
        if model_class is None:
            model_class = self.model_class
            
        if model_class is None:
            return True  # No validation possible
        
        try:
            if isinstance(state, dict):
                model_class(**state)
            elif isinstance(state, model_class):
                pass  # Already the right type
            else:
                return False
            return True
        except Exception:
            return False


class WorkflowStateAdapter(StateAdapter):
    """
    Specialized adapter for workflow states with common workflow fields.
    
    This adapter provides additional utilities for handling workflow-specific state
    conversions and validations.
    """
    
    def __init__(self, model_class: Optional[Type[BaseModel]] = None):
        super().__init__(model_class)
        self.workflow_fields = {
            'ticket_id', 'current_step', 'ticket', 'intent_analysis',
            'generated_solution', 'quality_check', 'response_sent',
            'follow_up_scheduled', 'error_message', 'workflow_started_at',
            'last_updated_at'
        }
    
    def extract_workflow_fields(self, state: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract only workflow-related fields from a state.
        
        Args:
            state: State object or dictionary
            
        Returns:
            Dictionary containing only workflow fields
        """
        state_dict = self.to_dict(state)
        return {k: v for k, v in state_dict.items() if k in self.workflow_fields}
    
    def merge_workflow_state(self, base_state: Union[BaseModel, Dict[str, Any]], 
                           updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge workflow updates into a base state.
        
        Args:
            base_state: Base state to merge into
            updates: Updates to apply
            
        Returns:
            Merged state as dictionary
        """
        base_dict = self.to_dict(base_state)
        # Only allow updates to workflow fields
        filtered_updates = {k: v for k, v in updates.items() if k in self.workflow_fields}
        return {**base_dict, **filtered_updates}


def create_state_adapter(model_class: Optional[Type[BaseModel]] = None) -> StateAdapter:
    """
    Factory function to create a StateAdapter.
    
    Args:
        model_class: Optional Pydantic model class
        
    Returns:
        StateAdapter instance
    """
    return StateAdapter(model_class)


def create_workflow_adapter(model_class: Optional[Type[BaseModel]] = None) -> WorkflowStateAdapter:
    """
    Factory function to create a WorkflowStateAdapter.
    
    Args:
        model_class: Optional Pydantic model class
        
    Returns:
        WorkflowStateAdapter instance
    """
    return WorkflowStateAdapter(model_class)


# Utility functions for common conversions
def pydantic_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Convert a Pydantic model to dictionary.
    
    Args:
        model: Pydantic model instance
        
    Returns:
        Dictionary representation
    """
    adapter = StateAdapter()
    return adapter.to_dict(model)


def dict_to_pydantic(state_dict: Dict[str, Any], model_class: Type[T]) -> T:
    """
    Convert a dictionary to Pydantic model.
    
    Args:
        state_dict: Dictionary representation
        model_class: Pydantic model class
        
    Returns:
        Pydantic model instance
    """
    adapter = StateAdapter(model_class)
    result = adapter.from_dict(state_dict, model_class)
    if isinstance(result, dict):
        raise ValueError(f"Could not convert dictionary to {model_class.__name__}")
    return result


def validate_workflow_state(state: Union[BaseModel, Dict[str, Any]], 
                          expected_fields: Optional[set] = None) -> bool:
    """
    Validate that a state has the expected workflow fields.
    
    Args:
        state: State to validate
        expected_fields: Set of expected field names (if None, uses default workflow fields)
        
    Returns:
        True if valid, False otherwise
    """
    if expected_fields is None:
        expected_fields = {
            'ticket_id', 'current_step', 'ticket', 'intent_analysis',
            'generated_solution', 'quality_check', 'response_sent',
            'follow_up_scheduled', 'error_message', 'workflow_started_at',
            'last_updated_at'
        }
    
    state_dict = StateAdapter().to_dict(state)
    return all(field in state_dict for field in expected_fields)
