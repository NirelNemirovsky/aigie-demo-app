from langgraph.graph import StateGraph
from .aigie_node import PolicyNode
from typing import Any, Dict, Optional, TypedDict, List

class AigieStateGraph(StateGraph):
    """
    A wrapper around the StateGraph class to extend or modify its functionality.
    """

    def __init__(self, state_schema: Optional[type] = None):
        """
        Initialize the AigieStateGraph with an optional state schema.
        
        Args:
            state_schema: Optional schema type for the state. If None, uses a default schema.
        """
        if state_schema is None:
            # Use a simple TypedDict schema that's compatible with langgraph
            class DefaultState(TypedDict):
                messages: List[str]
                state: Dict[str, Any]
            
            state_schema = DefaultState
        
        super().__init__(state_schema)

    def add_node(self, node_id: str, node_data=None):
        """
        Overrides the add_node method to wrap the node with PolicyNode before adding it.
        
        Args:
            node_id: The identifier for the node
            node_data: The function or runnable to wrap with PolicyNode
        """
        if node_data is not None:
            # Wrap the node with PolicyNode
            wrapped_node = PolicyNode(inner=node_data, name=node_id)
            # Call the original add_node method with the wrapped node
            super().add_node(node_id, wrapped_node)
            print(f"Node {node_id} wrapped with PolicyNode and added successfully.")
        else:
            # If no node_data, just add the node_id (for conditional nodes)
            super().add_node(node_id, node_data)
            print(f"Node {node_id} added successfully.")