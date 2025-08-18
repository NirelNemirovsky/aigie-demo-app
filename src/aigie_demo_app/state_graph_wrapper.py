from langgraph import StateGraph
from .aigie_node import PolicyNode

class StateGraphWrapper(StateGraph):
    """
    A wrapper around the StateGraph class to extend or modify its functionality.
    """

    def add_node(self, node_id, node_data=None):
        """
        Overrides the add_node method to wrap the node with PolicyNode before adding it.
        """
        # Wrap the node with PolicyNode
        wrapped_node = PolicyNode(inner=node_data, name=node_id)

        # Call the original add_node method with the wrapped node
        super().add_node(node_id, wrapped_node)

        # Optional: Log or perform additional actions
        print(f"Node {node_id} wrapped with PolicyNode and added successfully.")
