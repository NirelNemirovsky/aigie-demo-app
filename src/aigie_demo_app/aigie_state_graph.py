from langgraph import StateGraph

class AigieStateGraph(StateGraph):
    """
    A wrapper around the StateGraph class to extend or modify its functionality.
    """

    def add_node(self, node_id, node_data=None):
        """
        Overrides the add_node method to include custom behavior.
        """
        # Custom logic before adding the node
        print(f"Adding node with ID: {node_id}")

        # Call the original add_node method
        super().add_node(node_id, node_data)

        # Custom logic after adding the node
        print(f"Node {node_id} added successfully with data: {node_data}")