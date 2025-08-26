#!/usr/bin/env python3
"""
Basic usage example for the aigie package
"""

from aigie import PolicyNode, AigieStateGraph
from pydantic import BaseModel

def main():
    """Demonstrate basic aigie functionality"""
    
    print("🚀 Aigie Package Demo")
    print("=" * 50)
    
    # Define a simple Pydantic model for state
    class SimpleState(BaseModel):
        input: str = ""
        processed: bool = False
        result: str = ""
        validated: bool = False
        last_node: str = ""
        error: dict = None
    
    # Example 1: Basic PolicyNode usage
    print("\n1. Creating a PolicyNode...")
    
    def simple_ai_function(state: SimpleState) -> SimpleState:
        """A simple AI function that processes state"""
        # Simulate some AI processing
        if not state.input:
            raise ValueError("Missing input in state")
        
        state.processed = True
        state.result = f"Processed: {state.input}"
        return state
    
    # Create a PolicyNode with retry logic
    node = PolicyNode(
        inner=simple_ai_function,
        name="ai_processor",
        max_attempts=3,
        fallback=lambda state: {"error": "Fallback executed", **state}
    )
    
    print("✅ PolicyNode created successfully")
    
    # Test the node
    test_state = SimpleState(input="Hello, AI World!")
    result = node.invoke(test_state)
    print(f"✅ Node execution result: {result}")
    
    # Example 2: AigieStateGraph usage
    print("\n2. Creating an AigieStateGraph...")
    
    graph = AigieStateGraph(state_schema=SimpleState)
    print("✅ StateGraph created successfully")
    
    # Add nodes to the graph
    graph.add_node("ai_processor", simple_ai_function)
    
    def validator_node(state: SimpleState) -> SimpleState:
        state.validated = True
        return state
    
    graph.add_node("validator", validator_node)
    
    print("✅ Nodes added to graph successfully")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe aigie package is working correctly and ready for use!")

if __name__ == "__main__":
    main()
