# Aigie - AI World Fixer

A Python package that helps fix your AI world with intelligent policy nodes and state graphs.

## Installation

### From GitHub
```bash
pip install git+https://github.com/yourusername/aigie-demo-app.git
```

### From source
```bash
git clone https://github.com/yourusername/aigie-demo-app.git
cd aigie-demo-app
pip install .
```

## Usage

```python
from aigie import PolicyNode, AigieStateGraph

# Create a policy node with retry logic and error handling
def my_function(state):
    # Your AI logic here
    return {"result": "success", **state}

node = PolicyNode(
    inner=my_function, 
    name="my_node",
    max_attempts=3,
    fallback=lambda state: {"result": "fallback", **state}
)

# Create a state graph
graph = AigieStateGraph()

# Add nodes to the graph (automatically wrapped with PolicyNode)
graph.add_node("my_node", my_function)

# The graph will automatically wrap your functions with PolicyNode
# for retry logic, error handling, and GCP logging
```

## Features

- **PolicyNode**: Wraps your functions with retry logic, error handling, and GCP logging
- **AigieStateGraph**: Extends LangGraph's StateGraph with automatic PolicyNode wrapping
- **GCP Integration**: Built-in Google Cloud Logging for production monitoring
- **Error Resilience**: Automatic retry with configurable fallback strategies

## Requirements

- Python 3.8+
- tenacity
- langchain-core
- langgraph
- google-cloud-logging

## Development

To set up the development environment:

```bash
pip install -e .
pytest
```

## License

MIT License