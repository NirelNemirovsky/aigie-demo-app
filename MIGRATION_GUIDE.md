# Aigie Migration Guide

This guide helps you migrate from previous versions of Aigie to the latest version with enhanced error handling and native Pydantic model support.

## 🎉 **GOOD NEWS: Aigie Now Uses Pydantic Models Natively!**

### **The Solution**

Aigie now properly uses Pydantic models with LangGraph, which supports them natively. No conversion is needed!

```python
from aigie import AigieStateGraph
from pydantic import BaseModel

# Define your Pydantic models
class WorkflowState(BaseModel):
    ticket_id: str
    current_step: str
    ticket: dict
    # ... other fields

# Create the graph with Pydantic schema
workflow_graph = AigieStateGraph(
    state_schema=WorkflowState,  # Pass your Pydantic model class!
    enable_gemini_remediation=True
)

# Add nodes that work with Pydantic models
def my_node(state: WorkflowState) -> WorkflowState:
    # Work with Pydantic models directly
    state.current_step = "next_step"
    return state

workflow_graph.add_node("my_node", my_node)

# Execute with Pydantic model
initial_state = WorkflowState(ticket_id="123", current_step="start", ticket={})
final_state = workflow_graph.invoke(initial_state)  # Returns Pydantic model
```

## 🔄 **General Migration Steps**

### **From Aigie 0.2.x to 0.3.x**

1. **Update your imports:**
   ```python
   # Old
   from aigie import StateGraph
   
   # New
   from aigie import AigieStateGraph
   ```

2. **Update graph creation:**
   ```python
   # Old
   graph = StateGraph()
   
   # New
   graph = AigieStateGraph(
       state_schema=YourPydanticModel,  # New parameter
       enable_gemini_remediation=True,  # New feature
       auto_apply_fixes=False,          # New feature
       log_remediation=True             # New feature
   )
   ```

3. **Your node functions can work with Pydantic models directly:**
   ```python
   # Your functions can work with Pydantic models directly
   def my_node(state: WorkflowState) -> WorkflowState:
       # Work with Pydantic models directly
       state.current_step = "next_step"
       return state
   ```

### **From LangGraph to Aigie**

1. **Replace StateGraph with AigieStateGraph:**
   ```python
   # Old
   from langgraph.graph import StateGraph
   graph = StateGraph(YourPydanticModel)
   
   # New
   from aigie import AigieStateGraph
   graph = AigieStateGraph(
       state_schema=YourPydanticModel,
       enable_gemini_remediation=True
   )
   ```

2. **Add enhanced error handling:**
   ```python
   # New: Enhanced error handling
   graph = AigieStateGraph(
       state_schema=YourPydanticModel,
       enable_gemini_remediation=True,
       auto_apply_fixes=False,
       log_remediation=True
   )
   ```

## 🧪 **Testing Your Migration**

### **Test Graph Execution**

```python
# Test with Pydantic model
initial_state = WorkflowState(...)
final_state = workflow_graph.invoke(initial_state)

# Verify the result is still a Pydantic model
assert isinstance(final_state, WorkflowState)
print("✅ Migration successful!")
```

## 🚨 **Common Issues and Solutions**

### **Issue 1: "Invalid state update" Error**

**Problem:** This usually means there's a mismatch in the state schema.

**Solution:** Make sure your Pydantic model matches the expected fields.

### **Issue 2: Enum values not being recognized**

**Problem:** Aigie sees enum objects instead of string values.

**Solution:** Use string enums or convert enum values to strings in your model.

### **Issue 3: Datetime serialization errors**

**Problem:** Aigie can't serialize datetime objects.

**Solution:** Use Pydantic's datetime field types or convert to ISO strings.

## 📚 **Complete Example**

See `examples/enhanced_usage.py` for a complete working example that demonstrates:

- Proper Pydantic model usage
- Native LangGraph integration
- Error handling with Gemini AI
- Analytics and monitoring

## 🔧 **Configuration Options**

### **AigieStateGraph Parameters**

```python
workflow_graph = AigieStateGraph(
    state_schema=YourPydanticModel,     # Pydantic model class
    enable_gemini_remediation=True,     # Enable AI-powered error remediation
    gemini_project_id="your-project",   # GCP project ID (optional)
    auto_apply_fixes=False,             # Auto-apply AI suggestions
    log_remediation=True                # Log remediation analysis
)
```

### **Node Configuration**

```python
workflow_graph.add_node(
    "my_node", 
    my_function,
    enable_gemini_remediation=True,  # Override graph setting
    max_attempts=3,                  # Retry attempts
    auto_apply_fixes=True,           # Auto-apply fixes for this node
    on_error=my_error_handler        # Custom error handler
)
```

## 📊 **Monitoring and Analytics**

### **Get Graph Analytics**

```python
analytics = workflow_graph.get_graph_analytics()
print(f"Total nodes: {analytics['graph_summary']['total_nodes']}")
print(f"Total errors: {analytics['graph_summary']['total_errors']}")
print(f"Total remediations: {analytics['graph_summary']['total_remediations']}")
```

### **Get Node Analytics**

```python
node_analytics = workflow_graph.get_node_analytics("my_node")
print(f"Node errors: {node_analytics['total_errors']}")
print(f"Error categories: {node_analytics['error_categories']}")
```

## 🎯 **Best Practices**

1. **Always use Pydantic models for type safety**
2. **Use the enhanced `AigieStateGraph` with `state_schema`**
3. **Enable Gemini remediation for production environments**
4. **Monitor error analytics regularly**
5. **Use the provided examples as templates**

## 🆘 **Need Help?**

If you encounter issues during migration:

1. Check the examples in the `examples/` directory
2. Verify your Pydantic model structure
3. Review the error analytics for insights
4. Enable detailed logging for debugging

The enhanced Aigie package provides comprehensive error handling and native Pydantic model support, making it easy to work with structured data while maintaining compatibility with LangGraph's architecture.
