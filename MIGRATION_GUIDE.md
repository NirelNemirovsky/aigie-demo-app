# Migration Guide: Unified Pydantic Standard

## 🎯 Overview

Aigie 0.3.0 introduces a **unified Pydantic standard** for state management, eliminating the need for multiple compatibility layers and simplifying the codebase.

## 🚀 What Changed

### Before (Multiple Standards)
```python
# ❌ Old approach with multiple formats
from aigie import AigieStateGraph, PydanticCompatibleAigieGraph, StateAdapter

# Dictionary-based approach
graph = AigieStateGraph()  # Uses TypedDict internally

# Pydantic compatibility layer
pydantic_graph = PydanticCompatibleAigieGraph(WorkflowState)
adapter = StateAdapter(WorkflowState)

# Manual conversions needed
state_dict = adapter.to_dict(pydantic_state)
final_state = adapter.from_dict(result_dict)
```

### After (Unified Standard)
```python
# ✅ New unified approach
from aigie import AigieStateGraph

# Single standard: Pydantic models
graph = AigieStateGraph(state_schema=WorkflowState)

# No conversions needed - works directly with Pydantic models
final_state = graph.invoke(initial_state)  # Returns Pydantic model
```

## 📋 Migration Steps

### Step 1: Update Imports

**Remove old compatibility imports:**
```python
# ❌ Remove these
from aigie import PydanticCompatibleAigieGraph, WorkflowCompatibleAigieGraph
from aigie import StateAdapter, WorkflowStateAdapter
from aigie import pydantic_to_dict, dict_to_pydantic, validate_workflow_state
```

**Use the unified import:**
```python
# ✅ Use this
from aigie import AigieStateGraph
```

### Step 2: Update Graph Creation

**Before:**
```python
# ❌ Old approach
workflow_graph = create_workflow_compatible_graph(
    WorkflowState,
    enable_gemini_remediation=True
)
```

**After:**
```python
# ✅ New approach
workflow_graph = AigieStateGraph(
    state_schema=WorkflowState,  # Pass Pydantic model class directly
    enable_gemini_remediation=True
)
```

### Step 3: Update Node Functions

**Before:**
```python
# ❌ Old approach with manual conversion
def ticket_reception_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Convert dictionary to Pydantic model
    state = WorkflowState(**state_dict)
    
    # Process the state
    state.current_step = WorkflowStep.INTENT_ANALYSIS
    
    # Convert back to dictionary
    return state.dict()
```

**After:**
```python
# ✅ New approach - direct Pydantic models
def ticket_reception_node(state: WorkflowState) -> WorkflowState:
    # Work directly with Pydantic model
    state.current_step = WorkflowStep.INTENT_ANALYSIS
    return state
```

### Step 4: Update Graph Execution

**Before:**
```python
# ❌ Old approach with manual conversion
initial_state_dict = pydantic_to_dict(initial_state)
result_dict = workflow_graph.invoke(initial_state_dict)
final_state = dict_to_pydantic(result_dict, WorkflowState)
```

**After:**
```python
# ✅ New approach - direct Pydantic models
final_state = workflow_graph.invoke(initial_state)  # Returns Pydantic model
```

## 🔧 Complete Migration Example

### Before Migration
```python
from aigie import create_workflow_compatible_graph, pydantic_to_dict, dict_to_pydantic
from pydantic import BaseModel

class WorkflowState(BaseModel):
    ticket_id: str
    current_step: str
    ticket: dict

# Create graph with compatibility layer
workflow_graph = create_workflow_compatible_graph(
    WorkflowState,
    enable_gemini_remediation=True
)

# Add nodes with manual conversion
def process_node(state_dict: dict) -> dict:
    state = WorkflowState(**state_dict)
    state.current_step = "processed"
    return state.dict()

workflow_graph.add_workflow_node("process", process_node)

# Execute with manual conversion
initial_state = WorkflowState(ticket_id="123", current_step="start", ticket={})
initial_dict = pydantic_to_dict(initial_state)
result_dict = workflow_graph.invoke(initial_dict)
final_state = dict_to_pydantic(result_dict, WorkflowState)
```

### After Migration
```python
from aigie import AigieStateGraph
from pydantic import BaseModel

class WorkflowState(BaseModel):
    ticket_id: str
    current_step: str
    ticket: dict

# Create graph with unified approach
workflow_graph = AigieStateGraph(
    state_schema=WorkflowState,
    enable_gemini_remediation=True
)

# Add nodes with direct Pydantic models
def process_node(state: WorkflowState) -> WorkflowState:
    state.current_step = "processed"
    return state

workflow_graph.add_node("process", process_node)

# Execute with direct Pydantic models
initial_state = WorkflowState(ticket_id="123", current_step="start", ticket={})
final_state = workflow_graph.invoke(initial_state)  # Returns WorkflowState
```

## 🎉 Benefits of Migration

### Code Simplification
- **50% less code**: No more compatibility layers
- **Cleaner functions**: Direct Pydantic model usage
- **Better IDE support**: Full autocomplete and type checking

### Performance Improvement
- **Faster execution**: No conversion overhead
- **Memory efficient**: No duplicate state representations
- **Reduced complexity**: Single code path

### Developer Experience
- **Type safety**: Compile-time error detection
- **Self-documenting**: Pydantic field descriptions
- **Validation**: Automatic data validation

## 🔍 Backward Compatibility

### What's Still Supported
- ✅ Dictionary inputs (automatically converted to Pydantic)
- ✅ All existing node functions (with automatic conversion)
- ✅ All configuration options
- ✅ All analytics and monitoring features

### What's Deprecated
- ❌ `PydanticCompatibleAigieGraph` (use `AigieStateGraph` with `state_schema`)
- ❌ `WorkflowCompatibleAigieGraph` (use `AigieStateGraph` with `state_schema`)
- ❌ `StateAdapter` and `WorkflowStateAdapter` (no longer needed)
- ❌ Manual conversion utilities (automatic conversion provided)

## 🚨 Breaking Changes

### Required Changes
1. **Graph Creation**: Must specify `state_schema` parameter
2. **Node Functions**: Should return Pydantic models (automatic conversion still works)
3. **Imports**: Remove compatibility layer imports

### Optional Improvements
1. **Node Functions**: Update to work directly with Pydantic models
2. **Type Hints**: Add proper type annotations
3. **Validation**: Use Pydantic field validators

## 📞 Support

If you encounter any issues during migration:

1. **Check the examples**: See `examples/unified_pydantic_example.py`
2. **Review the API**: All methods support both Pydantic models and dictionaries
3. **Enable debug mode**: Set `log_remediation=True` for detailed logs
4. **Contact support**: Open an issue with your specific use case

## 🎯 Migration Checklist

- [ ] Update imports to use `AigieStateGraph` only
- [ ] Add `state_schema` parameter to graph creation
- [ ] Update node functions to work with Pydantic models (optional)
- [ ] Remove manual conversion calls
- [ ] Test with your existing workflow
- [ ] Update documentation and examples
- [ ] Deploy and monitor

**Migration time estimate**: 30 minutes to 2 hours depending on complexity
