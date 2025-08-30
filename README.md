# Aigie - AI World Fixer with Trail Taxonomy & Advanced Proactive Remediation

A Python package that provides intelligent error handling and **advanced proactive remediation** for AI systems using Trail Taxonomy classification, Google's Gemini 2.5 Flash AI model, and CodeAct/ReAct agent-based automatic error fixing capabilities. **Aigie natively supports Pydantic models as the single standard for state management**, eliminating the need for multiple compatibility layers.

## 🚀 Features

- **Native Pydantic Support**: Direct integration with LangGraph's native Pydantic model support
- **Trail Taxonomy Error Classification**: Comprehensive error categorization based on the Trail Taxonomy paper
- **Latest Gemini AI Remediation**: AI-powered error analysis using Google's Gemini 2.5 Flash with Vertex AI SDK 1.50.0+
- **Real-time Error Handling**: Intelligent error detection and resolution in real-time
- **Advanced Proactive Error Remediation**: CodeAct/ReAct agent-based automatic error fixing with AI-powered dynamic code generation, intelligent analysis, and safe execution
- **Learning System**: Memory-based learning from successful fixes with confidence scoring
- **Automatic Error Fixing**: Optional automatic application of AI-suggested fixes
- **Comprehensive Analytics**: Detailed error analytics and monitoring
- **Seamless Integration**: Drop-in replacement for LangGraph's StateGraph
- **GCP Integration**: Built-in Google Cloud Logging and Vertex AI integration
- **Latest API Standards**: Updated to use the most current Gemini API patterns and response handling

## 🏗️ Architecture

### Latest Gemini AI Integration

Aigie now uses the most current Gemini API standards with Google Gen AI SDK 1.32.0+:

```python
import google.genai

# Initialize with latest standards
from aigie.gemini_remediator import GeminiRemediator

# For Gemini Developer API (recommended)
remediator = GeminiRemediator(api_key="your-api-key")

# For Vertex AI (optional)
remediator = GeminiRemediator(project_id="your-project-id", use_vertex_ai=True)

result = remediator.analyze_and_remediate(error_analysis, context)
```

**Key Improvements:**
- **SDK**: Google Gen AI SDK 1.32.0+ (latest recommended)
- **Dual Support**: Both Gemini Developer API and Vertex AI services
- **API Patterns**: Modern client-based approach with latest standards
- **Backward Compatibility**: All existing code continues to work

### Simplified Pydantic Integration

Aigie now works directly with LangGraph's native Pydantic support, eliminating the need for conversion layers:

```python
from aigie import AigieStateGraph
from pydantic import BaseModel

class WorkflowState(BaseModel):
    ticket_id: str
    current_step: str
    # ... other fields

# Create graph with Pydantic schema
graph = AigieStateGraph(state_schema=WorkflowState)

# Add nodes that work with Pydantic models directly
def my_node(state: WorkflowState) -> WorkflowState:
    state.current_step = "next_step"
    return state

graph.add_node("my_node", my_node)
```

### Advanced Proactive Remediation Engine

Aigie's advanced proactive remediation system uses a **CodeAct/ReAct agent pattern** with AI-powered dynamic code generation:

```
┌─────────────────────────────────────────────────────────────┐
│              Advanced Proactive Remediation Engine          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         AdvancedProactiveRemediationEngine         │   │
│  │           (CodeAct/ReAct Agent)                    │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │           CodeActAgent                      │   │   │
│  │  │  • Multi-step reasoning                     │   │   │
│  │  │  • Error analysis                           │   │   │
│  │  │  • Strategy generation                      │   │   │
│  │  │  • Code generation                          │   │   │
│  │  │  • Validation & execution                   │   │   │
│  │  │  • Learning & memory                        │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                              │                      │   │
│  │                              ▼                      │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │         AICodeGenerator                     │   │   │
│  │  │  (AI-Powered Code Generation)               │   │   │
│  │  │                                             │   │   │
│  │  │  • Gemini/Claude integration                │   │   │
│  │  │  • Dynamic code generation                  │   │   │
│  │  │  • Safety validation                        │   │   │
│  │  │  • Fallback mechanisms                      │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Agent Reasoning Process

The CodeAct/ReAct agent follows a sophisticated multi-step reasoning process:

1. **🔍 Error Analysis**: Deep analysis of error patterns, context, and root causes
2. **🎯 Strategy Generation**: AI-powered strategy development for fix approach
3. **💻 Code Generation**: Dynamic code generation using AI models
4. **✅ Validation**: Safety and logic validation of generated code
5. **🚀 Execution**: Safe execution of validated fixes
6. **🧠 Learning**: Memory storage and pattern recognition for future improvements

### Supported Error Types

The system can handle various error patterns with intelligent fixes:

- **Missing Fields**: Automatic addition with appropriate defaults
- **Type Conversion**: Smart type casting (string to int, etc.)
- **Validation Errors**: Default value provision and format fixes
- **API Errors**: Retry logic with exponential backoff
- **Timeout Errors**: Timeout multiplier adjustments
- **Generic Errors**: Fallback fixes with logging

## 📦 Installation

### Prerequisites

- Python 3.8+
- Google Cloud Platform account (for Gemini AI features)
- gcloud CLI (for authentication)

### Install Aigie

#### From GitHub
```bash
pip install git+https://github.com/yourusername/aigie-demo-app.git
```

#### From source
```bash
git clone https://github.com/yourusername/aigie-demo-app.git
cd aigie-demo-app
pip install -e .

# Optional: Run the interactive setup script
python3 setup_gemini.py
```

### Install Dependencies

The package will automatically install required dependencies:
- `tenacity` - Retry logic
- `langchain-core` - Core LangChain functionality
- `langgraph` - Graph-based AI workflows
- `google-cloud-logging` - GCP logging integration
- `google-cloud-aiplatform` - Vertex AI for Gemini

## 🔧 Setup

### 1. Gemini AI Configuration (Seamless Setup)

Aigie automatically detects your Gemini configuration for seamless usage:

#### **Option A: Environment Variable (Recommended)**
```bash
# Set your Gemini API key
export GOOGLE_API_KEY="your-api-key-from-google-ai-studio"

# That's it! Aigie will work automatically
```

#### **Option B: Configuration File**
```bash
# Create ~/.aigie/config.json
mkdir -p ~/.aigie
cat > ~/.aigie/config.json << EOF
{
    "api_key": "your-api-key-here",
    "use_vertex_ai": false
}
EOF
```

#### **Option C: GCP Vertex AI (Advanced)**
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Set your GCP project ID
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Authenticate with gcloud
gcloud auth application-default login

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable logging.googleapis.com
```

### 2. Environment Variables

```bash
# For Gemini Developer API (recommended)
export GOOGLE_API_KEY="your-api-key-from-google-ai-studio"

# For Vertex AI (optional)
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 3. Verify Installation

```python
from aigie import AigieStateGraph, EnhancedTrailTaxonomyClassifier

# Test basic import
print("✅ Aigie imported successfully!")

# Test Trail Taxonomy
classifier = EnhancedTrailTaxonomyClassifier()
print("✅ Trail Taxonomy ready!")

# Test graph creation with Gemini
graph = AigieStateGraph(enable_gemini_remediation=True)
print("✅ AigieStateGraph ready with Gemini AI!")
```

## 🎯 Quick Start

### Seamless Gemini Setup

Aigie automatically detects your Gemini configuration for zero-config usage:

```python
from aigie import AigieStateGraph

# Just enable Gemini - Aigie handles the rest!
graph = AigieStateGraph(
    enable_gemini_remediation=True,  # ✅ Works automatically!
    auto_apply_fixes=False,
    log_remediation=True
)

# Aigie will automatically:
# 1. Check for GOOGLE_API_KEY environment variable
# 2. Look for ~/.aigie/config.json configuration file
# 3. Detect GCP project from GOOGLE_CLOUD_PROJECT
# 4. Fall back to Trail Taxonomy if no Gemini config found
```

**Setup Options (choose one):**
- **Environment Variable**: `export GOOGLE_API_KEY="your-key"`
- **Config File**: Create `~/.aigie/config.json`
- **GCP Project**: `export GOOGLE_CLOUD_PROJECT="your-project"`

### Unified Pydantic Approach

Aigie uses **Pydantic models as the single standard** for state management. This eliminates the confusion of multiple state formats and provides:

- **Type Safety**: Full type checking and validation
- **IDE Support**: Autocomplete and error detection
- **Documentation**: Self-documenting code with field descriptions
- **Serialization**: Built-in JSON serialization/deserialization
- **Validation**: Automatic data validation and error messages

### Basic Usage (Seamless Integration)

```python
from aigie import AigieStateGraph
from pydantic import BaseModel

class WorkflowState(BaseModel):
    input: str
    result: str = ""
    error: dict = None

# Create an enhanced state graph with AI-powered error handling
graph = AigieStateGraph(
    state_schema=WorkflowState,
    enable_gemini_remediation=True,  # Enable Gemini AI remediation
    auto_apply_fixes=False,          # Don't auto-apply fixes
    log_remediation=True             # Log all remediation analysis
)

# Define your AI functions
def my_ai_function(state: WorkflowState) -> WorkflowState:
    # Your AI logic here
    if not state.input:
        raise ValueError("Missing input in state")
    state.result = "success"
    return state

# Add nodes (automatically wrapped with enhanced error handling)
graph.add_node("my_node", my_ai_function)

# The graph now has intelligent error handling with Trail Taxonomy and Gemini AI
```

### Advanced Usage with Custom Configuration

```python
from aigie import AigieStateGraph, PolicyNode

# Create graph with custom configuration
graph = AigieStateGraph(
    enable_gemini_remediation=True,
    gemini_project_id="my-gcp-project",
    auto_apply_fixes=True,  # Automatically apply AI-suggested fixes
    log_remediation=True
)

# Add node with custom error handling configuration
graph.add_node(
    "critical_node", 
    my_function,
    max_attempts=5,
    enable_gemini_remediation=True,
    auto_apply_fixes=True
)
```

## 🔍 Error Classification (Trail Taxonomy)

Aigie classifies errors into comprehensive categories:

- **Input Errors**: Validation, format, missing data
- **Processing Errors**: Algorithm failures, resource constraints  
- **Output Errors**: Format issues, quality problems
- **System Errors**: Infrastructure, network, configuration
- **External Errors**: API failures, third-party service issues

```python
from aigie import EnhancedTrailTaxonomyClassifier

classifier = EnhancedTrailTaxonomyClassifier()

# Classify an error
error = ValueError("Missing required field 'user_id'")
analysis = classifier.classify_error(error)

print(f"Category: {analysis.category.value}")
print(f"Severity: {analysis.severity.value}")
print(f"Confidence: {analysis.confidence}")
print(f"Keywords: {analysis.keywords}")
print(f"Remediation Hints: {analysis.remediation_hints}")
```

## 🤖 Advanced Proactive Remediation

### Direct Usage of Advanced Proactive Remediation Engine

```python
from aigie import AdvancedProactiveRemediationEngine
from aigie import ErrorAnalysis, ErrorCategory, ErrorSeverity

# Initialize the advanced remediation engine
engine = AdvancedProactiveRemediationEngine(
    max_iterations=3,           # Maximum reasoning iterations
    confidence_threshold=0.6,   # Minimum confidence for fixes
    ai_model="gemini-2.5-flash" # AI model for code generation
)

# Define an error to fix
error = ErrorAnalysis(
    category=ErrorCategory.INPUT_ERROR,
    severity=ErrorSeverity.HIGH,
    confidence=0.9,
    keywords=["missing", "required", "field"],
    description="Missing required field 'customer_id'",
    remediation_hints=["Add customer_id field"],
    context={}
)

# Current state that needs fixing
state = {"email": "test@example.com"}

# Apply proactive remediation
result = engine.apply_proactive_remediation(error, state)

print(f"Success: {result.success}")
print(f"Fixed State: {result.fixed_state}")
print(f"Generated Code: {result.generated_code}")
print(f"Confidence: {result.confidence_score}")
print(f"Strategy: {result.fix_strategy}")
print(f"Execution Time: {result.execution_time:.3f}s")

# Get learning insights
insights = engine.get_learning_insights()
print(f"Learning Patterns: {insights['total_patterns']}")
print(f"Success Rates: {insights['success_rates']}")
```

### CodeAct/ReAct Agent Reasoning Process

```python
from aigie import CodeActAgent

# Initialize the agent
agent = CodeActAgent(
    max_iterations=5,
    confidence_threshold=0.7
)

# Let the agent think and act
result = agent.think_and_act(error, state)

# Examine the reasoning process
for i, thought in enumerate(result.agent_thoughts, 1):
    print(f"Thought {i}: {thought.action.value}")
    print(f"  Reasoning: {thought.reasoning}")
    print(f"  Confidence: {thought.confidence}")
    if thought.output_data:
        print(f"  Output: {thought.output_data}")
    print()
```

### AI-Powered Code Generation

```python
from aigie import AICodeGenerator, AICodeGenerationRequest

# Initialize AI code generator
generator = AICodeGenerator(
    model_name="gemini-2.5-flash",
    project_id="your-gcp-project-id"
)

# Create a code generation request
request = AICodeGenerationRequest(
    error_analysis={
        "description": "Field 'age' must be an integer",
        "category": "input_error",
        "severity": "medium"
    },
    state={"age": "25", "name": "John"},
    code_context="",
    strategy="Convert field to appropriate type",
    constraints={
        "safe_only": True,
        "state_only": True,
        "logging_required": True
    }
)

# Generate fix code
response = generator.generate_fix_code(request)

print(f"Generated Code: {response.generated_code}")
print(f"Confidence: {response.confidence}")
print(f"Reasoning: {response.reasoning}")
print(f"Safety Score: {response.safety_score}")
```

### Error Type Examples

#### Missing Field Error
```python
# Error: Missing required field
error = ErrorAnalysis(
    category=ErrorCategory.INPUT_ERROR,
    severity=ErrorSeverity.HIGH,
    description="Missing required field 'customer_id'",
    keywords=["missing", "required", "field"],
    context={}
)

state = {"email": "test@example.com"}
result = engine.apply_proactive_remediation(error, state)
# Result: Adds customer_id with UUID
```

#### Type Conversion Error
```python
# Error: Type mismatch
error = ErrorAnalysis(
    category=ErrorCategory.INPUT_ERROR,
    severity=ErrorSeverity.MEDIUM,
    description="Field 'age' must be an integer",
    keywords=["type", "int", "conversion"],
    context={}
)

state = {"age": "25", "name": "John"}
result = engine.apply_proactive_remediation(error, state)
# Result: Converts age from string to integer
```

#### Validation Error
```python
# Error: Validation failure
error = ErrorAnalysis(
    category=ErrorCategory.INPUT_ERROR,
    severity=ErrorSeverity.MEDIUM,
    description="Invalid email format",
    keywords=["validation", "invalid", "format"],
    context={}
)

state = {"email": "invalid-email", "name": "John"}
result = engine.apply_proactive_remediation(error, state)
# Result: Applies validation fixes with defaults
```

### Integration with LangGraph Workflows

```python
from aigie import AigieStateGraph, PolicyNode

# Create graph with advanced proactive remediation
graph = AigieStateGraph(
    enable_proactive_remediation=True,  # Enable advanced remediation
    proactive_fix_types=[               # Specify which errors to fix
        'missing_field', 
        'type_error', 
        'validation_error'
    ],
    max_proactive_attempts=3,           # Maximum fix attempts
    confidence_threshold=0.6            # Minimum confidence
)

# Define a function that might fail
def process_user_data(state):
    # This might fail if customer_id is missing
    if "customer_id" not in state:
        raise ValueError("Missing required field 'customer_id'")
    
    # This might fail if age is a string
    if isinstance(state.get("age"), str):
        raise TypeError("Field 'age' must be an integer")
    
    return {"processed": True, **state}

# Add node with enhanced error handling
graph.add_node(
    "process_user", 
    process_user_data,
    enable_proactive_remediation=True,  # Enable for this node
    max_attempts=3
)

# The system will automatically:
# 1. Detect the error
# 2. Analyze it using Trail Taxonomy
# 3. Apply proactive remediation if possible
# 4. Retry the operation
# 5. Learn from the experience
```

### Custom Error Handling with Proactive Remediation

```python
from aigie import PolicyNode

# Create a custom node with specific remediation settings
class CustomNode(PolicyNode):
    def __init__(self, inner_function):
        super().__init__(
            inner=inner_function,
            name="custom_node",
            enable_proactive_remediation=True,
            proactive_fix_types=['missing_field', 'type_error'],
            max_proactive_attempts=2,
            confidence_threshold=0.8  # Higher confidence requirement
        )
    
    def custom_error_handler(self, error, state):
        # Custom error handling logic
        if "custom_error" in str(error):
            # Handle custom errors
            return {"custom_fix": True, **state}
        
        # Fall back to proactive remediation
        return super()._handle_error_with_remediation(error, 1, state, None)

# Usage
def my_function(state):
    # Your logic here
    pass

custom_node = CustomNode(my_function)
```

### Monitoring and Analytics

```python
# Get comprehensive analytics
node = graph.get_node("process_user")
analytics = node.get_error_analytics()

print("Error Analytics:")
print(f"Total Errors: {analytics['total_errors']}")
print(f"Proactive Fixes Applied: {analytics['proactive_fixes_applied']}")
print(f"Proactive Fixes Successful: {analytics['proactive_fixes_successful']}")
print(f"Success Rate: {analytics['success_rate']:.2%}")

# Get learning insights from the remediation engine
engine = AdvancedProactiveRemediationEngine()
insights = engine.get_learning_insights()

print("\nLearning Insights:")
print(f"Patterns Learned: {insights['total_patterns']}")
for pattern, success_rate in insights['success_rates'].items():
    print(f"  {pattern}: {success_rate:.2%} success rate")
```

## 🎯 Use Cases and Benefits

### When to Use Advanced Proactive Remediation

**Perfect for scenarios where:**
- **High Availability**: Zero-downtime error recovery
- **Complex Workflows**: Multi-step AI processes with dependencies
- **Data Processing**: ETL pipelines with variable data formats
- **API Integration**: Third-party service interactions
- **User Input**: Unpredictable user-provided data
- **Production Systems**: Critical systems requiring automatic recovery

### Benefits

| **Benefit** | **Description** |
|-------------|-----------------|
| **🚀 Zero Downtime** | Automatic error recovery without manual intervention |
| **🧠 Intelligent Learning** | System improves over time based on successful fixes |
| **🔒 Safety First** | All generated code is validated for security |
| **📊 Comprehensive Analytics** | Detailed insights into error patterns and fix success rates |
| **⚡ Fast Recovery** | Sub-second error detection and remediation |
| **🔄 Adaptive** | Handles new error types through AI-powered analysis |

### Real-World Examples

#### E-commerce Order Processing
```python
# Automatically handles missing customer IDs, invalid prices, etc.
def process_order(state):
    if "customer_id" not in state:
        raise ValueError("Missing customer_id")
    if not isinstance(state.get("total"), (int, float)):
        raise TypeError("Invalid price format")
    # Process order...
```

#### Data Pipeline Processing
```python
# Handles data type mismatches, missing fields, validation errors
def process_data(state):
    if "data" not in state:
        raise ValueError("Missing data field")
    if not isinstance(state["data"], list):
        raise TypeError("Data must be a list")
    # Process data...
```

## 🤖 Gemini AI Remediation

Get AI-powered error analysis and remediation suggestions:

remediator = GeminiRemediator(project_id="your-gcp-project")

# Get AI-powered remediation suggestions
remediation_result = remediator.analyze_and_remediate(error_analysis, node_context)

for suggestion in remediation_result.suggestions:
    print(f"Action: {suggestion.action}")
    print(f"Description: {suggestion.description}")
    print(f"Confidence: {suggestion.confidence}")
    print(f"Priority: {suggestion.priority}")
    print(f"Estimated Effort: {suggestion.estimated_effort}")
```

## 📊 Analytics and Monitoring

Get comprehensive error analytics:

```python
# Get analytics for specific node
node_analytics = graph.get_node_analytics("my_node")

# Get analytics for entire graph
graph_analytics = graph.get_graph_analytics()

print(f"Total Errors: {graph_analytics['graph_summary']['total_errors']}")
print(f"Error Categories: {graph_analytics['error_distribution']['categories']}")
print(f"Remediation Stats: {graph_analytics['graph_summary']['total_remediations']}")
```

## 🔧 Configuration Options

### AigieStateGraph Configuration
- `enable_gemini_remediation`: Enable/disable Gemini AI remediation
- `enable_proactive_remediation`: Enable/disable advanced proactive remediation
- `proactive_fix_types`: List of error types to fix proactively
- `max_proactive_attempts`: Maximum proactive fix attempts
- `confidence_threshold`: Minimum confidence for proactive fixes
- `gemini_project_id`: GCP project ID for Gemini (auto-detected if not provided)
- `auto_apply_fixes`: Automatically apply AI-suggested fixes
- `log_remediation`: Log remediation analysis to GCP

### PolicyNode Configuration
- `max_attempts`: Maximum retry attempts
- `enable_proactive_remediation`: Enable proactive remediation for this node
- `proactive_fix_types`: Specific error types to fix for this node
- `max_proactive_attempts`: Maximum proactive attempts for this node
- `confidence_threshold`: Confidence threshold for this node
- `fallback`: Fallback function when all attempts fail
- `tweak_input`: Function to modify input between attempts
- `on_error`: Custom error handler
- `enable_gemini_remediation`: Enable Gemini for this specific node
- `auto_apply_fixes`: Auto-apply fixes for this node

### Advanced Proactive Remediation Configuration
- `max_iterations`: Maximum reasoning iterations for the agent
- `confidence_threshold`: Minimum confidence for applying fixes
- `ai_model`: AI model for code generation (default: "gemini-2.5-flash")
- `project_id`: GCP project ID for AI services

## 🏗️ Architecture

### Complete System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your AI Node  │───▶│ PolicyNode│───▶│ Trail Taxonomy  │
│                 │    │                  │    │ Classification  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Advanced Proactive│   │ Error Analytics │
                       │ Remediation      │   │ & Monitoring    │
                       │ (CodeAct/ReAct)  │   │                 │
                       └──────────────────┘   └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Gemini 2.5 Flash │    │ GCP Logging     │
                       │ + AI Code Gen    │    │ & Metrics       │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Auto-fix         │    │ Learning        │
                       │ Application      │    │ & Memory        │
                       └──────────────────┘    └─────────────────┘
```

### Advanced Proactive Remediation Flow

```
┌─────────────────┐
│   Error Occurs  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Trail Taxonomy  │
│ Classification  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ CodeAct/ReAct   │
│ Agent Process   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Error Analysis  │───▶│ Strategy Gen    │───▶│ Code Generation │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Validation      │───▶│ Safe Execution  │───▶│ Learning        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- tenacity
- langchain-core
- langgraph
- google-cloud-logging
- google-cloud-aiplatform

## 🧪 Examples

See the `examples/` directory for comprehensive examples:

### 🚀 Latest Gemini API Examples
- **`examples/enhanced_usage.py`**: Demonstrates the most current Gemini API patterns and standards with Pydantic models
- **`examples/enhanced_usage.py`**: Shows Aigie integration with latest Gemini features
- **`examples/basic_usage.py`**: Basic functionality and setup

- `basic_usage.py`: Basic functionality demonstration
- `enhanced_usage.py`: Full Trail Taxonomy and Gemini AI demonstration

### Running Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Enhanced usage with Trail Taxonomy and Gemini
python examples/enhanced_usage.py

# Latest Gemini API patterns and standards
python examples/enhanced_usage.py
```

> **💡 Gemini Implementation**: For complete details on the consolidated Gemini implementation, see [GEMINI_IMPLEMENTATION_SUMMARY.md](GEMINI_IMPLEMENTATION_SUMMARY.md)

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Install test dependencies
pip install pytest

# Run tests
python -m pytest tests/ -v
```

## 🔒 Security

- All Gemini API calls are made through Google Cloud's secure infrastructure
- Error data is logged to GCP Cloud Logging with proper access controls
- Auto-fix code execution is sandboxed for safety
- No sensitive data is stored or transmitted outside your GCP project

## 📈 Performance

- **Real-time Processing**: Error classification and remediation in milliseconds
- **Intelligent Caching**: Gemini responses cached to reduce API calls
- **Efficient Classification**: Trail Taxonomy classification using optimized pattern matching
- **Minimal Overhead**: Lightweight wrapper with minimal performance impact

## 🚨 Troubleshooting

### Common Issues

1. **GCP Authentication Error**
   ```bash
   # Re-authenticate with gcloud
   gcloud auth application-default login
   ```

2. **Gemini API Not Enabled**
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   ```

3. **LangGraph Version Issues**
   ```bash
   # Update langgraph to latest version
   pip install --upgrade langgraph
   ```

4. **Import Errors**
   ```bash
   # Reinstall in development mode
   pip install -e .
   ```

5. **Compatibility Issues (Fixed in 0.5.0)**
   - ✅ **DynamicFixResult.fix_code** → **generated_code** attribute fixed
   - ✅ **JSON serialization** with datetime objects fixed
   - ✅ **Advanced proactive remediation** classes properly exported
   - See [COMPATIBILITY_FIXES_SUMMARY.md](./COMPATIBILITY_FIXES_SUMMARY.md) for details

### Getting Help

- Check the [examples](./examples/) directory for usage patterns
- Review the [test suite](./tests/) for implementation details
- Ensure all dependencies are properly installed
- Verify GCP configuration and authentication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- Email: support@aigie.ai
- Issues: GitHub Issues
- Documentation: This README and examples

---

**Aigie**: Making AI systems more reliable, one error at a time. 🚀