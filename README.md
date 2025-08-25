# Aigie - AI World Fixer with Trail Taxonomy & Gemini AI

A Python package that provides intelligent error handling and remediation for AI systems using Trail Taxonomy classification and Google's Gemini 2.5 Flash AI model. **Aigie natively supports Pydantic models as the single standard for state management**, eliminating the need for multiple compatibility layers.

## 🚀 Features

- **Unified Pydantic Standard**: Native support for Pydantic models as the single state management standard
- **Trail Taxonomy Error Classification**: Comprehensive error categorization based on the Trail Taxonomy paper
- **Gemini AI Remediation**: AI-powered error analysis and remediation suggestions using Google's Gemini 2.5 Flash
- **Real-time Error Handling**: Intelligent error detection and resolution in real-time
- **Automatic Error Fixing**: Optional automatic application of AI-suggested fixes
- **Comprehensive Analytics**: Detailed error analytics and monitoring
- **Seamless Integration**: Drop-in replacement for LangGraph's StateGraph
- **GCP Integration**: Built-in Google Cloud Logging and Vertex AI integration

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
```

### Install Dependencies

The package will automatically install required dependencies:
- `tenacity` - Retry logic
- `langchain-core` - Core LangChain functionality
- `langgraph` - Graph-based AI workflows
- `google-cloud-logging` - GCP logging integration
- `google-cloud-aiplatform` - Vertex AI for Gemini

## 🔧 Setup

### 1. GCP Configuration (for Gemini AI features)

```bash
# Install gcloud CLI (if not already installed)
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
# Set your GCP project ID
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Optional: Set GCP location (default: us-central1)
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 3. Verify Installation

```python
from aigie import AigieStateGraph, TrailTaxonomyClassifier

# Test basic import
print("✅ Aigie imported successfully!")

# Test Trail Taxonomy
classifier = TrailTaxonomyClassifier()
print("✅ Trail Taxonomy ready!")

# Test graph creation
graph = AigieStateGraph()
print("✅ AigieStateGraph ready!")
```

## 🎯 Quick Start

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

# Create an enhanced state graph with AI-powered error handling
graph = AigieStateGraph(
    enable_gemini_remediation=True,  # Enable Gemini AI remediation
    auto_apply_fixes=False,          # Don't auto-apply fixes
    log_remediation=True             # Log all remediation analysis
)

# Define your AI functions
def my_ai_function(state):
    # Your AI logic here
    if "input" not in state:
        raise ValueError("Missing input in state")
    return {"result": "success", **state}

# Add nodes (automatically wrapped with enhanced error handling)
graph.add_node("my_node", my_ai_function)

# The graph now has intelligent error handling with Trail Taxonomy and Gemini AI
```

### Advanced Usage with Custom Configuration

```python
from aigie import AigieStateGraph, EnhancedPolicyNode

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
from aigie import TrailTaxonomyClassifier

classifier = TrailTaxonomyClassifier()

# Classify an error
error = ValueError("Missing required field 'user_id'")
analysis = classifier.classify_error(error)

print(f"Category: {analysis.category.value}")
print(f"Severity: {analysis.severity.value}")
print(f"Confidence: {analysis.confidence}")
print(f"Keywords: {analysis.keywords}")
print(f"Remediation Hints: {analysis.remediation_hints}")
```

## 🤖 Gemini AI Remediation

Get AI-powered error analysis and remediation suggestions:

```python
from aigie import GeminiRemediator

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
- `gemini_project_id`: GCP project ID for Gemini (auto-detected if not provided)
- `auto_apply_fixes`: Automatically apply AI-suggested fixes
- `log_remediation`: Log remediation analysis to GCP

### EnhancedPolicyNode Configuration
- `max_attempts`: Maximum retry attempts
- `fallback`: Fallback function when all attempts fail
- `tweak_input`: Function to modify input between attempts
- `on_error`: Custom error handler
- `enable_gemini_remediation`: Enable Gemini for this specific node
- `auto_apply_fixes`: Auto-apply fixes for this node

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your AI Node  │───▶│ EnhancedPolicyNode│───▶│ Trail Taxonomy  │
│                 │    │                  │    │ Classification  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Gemini 2.5 Flash │    │ Error Analytics │
                       │ Remediation      │    │ & Monitoring    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Auto-fix         │    │ GCP Logging     │
                       │ Application      │    │ & Metrics       │
                       └──────────────────┘    └─────────────────┘
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

- `basic_usage.py`: Basic functionality demonstration
- `enhanced_usage.py`: Full Trail Taxonomy and Gemini AI demonstration

### Running Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Enhanced usage with Trail Taxonomy and Gemini
python examples/enhanced_usage.py
```

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