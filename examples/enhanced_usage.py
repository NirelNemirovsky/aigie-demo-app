#!/usr/bin/env python3
"""
Enhanced usage example for the aigie package with Trail Taxonomy and Gemini remediation
"""

import os
import time
from aigie import AigieStateGraph, EnhancedPolicyNode, TrailTaxonomyClassifier, GeminiRemediator

def main():
    """Demonstrate enhanced aigie functionality with Trail Taxonomy and Gemini"""
    
    print("🚀 Enhanced Aigie Package Demo with Trail Taxonomy & Gemini")
    print("=" * 70)
    
    # Example 1: Basic enhanced usage with automatic error classification
    print("\n1. Creating an Enhanced AigieStateGraph...")
    
    # Create graph with enhanced error handling
    graph = AigieStateGraph(
        enable_gemini_remediation=True,  # Enable Gemini AI remediation
        auto_apply_fixes=False,          # Don't auto-apply fixes for demo
        log_remediation=True             # Log all remediation analysis
    )
    
    print("✅ Enhanced StateGraph created successfully")
    
    # Example 2: Functions that will trigger different error types
    def input_validation_error_function(state):
        """Function that triggers input validation errors"""
        if "input_data" not in state:
            raise ValueError("Missing required input_data field")
        
        if not isinstance(state["input_data"], dict):
            raise TypeError("input_data must be a dictionary")
        
        return {"processed": True, "result": "Input validated", **state}
    
    def processing_error_function(state):
        """Function that triggers processing errors"""
        if "numbers" not in state:
            raise ValueError("Missing numbers for processing")
        
        numbers = state["numbers"]
        if not numbers:
            raise ZeroDivisionError("Cannot process empty list")
        
        # Simulate a processing error
        result = sum(numbers) / len(numbers)
        return {"average": result, "processed": True, **state}
    
    def external_api_error_function(state):
        """Function that simulates external API errors"""
        import requests
        
        # Simulate API call that might fail
        try:
            response = requests.get("https://httpbin.org/status/500", timeout=1)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"External API call failed: {str(e)}")
        
        return {"api_result": "success", **state}
    
    # Add nodes to the graph with enhanced error handling
    print("\n2. Adding nodes with enhanced error handling...")
    
    graph.add_node("input_validator", input_validation_error_function)
    graph.add_node("data_processor", processing_error_function)
    graph.add_node("api_caller", external_api_error_function)
    
    print("✅ Nodes added with enhanced error handling")
    
    # Example 3: Test error scenarios
    print("\n3. Testing error scenarios with Trail Taxonomy classification...")
    
    # Test 1: Input validation error
    print("\n   Testing Input Validation Error:")
    test_state_1 = {"step": 1}  # Missing input_data
    try:
        result = graph.nodes["input_validator"].invoke(test_state_1)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Expected error caught: {e}")
    
    # Test 2: Processing error
    print("\n   Testing Processing Error:")
    test_state_2 = {"numbers": []}  # Empty list will cause division by zero
    try:
        result = graph.nodes["data_processor"].invoke(test_state_2)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Expected error caught: {e}")
    
    # Example 4: Manual Trail Taxonomy classification
    print("\n4. Manual Trail Taxonomy Classification Demo:")
    
    classifier = TrailTaxonomyClassifier()
    
    # Test different error types
    test_errors = [
        ValueError("Missing required field 'user_id'"),
        ZeroDivisionError("Division by zero in calculation"),
        RuntimeError("External API timeout after 30 seconds"),
        FileNotFoundError("Configuration file not found: config.json"),
        TypeError("Expected string, got int")
    ]
    
    for error in test_errors:
        analysis = classifier.classify_error(error)
        print(f"   Error: {type(error).__name__}: {str(error)}")
        print(f"   → Category: {analysis.category.value}")
        print(f"   → Severity: {analysis.severity.value}")
        print(f"   → Confidence: {analysis.confidence:.2f}")
        print(f"   → Keywords: {analysis.keywords}")
        print()
    
    # Example 5: Gemini Remediation Demo (if available)
    print("\n5. Gemini AI Remediation Demo:")
    
    # Check if Gemini is available
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        print(f"   Using GCP Project: {project_id}")
        
        remediator = GeminiRemediator(project_id=project_id)
        
        # Test remediation on a sample error
        sample_error = ValueError("Invalid JSON format in API response")
        sample_context = {
            "node_name": "api_processor",
            "node_type": "function",
            "environment": "production"
        }
        
        # Get error analysis
        classifier = TrailTaxonomyClassifier()
        error_analysis = classifier.classify_error(sample_error, sample_context)
        
        print(f"   Analyzing error: {sample_error}")
        print(f"   → Trail Taxonomy Category: {error_analysis.category.value}")
        
        # Get Gemini remediation
        remediation_result = remediator.analyze_and_remediate(error_analysis, sample_context)
        
        print(f"   → Gemini Suggestions: {len(remediation_result.suggestions)}")
        print(f"   → Auto-fix Available: {remediation_result.auto_fix_available}")
        print(f"   → Reasoning: {remediation_result.reasoning[:100]}...")
        
        # Show top suggestion
        if remediation_result.suggestions:
            top_suggestion = remediation_result.suggestions[0]
            print(f"   → Top Suggestion: {top_suggestion.action}")
            print(f"   → Confidence: {top_suggestion.confidence:.2f}")
    else:
        print("   ⚠️  GCP Project ID not set. Set GOOGLE_CLOUD_PROJECT environment variable for Gemini demo.")
    
    # Example 6: Analytics and Monitoring
    print("\n6. Error Analytics Demo:")
    
    # Get analytics for the graph
    analytics = graph.get_graph_analytics()
    
    print(f"   Graph Summary:")
    print(f"   → Total Nodes: {analytics['graph_summary']['total_nodes']}")
    print(f"   → Configuration: Gemini={analytics['configuration']['enable_gemini_remediation']}")
    
    # Get node-specific analytics
    node_analytics = graph.get_node_analytics("input_validator")
    print(f"   → Input Validator Analytics: {node_analytics}")
    
    print("\n🎉 Enhanced Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Trail Taxonomy error classification")
    print("✅ Gemini AI-powered remediation suggestions")
    print("✅ Real-time error handling and logging")
    print("✅ Comprehensive error analytics")
    print("✅ Seamless integration with LangGraph")

if __name__ == "__main__":
    main()
