"""
Tests for enhanced aigie features with Trail Taxonomy and Gemini remediation
"""

import pytest
from unittest.mock import Mock, patch
import os

from aigie import (
    AigieStateGraph,
    PolicyNode, 
    EnhancedTrailTaxonomyClassifier, 
    GeminiRemediator,
    ErrorCategory,
    ErrorSeverity
)


class TestEnhancedTrailTaxonomyClassifier:
    """Test Trail Taxonomy error classification"""
    
    def test_input_error_classification(self):
        """Test classification of input validation errors"""
        classifier = EnhancedTrailTaxonomyClassifier()
        
        error = ValueError("Missing required field 'user_id'")
        analysis = classifier.classify_error(error)
        
        assert analysis.category == ErrorCategory.INPUT_ERROR
        assert analysis.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        assert analysis.confidence > 0.5
        assert "missing" in analysis.keywords or "required" in analysis.keywords
    
    def test_processing_error_classification(self):
        """Test classification of processing errors"""
        classifier = EnhancedTrailTaxonomyClassifier()
        
        error = ZeroDivisionError("Division by zero")
        analysis = classifier.classify_error(error)
        
        assert analysis.category == ErrorCategory.PROCESSING_ERROR
        assert analysis.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        assert analysis.confidence > 0.5
        assert "division" in analysis.keywords or "zero" in analysis.keywords
    
    def test_external_error_classification(self):
        """Test classification of external service errors"""
        classifier = EnhancedTrailTaxonomyClassifier()
        
        error = RuntimeError("API timeout after 30 seconds")
        analysis = classifier.classify_error(error)
        
        # Should classify as external error due to "API" and "timeout" keywords
        assert analysis.category in [ErrorCategory.EXTERNAL_ERROR, ErrorCategory.SYSTEM_ERROR]
        assert analysis.confidence > 0.3


class TestPolicyNode:
    """Test PolicyNode functionality"""
    
    def test_basic_functionality(self):
        """Test basic PolicyNode functionality"""
        def test_function(state):
            return {"result": "success", **state}
        
        node = PolicyNode(
            inner=test_function,
            name="test_node",
            enable_gemini_remediation=False  # Disable for testing
        )
        
        result = node.invoke({"input": "test"})
        assert result["result"] == "success"
        assert result["input"] == "test"
        assert result["error"] is None
    
    def test_error_handling_without_gemini(self):
        """Test error handling when Gemini is disabled"""
        def error_function(state):
            raise ValueError("Test error")
        
        node = PolicyNode(
            inner=error_function,
            name="error_node",
            enable_gemini_remediation=False,
            max_attempts=2
        )
        
        result = node.invoke({"input": "test"})
        assert "error" in result
        assert result["error"]["type"] == "ValueError"
        assert result["error"]["msg"] == "Test error"
    
    @patch('aigie.gemini_remediator.GEMINI_AVAILABLE', False)
    def test_fallback_when_gemini_unavailable(self):
        """Test fallback behavior when Gemini is not available"""
        def error_function(state):
            raise ValueError("Test error")
        
        node = PolicyNode(
            inner=error_function,
            name="fallback_node",
            enable_gemini_remediation=True,  # Try to enable
            max_attempts=1
        )
        
        result = node.invoke({"input": "test"})
        assert "error" in result
        # Should still work with fallback remediation


class TestAigieStateGraph:
    """Test enhanced AigieStateGraph functionality"""
    
    def test_enhanced_graph_creation(self):
        """Test creation of enhanced graph"""
        graph = AigieStateGraph(
            enable_gemini_remediation=False,  # Disable for testing
            auto_apply_fixes=False,
            log_remediation=False
        )
        
        assert graph.enable_gemini_remediation is False
        assert graph.auto_apply_fixes is False
        assert graph.log_remediation is False
    
    def test_enhanced_node_addition(self):
        """Test adding nodes with enhanced error handling"""
        graph = AigieStateGraph(enable_gemini_remediation=False)
        
        def test_function(state):
            return {"result": "success", **state}
        
        graph.add_node("test_node", test_function)
        
        assert "test_node" in graph.nodes
        node = graph.nodes["test_node"]
        assert isinstance(node, PolicyNode)
        assert node.name == "test_node"
    
    def test_analytics_functionality(self):
        """Test analytics functionality"""
        graph = AigieStateGraph(enable_gemini_remediation=False)
        
        def error_function(state):
            raise ValueError("Test error")
        
        graph.add_node("error_node", error_function)
        
        # Trigger an error
        try:
            graph.nodes["error_node"].invoke({"input": "test"})
        except:
            pass
        
        # Get analytics
        analytics = graph.get_node_analytics("error_node")
        assert "total_errors" in analytics or "message" in analytics
        
        graph_analytics = graph.get_graph_analytics()
        assert "graph_summary" in graph_analytics
        assert "configuration" in graph_analytics


class TestGeminiRemediator:
    """Test Gemini remediation functionality"""
    
    @patch('aigie.gemini_remediator.GEMINI_AVAILABLE', False)
    def test_fallback_remediation(self):
        """Test fallback remediation when Gemini is not available"""
        remediator = GeminiRemediator()
        
        from aigie import ErrorAnalysis, ErrorCategory, ErrorSeverity
        
        error_analysis = ErrorAnalysis(
            category=ErrorCategory.INPUT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            confidence=0.8,
            keywords=["missing", "required"],
            description="Test error",
            remediation_hints=["Add validation"],
            context={}
        )
        
        result = remediator.analyze_and_remediate(error_analysis, {})
        
        assert result.error_analysis == error_analysis
        assert len(result.suggestions) > 0
        assert result.auto_fix_available is False


if __name__ == "__main__":
    pytest.main([__file__])
