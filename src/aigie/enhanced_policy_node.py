"""
Enhanced PolicyNode with Trail Taxonomy and Gemini Remediation

This module provides an enhanced PolicyNode that integrates Trail Taxonomy error classification
and Gemini 2.5 Flash AI-powered remediation for real-time error handling.
"""

from typing import Any, Dict, Optional, Iterable, Union, List
from langchain_core.runnables import Runnable, RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential
from google.cloud import logging
import time
import json
from dataclasses import asdict

from .error_taxonomy import TrailTaxonomyClassifier, ErrorAnalysis
from .gemini_remediator import GeminiRemediator, GeminiRemediationResult, RemediationSuggestion

GraphLike = Dict[str, Any]

# Configure GCP logging
client = logging.Client()
gcp_logger = client.logger(__name__)


class EnhancedPolicyNode(Runnable[GraphLike, GraphLike]):
    """
    Enhanced PolicyNode with Trail Taxonomy error classification and Gemini AI remediation
    """
    
    def __init__(
        self,
        inner: Union[Runnable, callable],
        name: str,
        max_attempts: int = 3,
        fallback: Optional[Union[Runnable, callable]] = None,
        tweak_input: Optional[callable] = None,
        on_error: Optional[callable] = None,
        enable_gemini_remediation: bool = True,
        gemini_project_id: Optional[str] = None,
        auto_apply_fixes: bool = False,
        log_remediation: bool = True
    ):
        self.inner = inner
        self.name = name
        self.max_attempts = max_attempts
        self.fallback = fallback
        self.tweak_input = tweak_input
        self.on_error = on_error
        self.enable_gemini_remediation = enable_gemini_remediation
        self.auto_apply_fixes = auto_apply_fixes
        self.log_remediation = log_remediation
        
        # Initialize error classification and remediation
        self.taxonomy_classifier = TrailTaxonomyClassifier()
        self.gemini_remediator = None
        
        if enable_gemini_remediation:
            self.gemini_remediator = GeminiRemediator(project_id=gemini_project_id)
        
        # Performance tracking
        self.error_history = []
        self.remediation_history = []
    
    def _invoke_once(self, state: GraphLike, config: Optional[RunnableConfig]) -> GraphLike:
        """Execute the inner function once"""
        if hasattr(self.inner, "invoke"):
            return self.inner.invoke(state, config=config)
        return self.inner(state)
    
    def invoke(self, state: GraphLike, config: Optional[RunnableConfig] = None) -> GraphLike:
        """Enhanced invoke with Trail Taxonomy and Gemini remediation"""
        attempt = 0
        last_exc = None
        cur_state = {**state, "last_node": self.name}
        
        while attempt < self.max_attempts:
            try:
                out = self._invoke_once(cur_state, config)
                # Success - normalize output
                out = {**cur_state, **out, "error": None, "last_node": self.name}
                
                # Log successful execution
                if self.log_remediation:
                    gcp_logger.log_text(
                        f"Node '{self.name}' executed successfully on attempt {attempt + 1}",
                        severity="INFO"
                    )
                
                return out
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed in node '{self.name}': {str(e)}")
                last_exc = e
                attempt += 1
                
                # Enhanced error handling with Trail Taxonomy and Gemini
                error_result = self._handle_error_with_remediation(e, attempt, cur_state, config)
                
                # Update state with error analysis
                cur_state.update(error_result.get("state_updates", {}))
                
                # Apply automatic fixes if enabled
                if self.auto_apply_fixes and error_result.get("auto_fix_applied"):
                    cur_state = error_result["fixed_state"]
                    continue
                
                # Call custom error handler if provided
                if self.on_error:
                    self.on_error(e, attempt, cur_state)
                
                # Apply input tweaking if provided
                if self.tweak_input:
                    cur_state = self.tweak_input(cur_state, e, attempt)
        
        # All attempts exhausted - try fallback
        if self.fallback is not None:
            try:
                if hasattr(self.fallback, "invoke"):
                    out = self.fallback.invoke(cur_state, config=config)
                else:
                    out = self.fallback(cur_state)
                return {**cur_state, **out, "error": None, "last_node": self.name}
            except Exception as e2:
                last_exc = e2
        
        # Final error state
        return {
            **cur_state,
            "error": {
                "node": self.name,
                "type": type(last_exc).__name__,
                "msg": str(last_exc),
                "attempts": self.max_attempts,
                "final_attempt": True
            },
        }
    
    def _handle_error_with_remediation(self, exception: Exception, attempt: int, 
                                     state: GraphLike, config: Optional[RunnableConfig]) -> Dict[str, Any]:
        """
        Handle error using Trail Taxonomy classification and Gemini remediation
        """
        start_time = time.time()
        
        # Prepare node context for analysis
        node_context = self._prepare_node_context(state, config)
        
        # Step 1: Trail Taxonomy Classification
        error_analysis = self.taxonomy_classifier.classify_error(exception, node_context)
        
        # Step 2: Gemini AI Remediation (if enabled)
        remediation_result = None
        if self.gemini_remediator:
            remediation_result = self.gemini_remediator.analyze_and_remediate(
                error_analysis, node_context
            )
        
        # Step 3: Log comprehensive error information
        self._log_error_analysis(error_analysis, remediation_result, attempt, start_time)
        
        # Step 4: Prepare state updates
        state_updates = self._prepare_state_updates(error_analysis, remediation_result, attempt)
        
        # Step 5: Apply automatic fixes if available and enabled
        auto_fix_applied = False
        fixed_state = state
        if (self.auto_apply_fixes and remediation_result and 
            remediation_result.auto_fix_available and 
            remediation_result.auto_fix_code):
            
            fixed_state = self._apply_auto_fix(state, remediation_result.auto_fix_code)
            auto_fix_applied = True
        
        # Step 6: Track error history
        self._track_error_history(error_analysis, remediation_result, attempt, start_time)
        
        return {
            "state_updates": state_updates,
            "auto_fix_applied": auto_fix_applied,
            "fixed_state": fixed_state,
            "error_analysis": error_analysis,
            "remediation_result": remediation_result
        }
    
    def _prepare_node_context(self, state: GraphLike, config: Optional[RunnableConfig]) -> Dict[str, Any]:
        """Prepare context information for error analysis"""
        return {
            "node_name": self.name,
            "node_type": type(self.inner).__name__,
            "environment": "production",  # Could be made configurable
            "state_keys": list(state.keys()) if state else [],
            "state_size": len(str(state)) if state else 0,
            "config": config
        }
    
    def _log_error_analysis(self, error_analysis: ErrorAnalysis, 
                          remediation_result: Optional[GeminiRemediationResult], 
                          attempt: int, start_time: float):
        """Log comprehensive error analysis to GCP"""
        
        # Log Trail Taxonomy analysis
        taxonomy_log = {
            "node": self.name,
            "attempt": attempt,
            "error_category": error_analysis.category.value,
            "error_severity": error_analysis.severity.value,
            "confidence": error_analysis.confidence,
            "description": error_analysis.description,
            "keywords": error_analysis.keywords,
            "remediation_hints": error_analysis.remediation_hints
        }
        
        gcp_logger.log_text(
            f"Trail Taxonomy Analysis: {json.dumps(taxonomy_log, indent=2)}",
            severity="ERROR"
        )
        
        # Log Gemini remediation if available
        if remediation_result:
            remediation_log = {
                "node": self.name,
                "attempt": attempt,
                "suggestions_count": len(remediation_result.suggestions),
                "auto_fix_available": remediation_result.auto_fix_available,
                "reasoning": remediation_result.reasoning,
                "execution_time": remediation_result.execution_time,
                "suggestions": [
                    {
                        "action": sugg.action,
                        "confidence": sugg.confidence,
                        "priority": sugg.priority,
                        "estimated_effort": sugg.estimated_effort
                    }
                    for sugg in remediation_result.suggestions
                ]
            }
            
            gcp_logger.log_text(
                f"Gemini Remediation Analysis: {json.dumps(remediation_log, indent=2)}",
                severity="INFO"
            )
    
    def _prepare_state_updates(self, error_analysis: ErrorAnalysis, 
                             remediation_result: Optional[GeminiRemediationResult], 
                             attempt: int) -> Dict[str, Any]:
        """Prepare state updates with error analysis and remediation info"""
        updates = {
            "error_details": {
                "category": error_analysis.category.value,
                "severity": error_analysis.severity.value,
                "confidence": error_analysis.confidence,
                "description": error_analysis.description,
                "keywords": error_analysis.keywords,
                "remediation_hints": error_analysis.remediation_hints,
                "attempt": attempt,
                "timestamp": time.time()
            }
        }
        
        if remediation_result:
            updates["remediation_details"] = {
                "suggestions": [
                    {
                        "action": sugg.action,
                        "description": sugg.description,
                        "confidence": sugg.confidence,
                        "priority": sugg.priority,
                        "estimated_effort": sugg.estimated_effort
                    }
                    for sugg in remediation_result.suggestions
                ],
                "reasoning": remediation_result.reasoning,
                "auto_fix_available": remediation_result.auto_fix_available,
                "execution_time": remediation_result.execution_time
            }
        
        return updates
    
    def _apply_auto_fix(self, state: GraphLike, auto_fix_code: str) -> GraphLike:
        """Apply automatic fix code to state"""
        try:
            # Create a safe execution environment
            safe_globals = {
                "state": state.copy(),
                "json": json,
                "time": time
            }
            
            # Execute the auto-fix code
            exec(auto_fix_code, safe_globals)
            
            # Return the potentially modified state
            return safe_globals.get("state", state)
            
        except Exception as e:
            gcp_logger.log_text(
                f"Auto-fix application failed for node '{self.name}': {str(e)}",
                severity="WARNING"
            )
            return state
    
    def _track_error_history(self, error_analysis: ErrorAnalysis, 
                           remediation_result: Optional[GeminiRemediationResult], 
                           attempt: int, start_time: float):
        """Track error history for analytics"""
        error_record = {
            "timestamp": time.time(),
            "node": self.name,
            "attempt": attempt,
            "category": error_analysis.category.value,
            "severity": error_analysis.severity.value,
            "confidence": error_analysis.confidence,
            "processing_time": time.time() - start_time
        }
        
        self.error_history.append(error_record)
        
        if remediation_result:
            remediation_record = {
                "timestamp": time.time(),
                "node": self.name,
                "suggestions_count": len(remediation_result.suggestions),
                "auto_fix_available": remediation_result.auto_fix_available,
                "execution_time": remediation_result.execution_time
            }
            self.remediation_history.append(remediation_record)
    
    def stream(self, state: GraphLike, config: Optional[RunnableConfig] = None) -> Iterable:
        """Enhanced streaming with error handling"""
        if hasattr(self.inner, "stream"):
            try:
                yield from self.inner.stream(state, config=config)
                return
            except Exception as e:
                # Handle streaming errors with remediation
                error_result = self._handle_error_with_remediation(e, 1, state, config)
                yield {
                    "event": "error", 
                    "error": {
                        "node": self.name, 
                        "type": type(e).__name__, 
                        "msg": str(e),
                        "analysis": error_result.get("error_analysis"),
                        "remediation": error_result.get("remediation_result")
                    }
                }
        
        # Fallback to non-stream invoke
        yield self.invoke(state, config=config)
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get error analytics and statistics"""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        # Calculate statistics
        total_errors = len(self.error_history)
        categories = {}
        severities = {}
        
        for error in self.error_history:
            cat = error["category"]
            sev = error["severity"]
            
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1
        
        return {
            "total_errors": total_errors,
            "error_categories": categories,
            "error_severities": severities,
            "remediation_stats": {
                "total_remediations": len(self.remediation_history),
                "auto_fix_available_count": sum(
                    1 for r in self.remediation_history if r["auto_fix_available"]
                )
            },
            "gemini_available": self.gemini_remediator is not None
        }
