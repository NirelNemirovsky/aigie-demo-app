"""
Enhanced PolicyNode with Trail Taxonomy and Gemini Remediation

This module provides an enhanced PolicyNode that integrates Trail Taxonomy error classification
and Gemini 2.5 Flash AI-powered remediation for real-time error handling.
"""

from typing import Any, Dict, Optional, Iterable, Union, List, Type, TypeVar
from langchain_core.runnables import Runnable, RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
from dataclasses import asdict
from pydantic import BaseModel
import logging

from .error_taxonomy import TrailTaxonomyClassifier, ErrorAnalysis
from .gemini_remediator import GeminiRemediator, GeminiRemediationResult, RemediationSuggestion
from .advanced_proactive_remediation import AdvancedProactiveRemediationEngine as ProactiveRemediationEngine
from .advanced_proactive_remediation import DynamicFixResult as FixResult

# Support both Pydantic models and dictionaries for backward compatibility
GraphLike = Union[BaseModel, Dict[str, Any]]
T = TypeVar('T', bound=BaseModel)

# Configure logging
logger = logging.getLogger(__name__)

# Try to import GCP logging, but don't fail if not available
try:
    from google.cloud import logging as gcp_logging
    GCP_LOGGING_AVAILABLE = True
    gcp_client = gcp_logging.Client()
    gcp_logger = gcp_client.logger(__name__)
except ImportError:
    GCP_LOGGING_AVAILABLE = False
    gcp_logger = None


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
        log_remediation: bool = True,
        enable_proactive_remediation: bool = True,
        proactive_fix_types: Optional[List[str]] = None,
        max_proactive_attempts: int = 3
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
        self.enable_proactive_remediation = enable_proactive_remediation
        self.proactive_fix_types = proactive_fix_types or [
            'missing_field', 'type_error', 'validation_error', 'api_error', 'timeout_error'
        ]
        self.max_proactive_attempts = max_proactive_attempts
        
        # Initialize error classification and remediation
        self.taxonomy_classifier = TrailTaxonomyClassifier()
        self.gemini_remediator = None
        self.proactive_remediation_engine = None
        
        if enable_gemini_remediation:
            self.gemini_remediator = GeminiRemediator(project_id=gemini_project_id)
        
        if enable_proactive_remediation:
                                self.proactive_remediation_engine = ProactiveRemediationEngine(
                        max_iterations=3,
                        confidence_threshold=0.6
                    )
        
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
        
        # Convert state to dictionary for processing
        if isinstance(state, BaseModel):
            cur_state = state.model_dump() if hasattr(state, 'model_dump') else state.dict()
        else:
            cur_state = state.copy()
        
        cur_state["last_node"] = self.name
        
        while attempt < self.max_attempts:
            try:
                out = self._invoke_once(cur_state, config)
                # Success - normalize output
                out = {**cur_state, **out, "error": None, "last_node": self.name}
                
                # Log successful execution
                if self.log_remediation:
                    logger.info(f"Node '{self.name}' executed successfully on attempt {attempt + 1}")
                    
                    # GCP Logging (if available)
                    if GCP_LOGGING_AVAILABLE and gcp_logger:
                        try:
                            gcp_logger.log_text(
                                f"Node '{self.name}' executed successfully on attempt {attempt + 1}",
                                severity="INFO"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to log to GCP: {e}")
                
                return out
                
            except Exception as e:
                print(f"\n❌ Attempt {attempt + 1} failed in node '{self.name}': {str(e)}")
                print(f"🚀 Starting Aigie error handling and remediation...")
                last_exc = e
                attempt += 1
                
                # Enhanced error handling with Trail Taxonomy and Gemini
                error_result = self._handle_error_with_remediation(e, attempt, cur_state, config)
                
                # Update state with error analysis
                cur_state.update(error_result.get("state_updates", {}))
                
                # Apply automatic fixes if enabled
                if self.auto_apply_fixes and error_result.get("auto_fix_applied"):
                    print(f"🔧 Auto-fix applied! Retrying with fixed state...")
                    cur_state = error_result["fixed_state"]
                    continue
                
                # Call custom error handler if provided
                if self.on_error:
                    self.on_error(e, attempt, cur_state)
                
                # Apply input tweaking if provided
                if self.tweak_input:
                    cur_state = self.tweak_input(cur_state, e, attempt)
        
        # All attempts exhausted - try fallback
        print(f"\n⚠️  All {self.max_attempts} attempts exhausted for node '{self.name}'")
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
        Handle error using Trail Taxonomy classification, Gemini remediation, and proactive remediation
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
        
        # Step 3: PROACTIVE REMEDIATION (NEW)
        proactive_fix_applied = False
        proactive_fix_result = None
        fixed_state = state
        
        if (self.enable_proactive_remediation and 
            self.proactive_remediation_engine and 
            self.proactive_remediation_engine.can_fix_proactively(error_analysis)):
            
            print(f"\n🔧 PROACTIVE REMEDIATION - Attempting automatic fix...")
            proactive_fix_result = self.proactive_remediation_engine.apply_proactive_remediation(
                error_analysis, state, ""
            )
            
            if proactive_fix_result.success:
                proactive_fix_applied = True
                fixed_state = proactive_fix_result.fixed_state
                print(f"✅ PROACTIVE FIX SUCCESSFUL!")
                print(f"   Generated Code: {proactive_fix_result.generated_code.strip()}")
                print(f"   State Changes: {proactive_fix_result.state_changes}")
                print(f"   Execution Time: {proactive_fix_result.execution_time:.3f}s")
            else:
                print(f"❌ PROACTIVE FIX FAILED: {proactive_fix_result.error_message}")
        
        # Step 4: Log comprehensive error information
        self._log_error_analysis(error_analysis, remediation_result, attempt, start_time, proactive_fix_result)
        
        # Step 5: Prepare state updates
        state_updates = self._prepare_state_updates(error_analysis, remediation_result, attempt, proactive_fix_result)
        
        # Step 6: Apply Gemini automatic fixes if available and enabled (fallback to proactive)
        auto_fix_applied = False
        if (not proactive_fix_applied and self.auto_apply_fixes and remediation_result and 
            remediation_result.auto_fix_available and 
            remediation_result.auto_fix_code):
            
            fixed_state = self._apply_auto_fix(state, remediation_result.auto_fix_code)
            auto_fix_applied = True
        
        # Step 7: Track error history
        self._track_error_history(error_analysis, remediation_result, attempt, start_time, proactive_fix_result)
        
        return {
            "state_updates": state_updates,
            "auto_fix_applied": auto_fix_applied or proactive_fix_applied,
            "fixed_state": fixed_state,
            "error_analysis": error_analysis,
            "remediation_result": remediation_result,
            "proactive_fix_applied": proactive_fix_applied,
            "proactive_fix_result": proactive_fix_result
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
                          attempt: int, start_time: float,
                          proactive_fix_result: Optional[FixResult] = None):
        """Log comprehensive error analysis to GCP and console"""
        
        # Console output for immediate visibility
        print(f"\n🚨 AIGIE ERROR HANDLING - Node: {self.name} (Attempt {attempt})")
        print("=" * 60)
        
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
        
        # Console output for Trail Taxonomy
        print(f"🔍 TRAIL TAXONOMY ANALYSIS:")
        print(f"   Category: {error_analysis.category.value.upper()}")
        print(f"   Severity: {error_analysis.severity.value.upper()}")
        print(f"   Confidence: {error_analysis.confidence:.2f}")
        print(f"   Description: {error_analysis.description}")
        print(f"   Keywords: {', '.join(error_analysis.keywords)}")
        print(f"   Hints: {', '.join(error_analysis.remediation_hints)}")
        
        # Standard Python logging
        logger.error(f"Trail Taxonomy Analysis: {json.dumps(taxonomy_log, indent=2)}")
        
        # GCP Logging (if available)
        if GCP_LOGGING_AVAILABLE and gcp_logger:
            try:
                gcp_logger.log_text(
                    f"Trail Taxonomy Analysis: {json.dumps(taxonomy_log, indent=2)}",
                    severity="ERROR"
                )
            except Exception as e:
                logger.warning(f"Failed to log to GCP: {e}")
        
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
            
            # Console output for Gemini remediation
            print(f"\n🤖 GEMINI AI REMEDIATION:")
            print(f"   Suggestions: {len(remediation_result.suggestions)}")
            print(f"   Auto-fix available: {remediation_result.auto_fix_available}")
            print(f"   Execution time: {remediation_result.execution_time:.2f}s")
            print(f"   Reasoning: {remediation_result.reasoning}")
            
            if remediation_result.suggestions:
                print(f"\n💡 SUGGESTIONS:")
                for i, sugg in enumerate(remediation_result.suggestions, 1):
                    print(f"   {i}. {sugg.action}")
                    print(f"      Confidence: {sugg.confidence:.2f}")
                    print(f"      Priority: {sugg.priority}")
                    print(f"      Effort: {sugg.estimated_effort}")
                    if sugg.code_example:
                        print(f"      Code: {sugg.code_example}")
            
            # Standard Python logging
            logger.info(f"Gemini Remediation Analysis: {json.dumps(remediation_log, indent=2)}")
            
            # GCP Logging (if available)
            if GCP_LOGGING_AVAILABLE and gcp_logger:
                try:
                    gcp_logger.log_text(
                        f"Gemini Remediation Analysis: {json.dumps(remediation_log, indent=2)}",
                        severity="INFO"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to GCP: {e}")
        else:
            print(f"\n⚠️  GEMINI REMEDIATION: Disabled or unavailable")
        
        # Log proactive remediation if available
        if proactive_fix_result:
            proactive_log = {
                "node": self.name,
                "attempt": attempt,
                "success": proactive_fix_result.success,
                "fix_code": proactive_fix_result.fix_code,
                "execution_time": proactive_fix_result.execution_time,
                "state_changes": proactive_fix_result.state_changes,
                "error_message": proactive_fix_result.error_message
            }
            
            # Console output for proactive remediation
            print(f"\n🔧 PROACTIVE REMEDIATION LOG:")
            print(f"   Success: {proactive_fix_result.success}")
            print(f"   Execution time: {proactive_fix_result.execution_time:.3f}s")
            print(f"   State changes: {proactive_fix_result.state_changes}")
            if proactive_fix_result.error_message:
                print(f"   Error: {proactive_fix_result.error_message}")
            
            # Standard Python logging
            logger.info(f"Proactive Remediation Result: {json.dumps(proactive_log, indent=2)}")
            
            # GCP Logging (if available)
            if GCP_LOGGING_AVAILABLE and gcp_logger:
                try:
                    gcp_logger.log_text(
                        f"Proactive Remediation Result: {json.dumps(proactive_log, indent=2)}",
                        severity="INFO"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to GCP: {e}")
        else:
            print(f"\n⚠️  PROACTIVE REMEDIATION: Disabled or not applicable")
        
        print("=" * 60)
        print(f"⏱️  Total error handling time: {time.time() - start_time:.2f}s\n")
    
    def _prepare_state_updates(self, error_analysis: ErrorAnalysis, 
                             remediation_result: Optional[GeminiRemediationResult], 
                             attempt: int,
                             proactive_fix_result: Optional[FixResult] = None) -> Dict[str, Any]:
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
        
        if proactive_fix_result:
            updates["proactive_remediation_details"] = {
                "success": proactive_fix_result.success,
                "fix_code": proactive_fix_result.fix_code,
                "execution_time": proactive_fix_result.execution_time,
                "state_changes": proactive_fix_result.state_changes,
                "error_message": proactive_fix_result.error_message
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
                           attempt: int, start_time: float,
                           proactive_fix_result: Optional[FixResult] = None):
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
        
        if proactive_fix_result:
            proactive_record = {
                "timestamp": time.time(),
                "node": self.name,
                "success": proactive_fix_result.success,
                "execution_time": proactive_fix_result.execution_time,
                "state_changes": proactive_fix_result.state_changes
            }
            self.remediation_history.append(proactive_record)
    
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
                    1 for r in self.remediation_history if r.get("auto_fix_available", False)
                ),
                "proactive_fixes_successful": sum(
                    1 for r in self.remediation_history if r.get("success", False)
                ),
                "proactive_fixes_total": sum(
                    1 for r in self.remediation_history if "success" in r
                )
            },
            "gemini_available": self.gemini_remediator is not None,
            "proactive_remediation_available": self.proactive_remediation_engine is not None
        }
