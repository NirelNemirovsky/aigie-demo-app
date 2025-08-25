"""
Enhanced PolicyNode V2 with Advanced Error Handling and Learning

This module provides an enhanced PolicyNode that integrates the improved error classification
and adaptive remediation system to better handle architectural issues, learn from failures,
and provide fallback mechanisms.
"""

from typing import Any, Dict, Optional, Iterable, Union, List, Type, TypeVar
from langchain_core.runnables import Runnable, RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
from dataclasses import asdict
from pydantic import BaseModel
import logging
from datetime import datetime

from .error_taxonomy import EnhancedTrailTaxonomyClassifier, ErrorAnalysis, ErrorCategory
from .advanced_proactive_remediation import AdaptiveRemediationEngine, EnhancedRemediationResult, FixStrategy
from .gemini_remediator import GeminiRemediator, GeminiRemediationResult, RemediationSuggestion

# Support both Pydantic models and dictionaries for backward compatibility
GraphLike = Union[BaseModel, Dict[str, Any]]
T = TypeVar('T', bound=BaseModel)

# Configure logging
logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Try to import GCP logging, but don't fail if not available
try:
    from google.cloud import logging as gcp_logging
    GCP_LOGGING_AVAILABLE = True
    # Don't initialize client at module level - do it lazily when needed
    gcp_client = None
    gcp_logger = None
except ImportError:
    GCP_LOGGING_AVAILABLE = False
    gcp_client = None
    gcp_logger = None


class EnhancedPolicyNode(Runnable[GraphLike, GraphLike]):
    """
    Enhanced PolicyNode V2 with advanced error handling, learning, and architectural awareness
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
        auto_apply_fixes: bool = True,  # Changed to True by default
        log_remediation: bool = True,
        enable_adaptive_remediation: bool = True,  # NEW: Enable adaptive remediation
        enable_learning: bool = True,  # NEW: Enable learning from failures
        max_adaptive_attempts: int = 3,
        confidence_threshold: float = 0.7
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
        self.enable_adaptive_remediation = enable_adaptive_remediation
        self.enable_learning = enable_learning
        self.max_adaptive_attempts = max_adaptive_attempts
        self.confidence_threshold = confidence_threshold
        
        # Initialize enhanced error classification and remediation
        self.enhanced_taxonomy_classifier = EnhancedTrailTaxonomyClassifier()
        self.adaptive_remediation_engine = None
        self.gemini_remediator = None
        
        if enable_adaptive_remediation:
            self.adaptive_remediation_engine = AdaptiveRemediationEngine(
                max_attempts=max_adaptive_attempts,
                confidence_threshold=confidence_threshold
            )
        
        if enable_gemini_remediation:
            self.gemini_remediator = GeminiRemediator(project_id=gemini_project_id)
        
        # Performance tracking
        self.error_history = []
        self.remediation_history = []
        self.learning_statistics = {}
        
        # Lazy GCP logger initialization
        self._gcp_logger = None
    
    def _get_gcp_logger(self):
        """Lazily initialize and return GCP logger"""
        if self._gcp_logger is None and GCP_LOGGING_AVAILABLE:
            try:
                if gcp_client is None:
                    # Initialize client only when needed
                    client = gcp_logging.Client()
                else:
                    client = gcp_client
                self._gcp_logger = client.logger(__name__)
            except Exception as e:
                logger.warning(f"Failed to initialize GCP logger: {e}")
                self._gcp_logger = None
        return self._gcp_logger
    
    def _invoke_once(self, state: GraphLike, config: Optional[RunnableConfig]) -> GraphLike:
        """Execute the inner function once"""
        if hasattr(self.inner, "invoke"):
            return self.inner.invoke(state, config=config)
        return self.inner(state)
    
    def invoke(self, state: GraphLike, config: Optional[RunnableConfig] = None) -> GraphLike:
        """Enhanced invoke with advanced error handling and learning"""
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
                    gcp_logger = self._get_gcp_logger()
                    if gcp_logger:
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
                print(f"🚀 Starting Enhanced Aigie error handling and remediation...")
                last_exc = e
                attempt += 1
                
                # Enhanced error handling with learning and adaptive strategies
                error_result = self._handle_error_with_enhanced_remediation(e, attempt, cur_state, config)
                
                # Update state with error analysis
                cur_state.update(error_result.get("state_updates", {}))
                
                # Apply automatic fixes if enabled and available
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
    
    def _handle_error_with_enhanced_remediation(self, exception: Exception, attempt: int, 
                                              state: GraphLike, config: Optional[RunnableConfig]) -> Dict[str, Any]:
        """
        Handle error using enhanced error classification, adaptive remediation, and learning
        """
        start_time = time.time()
        
        # Prepare node context for analysis
        node_context = self._prepare_node_context(state, config)
        
        # Step 1: Enhanced Trail Taxonomy Classification
        error_analysis = self.enhanced_taxonomy_classifier.classify_error(exception, node_context)
        
        # Step 2: ADAPTIVE REMEDIATION (NEW - Primary approach)
        adaptive_fix_applied = False
        adaptive_fix_result = None
        fixed_state = state
        
        if self.enable_adaptive_remediation and self.adaptive_remediation_engine:
            print(f"\n🧠 ADAPTIVE REMEDIATION - Using learning-based strategies...")
            adaptive_fix_result = self.adaptive_remediation_engine.remediate_error(
                exception, state, node_context
            )
            
            if adaptive_fix_result.success:
                adaptive_fix_applied = True
                fixed_state = adaptive_fix_result.fixed_state
                print(f"✅ ADAPTIVE FIX SUCCESSFUL!")
                print(f"   Strategy Used: {adaptive_fix_result.strategy_used.value}")
                print(f"   Confidence: {adaptive_fix_result.confidence:.2f}")
                print(f"   Execution Time: {adaptive_fix_result.execution_time:.3f}s")
                print(f"   State Changes: {len(adaptive_fix_result.state_changes) if adaptive_fix_result.state_changes else 0} changes")
                
                # Log learning insights
                if adaptive_fix_result.learning_insights:
                    print(f"   Learning: {adaptive_fix_result.learning_insights.get('total_attempts', 0)} previous attempts")
            else:
                print(f"❌ ADAPTIVE FIX FAILED: {adaptive_fix_result.error_message}")
        
        # Step 3: Gemini AI Remediation (fallback)
        gemini_remediation_result = None
        if (not adaptive_fix_applied and self.enable_gemini_remediation and 
            self.gemini_remediator):
            gemini_remediation_result = self.gemini_remediator.analyze_and_remediate(
                error_analysis, node_context
            )
        
        # Step 4: Log comprehensive error information
        self._log_enhanced_error_analysis(error_analysis, adaptive_fix_result, gemini_remediation_result, attempt, start_time)
        
        # Step 5: Prepare state updates
        state_updates = self._prepare_enhanced_state_updates(error_analysis, adaptive_fix_result, gemini_remediation_result, attempt)
        
        # Step 6: Apply automatic fixes (prioritize adaptive over Gemini)
        auto_fix_applied = False
        if adaptive_fix_applied:
            auto_fix_applied = True
        elif (self.auto_apply_fixes and gemini_remediation_result and 
              gemini_remediation_result.auto_fix_available and 
              gemini_remediation_result.auto_fix_code):
            fixed_state = self._apply_auto_fix(state, gemini_remediation_result.auto_fix_code)
            auto_fix_applied = True
        
        # Step 7: Track error history and learning
        self._track_enhanced_error_history(error_analysis, adaptive_fix_result, gemini_remediation_result, attempt, start_time)
        
        return {
            "state_updates": state_updates,
            "auto_fix_applied": auto_fix_applied,
            "fixed_state": fixed_state,
            "error_analysis": error_analysis,
            "adaptive_fix_result": adaptive_fix_result,
            "gemini_remediation_result": gemini_remediation_result
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
    
    def _log_enhanced_error_analysis(self, error_analysis: ErrorAnalysis, 
                                   adaptive_fix_result: Optional[EnhancedRemediationResult],
                                   gemini_remediation_result: Optional[GeminiRemediationResult], 
                                   attempt: int, start_time: float):
        """Log comprehensive error analysis with enhanced information"""
        
        # Console output for immediate visibility
        print(f"\n🚨 ENHANCED AIGIE ERROR HANDLING - Node: {self.name} (Attempt {attempt})")
        print("=" * 70)
        
        # Log Enhanced Trail Taxonomy analysis
        taxonomy_log = {
            "node": self.name,
            "attempt": attempt,
            "error_category": error_analysis.category.value,
            "error_severity": error_analysis.severity.value,
            "confidence": error_analysis.confidence,
            "description": error_analysis.description,
            "keywords": error_analysis.keywords,
            "remediation_hints": error_analysis.remediation_hints,
            "root_cause": error_analysis.root_cause,
            "architectural_issue": error_analysis.architectural_issue,
            "suggested_fix_type": error_analysis.suggested_fix_type
        }
        
        # Console output for Enhanced Trail Taxonomy
        print(f"🔍 ENHANCED TRAIL TAXONOMY ANALYSIS:")
        print(f"   Category: {error_analysis.category.value.upper()}")
        print(f"   Severity: {error_analysis.severity.value.upper()}")
        print(f"   Confidence: {error_analysis.confidence:.2f}")
        print(f"   Description: {error_analysis.description}")
        print(f"   Root Cause: {error_analysis.root_cause or 'Unknown'}")
        print(f"   Architectural Issue: {error_analysis.architectural_issue or 'None'}")
        print(f"   Suggested Fix: {error_analysis.suggested_fix_type or 'None'}")
        print(f"   Keywords: {', '.join(error_analysis.keywords)}")
        print(f"   Hints: {', '.join(error_analysis.remediation_hints)}")
        
        # Standard Python logging
        logger.error(f"Enhanced Trail Taxonomy Analysis: {json.dumps(taxonomy_log, indent=2, cls=DateTimeEncoder)}")
        
        # GCP Logging (if available)
        gcp_logger = self._get_gcp_logger()
        if gcp_logger:
            try:
                gcp_logger.log_text(
                    f"Enhanced Trail Taxonomy Analysis: {json.dumps(taxonomy_log, indent=2, cls=DateTimeEncoder)}",
                    severity="ERROR"
                )
            except Exception as e:
                logger.warning(f"Failed to log to GCP: {e}")
        
        # Log Adaptive Remediation if available
        if adaptive_fix_result:
            adaptive_log = {
                "node": self.name,
                "attempt": attempt,
                "success": adaptive_fix_result.success,
                "strategy_used": adaptive_fix_result.strategy_used.value,
                "confidence": adaptive_fix_result.confidence,
                "execution_time": adaptive_fix_result.execution_time,
                "fix_attempts_count": len(adaptive_fix_result.fix_attempts),
                "state_changes_count": len(adaptive_fix_result.state_changes) if adaptive_fix_result.state_changes else 0,
                "learning_insights": adaptive_fix_result.learning_insights
            }
            
            # Console output for Adaptive Remediation
            print(f"\n🧠 ADAPTIVE REMEDIATION LOG:")
            print(f"   Success: {adaptive_fix_result.success}")
            print(f"   Strategy: {adaptive_fix_result.strategy_used.value}")
            print(f"   Confidence: {adaptive_fix_result.confidence:.2f}")
            print(f"   Execution Time: {adaptive_fix_result.execution_time:.3f}s")
            print(f"   Fix Attempts: {len(adaptive_fix_result.fix_attempts)}")
            print(f"   State Changes: {len(adaptive_fix_result.state_changes) if adaptive_fix_result.state_changes else 0}")
            
            if adaptive_fix_result.learning_insights:
                insights = adaptive_fix_result.learning_insights
                print(f"   Learning: {insights.get('total_attempts', 0)} previous attempts")
                if 'strategy_success_rates' in insights:
                    rates = insights['strategy_success_rates']
                    for strategy, data in rates.items():
                        rate = data.get('rate', 0)
                        print(f"     {strategy}: {rate:.2f} success rate")
            
            # Standard Python logging
            logger.info(f"Adaptive Remediation Result: {json.dumps(adaptive_log, indent=2, cls=DateTimeEncoder)}")
            
            # GCP Logging (if available)
            gcp_logger = self._get_gcp_logger()
            if gcp_logger:
                try:
                    gcp_logger.log_text(
                        f"Adaptive Remediation Result: {json.dumps(adaptive_log, indent=2, cls=DateTimeEncoder)}",
                        severity="INFO"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to GCP: {e}")
        else:
            print(f"\n⚠️  ADAPTIVE REMEDIATION: Disabled or not applicable")
        
        # Log Gemini remediation if available
        if gemini_remediation_result:
            gemini_log = {
                "node": self.name,
                "attempt": attempt,
                "suggestions_count": len(gemini_remediation_result.suggestions),
                "auto_fix_available": gemini_remediation_result.auto_fix_available,
                "reasoning": gemini_remediation_result.reasoning,
                "execution_time": gemini_remediation_result.execution_time,
                "suggestions": [
                    {
                        "action": sugg.action,
                        "confidence": sugg.confidence,
                        "priority": sugg.priority,
                        "estimated_effort": sugg.estimated_effort
                    }
                    for sugg in gemini_remediation_result.suggestions
                ]
            }
            
            # Console output for Gemini remediation
            print(f"\n🤖 GEMINI AI REMEDIATION (Fallback):")
            print(f"   Suggestions: {len(gemini_remediation_result.suggestions)}")
            print(f"   Auto-fix available: {gemini_remediation_result.auto_fix_available}")
            print(f"   Execution time: {gemini_remediation_result.execution_time:.2f}s")
            print(f"   Reasoning: {gemini_remediation_result.reasoning}")
            
            if gemini_remediation_result.suggestions:
                print(f"\n💡 SUGGESTIONS:")
                for i, sugg in enumerate(gemini_remediation_result.suggestions, 1):
                    print(f"   {i}. {sugg.action}")
                    print(f"      Confidence: {sugg.confidence:.2f}")
                    print(f"      Priority: {sugg.priority}")
                    print(f"      Effort: {sugg.estimated_effort}")
                    if sugg.code_example:
                        print(f"      Code: {sugg.code_example}")
            
            # Standard Python logging
            logger.info(f"Gemini Remediation Analysis: {json.dumps(gemini_log, indent=2, cls=DateTimeEncoder)}")
            
            # GCP Logging (if available)
            gcp_logger = self._get_gcp_logger()
            if gcp_logger:
                try:
                    gcp_logger.log_text(
                        f"Gemini Remediation Analysis: {json.dumps(gemini_log, indent=2, cls=DateTimeEncoder)}",
                        severity="INFO"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to GCP: {e}")
        else:
            print(f"\n⚠️  GEMINI REMEDIATION: Disabled or not used")
        
        print("=" * 70)
        print(f"⏱️  Total error handling time: {time.time() - start_time:.2f}s\n")
    
    def _prepare_enhanced_state_updates(self, error_analysis: ErrorAnalysis, 
                                      adaptive_fix_result: Optional[EnhancedRemediationResult],
                                      gemini_remediation_result: Optional[GeminiRemediationResult], 
                                      attempt: int) -> Dict[str, Any]:
        """Prepare enhanced state updates with learning information"""
        updates = {
            "error_details": {
                "category": error_analysis.category.value,
                "severity": error_analysis.severity.value,
                "confidence": error_analysis.confidence,
                "description": error_analysis.description,
                "keywords": error_analysis.keywords,
                "remediation_hints": error_analysis.remediation_hints,
                "root_cause": error_analysis.root_cause,
                "architectural_issue": error_analysis.architectural_issue,
                "suggested_fix_type": error_analysis.suggested_fix_type,
                "attempt": attempt,
                "timestamp": time.time()
            }
        }
        
        if adaptive_fix_result:
            updates["adaptive_remediation_details"] = {
                "success": adaptive_fix_result.success,
                "strategy_used": adaptive_fix_result.strategy_used.value,
                "confidence": adaptive_fix_result.confidence,
                "execution_time": adaptive_fix_result.execution_time,
                "fix_attempts_count": len(adaptive_fix_result.fix_attempts),
                "state_changes": adaptive_fix_result.state_changes,
                "learning_insights": adaptive_fix_result.learning_insights
            }
        
        if gemini_remediation_result:
            updates["gemini_remediation_details"] = {
                "suggestions": [
                    {
                        "action": sugg.action,
                        "description": sugg.description,
                        "confidence": sugg.confidence,
                        "priority": sugg.priority,
                        "estimated_effort": sugg.estimated_effort
                    }
                    for sugg in gemini_remediation_result.suggestions
                ],
                "reasoning": gemini_remediation_result.reasoning,
                "auto_fix_available": gemini_remediation_result.auto_fix_available,
                "execution_time": gemini_remediation_result.execution_time
            }
        
        return updates
    
    def _apply_auto_fix(self, state: GraphLike, auto_fix_code: str) -> GraphLike:
        """Apply automatic fix code to state"""
        try:
            # Create a safe execution environment
            safe_globals = {
                "state": state.copy(),
                "json": json,
                "time": time,
                "datetime": datetime
            }
            
            # Execute the auto-fix code
            exec(auto_fix_code, safe_globals)
            
            # Return the potentially modified state
            return safe_globals.get("state", state)
            
        except Exception as e:
            gcp_logger = self._get_gcp_logger()
            if gcp_logger:
                gcp_logger.log_text(
                    f"Auto-fix application failed for node '{self.name}': {str(e)}",
                    severity="WARNING"
                )
            return state
    
    def _track_enhanced_error_history(self, error_analysis: ErrorAnalysis, 
                                    adaptive_fix_result: Optional[EnhancedRemediationResult],
                                    gemini_remediation_result: Optional[GeminiRemediationResult], 
                                    attempt: int, start_time: float):
        """Track enhanced error history for analytics and learning"""
        error_record = {
            "timestamp": time.time(),
            "node": self.name,
            "attempt": attempt,
            "category": error_analysis.category.value,
            "severity": error_analysis.severity.value,
            "confidence": error_analysis.confidence,
            "root_cause": error_analysis.root_cause,
            "architectural_issue": error_analysis.architectural_issue,
            "suggested_fix_type": error_analysis.suggested_fix_type,
            "processing_time": time.time() - start_time
        }
        
        self.error_history.append(error_record)
        
        if adaptive_fix_result:
            adaptive_record = {
                "timestamp": time.time(),
                "node": self.name,
                "success": adaptive_fix_result.success,
                "strategy_used": adaptive_fix_result.strategy_used.value,
                "confidence": adaptive_fix_result.confidence,
                "execution_time": adaptive_fix_result.execution_time,
                "fix_attempts_count": len(adaptive_fix_result.fix_attempts)
            }
            self.remediation_history.append(adaptive_record)
        
        if gemini_remediation_result:
            gemini_record = {
                "timestamp": time.time(),
                "node": self.name,
                "suggestions_count": len(gemini_remediation_result.suggestions),
                "auto_fix_available": gemini_remediation_result.auto_fix_available,
                "execution_time": gemini_remediation_result.execution_time
            }
            self.remediation_history.append(gemini_record)
    
    def stream(self, state: GraphLike, config: Optional[RunnableConfig] = None) -> Iterable:
        """Enhanced streaming with error handling"""
        if hasattr(self.inner, "stream"):
            try:
                yield from self.inner.stream(state, config=config)
                return
            except Exception as e:
                # Handle streaming errors with enhanced remediation
                error_result = self._handle_error_with_enhanced_remediation(e, 1, state, config)
                yield {
                    "event": "error", 
                    "error": {
                        "node": self.name, 
                        "type": type(e).__name__, 
                        "msg": str(e),
                        "analysis": error_result.get("error_analysis"),
                        "adaptive_result": error_result.get("adaptive_fix_result"),
                        "gemini_result": error_result.get("gemini_remediation_result")
                    }
                }
        
        # Fallback to non-stream invoke
        yield self.invoke(state, config=config)
    
    def get_enhanced_error_analytics(self) -> Dict[str, Any]:
        """Get enhanced error analytics and learning statistics"""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        # Calculate basic statistics
        total_errors = len(self.error_history)
        categories = {}
        severities = {}
        architectural_issues = {}
        root_causes = {}
        
        for error in self.error_history:
            cat = error["category"]
            sev = error["severity"]
            arch_issue = error.get("architectural_issue", "none")
            root_cause = error.get("root_cause", "unknown")
            
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1
            architectural_issues[arch_issue] = architectural_issues.get(arch_issue, 0) + 1
            root_causes[root_cause] = root_causes.get(root_cause, 0) + 1
        
        # Get learning statistics from adaptive remediation engine
        learning_stats = {}
        if self.adaptive_remediation_engine:
            learning_stats = self.adaptive_remediation_engine.get_learning_statistics()
        
        return {
            "total_errors": total_errors,
            "error_categories": categories,
            "error_severities": severities,
            "architectural_issues": architectural_issues,
            "root_causes": root_causes,
            "remediation_stats": {
                "total_remediations": len(self.remediation_history),
                "adaptive_remediation_available": self.adaptive_remediation_engine is not None,
                "gemini_available": self.gemini_remediator is not None,
                "learning_enabled": self.enable_learning
            },
            "learning_statistics": learning_stats,
            "enhanced_features": {
                "architectural_awareness": True,
                "adaptive_strategies": True,
                "learning_capabilities": self.enable_learning,
                "fallback_mechanisms": True
            }
        }
