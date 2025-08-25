"""
Advanced Proactive Remediation Engine for Aigie

This module implements a CodeAct/ReAct agent-based proactive remediation system
that can dynamically analyze errors, generate intelligent fixes, and execute them safely.
"""

import re
import time
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

try:
    from .error_taxonomy import ErrorAnalysis, ErrorCategory
    from .ai_code_generator import AICodeGenerator, AICodeGenerationRequest, AICodeGenerationResponse
except ImportError:
    from error_taxonomy import ErrorAnalysis, ErrorCategory
    from ai_code_generator import AICodeGenerator, AICodeGenerationRequest, AICodeGenerationResponse

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Types of actions the agent can take"""
    ANALYZE_ERROR = "analyze_error"
    GENERATE_FIX = "generate_fix"
    VALIDATE_FIX = "validate_fix"
    EXECUTE_FIX = "execute_fix"
    LEARN = "learn"


@dataclass
class AgentThought:
    """Represents a thought/step in the agent's reasoning process"""
    action: AgentAction
    reasoning: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DynamicFixResult:
    """Result of a dynamic proactive fix attempt"""
    success: bool
    fixed_state: Dict[str, Any]
    generated_code: str
    execution_time: float
    agent_thoughts: List[AgentThought] = field(default_factory=list)
    error_message: Optional[str] = None
    state_changes: Dict[str, Any] = None
    confidence_score: float = 0.0
    fix_strategy: str = ""


class CodeActAgent:
    """CodeAct agent for proactive remediation"""
    
    def __init__(self, max_iterations: int = 5, confidence_threshold: float = 0.7, 
                 ai_model: str = "gemini-2.5-flash", project_id: Optional[str] = None):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.thought_history: List[AgentThought] = []
        self.learning_memory: Dict[str, Any] = {}
        self.ai_code_generator = AICodeGenerator(ai_model, project_id)
    
    def think_and_act(self, error_analysis: ErrorAnalysis, 
                     state: Dict[str, Any], 
                     code_context: str = "") -> DynamicFixResult:
        """Main reasoning loop following CodeAct/ReAct pattern"""
        
        start_time = time.time()
        original_state = state.copy()
        current_state = state.copy()
        
        # Clear thought history for new error
        self.thought_history = []
        
        try:
            # Step 1: Analyze the error deeply
            analysis_thought = self._analyze_error_deeply(error_analysis, current_state, code_context)
            self.thought_history.append(analysis_thought)
            
            if analysis_thought.confidence < self.confidence_threshold:
                return self._create_failure_result(
                    original_state, 
                    "Error analysis confidence too low",
                    start_time
                )
            
            # Step 2: Generate dynamic fix strategy
            strategy_thought = self._generate_fix_strategy(analysis_thought, current_state)
            self.thought_history.append(strategy_thought)
            
            if strategy_thought.confidence < self.confidence_threshold:
                return self._create_failure_result(
                    original_state,
                    "Fix strategy generation confidence too low",
                    start_time
                )
            
            # Step 3: Generate the actual fix code
            code_thought = self._generate_fix_code(strategy_thought, current_state)
            self.thought_history.append(code_thought)
            
            if code_thought.confidence < self.confidence_threshold:
                return self._create_failure_result(
                    original_state,
                    "Code generation confidence too low",
                    start_time
                )
            
            # Step 4: Validate the generated code
            validation_thought = self._validate_generated_code(code_thought, current_state)
            self.thought_history.append(validation_thought)
            
            if not validation_thought.output_data.get("is_valid", False):
                return self._create_failure_result(
                    original_state,
                    f"Generated code validation failed: {validation_thought.output_data.get('validation_error', 'Unknown error')}",
                    start_time
                )
            
            # Step 5: Execute the fix
            execution_thought = self._execute_fix(code_thought, current_state)
            self.thought_history.append(execution_thought)
            
            if not execution_thought.output_data.get("success", False):
                return self._create_failure_result(
                    original_state,
                    f"Fix execution failed: {execution_thought.output_data.get('error', 'Unknown error')}",
                    start_time
                )
            
            # Step 6: Learn from the experience
            learn_thought = self._learn_from_experience(
                error_analysis, 
                execution_thought.output_data.get("fixed_state", current_state),
                success=True
            )
            self.thought_history.append(learn_thought)
            
            # Calculate final result
            execution_time = time.time() - start_time
            fixed_state = execution_thought.output_data.get("fixed_state", current_state)
            state_changes = self._calculate_state_changes(original_state, fixed_state)
            
            return DynamicFixResult(
                success=True,
                fixed_state=fixed_state,
                generated_code=code_thought.output_data.get("generated_code", ""),
                execution_time=execution_time,
                agent_thoughts=self.thought_history.copy(),
                state_changes=state_changes,
                confidence_score=execution_thought.confidence,
                fix_strategy=strategy_thought.output_data.get("strategy", "")
            )
            
        except Exception as e:
            logger.error(f"Agent reasoning failed: {e}")
            return self._create_failure_result(
                original_state,
                f"Agent reasoning failed: {str(e)}",
                start_time
            )
    
    def _analyze_error_deeply(self, error_analysis: ErrorAnalysis, 
                            state: Dict[str, Any], 
                            code_context: str) -> AgentThought:
        """Deep error analysis using AI reasoning"""
        
        # Analyze error patterns, context, and potential root causes
        analysis_prompt = self._build_analysis_prompt(error_analysis, state, code_context)
        
        # Use AI for analysis
        analysis_result = self._analyze_error_with_ai(analysis_prompt)
        
        return AgentThought(
            action=AgentAction.ANALYZE_ERROR,
            reasoning=analysis_result["reasoning"],
            input_data={
                "error_analysis": error_analysis,
                "state": state,
                "code_context": code_context
            },
            output_data=analysis_result,
            confidence=analysis_result.get("confidence", 0.8)
        )
    
    def _generate_fix_strategy(self, analysis_thought: AgentThought, 
                             state: Dict[str, Any]) -> AgentThought:
        """Generate a fix strategy based on error analysis"""
        
        strategy_prompt = self._build_strategy_prompt(analysis_thought, state)
        strategy_result = self._generate_strategy_with_ai(strategy_prompt)
        
        return AgentThought(
            action=AgentAction.GENERATE_FIX,
            reasoning=strategy_result["reasoning"],
            input_data={
                "analysis": analysis_thought.output_data,
                "state": state
            },
            output_data=strategy_result,
            confidence=strategy_result.get("confidence", 0.8)
        )
    
    def _generate_fix_code(self, strategy_thought: AgentThought, 
                          state: Dict[str, Any]) -> AgentThought:
        """Generate actual fix code based on strategy using AI"""
        
        # Create AI code generation request
        # Get the original error analysis from the first thought
        original_error_analysis = None
        for thought in self.thought_history:
            if thought.action == AgentAction.ANALYZE_ERROR:
                original_error_analysis = thought.input_data.get("error_analysis")
                break
        

        
        request = AICodeGenerationRequest(
            error_analysis=original_error_analysis or strategy_thought.input_data.get("analysis", {}),
            state=state,
            code_context="",  # Could be enhanced with actual code context
            strategy=strategy_thought.output_data.get("strategy", ""),
            constraints={
                "safe_only": True,
                "state_only": True,
                "logging_required": True
            }
        )
        
        # Generate code using AI
        ai_response = self.ai_code_generator.generate_fix_code(request)
        
        return AgentThought(
            action=AgentAction.GENERATE_FIX,
            reasoning=ai_response.reasoning,
            input_data={
                "strategy": strategy_thought.output_data,
                "state": state,
                "ai_request": request
            },
            output_data={
                "generated_code": ai_response.generated_code,
                "confidence": ai_response.confidence,
                "reasoning": ai_response.reasoning,
                "safety_score": ai_response.safety_score,
                "execution_plan": ai_response.execution_plan
            },
            confidence=ai_response.confidence
        )
    
    def _validate_generated_code(self, code_thought: AgentThought, 
                               state: Dict[str, Any]) -> AgentThought:
        """Validate the generated code for safety and correctness"""
        
        generated_code = code_thought.output_data.get("generated_code", "")
        
        # Validate code safety
        safety_validator = CodeSafetyValidator()
        is_safe, safety_message = safety_validator.validate_code(generated_code)
        
        # Validate code logic
        logic_validator = CodeLogicValidator()
        is_logical, logic_message = logic_validator.validate_code(generated_code, state)
        
        is_valid = is_safe and is_logical
        validation_message = f"Safety: {safety_message}, Logic: {logic_message}"
        
        return AgentThought(
            action=AgentAction.VALIDATE_FIX,
            reasoning=f"Code validation: {validation_message}",
            input_data={
                "generated_code": generated_code,
                "state": state
            },
            output_data={
                "is_valid": is_valid,
                "safety_valid": is_safe,
                "logic_valid": is_logical,
                "validation_error": None if is_valid else validation_message
            },
            confidence=1.0 if is_valid else 0.0
        )
    
    def _execute_fix(self, code_thought: AgentThought, 
                    state: Dict[str, Any]) -> AgentThought:
        """Execute the validated fix code"""
        
        generated_code = code_thought.output_data.get("generated_code", "")
        safe_executor = SafeCodeExecutor()
        
        try:
            result = safe_executor.execute_code(generated_code, state)
            
            return AgentThought(
                action=AgentAction.EXECUTE_FIX,
                reasoning=f"Fix executed successfully in {result.execution_time:.3f}s",
                input_data={
                    "generated_code": generated_code,
                    "state": state
                },
                output_data={
                    "success": result.success,
                    "fixed_state": result.fixed_state,
                    "execution_time": result.execution_time,
                    "error": result.error_message
                },
                confidence=1.0 if result.success else 0.0
            )
            
        except Exception as e:
            return AgentThought(
                action=AgentAction.EXECUTE_FIX,
                reasoning=f"Fix execution failed: {str(e)}",
                input_data={
                    "generated_code": generated_code,
                    "state": state
                },
                output_data={
                    "success": False,
                    "error": str(e)
                },
                confidence=0.0
            )
    
    def _learn_from_experience(self, error_analysis: ErrorAnalysis, 
                             fixed_state: Dict[str, Any], 
                             success: bool) -> AgentThought:
        """Learn from the experience to improve future fixes"""
        
        # Store learning in memory
        learning_key = f"{error_analysis.category.value}_{error_analysis.description[:50]}"
        
        if learning_key not in self.learning_memory:
            self.learning_memory[learning_key] = {
                "attempts": 0,
                "successes": 0,
                "strategies": [],
                "patterns": []
            }
        
        self.learning_memory[learning_key]["attempts"] += 1
        if success:
            self.learning_memory[learning_key]["successes"] += 1
        
        return AgentThought(
            action=AgentAction.LEARN,
            reasoning=f"Learned from {learning_key}: success={success}",
            input_data={
                "error_analysis": error_analysis,
                "success": success
            },
            output_data={
                "learning_key": learning_key,
                "memory_updated": True
            },
            confidence=1.0
        )
    
    def _build_analysis_prompt(self, error_analysis: ErrorAnalysis, 
                             state: Dict[str, Any], 
                             code_context: str) -> str:
        """Build prompt for deep error analysis"""
        return f"""
Analyze this error deeply and identify the root cause:

ERROR:
- Category: {error_analysis.category.value}
- Description: {error_analysis.description}
- Keywords: {error_analysis.keywords}
- Severity: {error_analysis.severity.value}

STATE:
{json.dumps(state, indent=2, cls=DateTimeEncoder)}

CONTEXT:
{code_context}

Provide:
1. Root cause analysis
2. Impact assessment
3. Potential fix approaches
4. Confidence level (0-1)
"""

    def _build_strategy_prompt(self, analysis_thought: AgentThought, 
                             state: Dict[str, Any]) -> str:
        """Build prompt for fix strategy generation"""
        return f"""
Based on this error analysis, generate a fix strategy:

ANALYSIS:
{json.dumps(analysis_thought.output_data, indent=2, cls=DateTimeEncoder)}

STATE:
{json.dumps(state, indent=2, cls=DateTimeEncoder)}

Generate:
1. Fix strategy description
2. Approach (additive/modifying/rollback)
3. Risk assessment
4. Confidence level (0-1)
"""

    def _build_code_generation_prompt(self, strategy_thought: AgentThought, 
                                    state: Dict[str, Any]) -> str:
        """Build prompt for code generation"""
        return f"""
Generate Python code to fix this issue:

STRATEGY:
{json.dumps(strategy_thought.output_data, indent=2, cls=DateTimeEncoder)}

STATE:
{json.dumps(state, indent=2, cls=DateTimeEncoder)}

Requirements:
1. Safe code that only manipulates the state dictionary
2. Handle edge cases and errors gracefully
3. Provide meaningful logging
4. Follow Python best practices
5. Confidence level (0-1)
"""

    def _analyze_error_with_ai(self, prompt: str) -> Dict[str, Any]:
        """Analyze error using AI"""
        try:
            # Use the AI code generator for analysis
            request = AICodeGenerationRequest(
                error_analysis={"prompt": prompt},
                state={},
                code_context="",
                strategy="analyze_error",
                constraints={"analysis_only": True}
            )
            
            response = self.ai_code_generator.generate_fix_code(request)
            
            # Parse the analysis from the response
            return {
                "root_cause": self._extract_root_cause(response.reasoning),
                "impact": self._extract_impact(response.reasoning),
                "fix_approaches": self._extract_fix_approaches(response.reasoning),
                "confidence": response.confidence,
                "reasoning": response.reasoning
            }
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}, using fallback")
            return self._fallback_analysis(prompt)
    
    def _generate_strategy_with_ai(self, prompt: str) -> Dict[str, Any]:
        """Generate strategy using AI"""
        try:
            # Use the AI code generator for strategy
            request = AICodeGenerationRequest(
                error_analysis={"prompt": prompt},
                state={},
                code_context="",
                strategy="generate_strategy",
                constraints={"strategy_only": True}
            )
            
            response = self.ai_code_generator.generate_fix_code(request)
            
            return {
                "strategy": self._extract_strategy(response.reasoning),
                "approach": self._extract_approach(response.reasoning),
                "risk": self._extract_risk(response.reasoning),
                "confidence": response.confidence,
                "reasoning": response.reasoning
            }
        except Exception as e:
            logger.warning(f"AI strategy generation failed: {e}, using fallback")
            return self._fallback_strategy(prompt)
    
    def _extract_root_cause(self, reasoning: str) -> str:
        """Extract root cause from AI reasoning"""
        if "missing" in reasoning.lower():
            return "Missing required field in state"
        elif "type" in reasoning.lower():
            return "Type conversion error"
        elif "validation" in reasoning.lower():
            return "Validation error"
        else:
            return "Unknown error pattern"
    
    def _extract_impact(self, reasoning: str) -> str:
        """Extract impact from AI reasoning"""
        if "high" in reasoning.lower() or "critical" in reasoning.lower():
            return "High - prevents workflow execution"
        elif "medium" in reasoning.lower():
            return "Medium - may cause issues"
        else:
            return "Low - minor issue"
    
    def _extract_fix_approaches(self, reasoning: str) -> List[str]:
        """Extract fix approaches from AI reasoning"""
        approaches = []
        if "add" in reasoning.lower() or "missing" in reasoning.lower():
            approaches.append("Add missing field")
        if "convert" in reasoning.lower() or "type" in reasoning.lower():
            approaches.append("Convert data type")
        if "default" in reasoning.lower():
            approaches.append("Provide default value")
        if not approaches:
            approaches.append("Generic fix")
        return approaches
    
    def _extract_strategy(self, reasoning: str) -> str:
        """Extract strategy from AI reasoning"""
        if "add" in reasoning.lower():
            return "Add missing field with appropriate default value"
        elif "convert" in reasoning.lower():
            return "Convert field to appropriate type"
        else:
            return "Apply generic fix strategy"
    
    def _extract_approach(self, reasoning: str) -> str:
        """Extract approach from AI reasoning"""
        if "add" in reasoning.lower():
            return "additive"
        elif "modify" in reasoning.lower():
            return "modifying"
        else:
            return "generic"
    
    def _extract_risk(self, reasoning: str) -> str:
        """Extract risk from AI reasoning"""
        if "safe" in reasoning.lower() or "low" in reasoning.lower():
            return "low"
        elif "medium" in reasoning.lower():
            return "medium"
        else:
            return "high"
    
    def _fallback_analysis(self, prompt: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        return {
            "root_cause": "Unknown error pattern",
            "impact": "Unknown impact",
            "fix_approaches": ["Generic fix"],
            "confidence": 0.5,
            "reasoning": "Fallback analysis due to AI unavailability"
        }
    
    def _fallback_strategy(self, prompt: str) -> Dict[str, Any]:
        """Fallback strategy when AI fails"""
        return {
            "strategy": "Apply generic fix strategy",
            "approach": "generic",
            "risk": "medium",
            "confidence": 0.5,
            "reasoning": "Fallback strategy due to AI unavailability"
        }



    def _calculate_state_changes(self, original: Dict[str, Any], 
                               modified: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between original and modified states"""
        changes = {}
        
        for key in modified:
            if key not in original:
                changes[f"added_{key}"] = modified[key]
            elif original[key] != modified[key]:
                changes[f"modified_{key}"] = {
                    "from": original[key],
                    "to": modified[key]
                }
        
        for key in original:
            if key not in modified:
                changes[f"removed_{key}"] = original[key]
        
        return changes

    def _create_failure_result(self, original_state: Dict[str, Any], 
                             error_message: str, 
                             start_time: float) -> DynamicFixResult:
        """Create a failure result"""
        return DynamicFixResult(
            success=False,
            fixed_state=original_state,
            generated_code="",
            execution_time=time.time() - start_time,
            agent_thoughts=self.thought_history.copy(),
            error_message=error_message,
            confidence_score=0.0
        )


class CodeSafetyValidator:
    """Validates generated code for safety"""
    
    def __init__(self):
        self.forbidden_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'subprocess',
            r'os\.',
            r'sys\.',
            r'globals\s*\(',
            r'locals\s*\(',
        ]
        
        self.allowed_patterns = [
            r'state\[.*\]\s*=.*',
            r'if.*in state.*',
            r'print\s*\(',
            r'time\.sleep\s*\(',
            r'uuid\.uuid4\s*\(',
            r'datetime\.now\s*\(',
            r'str\s*\(',
            r'int\s*\(',
            r'float\s*\(',
            r'bool\s*\(',
        ]
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code for safety"""
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern found: {pattern}"
        
        # Check for allowed patterns
        has_allowed = False
        for pattern in self.allowed_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                has_allowed = True
                break
        
        if not has_allowed:
            return False, "No allowed patterns found"
        
        return True, "Code is safe"


class CodeLogicValidator:
    """Validates generated code for logical correctness"""
    
    def validate_code(self, code: str, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate code logic"""
        # Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for basic logical issues
        if 'state[' not in code:
            return False, "Code doesn't manipulate state"
        
        if 'print(' not in code and 'logger.' not in code:
            return False, "Code doesn't provide logging"
        
        return True, "Code logic is valid"


class SafeCodeExecutor:
    """Safely executes generated code"""
    
    def __init__(self):
        self.validator = CodeSafetyValidator()
    
    def execute_code(self, code: str, state: Dict[str, Any]) -> DynamicFixResult:
        """Execute code safely"""
        start_time = time.time()
        original_state = state.copy()
        
        try:
            # Validate code
            is_safe, safety_message = self.validator.validate_code(code)
            if not is_safe:
                return DynamicFixResult(
                    success=False,
                    fixed_state=original_state,
                    generated_code=code,
                    execution_time=time.time() - start_time,
                    error_message=f"Code validation failed: {safety_message}"
                )
            
            # Create safe execution environment
            safe_globals = {
                'state': state.copy(),
                'json': json,
                'time': time,
                'datetime': datetime,
                'uuid': uuid,
                'print': lambda *args: logger.info(" ".join(map(str, args)))
            }
            
            # Execute code
            exec(code, safe_globals)
            fixed_state = safe_globals['state']
            
            execution_time = time.time() - start_time
            
            return DynamicFixResult(
                success=True,
                fixed_state=fixed_state,
                generated_code=code,
                execution_time=execution_time,
                state_changes=self._calculate_changes(original_state, fixed_state)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return DynamicFixResult(
                success=False,
                fixed_state=original_state,
                generated_code=code,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _calculate_changes(self, original: Dict[str, Any], 
                          modified: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate state changes"""
        changes = {}
        
        for key in modified:
            if key not in original:
                changes[f"added_{key}"] = modified[key]
            elif original[key] != modified[key]:
                changes[f"modified_{key}"] = {
                    "from": original[key],
                    "to": modified[key]
                }
        
        return changes


class AdvancedProactiveRemediationEngine:
    """Advanced proactive remediation engine using CodeAct/ReAct agent"""
    
    def __init__(self, max_iterations: int = 5, confidence_threshold: float = 0.7,
                 ai_model: str = "gemini-2.5-flash", project_id: Optional[str] = None):
        self.agent = CodeActAgent(max_iterations, confidence_threshold, ai_model, project_id)
        self.learning_memory = {}
    
    def can_fix_proactively(self, error_analysis: ErrorAnalysis) -> bool:
        """Determine if this error can be fixed proactively"""
        # More sophisticated logic based on error patterns and learning
        fixable_categories = [
            ErrorCategory.INPUT_ERROR,
            ErrorCategory.PROCESSING_ERROR
        ]
        
        # Check learning memory for similar errors
        learning_key = f"{error_analysis.category.value}_{error_analysis.description[:50]}"
        if learning_key in self.agent.learning_memory:
            memory = self.agent.learning_memory[learning_key]
            success_rate = memory["successes"] / memory["attempts"] if memory["attempts"] > 0 else 0
            return success_rate > 0.5  # Only if we have >50% success rate
        
        return error_analysis.category in fixable_categories
    
    def apply_proactive_remediation(self, error_analysis: ErrorAnalysis, 
                                  state: Dict[str, Any],
                                  code_context: str = "") -> DynamicFixResult:
        """Apply advanced proactive remediation"""
        
        if not self.can_fix_proactively(error_analysis):
            return DynamicFixResult(
                success=False,
                fixed_state=state,
                generated_code="",
                execution_time=0.0,
                error_message="Error type not supported for proactive remediation"
            )
        
        # Use the CodeAct agent to think and act
        result = self.agent.think_and_act(error_analysis, state, code_context)
        
        # Log the result
        if result.success:
            logger.info(f"Advanced proactive remediation successful: {error_analysis.description}")
            logger.info(f"Strategy: {result.fix_strategy}")
            logger.info(f"Confidence: {result.confidence_score:.2f}")
        else:
            logger.warning(f"Advanced proactive remediation failed: {result.error_message}")
        
        return result
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning memory"""
        return {
            "total_patterns": len(self.agent.learning_memory),
            "patterns": self.agent.learning_memory,
            "success_rates": {
                key: memory["successes"] / memory["attempts"] if memory["attempts"] > 0 else 0
                for key, memory in self.agent.learning_memory.items()
            }
        }
