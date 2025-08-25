"""
AI-Powered Code Generator for Proactive Remediation

This module integrates with AI models (Gemini/Claude) to generate dynamic fix code
based on error analysis and context.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AICodeGenerationRequest:
    """Request for AI code generation"""
    error_analysis: Dict[str, Any]
    state: Dict[str, Any]
    code_context: str
    strategy: str
    constraints: Dict[str, Any]


@dataclass
class AICodeGenerationResponse:
    """Response from AI code generation"""
    generated_code: str
    confidence: float
    reasoning: str
    safety_score: float
    execution_plan: str


class AICodeGenerator:
    """AI-powered code generator using Gemini/Claude"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", project_id: Optional[str] = None):
        self.model_name = model_name
        self.project_id = project_id
        self.gemini_client = None
        self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize the AI client (Gemini or Claude)"""
        try:
            # Try to initialize Gemini
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            if self.project_id:
                vertexai.init(project=self.project_id)
            
            self.gemini_client = GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini AI client with model: {self.model_name}")
            
        except ImportError:
            logger.warning("Vertex AI not available, falling back to simulation mode")
            self.gemini_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}, falling back to simulation mode")
            self.gemini_client = None
    
    def generate_fix_code(self, request: AICodeGenerationRequest) -> AICodeGenerationResponse:
        """Generate fix code using AI"""
        
        if self.gemini_client:
            return self._generate_with_gemini(request)
        else:
            return self._generate_simulated(request)
    
    def _generate_with_gemini(self, request: AICodeGenerationRequest) -> AICodeGenerationResponse:
        """Generate code using Gemini AI"""
        
        prompt = self._build_gemini_prompt(request)
        
        try:
            response = self.gemini_client.generate_content(prompt)
            
            # Parse the response
            generated_code = self._extract_code_from_response(response.text)
            confidence = self._extract_confidence_from_response(response.text)
            reasoning = self._extract_reasoning_from_response(response.text)
            
            return AICodeGenerationResponse(
                generated_code=generated_code,
                confidence=confidence,
                reasoning=reasoning,
                safety_score=self._calculate_safety_score(generated_code),
                execution_plan=self._extract_execution_plan_from_response(response.text)
            )
            
        except Exception as e:
            logger.error(f"Gemini code generation failed: {e}")
            return self._generate_simulated(request)
    
    def _generate_simulated(self, request: AICodeGenerationRequest) -> AICodeGenerationResponse:
        """Generate simulated code when AI is not available"""
        
        # Check if this is an analysis or strategy request
        if request.constraints.get("analysis_only", False):
            return self._generate_analysis_response(request)
        elif request.constraints.get("strategy_only", False):
            return self._generate_strategy_response(request)
        
        # Analyze the error and generate appropriate code
        # Handle both dictionary and object formats
        if isinstance(request.error_analysis, dict):
            error_desc = request.error_analysis.get("description", "").lower()
        else:
            error_desc = getattr(request.error_analysis, "description", "").lower()
        state = request.state
        

        
        # Check for specific error patterns and generate appropriate fixes
        if "missing" in error_desc and "field" in error_desc:
            # Generate missing field fix
            field_name = self._extract_field_name(error_desc)
            if field_name:
                generated_code = self._generate_missing_field_code(field_name)
                reasoning = f"Detected missing field '{field_name}', generating code to add it with appropriate default"
            else:
                generated_code = self._generate_generic_fix_code()
                reasoning = "Detected missing field error, generating generic fix"
        
        elif "must be" in error_desc and ("int" in error_desc or "integer" in error_desc):
            # Generate type conversion fix for integer
            field_name = self._extract_field_name(error_desc)
            if field_name:
                generated_code = self._generate_type_conversion_code(field_name, "int")
                reasoning = f"Detected type error for field '{field_name}', generating conversion to int"
            else:
                generated_code = self._generate_generic_fix_code()
                reasoning = "Detected type error, generating generic fix"
        
        elif "validation" in error_desc or "invalid" in error_desc:
            # Generate validation fix
            generated_code = self._generate_validation_fix_code()
            reasoning = "Detected validation error, generating validation fix"
        
        else:
            # Generate generic fix
            generated_code = self._generate_generic_fix_code()
            reasoning = "Unknown error type, generating generic fix"
        
        return AICodeGenerationResponse(
            generated_code=generated_code,
            confidence=0.7,  # Lower confidence for simulated generation
            reasoning=reasoning,
            safety_score=0.9,  # High safety for our predefined patterns
            execution_plan="Simulated execution plan"
        )
    
    def _generate_analysis_response(self, request: AICodeGenerationRequest) -> AICodeGenerationResponse:
        """Generate analysis response"""
        prompt = request.error_analysis.get("prompt", "")
        
        if "missing" in prompt.lower():
            reasoning = "Root cause: Missing required field in state. Impact: High - prevents workflow execution. Fix approaches: Add missing field, Provide default value."
        elif "type" in prompt.lower():
            reasoning = "Root cause: Type conversion error. Impact: Medium - may cause processing issues. Fix approaches: Convert data type, Provide default value."
        elif "validation" in prompt.lower():
            reasoning = "Root cause: Validation error. Impact: Medium - may cause downstream issues. Fix approaches: Fix validation, Provide default value."
        else:
            reasoning = "Root cause: Unknown error pattern. Impact: Unknown. Fix approaches: Generic fix."
        
        return AICodeGenerationResponse(
            generated_code="",  # No code for analysis
            confidence=0.8,
            reasoning=reasoning,
            safety_score=1.0,
            execution_plan="Analysis completed"
        )
    
    def _generate_strategy_response(self, request: AICodeGenerationRequest) -> AICodeGenerationResponse:
        """Generate strategy response"""
        prompt = request.error_analysis.get("prompt", "")
        
        if "missing" in prompt.lower():
            reasoning = "Strategy: Add missing field with appropriate default value. Approach: additive. Risk: low. This is safe and addresses the root cause."
        elif "type" in prompt.lower():
            reasoning = "Strategy: Convert field to appropriate type. Approach: modifying. Risk: low. Type conversion is generally safe."
        elif "validation" in prompt.lower():
            reasoning = "Strategy: Fix validation issues. Approach: modifying. Risk: medium. Validation fixes may have side effects."
        else:
            reasoning = "Strategy: Apply generic fix strategy. Approach: generic. Risk: medium. Generic approach for unknown issues."
        
        return AICodeGenerationResponse(
            generated_code="",  # No code for strategy
            confidence=0.8,
            reasoning=reasoning,
            safety_score=1.0,
            execution_plan="Strategy generated"
        )
    
    def _build_gemini_prompt(self, request: AICodeGenerationRequest) -> str:
        """Build a comprehensive prompt for Gemini"""
        
        return f"""
You are an expert Python developer tasked with generating safe, effective code to fix runtime errors in AI workflows.

ERROR ANALYSIS:
{request.error_analysis}

CURRENT STATE:
{request.state}

CODE CONTEXT:
{request.code_context}

FIX STRATEGY:
{request.strategy}

CONSTRAINTS:
- Only manipulate the 'state' dictionary
- Use safe operations only (no file I/O, network calls, etc.)
- Include proper error handling
- Add meaningful logging with print() statements
- Follow Python best practices
- Handle edge cases gracefully

REQUIRED OUTPUT FORMAT:
```python
# Generated fix code here
```

CONFIDENCE: [0.0-1.0]
REASONING: [Explanation of the fix approach]
EXECUTION_PLAN: [Step-by-step plan for the fix]

Generate the fix code:
"""
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from AI response"""
        # Look for code blocks
        import re
        
        # Try to find Python code blocks
        code_pattern = r'```python\s*(.*?)\s*```'
        match = re.search(code_pattern, response_text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for any code-like content
        lines = response_text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('#') or 'state[' in line or 'print(' in line:
                in_code = True
            if in_code and line.strip():
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Last resort: return a safe default
        return """
# Default safe fix
if 'proactive_fix_applied' not in state:
    state['proactive_fix_applied'] = True
    state['fix_timestamp'] = datetime.now().isoformat()
    print("Applied default proactive fix")
"""
    
    def _extract_confidence_from_response(self, response_text: str) -> float:
        """Extract confidence score from AI response"""
        import re
        
        confidence_pattern = r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)'
        match = re.search(confidence_pattern, response_text)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return 0.7  # Default confidence
    
    def _extract_reasoning_from_response(self, response_text: str) -> str:
        """Extract reasoning from AI response"""
        import re
        
        reasoning_pattern = r'REASONING:\s*(.*?)(?:\n|$)'
        match = re.search(reasoning_pattern, response_text)
        
        if match:
            return match.group(1).strip()
        
        return "AI-generated fix based on error analysis"
    
    def _extract_execution_plan_from_response(self, response_text: str) -> str:
        """Extract execution plan from AI response"""
        import re
        
        plan_pattern = r'EXECUTION_PLAN:\s*(.*?)(?:\n|$)'
        match = re.search(plan_pattern, response_text)
        
        if match:
            return match.group(1).strip()
        
        return "Execute generated code in safe environment"
    
    def _calculate_safety_score(self, code: str) -> float:
        """Calculate safety score for generated code"""
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'subprocess',
        ]
        
        safe_patterns = [
            r'state\[.*\]\s*=.*',
            r'print\s*\(',
            r'if.*in state.*',
            r'uuid\.uuid4\s*\(',
            r'datetime\.now\s*\(',
        ]
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return 0.0
        
        # Check for safe patterns
        safe_count = 0
        for pattern in safe_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                safe_count += 1
        
        return min(1.0, safe_count / len(safe_patterns))
    
    def _extract_field_name(self, error_description: str) -> Optional[str]:
        """Extract field name from error description"""
        import re
        
        patterns = [
            r"field ['\"]([^'\"]+)['\"]",
            r"([a-zA-Z_][a-zA-Z0-9_]*) is required",
            r"missing ([a-zA-Z_][a-zA-Z0-9_]*)",
            r"([a-zA-Z_][a-zA-Z0-9_]*) must be",
            r"Field ['\"]([^'\"]+)['\"]",
            r"([a-zA-Z_][a-zA-Z0-9_]*) must be an",
            r"([a-zA-Z_][a-zA-Z0-9_]*) format",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_description)
            if match:
                return match.group(1)
        
        return None
    
    def _generate_missing_field_code(self, field_name: str) -> str:
        """Generate code for missing field fix"""
        return f"""
# Fix missing field: {field_name}
if '{field_name}' not in state or state['{field_name}'] is None:
    # Generate appropriate default value
    if '{field_name}' in ['customer_id', 'user_id', 'id']:
        state['{field_name}'] = str(uuid.uuid4())
    elif '{field_name}' in ['created_at', 'timestamp', 'date']:
        state['{field_name}'] = datetime.now().isoformat()
    elif '{field_name}' in ['status', 'state']:
        state['{field_name}'] = 'pending'
    elif '{field_name}' in ['email']:
        state['{field_name}'] = 'default@example.com'
    elif '{field_name}' in ['name', 'title']:
        state['{field_name}'] = 'Default Name'
    else:
        state['{field_name}'] = ''
    
    print(f"Proactively added missing field: {field_name} = {{state['{field_name}']}}")
"""
    
    def _generate_type_conversion_code(self, field_name: str, expected_type: str) -> str:
        """Generate code for type conversion fix"""
        return f"""
# Fix type conversion for field: {field_name}
try:
    if '{field_name}' in state and state['{field_name}'] is not None:
        if '{expected_type}' == 'int':
            state['{field_name}'] = int(state['{field_name}'])
        elif '{expected_type}' == 'str':
            state['{field_name}'] = str(state['{field_name}'])
        elif '{expected_type}' == 'float':
            state['{field_name}'] = float(state['{field_name}'])
        elif '{expected_type}' == 'bool':
            state['{field_name}'] = bool(state['{field_name}'])
        
        print(f"Proactively converted {field_name} to {expected_type}: {{state['{field_name}']}}")
except (ValueError, TypeError) as e:
    # Provide safe default if conversion fails
    if '{expected_type}' == 'int':
        state['{field_name}'] = 0
    elif '{expected_type}' == 'str':
        state['{field_name}'] = ''
    elif '{expected_type}' == 'float':
        state['{field_name}'] = 0.0
    elif '{expected_type}' == 'bool':
        state['{field_name}'] = False
    
    print(f"Proactively set default value for {field_name}: {{state['{field_name}']}}")
"""
    
    def _generate_validation_fix_code(self) -> str:
        """Generate code for validation fix"""
        return f"""
# Fix validation errors by providing default values
validation_defaults = {{
    'email': 'default@example.com',
    'phone': '+1234567890',
    'url': 'https://example.com',
    'date': datetime.now().isoformat(),
    'status': 'pending'
}}

for field, default_value in validation_defaults.items():
    if field in state and (state[field] is None or state[field] == ''):
        state[field] = default_value
        print(f"Proactively set validation default for {{field}}: {{default_value}}")

print("Proactively applied validation fixes")
"""
    
    def _generate_generic_fix_code(self) -> str:
        """Generate generic fix code"""
        return f"""
# Generic proactive fix
if 'proactive_fix_count' not in state:
    state['proactive_fix_count'] = 0

state['proactive_fix_count'] += 1
state['last_fix_time'] = datetime.now().isoformat()
state['fix_applied'] = True

print(f"Applied generic proactive fix (attempt {{state['proactive_fix_count']}})")
"""
