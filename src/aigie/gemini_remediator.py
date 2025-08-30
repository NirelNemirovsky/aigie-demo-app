"""
Google Gen AI Integration for Intelligent Error Remediation

This module provides AI-powered error analysis and remediation suggestions using Google's Gemini models
through the new Google Gen AI SDK, supporting both Gemini Developer API and Vertex AI services.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import functools

try:
    import google.genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gen AI SDK not available. Install with: pip install google-genai>=0.3.0")

from .error_taxonomy import ErrorAnalysis, ErrorCategory, ErrorSeverity


@dataclass
class RemediationSuggestion:
    """Structured remediation suggestion from Gemini"""
    action: str
    description: str
    confidence: float
    code_example: Optional[str] = None
    priority: str = "medium"
    estimated_effort: str = "low"


@dataclass
class GeminiRemediationResult:
    """Complete remediation result from Gemini analysis"""
    error_analysis: ErrorAnalysis
    suggestions: List[RemediationSuggestion]
    reasoning: str
    auto_fix_available: bool
    auto_fix_code: Optional[str] = None
    execution_time: float = 0.0


class GeminiRemediator:
    """
    AI-powered error remediation using Google's Gemini models via Google Gen AI SDK
    """
    
    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None, 
                 location: str = "us-central1", use_vertex_ai: bool = False):
        """
        Initialize Gemini remediator
        
        Args:
            api_key: Google API key for Gemini Developer API (if None, will try to auto-detect)
            project_id: GCP project ID for Vertex AI (if use_vertex_ai=True)
            location: GCP location for Vertex AI
            use_vertex_ai: Whether to use Vertex AI instead of Gemini Developer API
        """
        self.api_key = api_key or self._get_default_api_key()
        self.project_id = project_id or self._get_default_project()
        self.location = location
        self.use_vertex_ai = use_vertex_ai
        self.client = None
        self.model_name = "gemini-2.0-flash-exp"
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        if GEMINI_AVAILABLE:
            self._initialize_gemini()
    
    def _get_default_api_key(self) -> Optional[str]:
        """Try to get default Google API key from environment"""
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    def _get_default_project(self) -> Optional[str]:
        """Try to get default GCP project from environment"""
        return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    
    def _initialize_gemini(self):
        """Initialize Gemini model using Google Gen AI SDK"""
        try:
            if self.use_vertex_ai:
                # Use Vertex AI service
                if not self.project_id:
                    raise ValueError("Project ID required for Vertex AI")
                
                # Configure for Vertex AI
                self.client = google.genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                print(f"✅ Gemini initialized for Vertex AI project: {self.project_id}")
                print(f"📍 Location: {self.location}")
            else:
                # Use Gemini Developer API
                if not self.api_key:
                    raise ValueError("API key required for Gemini Developer API")
                
                # Configure for Gemini Developer API
                self.client = google.genai.Client(
                    api_key=self.api_key
                )
                print(f"✅ Gemini initialized for Developer API")
            
            print(f"🤖 Model: {self.model_name}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini: {e}")
            self.client = None
    
    def analyze_and_remediate(self, error_analysis: ErrorAnalysis, 
                            node_context: Dict[str, Any] = None) -> GeminiRemediationResult:
        """
        Analyze error and provide AI-powered remediation suggestions
        
        Args:
            error_analysis: Error analysis from Trail Taxonomy classifier
            node_context: Additional context about the node and execution
            
        Returns:
            GeminiRemediationResult with suggestions and reasoning
        """
        start_time = time.time()
        
        if not self.client:
            return self._fallback_remediation(error_analysis, node_context)
        
        # Create cache key
        cache_key = self._create_cache_key(error_analysis, node_context)
        
        # Check cache first
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                return cached_result["result"]
        
        try:
            # Prepare prompt for Gemini
            prompt = self._create_remediation_prompt(error_analysis, node_context)
            
            # Get response from Gemini using Google Gen AI SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}]}]
            )
            
            # Parse response using the new response format
            result = self._parse_gemini_response(response, error_analysis)
            result.execution_time = time.time() - start_time
            
            # Cache result
            self.cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Gemini analysis failed: {e}")
            return self._fallback_remediation(error_analysis, node_context)
    
    def _create_remediation_prompt(self, error_analysis: ErrorAnalysis, 
                                  node_context: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for Gemini analysis"""
        
        context_info = ""
        if node_context:
            context_info = f"""
Node Context:
- Node Name: {node_context.get('node_name', 'Unknown')}
- Node Type: {node_context.get('node_type', 'Unknown')}
- Execution Environment: {node_context.get('environment', 'Unknown')}
- State Keys: {list(node_context.get('state_keys', []))}
"""
        
        prompt = f"""
You are an expert AI system error remediation specialist. Analyze the following error and provide intelligent remediation suggestions.

ERROR ANALYSIS:
{json.dumps(asdict(error_analysis), indent=2)}

{context_info}

TASK:
1. Analyze the error based on the Trail Taxonomy classification
2. Provide 3-5 specific, actionable remediation suggestions
3. For each suggestion, provide:
   - Action: What to do
   - Description: Why this helps
   - Confidence: 0.0-1.0
   - Code Example: If applicable
   - Priority: low/medium/high
   - Estimated Effort: low/medium/high
4. Determine if an automatic fix is possible
5. Provide reasoning for your suggestions

RESPONSE FORMAT (JSON):
{{
    "suggestions": [
        {{
            "action": "string",
            "description": "string", 
            "confidence": 0.0-1.0,
            "code_example": "string or null",
            "priority": "low/medium/high",
            "estimated_effort": "low/medium/high"
        }}
    ],
    "reasoning": "string",
    "auto_fix_available": true/false,
    "auto_fix_code": "string or null"
}}

Focus on practical, implementable solutions that address the root cause of the error.
"""
        return prompt
    
    def _parse_gemini_response(self, response, error_analysis: ErrorAnalysis) -> GeminiRemediationResult:
        """Parse Gemini response into structured format using Google Gen AI SDK response object"""
        try:
            # Extract text from the response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
                    else:
                        response_text = str(candidate.content)
                else:
                    response_text = str(candidate)
            else:
                response_text = str(response)
            
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                # Fallback parsing
                data = self._fallback_parse_response(response_text)
            
            # Parse suggestions
            suggestions = []
            for sugg_data in data.get("suggestions", []):
                suggestion = RemediationSuggestion(
                    action=sugg_data.get("action", ""),
                    description=sugg_data.get("description", ""),
                    confidence=float(sugg_data.get("confidence", 0.5)),
                    code_example=sugg_data.get("code_example"),
                    priority=sugg_data.get("priority", "medium"),
                    estimated_effort=sugg_data.get("estimated_effort", "low")
                )
                suggestions.append(suggestion)
            
            return GeminiRemediationResult(
                error_analysis=error_analysis,
                suggestions=suggestions,
                reasoning=data.get("reasoning", ""),
                auto_fix_available=data.get("auto_fix_available", False),
                auto_fix_code=data.get("auto_fix_code"),
                execution_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            print(f"⚠️ Failed to parse Gemini response: {e}")
            return self._fallback_remediation(error_analysis, {})
    
    def _fallback_parse_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        # Simple keyword-based parsing
        suggestions = []
        
        # Look for action patterns
        lines = response_text.split('\n')
        current_suggestion = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Action:') or line.startswith('Suggestion:'):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                current_suggestion = {"action": line.split(':', 1)[1].strip()}
            elif line.startswith('Description:'):
                current_suggestion["description"] = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                try:
                    current_suggestion["confidence"] = float(line.split(':', 1)[1].strip())
                except:
                    current_suggestion["confidence"] = 0.5
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return {
            "suggestions": suggestions,
            "reasoning": "Fallback parsing used due to response format issues",
            "auto_fix_available": False,
            "auto_fix_code": None
        }
    
    def _fallback_remediation(self, error_analysis: ErrorAnalysis, 
                            node_context: Dict[str, Any]) -> GeminiRemediationResult:
        """Fallback remediation when Gemini is not available"""
        suggestions = []
        
        # Use the remediation hints from error analysis
        for hint in error_analysis.remediation_hints:
            suggestion = RemediationSuggestion(
                action="Follow remediation hint",
                description=hint,
                confidence=0.7,
                priority="medium",
                estimated_effort="low"
            )
            suggestions.append(suggestion)
        
        return GeminiRemediationResult(
            error_analysis=error_analysis,
            suggestions=suggestions,
            reasoning="Fallback remediation using Trail Taxonomy hints",
            auto_fix_available=False,
            auto_fix_code=None,
            execution_time=0.0
        )
    
    def _create_cache_key(self, error_analysis: ErrorAnalysis, 
                         node_context: Dict[str, Any]) -> str:
        """Create a cache key for the error analysis"""
        key_parts = [
            error_analysis.category.value,
            error_analysis.severity.value,
            str(hash(error_analysis.description)),
            str(hash(str(node_context.get('node_name', ''))))
        ]
        return "|".join(key_parts)
    
    def clear_cache(self):
        """Clear the remediation cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl,
            "gemini_available": self.client is not None,
            "service_type": "vertex_ai" if self.use_vertex_ai else "developer_api"
        }
    
    # Alias for backward compatibility
    analyzeAndRemediate = analyze_and_remediate
