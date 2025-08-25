"""
Enhanced Error Taxonomy Classification System

This module implements an improved error classification system that can better identify
async/await issues, architectural problems, import issues, and other complex error patterns
that the original system was missing.
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import re
import traceback
from dataclasses import dataclass
import ast


class ErrorCategory(Enum):
    """Enhanced error categories based on Trail Taxonomy with architectural awareness"""
    INPUT_ERROR = "input_error"
    PROCESSING_ERROR = "processing_error"
    OUTPUT_ERROR = "output_error"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_ERROR = "external_error"
    ARCHITECTURAL_ERROR = "architectural_error"  # NEW: For async/await, imports, etc.
    CODE_GENERATION_ERROR = "code_generation_error"  # NEW: For AI generation failures
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorAnalysis:
    """Enhanced structured error analysis result"""
    category: ErrorCategory
    severity: ErrorSeverity
    confidence: float
    keywords: List[str]
    description: str
    remediation_hints: List[str]
    context: Dict[str, Any]
    root_cause: Optional[str] = None  # NEW: Specific root cause identification
    architectural_issue: Optional[str] = None  # NEW: For async/await, import issues
    suggested_fix_type: Optional[str] = None  # NEW: Type of fix needed


class EnhancedTrailTaxonomyClassifier:
    """
    Enhanced Trail Taxonomy error classification with architectural awareness
    """
    
    def __init__(self):
        # Enhanced error patterns and keywords for classification
        self.error_patterns = {
            ErrorCategory.INPUT_ERROR: {
                "keywords": [
                    "validation", "invalid", "missing", "required", "format", "type",
                    "schema", "constraint", "boundary", "range", "empty", "null",
                    "undefined", "malformed", "corrupted", "incomplete"
                ],
                "exceptions": [
                    "ValueError", "TypeError", "AttributeError", "KeyError",
                    "IndexError", "AssertionError"
                ],
                "patterns": [
                    r"missing.*required",
                    r"invalid.*format",
                    r"type.*error",
                    r"validation.*failed",
                    r"required.*field"
                ]
            },
            ErrorCategory.PROCESSING_ERROR: {
                "keywords": [
                    "algorithm", "computation", "calculation", "processing",
                    "memory", "overflow", "underflow", "division", "zero",
                    "recursion", "stack", "timeout", "deadlock", "race"
                ],
                "exceptions": [
                    "ZeroDivisionError", "OverflowError", "RecursionError",
                    "MemoryError", "TimeoutError"
                ],
                "patterns": [
                    r"division.*zero",
                    r"memory.*overflow",
                    r"recursion.*limit",
                    r"timeout.*error",
                    r"stack.*overflow"
                ]
            },
            ErrorCategory.OUTPUT_ERROR: {
                "keywords": [
                    "output", "result", "format", "serialization", "encoding",
                    "decoding", "json", "xml", "parsing", "rendering",
                    "display", "presentation", "quality", "accuracy"
                ],
                "exceptions": [
                    "JSONDecodeError", "UnicodeError", "SyntaxError",
                    "IndentationError"
                ],
                "patterns": [
                    r"json.*decode",
                    r"unicode.*error",
                    r"format.*error",
                    r"parsing.*failed"
                ]
            },
            ErrorCategory.SYSTEM_ERROR: {
                "keywords": [
                    "system", "os", "file", "permission", "access", "network",
                    "connection", "socket", "database", "disk", "resource",
                    "configuration", "environment", "path", "directory"
                ],
                "exceptions": [
                    "OSError", "FileNotFoundError", "PermissionError",
                    "ConnectionError", "TimeoutError", "ResourceWarning"
                ],
                "patterns": [
                    r"file.*not.*found",
                    r"permission.*denied",
                    r"connection.*refused",
                    r"resource.*unavailable",
                    r"disk.*full"
                ]
            },
            ErrorCategory.EXTERNAL_ERROR: {
                "keywords": [
                    "api", "http", "request", "response", "status", "code",
                    "external", "service", "third-party", "network", "timeout",
                    "rate", "limit", "quota", "authentication", "authorization"
                ],
                "exceptions": [
                    "requests.exceptions.RequestException",
                    "urllib.error.URLError", "socket.error"
                ],
                "patterns": [
                    r"http.*error",
                    r"api.*limit",
                    r"rate.*limit",
                    r"authentication.*failed",
                    r"service.*unavailable"
                ]
            },
            # NEW: Architectural error patterns
            ErrorCategory.ARCHITECTURAL_ERROR: {
                "keywords": [
                    "coroutine", "async", "await", "never awaited", "event loop",
                    "import", "module", "name.*not.*defined", "attribute",
                    "method", "function", "class", "inheritance", "interface"
                ],
                "exceptions": [
                    "RuntimeWarning", "ImportError", "ModuleNotFoundError",
                    "NameError", "AttributeError", "TypeError"
                ],
                "patterns": [
                    r"coroutine.*never.*awaited",
                    r"name.*not.*defined",
                    r"module.*not.*found",
                    r"cannot.*import",
                    r"coroutine.*object.*not.*mapping",
                    r"async.*function.*called.*without.*await"
                ]
            },
            # NEW: Code generation error patterns
            ErrorCategory.CODE_GENERATION_ERROR: {
                "keywords": [
                    "generation", "failed", "ai", "model", "response",
                    "parse", "format", "invalid", "malformed", "timeout"
                ],
                "exceptions": [
                    "ValueError", "TypeError", "RuntimeError", "TimeoutError"
                ],
                "patterns": [
                    r"generation.*failed",
                    r"ai.*failed",
                    r"model.*error",
                    r"response.*invalid",
                    r"parse.*failed"
                ]
            }
        }
        
        # Enhanced severity assessment patterns
        self.severity_patterns = {
            ErrorSeverity.CRITICAL: [
                "fatal", "critical", "emergency", "panic", "abort",
                "corruption", "data.*loss", "security", "breach"
            ],
            ErrorSeverity.HIGH: [
                "error", "failed", "exception", "crash", "break",
                "invalid", "corrupted", "malformed", "coroutine.*never.*awaited"
            ],
            ErrorSeverity.MEDIUM: [
                "warning", "deprecated", "timeout", "retry", "fallback",
                "import.*error", "name.*not.*defined"
            ],
            ErrorSeverity.LOW: [
                "info", "debug", "notice", "suggestion", "hint"
            ]
        }
        
        # NEW: Specific architectural issue patterns
        self.architectural_patterns = {
            "async_await": {
                "patterns": [
                    r"coroutine.*never.*awaited",
                    r"coroutine.*object.*not.*mapping",
                    r"async.*function.*called.*without.*await",
                    r"await.*outside.*async.*function"
                ],
                "keywords": ["coroutine", "async", "await", "never awaited"],
                "root_cause": "Async/await pattern issue",
                "suggested_fix": "architectural_fix"
            },
            "import_issue": {
                "patterns": [
                    r"name.*not.*defined",
                    r"module.*not.*found",
                    r"cannot.*import",
                    r"import.*error"
                ],
                "keywords": ["import", "module", "name", "defined"],
                "root_cause": "Missing import or module issue",
                "suggested_fix": "import_fix"
            },
            "code_generation": {
                "patterns": [
                    r"generation.*failed",
                    r"ai.*failed",
                    r"model.*error",
                    r"response.*invalid"
                ],
                "keywords": ["generation", "failed", "ai", "model"],
                "root_cause": "AI code generation failure",
                "suggested_fix": "fallback_generation"
            }
        }

    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorAnalysis:
        """
        Enhanced error classification with architectural awareness
        
        Args:
            exception: The exception that occurred
            context: Additional context about the error (state, node info, etc.)
            
        Returns:
            ErrorAnalysis object with enhanced classification results
        """
        if context is None:
            context = {}
            
        error_message = str(exception)
        error_type = type(exception).__name__
        traceback_str = traceback.format_exc()
        
        # NEW: Check for architectural issues first
        architectural_issue = self._detect_architectural_issue(error_message, error_type, traceback_str)
        
        # Analyze error using multiple classification methods
        category_scores = self._calculate_category_scores(error_message, error_type, traceback_str, architectural_issue)
        severity = self._assess_severity(error_message, error_type, context)
        
        # Get the most likely category
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        confidence = category_scores[best_category]
        
        # Extract relevant keywords
        keywords = self._extract_keywords(error_message, best_category)
        
        # Generate description
        description = self._generate_description(best_category, error_type, error_message)
        
        # NEW: Identify root cause and suggested fix type
        root_cause = self._identify_root_cause(error_message, architectural_issue)
        suggested_fix_type = self._suggest_fix_type(best_category, architectural_issue)
        
        # Generate remediation hints
        remediation_hints = self._generate_remediation_hints(best_category, error_message, context, architectural_issue)
        
        return ErrorAnalysis(
            category=best_category,
            severity=severity,
            confidence=confidence,
            keywords=keywords,
            description=description,
            remediation_hints=remediation_hints,
            context=context,
            root_cause=root_cause,
            architectural_issue=architectural_issue,
            suggested_fix_type=suggested_fix_type
        )
    
    def _detect_architectural_issue(self, error_message: str, error_type: str, traceback_str: str) -> Optional[str]:
        """Detect specific architectural issues like async/await, imports, etc."""
        error_lower = error_message.lower()
        traceback_lower = traceback_str.lower()
        
        # Check for async/await issues
        if any(re.search(pattern, error_lower) for pattern in self.architectural_patterns["async_await"]["patterns"]):
            return "async_await"
        
        # Check for import issues
        if any(re.search(pattern, error_lower) for pattern in self.architectural_patterns["import_issue"]["patterns"]):
            return "import_issue"
        
        # Check for code generation issues
        if any(re.search(pattern, error_lower) for pattern in self.architectural_patterns["code_generation"]["patterns"]):
            return "code_generation"
        
        # Check traceback for additional clues
        if "coroutine" in traceback_lower and "never awaited" in traceback_lower:
            return "async_await"
        
        if "name" in traceback_lower and "not defined" in traceback_lower:
            return "import_issue"
        
        return None
    
    def _calculate_category_scores(self, error_message: str, error_type: str, traceback_str: str, 
                                 architectural_issue: Optional[str] = None) -> Dict[ErrorCategory, float]:
        """Calculate confidence scores for each error category with architectural awareness"""
        scores = {category: 0.0 for category in ErrorCategory}
        
        # Boost architectural category if architectural issue detected
        if architectural_issue:
            if architectural_issue == "async_await":
                scores[ErrorCategory.ARCHITECTURAL_ERROR] += 0.8
            elif architectural_issue == "import_issue":
                scores[ErrorCategory.ARCHITECTURAL_ERROR] += 0.8
            elif architectural_issue == "code_generation":
                scores[ErrorCategory.CODE_GENERATION_ERROR] += 0.8
        
        # Check exception type matches
        for category, patterns in self.error_patterns.items():
            if error_type in patterns["exceptions"]:
                scores[category] += 0.4
                
            # Check keyword matches
            keyword_matches = sum(1 for keyword in patterns["keywords"] 
                                if keyword.lower() in error_message.lower())
            scores[category] += (keyword_matches / len(patterns["keywords"])) * 0.3
            
            # Check pattern matches
            pattern_matches = sum(1 for pattern in patterns["patterns"] 
                                if re.search(pattern, error_message.lower()))
            scores[category] += (pattern_matches / len(patterns["patterns"])) * 0.3
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _identify_root_cause(self, error_message: str, architectural_issue: Optional[str] = None) -> Optional[str]:
        """Identify specific root cause of the error"""
        if architectural_issue:
            return self.architectural_patterns[architectural_issue]["root_cause"]
        
        # Fallback root cause identification
        error_lower = error_message.lower()
        
        if "coroutine" in error_lower and "never awaited" in error_lower:
            return "Async function called without await"
        elif "name" in error_lower and "not defined" in error_lower:
            return "Missing import or undefined variable"
        elif "module" in error_lower and "not found" in error_lower:
            return "Missing module or package"
        elif "generation" in error_lower and "failed" in error_lower:
            return "AI code generation failure"
        
        return None
    
    def _suggest_fix_type(self, category: ErrorCategory, architectural_issue: Optional[str] = None) -> Optional[str]:
        """Suggest the type of fix needed"""
        if architectural_issue:
            return self.architectural_patterns[architectural_issue]["suggested_fix"]
        
        # Category-based fix suggestions
        fix_types = {
            ErrorCategory.ARCHITECTURAL_ERROR: "architectural_fix",
            ErrorCategory.CODE_GENERATION_ERROR: "fallback_generation",
            ErrorCategory.INPUT_ERROR: "data_fix",
            ErrorCategory.PROCESSING_ERROR: "logic_fix",
            ErrorCategory.OUTPUT_ERROR: "format_fix",
            ErrorCategory.SYSTEM_ERROR: "system_fix",
            ErrorCategory.EXTERNAL_ERROR: "retry_fix"
        }
        
        return fix_types.get(category)
    
    def _assess_severity(self, error_message: str, error_type: str, context: Dict[str, Any]) -> ErrorSeverity:
        """Assess error severity based on patterns and context"""
        error_lower = error_message.lower()
        
        # Check for critical patterns
        for pattern in self.severity_patterns[ErrorSeverity.CRITICAL]:
            if re.search(pattern, error_lower):
                return ErrorSeverity.CRITICAL
        
        # Check for high severity patterns (including architectural issues)
        for pattern in self.severity_patterns[ErrorSeverity.HIGH]:
            if re.search(pattern, error_lower):
                return ErrorSeverity.HIGH
        
        # Check for medium severity patterns
        for pattern in self.severity_patterns[ErrorSeverity.MEDIUM]:
            if re.search(pattern, error_lower):
                return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _extract_keywords(self, error_message: str, category: ErrorCategory) -> List[str]:
        """Extract relevant keywords from error message"""
        keywords = []
        patterns = self.error_patterns[category]
        
        for keyword in patterns["keywords"]:
            if keyword.lower() in error_message.lower():
                keywords.append(keyword)
        
        return keywords[:5]  # Limit to top 5 keywords
    
    def _generate_description(self, category: ErrorCategory, error_type: str, error_message: str) -> str:
        """Generate a human-readable description of the error"""
        descriptions = {
            ErrorCategory.INPUT_ERROR: f"Input validation error ({error_type}): {error_message}",
            ErrorCategory.PROCESSING_ERROR: f"Processing/computation error ({error_type}): {error_message}",
            ErrorCategory.OUTPUT_ERROR: f"Output formatting error ({error_type}): {error_message}",
            ErrorCategory.SYSTEM_ERROR: f"System/infrastructure error ({error_type}): {error_message}",
            ErrorCategory.EXTERNAL_ERROR: f"External service error ({error_type}): {error_message}",
            ErrorCategory.ARCHITECTURAL_ERROR: f"Architectural/code structure error ({error_type}): {error_message}",
            ErrorCategory.CODE_GENERATION_ERROR: f"AI code generation error ({error_type}): {error_message}",
            ErrorCategory.UNKNOWN_ERROR: f"Unknown error type ({error_type}): {error_message}"
        }
        
        return descriptions.get(category, f"Error ({error_type}): {error_message}")
    
    def _generate_remediation_hints(self, category: ErrorCategory, error_message: str, 
                                  context: Dict[str, Any], architectural_issue: Optional[str] = None) -> List[str]:
        """Generate remediation hints based on error category and architectural issues"""
        
        # Base hints by category
        base_hints = {
            ErrorCategory.INPUT_ERROR: [
                "Validate input data before processing",
                "Check for required fields and data types",
                "Ensure input format matches expected schema",
                "Add input sanitization and validation"
            ],
            ErrorCategory.PROCESSING_ERROR: [
                "Check for division by zero or overflow conditions",
                "Implement proper error handling for calculations",
                "Add timeout handling for long-running operations",
                "Consider using more robust algorithms"
            ],
            ErrorCategory.OUTPUT_ERROR: [
                "Validate output format before returning",
                "Add proper encoding/decoding error handling",
                "Check JSON/XML structure validity",
                "Implement output quality checks"
            ],
            ErrorCategory.SYSTEM_ERROR: [
                "Check file permissions and paths",
                "Verify system resources availability",
                "Ensure proper configuration settings",
                "Add system health checks"
            ],
            ErrorCategory.EXTERNAL_ERROR: [
                "Implement retry logic with exponential backoff",
                "Add circuit breaker pattern for external services",
                "Check API rate limits and quotas",
                "Verify authentication and authorization"
            ],
            ErrorCategory.ARCHITECTURAL_ERROR: [
                "Review async/await patterns and ensure proper usage",
                "Check for missing imports and dependencies",
                "Verify function signatures and method calls",
                "Ensure proper code structure and organization"
            ],
            ErrorCategory.CODE_GENERATION_ERROR: [
                "Implement fallback code generation mechanisms",
                "Add retry logic for AI model calls",
                "Provide alternative code templates",
                "Implement manual code generation as backup"
            ],
            ErrorCategory.UNKNOWN_ERROR: [
                "Add comprehensive logging for debugging",
                "Implement graceful error handling",
                "Consider adding monitoring and alerting",
                "Review error patterns for classification"
            ]
        }
        
        hints = base_hints.get(category, ["Implement general error handling"])
        
        # Add architectural-specific hints
        if architectural_issue == "async_await":
            hints.extend([
                "Ensure async functions are called with await",
                "Check for missing await keywords",
                "Verify event loop is properly configured",
                "Consider using asyncio.run() for top-level async calls"
            ])
        elif architectural_issue == "import_issue":
            hints.extend([
                "Add missing import statements",
                "Check module installation and dependencies",
                "Verify import paths and module names",
                "Consider using try/except for optional imports"
            ])
        elif architectural_issue == "code_generation":
            hints.extend([
                "Implement fallback code generation",
                "Add retry mechanisms for AI calls",
                "Provide template-based code generation",
                "Consider manual code generation as backup"
            ])
        
        return hints[:6]  # Limit to top 6 hints
