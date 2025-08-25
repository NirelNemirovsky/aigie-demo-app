"""
Trail Taxonomy Error Classification System

This module implements a comprehensive error classification system based on the Trail Taxonomy paper
for intelligent error detection and remediation in AI systems.
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import re
import traceback
from dataclasses import dataclass


class ErrorCategory(Enum):
    """Error categories based on Trail Taxonomy"""
    INPUT_ERROR = "input_error"
    PROCESSING_ERROR = "processing_error"
    OUTPUT_ERROR = "output_error"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_ERROR = "external_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorAnalysis:
    """Structured error analysis result"""
    category: ErrorCategory
    severity: ErrorSeverity
    confidence: float
    keywords: List[str]
    description: str
    remediation_hints: List[str]
    context: Dict[str, Any]


class TrailTaxonomyClassifier:
    """
    Implements Trail Taxonomy error classification for intelligent error handling
    """
    
    def __init__(self):
        # Error patterns and keywords for classification
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
            }
        }
        
        # Severity assessment patterns
        self.severity_patterns = {
            ErrorSeverity.CRITICAL: [
                "fatal", "critical", "emergency", "panic", "abort",
                "corruption", "data.*loss", "security", "breach"
            ],
            ErrorSeverity.HIGH: [
                "error", "failed", "exception", "crash", "break",
                "invalid", "corrupted", "malformed"
            ],
            ErrorSeverity.MEDIUM: [
                "warning", "deprecated", "timeout", "retry", "fallback"
            ],
            ErrorSeverity.LOW: [
                "info", "debug", "notice", "suggestion", "hint"
            ]
        }

    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorAnalysis:
        """
        Classify an error using Trail Taxonomy approach
        
        Args:
            exception: The exception that occurred
            context: Additional context about the error (state, node info, etc.)
            
        Returns:
            ErrorAnalysis object with classification results
        """
        if context is None:
            context = {}
            
        error_message = str(exception)
        error_type = type(exception).__name__
        traceback_str = traceback.format_exc()
        
        # Analyze error using multiple classification methods
        category_scores = self._calculate_category_scores(error_message, error_type, traceback_str)
        severity = self._assess_severity(error_message, error_type, context)
        
        # Get the most likely category
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        confidence = category_scores[best_category]
        
        # Extract relevant keywords
        keywords = self._extract_keywords(error_message, best_category)
        
        # Generate description
        description = self._generate_description(best_category, error_type, error_message)
        
        # Generate remediation hints
        remediation_hints = self._generate_remediation_hints(best_category, error_message, context)
        
        return ErrorAnalysis(
            category=best_category,
            severity=severity,
            confidence=confidence,
            keywords=keywords,
            description=description,
            remediation_hints=remediation_hints,
            context=context
        )
    
    def _calculate_category_scores(self, error_message: str, error_type: str, traceback_str: str) -> Dict[ErrorCategory, float]:
        """Calculate confidence scores for each error category"""
        scores = {category: 0.0 for category in ErrorCategory}
        
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
    
    def _assess_severity(self, error_message: str, error_type: str, context: Dict[str, Any]) -> ErrorSeverity:
        """Assess error severity based on patterns and context"""
        error_lower = error_message.lower()
        
        # Check for critical patterns
        for pattern in self.severity_patterns[ErrorSeverity.CRITICAL]:
            if re.search(pattern, error_lower):
                return ErrorSeverity.CRITICAL
        
        # Check for high severity patterns
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
            ErrorCategory.UNKNOWN_ERROR: f"Unknown error type ({error_type}): {error_message}"
        }
        
        return descriptions.get(category, f"Error ({error_type}): {error_message}")
    
    def _generate_remediation_hints(self, category: ErrorCategory, error_message: str, context: Dict[str, Any]) -> List[str]:
        """Generate remediation hints based on error category"""
        hints = {
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
            ErrorCategory.UNKNOWN_ERROR: [
                "Add comprehensive logging for debugging",
                "Implement graceful error handling",
                "Consider adding monitoring and alerting",
                "Review error patterns for classification"
            ]
        }
        
        return hints.get(category, ["Implement general error handling"])
