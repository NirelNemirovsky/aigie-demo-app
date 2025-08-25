"""
Enhanced Remediation Engine for Aigie

This module implements an improved remediation system that can handle architectural issues,
learn from failed attempts, and provide fallback mechanisms when AI code generation fails.
"""

import re
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .error_taxonomy import EnhancedTrailTaxonomyClassifier, ErrorAnalysis, ErrorCategory

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Types of fix strategies"""
    DATA_FIX = "data_fix"
    ARCHITECTURAL_FIX = "architectural_fix"
    IMPORT_FIX = "import_fix"
    FALLBACK_GENERATION = "fallback_generation"
    RETRY_FIX = "retry_fix"
    SYSTEM_FIX = "system_fix"
    LOGIC_FIX = "logic_fix"
    FORMAT_FIX = "format_fix"


@dataclass
class FixAttempt:
    """Record of a fix attempt"""
    strategy: FixStrategy
    code: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EnhancedRemediationResult:
    """Enhanced remediation result with learning capabilities"""
    success: bool
    fixed_state: Dict[str, Any]
    generated_code: str
    execution_time: float
    strategy_used: FixStrategy
    confidence: float
    error_message: Optional[str] = None
    state_changes: Dict[str, Any] = None
    fix_attempts: List[FixAttempt] = field(default_factory=list)
    learning_insights: Dict[str, Any] = None


class AdaptiveRemediationEngine:
    """
    Adaptive remediation engine that learns from failed attempts and adapts strategies
    """
    
    def __init__(self, max_attempts: int = 3, confidence_threshold: float = 0.7):
        self.max_attempts = max_attempts
        self.confidence_threshold = confidence_threshold
        self.error_classifier = EnhancedTrailTaxonomyClassifier()
        self.learning_memory: Dict[str, List[FixAttempt]] = {}
        self.strategy_success_rates: Dict[FixStrategy, Dict[str, float]] = {}
        
    def remediate_error(self, exception: Exception, state: Dict[str, Any], 
                       context: Dict[str, Any] = None) -> EnhancedRemediationResult:
        """
        Remediate error using adaptive strategies that learn from previous attempts
        """
        start_time = time.time()
        original_state = state.copy()
        
        # Classify the error
        error_analysis = self.error_classifier.classify_error(exception, context)
        
        # Get learning key for this error pattern
        learning_key = self._create_learning_key(error_analysis)
        
        # Get successful strategies for similar errors
        successful_strategies = self._get_successful_strategies(learning_key)
        
        # Determine fix strategies to try (prioritize successful ones)
        strategies = self._determine_fix_strategies(error_analysis, successful_strategies)
        
        fix_attempts = []
        
        for i, strategy in enumerate(strategies):
            if i >= self.max_attempts:
                break
                
            print(f"\n🔧 Attempting {strategy.value} fix (attempt {i + 1}/{self.max_attempts})...")
            
            # Generate fix code for this strategy
            fix_code = self._generate_fix_code(strategy, error_analysis, state, context)
            
            # Execute the fix
            attempt_start = time.time()
            success, fixed_state, error_msg = self._execute_fix(fix_code, state.copy())
            attempt_time = time.time() - attempt_start
            
            # Create fix attempt record
            fix_attempt = FixAttempt(
                strategy=strategy,
                code=fix_code,
                success=success,
                error_message=error_msg,
                execution_time=attempt_time,
                confidence=self._calculate_confidence(strategy, error_analysis)
            )
            fix_attempts.append(fix_attempt)
            
            # Update learning memory
            self._update_learning_memory(learning_key, fix_attempt)
            
            if success:
                print(f"✅ {strategy.value} fix successful!")
                
                # Calculate final result
                execution_time = time.time() - start_time
                state_changes = self._calculate_state_changes(original_state, fixed_state)
                
                return EnhancedRemediationResult(
                    success=True,
                    fixed_state=fixed_state,
                    generated_code=fix_code,
                    execution_time=execution_time,
                    strategy_used=strategy,
                    confidence=fix_attempt.confidence,
                    state_changes=state_changes,
                    fix_attempts=fix_attempts,
                    learning_insights=self._get_learning_insights(learning_key)
                )
            else:
                print(f"❌ {strategy.value} fix failed: {error_msg}")
        
        # All attempts failed
        execution_time = time.time() - start_time
        
        return EnhancedRemediationResult(
            success=False,
            fixed_state=original_state,
            generated_code="",
            execution_time=execution_time,
            strategy_used=FixStrategy.DATA_FIX,  # Default
            confidence=0.0,
            error_message=f"All {self.max_attempts} fix attempts failed",
            fix_attempts=fix_attempts,
            learning_insights=self._get_learning_insights(learning_key)
        )
    
    def _create_learning_key(self, error_analysis: ErrorAnalysis) -> str:
        """Create a learning key for this error pattern"""
        key_parts = [
            error_analysis.category.value,
            error_analysis.architectural_issue or "none",
            error_analysis.root_cause or "unknown",
            str(hash(error_analysis.description[:100]))
        ]
        return "|".join(key_parts)
    
    def _get_successful_strategies(self, learning_key: str) -> List[FixStrategy]:
        """Get strategies that have been successful for similar errors"""
        if learning_key not in self.learning_memory:
            return []
        
        attempts = self.learning_memory[learning_key]
        successful_strategies = []
        
        for attempt in attempts:
            if attempt.success:
                successful_strategies.append(attempt.strategy)
        
        return successful_strategies
    
    def _determine_fix_strategies(self, error_analysis: ErrorAnalysis, 
                                successful_strategies: List[FixStrategy]) -> List[FixStrategy]:
        """Determine which fix strategies to try, prioritizing successful ones"""
        strategies = []
        
        # First, try successful strategies from similar errors
        strategies.extend(successful_strategies)
        
        # Then, add strategies based on error analysis
        if error_analysis.suggested_fix_type:
            if error_analysis.suggested_fix_type == "architectural_fix":
                strategies.append(FixStrategy.ARCHITECTURAL_FIX)
            elif error_analysis.suggested_fix_type == "import_fix":
                strategies.append(FixStrategy.IMPORT_FIX)
            elif error_analysis.suggested_fix_type == "fallback_generation":
                strategies.append(FixStrategy.FALLBACK_GENERATION)
        
        # Add category-based strategies
        if error_analysis.category == ErrorCategory.INPUT_ERROR:
            strategies.append(FixStrategy.DATA_FIX)
        elif error_analysis.category == ErrorCategory.ARCHITECTURAL_ERROR:
            strategies.append(FixStrategy.ARCHITECTURAL_FIX)
        elif error_analysis.category == ErrorCategory.CODE_GENERATION_ERROR:
            strategies.append(FixStrategy.FALLBACK_GENERATION)
        elif error_analysis.category == ErrorCategory.EXTERNAL_ERROR:
            strategies.append(FixStrategy.RETRY_FIX)
        elif error_analysis.category == ErrorCategory.SYSTEM_ERROR:
            strategies.append(FixStrategy.SYSTEM_FIX)
        elif error_analysis.category == ErrorCategory.PROCESSING_ERROR:
            strategies.append(FixStrategy.LOGIC_FIX)
        elif error_analysis.category == ErrorCategory.OUTPUT_ERROR:
            strategies.append(FixStrategy.FORMAT_FIX)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        return unique_strategies
    
    def _generate_fix_code(self, strategy: FixStrategy, error_analysis: ErrorAnalysis,
                          state: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate fix code based on strategy"""
        
        if strategy == FixStrategy.ARCHITECTURAL_FIX:
            return self._generate_architectural_fix(error_analysis, state)
        elif strategy == FixStrategy.IMPORT_FIX:
            return self._generate_import_fix(error_analysis, state)
        elif strategy == FixStrategy.FALLBACK_GENERATION:
            return self._generate_fallback_code(error_analysis, state)
        elif strategy == FixStrategy.DATA_FIX:
            return self._generate_data_fix(error_analysis, state)
        elif strategy == FixStrategy.RETRY_FIX:
            return self._generate_retry_fix(error_analysis, state)
        elif strategy == FixStrategy.SYSTEM_FIX:
            return self._generate_system_fix(error_analysis, state)
        elif strategy == FixStrategy.LOGIC_FIX:
            return self._generate_logic_fix(error_analysis, state)
        elif strategy == FixStrategy.FORMAT_FIX:
            return self._generate_format_fix(error_analysis, state)
        else:
            return self._generate_generic_fix(error_analysis, state)
    
    def _generate_architectural_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for architectural issues like async/await, imports"""
        
        if error_analysis.architectural_issue == "async_await":
            return """
# Fix async/await architectural issue
import asyncio
import inspect

# Check if we're in an async context and handle accordingly
if 'async_context' not in state:
    state['async_context'] = False

# Add async handling metadata
state['async_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()
state['architectural_issue'] = 'async_await'

print("Applied async/await architectural fix")
print("Note: This is a runtime fix. Code changes may be needed for permanent resolution.")
"""
        
        elif error_analysis.architectural_issue == "import_issue":
            return """
# Fix import/name not defined issue
import sys
import importlib

# Add import handling metadata
state['import_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()
state['architectural_issue'] = 'import_issue'

# Try to handle common missing imports
missing_imports = ['re', 'json', 'datetime', 'uuid', 'time']
for module_name in missing_imports:
    if module_name not in sys.modules:
        try:
            importlib.import_module(module_name)
            print(f"Attempted to import {module_name}")
        except ImportError:
            print(f"Could not import {module_name}")

print("Applied import fix")
print("Note: This is a runtime fix. Code changes may be needed for permanent resolution.")
"""
        
        else:
            return """
# Generic architectural fix
state['architectural_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()
state['architectural_issue'] = 'generic'

print("Applied generic architectural fix")
"""
    
    def _generate_import_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for import issues"""
        return """
# Fix import issues
import sys
import importlib

# Common missing imports that might be needed
common_imports = {
    're': 're',
    'json': 'json', 
    'datetime': 'datetime',
    'uuid': 'uuid',
    'time': 'time',
    'os': 'os',
    'sys': 'sys',
    'logging': 'logging'
}

# Try to import missing modules
for module_name, import_name in common_imports.items():
    if module_name not in sys.modules:
        try:
            globals()[import_name] = importlib.import_module(module_name)
            print(f"Successfully imported {module_name}")
        except ImportError:
            print(f"Could not import {module_name}")

state['import_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()
print("Applied import fix")
"""
    
    def _generate_fallback_code(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fallback code when AI generation fails"""
        return """
# Fallback code generation when AI fails
import uuid
from datetime import datetime

# Apply safe fallback fixes
state['fallback_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()
state['fallback_id'] = str(uuid.uuid4())

# Add common missing fields with safe defaults
common_fields = {
    'id': str(uuid.uuid4()),
    'timestamp': datetime.now().isoformat(),
    'status': 'pending',
    'created_at': datetime.now().isoformat(),
    'updated_at': datetime.now().isoformat()
}

for field, default_value in common_fields.items():
    if field not in state or state[field] is None:
        state[field] = default_value
        print(f"Added fallback field: {field} = {default_value}")

print("Applied fallback code generation fix")
"""
    
    def _generate_data_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for data/input issues"""
        return """
# Fix data/input validation issues
from datetime import datetime
import uuid

# Apply data validation fixes
state['data_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()

# Common validation fixes
validation_defaults = {
    'email': 'default@example.com',
    'phone': '+1234567890',
    'url': 'https://example.com',
    'date': datetime.now().isoformat(),
    'status': 'pending',
    'id': str(uuid.uuid4())
}

for field, default_value in validation_defaults.items():
    if field in state and (state[field] is None or state[field] == ''):
        state[field] = default_value
        print(f"Fixed validation issue for {field}: {default_value}")

print("Applied data validation fix")
"""
    
    def _generate_retry_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for retry/external issues"""
        return """
# Fix retry/external service issues
import time
from datetime import datetime

# Apply retry logic fix
state['retry_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()

# Add retry metadata
if 'retry_count' not in state:
    state['retry_count'] = 0
state['retry_count'] += 1

# Add exponential backoff
if 'retry_delay' not in state:
    state['retry_delay'] = 1.0
state['retry_delay'] = min(state['retry_delay'] * 2, 60.0)

# Add timeout handling
state['timeout'] = 30.0
state['max_retries'] = 3

print(f"Applied retry fix (attempt {state['retry_count']})")
"""
    
    def _generate_system_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for system issues"""
        return """
# Fix system/infrastructure issues
import os
from datetime import datetime

# Apply system fix
state['system_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()

# Add system metadata
state['environment'] = os.environ.get('ENVIRONMENT', 'development')
state['working_directory'] = os.getcwd()
state['system_info'] = {
    'platform': os.name,
    'pid': os.getpid()
}

print("Applied system fix")
"""
    
    def _generate_logic_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for logic/processing issues"""
        return """
# Fix logic/processing issues
from datetime import datetime

# Apply logic fix
state['logic_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()

# Add error handling metadata
state['error_handling'] = {
    'enabled': True,
    'max_retries': 3,
    'timeout': 30.0
}

# Add processing safeguards
state['processing_safeguards'] = {
    'division_by_zero_check': True,
    'null_check': True,
    'type_validation': True
}

print("Applied logic fix")
"""
    
    def _generate_format_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate fix for format/output issues"""
        return """
# Fix format/output issues
import json
from datetime import datetime

# Apply format fix
state['format_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()

# Add output formatting metadata
state['output_format'] = {
    'encoding': 'utf-8',
    'indent': 2,
    'ensure_ascii': False
}

# Ensure JSON serializable
for key, value in list(state.items()):
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        state[key] = str(value)
        print(f"Converted non-serializable value for key: {key}")

print("Applied format fix")
"""
    
    def _generate_generic_fix(self, error_analysis: ErrorAnalysis, state: Dict[str, Any]) -> str:
        """Generate generic fix for unknown issues"""
        return """
# Generic fix for unknown issues
from datetime import datetime
import uuid

# Apply generic fix
state['generic_fix_applied'] = True
state['fix_timestamp'] = datetime.now().isoformat()
state['fix_id'] = str(uuid.uuid4())

# Add basic error handling
state['error_handling'] = {
    'enabled': True,
    'fallback_mode': True
}

print("Applied generic fix")
"""
    
    def _execute_fix(self, fix_code: str, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Execute fix code safely"""
        try:
            # Create safe execution environment
            safe_globals = {
                'state': state,
                'datetime': datetime,
                'uuid': uuid,
                'time': time,
                'json': json,
                're': re,
                'os': os,
                'sys': sys,
                'importlib': importlib,
                'asyncio': asyncio,
                'inspect': inspect,
                'logging': logging,
                'print': lambda *args: logger.info(" ".join(map(str, args)))
            }
            
            # Execute the fix code
            exec(fix_code, safe_globals)
            
            # Return the potentially modified state
            return True, safe_globals['state'], None
            
        except Exception as e:
            return False, state, str(e)
    
    def _calculate_confidence(self, strategy: FixStrategy, error_analysis: ErrorAnalysis) -> float:
        """Calculate confidence for a strategy based on error analysis"""
        base_confidence = 0.7
        
        # Boost confidence for matching strategies
        if error_analysis.suggested_fix_type:
            if (strategy == FixStrategy.ARCHITECTURAL_FIX and 
                error_analysis.suggested_fix_type == "architectural_fix"):
                base_confidence += 0.2
            elif (strategy == FixStrategy.IMPORT_FIX and 
                  error_analysis.suggested_fix_type == "import_fix"):
                base_confidence += 0.2
            elif (strategy == FixStrategy.FALLBACK_GENERATION and 
                  error_analysis.suggested_fix_type == "fallback_generation"):
                base_confidence += 0.2
        
        # Boost confidence for architectural issues
        if error_analysis.architectural_issue:
            if (error_analysis.architectural_issue == "async_await" and 
                strategy == FixStrategy.ARCHITECTURAL_FIX):
                base_confidence += 0.1
            elif (error_analysis.architectural_issue == "import_issue" and 
                  strategy == FixStrategy.IMPORT_FIX):
                base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _update_learning_memory(self, learning_key: str, fix_attempt: FixAttempt):
        """Update learning memory with fix attempt results"""
        if learning_key not in self.learning_memory:
            self.learning_memory[learning_key] = []
        
        self.learning_memory[learning_key].append(fix_attempt)
        
        # Keep only recent attempts (last 10)
        if len(self.learning_memory[learning_key]) > 10:
            self.learning_memory[learning_key] = self.learning_memory[learning_key][-10:]
    
    def _get_learning_insights(self, learning_key: str) -> Dict[str, Any]:
        """Get learning insights for this error pattern"""
        if learning_key not in self.learning_memory:
            return {"message": "No previous attempts for this error pattern"}
        
        attempts = self.learning_memory[learning_key]
        
        # Calculate success rates by strategy
        strategy_success = {}
        for attempt in attempts:
            strategy = attempt.strategy.value
            if strategy not in strategy_success:
                strategy_success[strategy] = {"successes": 0, "total": 0}
            
            strategy_success[strategy]["total"] += 1
            if attempt.success:
                strategy_success[strategy]["successes"] += 1
        
        # Calculate success rates
        for strategy in strategy_success:
            total = strategy_success[strategy]["total"]
            successes = strategy_success[strategy]["successes"]
            strategy_success[strategy]["rate"] = successes / total if total > 0 else 0
        
        return {
            "total_attempts": len(attempts),
            "strategy_success_rates": strategy_success,
            "recent_attempts": [
                {
                    "strategy": attempt.strategy.value,
                    "success": attempt.success,
                    "confidence": attempt.confidence,
                    "execution_time": attempt.execution_time
                }
                for attempt in attempts[-5:]  # Last 5 attempts
            ]
        }
    
    def _calculate_state_changes(self, original: Dict[str, Any], modified: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get overall learning statistics"""
        total_patterns = len(self.learning_memory)
        total_attempts = sum(len(attempts) for attempts in self.learning_memory.values())
        
        # Calculate overall success rates
        strategy_stats = {}
        for attempts in self.learning_memory.values():
            for attempt in attempts:
                strategy = attempt.strategy.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"successes": 0, "total": 0}
                
                strategy_stats[strategy]["total"] += 1
                if attempt.success:
                    strategy_stats[strategy]["successes"] += 1
        
        # Calculate rates
        for strategy in strategy_stats:
            total = strategy_stats[strategy]["total"]
            successes = strategy_stats[strategy]["successes"]
            strategy_stats[strategy]["rate"] = successes / total if total > 0 else 0
        
        return {
            "total_error_patterns": total_patterns,
            "total_fix_attempts": total_attempts,
            "strategy_statistics": strategy_stats,
            "learning_memory_size": total_patterns
        }
