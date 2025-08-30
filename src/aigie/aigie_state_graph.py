from langgraph.graph import StateGraph
from .enhanced_policy_node import PolicyNode
from typing import Any, Dict, Optional, Type, TypeVar, Union, List
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class AigieStateGraph(StateGraph):
    """
    Enhanced AigieStateGraph with Trail Taxonomy error classification and Gemini AI remediation.
    
    This wrapper extends LangGraph's StateGraph with intelligent error handling that:
    1. Classifies errors using Trail Taxonomy approach
    2. Provides AI-powered remediation suggestions using Gemini 2.5 Flash
    3. Offers automatic error fixing capabilities
    4. Maintains comprehensive error analytics and logging
    5. Natively supports Pydantic models for state management
    """

    def __init__(self, 
                 state_schema: Optional[Type[BaseModel]] = None,
                 enable_gemini_remediation: bool = True,
                 gemini_project_id: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 auto_apply_fixes: bool = False,
                 log_remediation: bool = True,
                 enable_proactive_remediation: bool = True,
                 proactive_fix_types: Optional[List[str]] = None,
                 max_proactive_attempts: int = 3):
        """
        Initialize the AigieStateGraph with enhanced error handling capabilities.
        
        Args:
            state_schema: Pydantic model class for the state schema. If None, uses a default schema.
            enable_gemini_remediation: Whether to enable Gemini AI-powered error remediation
            gemini_project_id: GCP project ID for Vertex AI (if None, will auto-detect)
            gemini_api_key: Google API key for Gemini Developer API (if None, will auto-detect)
            auto_apply_fixes: Whether to automatically apply AI-suggested fixes
            log_remediation: Whether to log remediation analysis to GCP
        """
        if state_schema is None:
            # Use a simple Pydantic model as default schema
            class DefaultState(BaseModel):
                messages: list[str] = []
                state: Dict[str, Any] = {}
            
            state_schema = DefaultState
        
        # Store the state schema for validation
        self.state_schema = state_schema
        
        # Initialize the underlying StateGraph with the Pydantic schema
        # Pass the class directly to parent - LangGraph handles Pydantic models natively
        super().__init__(state_schema)
        
        # Auto-detect Gemini configuration for seamless setup
        self.gemini_config = self._auto_detect_gemini_config(
            enable_gemini_remediation, 
            gemini_project_id, 
            gemini_api_key
        )
        
        # Store configuration for enhanced error handling
        self.enable_gemini_remediation = self.gemini_config["enabled"]
        self.gemini_project_id = self.gemini_config["project_id"]
        self.gemini_api_key = self.gemini_config["api_key"]
        self.auto_apply_fixes = auto_apply_fixes
        self.log_remediation = log_remediation
        self.enable_proactive_remediation = enable_proactive_remediation
        self.proactive_fix_types = proactive_fix_types or [
            'missing_field', 'type_error', 'validation_error', 'api_error', 'timeout_error'
        ]
        self.max_proactive_attempts = max_proactive_attempts
        
        # Track nodes for analytics
        self.node_analytics = {}
        
        # Log Gemini configuration status
        self._log_gemini_status()
        
        # Store the original Pydantic schema for conversion
        self._pydantic_schema = state_schema if state_schema and issubclass(state_schema, BaseModel) else None

    def _auto_detect_gemini_config(self, enable_gemini: bool, project_id: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
        """
        Automatically detect Gemini configuration for seamless setup.
        
        Priority order:
        1. Explicit parameters (highest priority)
        2. Environment variables
        3. Configuration file
        4. GCP auto-detection
        5. Fallback to Trail Taxonomy only
        """
        if not enable_gemini:
            return {"enabled": False, "project_id": None, "api_key": None, "service_type": "disabled"}
        
        # Check explicit parameters first
        if api_key:
            return {"enabled": True, "project_id": None, "api_key": api_key, "service_type": "developer_api"}
        
        if project_id:
            return {"enabled": True, "project_id": project_id, "api_key": None, "service_type": "vertex_ai"}
        
        # Check environment variables
        env_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if env_api_key:
            return {"enabled": True, "project_id": None, "api_key": env_api_key, "service_type": "developer_api"}
        
        env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        if env_project_id:
            return {"enabled": True, "project_id": env_project_id, "api_key": None, "service_type": "vertex_ai"}
        
        # Check configuration file
        config = self._load_config_file()
        if config.get("api_key"):
            return {"enabled": True, "project_id": None, "api_key": config["api_key"], "service_type": "developer_api"}
        
        if config.get("project_id"):
            return {"enabled": True, "project_id": config["project_id"], "api_key": None, "service_type": "vertex_ai"}
        
        # Fallback: Gemini enabled but no configuration found
        return {"enabled": True, "project_id": None, "api_key": None, "service_type": "fallback"}
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from ~/.aigie/config.json if it exists"""
        try:
            config_path = os.path.expanduser("~/.aigie/config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.loads(f.read())
        except Exception:
            pass
        return {}
    
    def _log_gemini_status(self):
        """Log the Gemini configuration status for user awareness"""
        if not self.enable_gemini_remediation:
            print("ℹ️  Gemini AI remediation: Disabled")
            return
        
        config = self.gemini_config
        service_type = config["service_type"]
        
        if service_type == "developer_api":
            print(f"✅ Gemini AI remediation: Enabled (Developer API)")
            print(f"   🔑 API Key: {'Configured' if config['api_key'] else 'Not found'}")
        elif service_type == "vertex_ai":
            print(f"✅ Gemini AI remediation: Enabled (Vertex AI)")
            print(f"   🏗️  Project: {config['project_id']}")
        elif service_type == "fallback":
            print("⚠️  Gemini AI remediation: Enabled but no configuration found")
            print("   💡 Set GOOGLE_API_KEY environment variable for AI features")
            print("   💡 Or create ~/.aigie/config.json with your API key")
        else:
            print("ℹ️  Gemini AI remediation: Disabled")

    def add_node(self, node_id: str, node_data=None, **node_config):
        """
        Enhanced add_node method that wraps nodes with intelligent error handling.
        
        Args:
            node_id: The identifier for the node
            node_data: The function or runnable to wrap with PolicyNode
            **node_config: Additional configuration for the PolicyNode
        """
        if node_data is not None:
            # Extract policy node specific config
            enhanced_config = {
                "enable_gemini_remediation": node_config.get("enable_gemini_remediation", self.enable_gemini_remediation),
                "gemini_project_id": node_config.get("gemini_project_id", self.gemini_project_id),
                "gemini_api_key": node_config.get("gemini_api_key", self.gemini_api_key),
                "auto_apply_fixes": node_config.get("auto_apply_fixes", self.auto_apply_fixes),
                "log_remediation": node_config.get("log_remediation", self.log_remediation),
                "enable_adaptive_remediation": node_config.get("enable_adaptive_remediation", self.enable_proactive_remediation),
                "max_adaptive_attempts": node_config.get("max_adaptive_attempts", self.max_proactive_attempts),
                "max_attempts": node_config.get("max_attempts", 3),
                "fallback": node_config.get("fallback"),
                "tweak_input": node_config.get("tweak_input"),
                "on_error": node_config.get("on_error")
            }
            
            # Wrap the node with PolicyNode
            wrapped_node = PolicyNode(
                inner=node_data, 
                name=node_id,
                pydantic_schema=self._pydantic_schema,
                **enhanced_config
            )
            
            # Call the original add_node method with the wrapped node
            super().add_node(node_id, wrapped_node)
            
            # Initialize analytics tracking for this node
            self.node_analytics[node_id] = {
                "node_type": type(node_data).__name__,
                "enhanced_config": enhanced_config,
                "creation_time": time.time()
            }
            
            print(f"✅ Node '{node_id}' wrapped with PolicyNode and added successfully.")
            print(f"   - Gemini remediation: {enhanced_config['enable_gemini_remediation']}")
            print(f"   - Adaptive remediation: {enhanced_config['enable_adaptive_remediation']}")
            print(f"   - Auto-apply fixes: {enhanced_config['auto_apply_fixes']}")
            print(f"   - Max attempts: {enhanced_config['max_attempts']}")
            print(f"   - Pydantic schema: {self.state_schema.__name__}")
        else:
            # If no node_data, just add the node_id (for conditional nodes)
            super().add_node(node_id, node_data)
            print(f"✅ Conditional node '{node_id}' added successfully.")





    def get_node_analytics(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get error analytics for nodes.
        
        Args:
            node_id: Specific node ID to get analytics for. If None, returns all nodes.
            
        Returns:
            Dictionary containing error analytics and statistics
        """
        if node_id:
            if node_id not in self.nodes:
                return {"error": f"Node '{node_id}' not found"}
            
            node = self.nodes[node_id]
            if hasattr(node, 'get_error_analytics'):
                return node.get_error_analytics()
            else:
                return {"error": f"Node '{node_id}' does not support analytics"}
        
        # Return analytics for all nodes
        all_analytics = {}
        for nid, node in self.nodes.items():
            if hasattr(node, 'get_error_analytics'):
                all_analytics[nid] = node.get_error_analytics()
            else:
                all_analytics[nid] = {"error": "Node does not support analytics"}
        
        return all_analytics
    
    def to_dict(self, state: Any) -> Dict[str, Any]:
        """Convert Pydantic model to dictionary for LangGraph compatibility"""
        if self._pydantic_schema and isinstance(state, self._pydantic_schema):
            return state.dict()
        elif isinstance(state, dict):
            return state
        else:
            return state
    
    def invoke(self, state: Any, config: Optional[Any] = None) -> Any:
        """Custom invoke method that handles Pydantic model conversion"""
        # Convert Pydantic model to dictionary for LangGraph compatibility
        if self._pydantic_schema and isinstance(state, self._pydantic_schema):
            state_dict = state.dict()
        else:
            state_dict = state
        
        # Compile and invoke
        compiled_workflow = self.compile()
        result = compiled_workflow.invoke(state_dict, config=config)
        
        # Convert result back to Pydantic model if applicable
        if self._pydantic_schema and isinstance(result, dict):
            return self._pydantic_schema(**result)
        return result
    
    def invoke_with_pydantic(self, state: Any, config: Optional[Any] = None) -> Any:
        """Alternative invoke method that automatically handles Pydantic conversion"""
        return self.invoke(state, config)
    
    def from_dict(self, state_dict: Dict[str, Any]) -> Any:
        """Convert dictionary back to Pydantic model"""
        if self._pydantic_schema and isinstance(state_dict, dict):
            return self._pydantic_schema(**state_dict)
        else:
            return state_dict
    
    def validate_state(self, state: Any) -> bool:
        """Validate state against the schema"""
        if self._pydantic_schema:
            try:
                if isinstance(state, self._pydantic_schema):
                    return True
                elif isinstance(state, dict):
                    # Validate by creating a temporary instance
                    self._pydantic_schema(**state)
                    return True
                else:
                    return False
            except Exception:
                return False
        return True

    def get_graph_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics for the entire graph.
        
        Returns:
            Dictionary containing graph-wide error analytics and statistics
        """
        node_analytics = self.get_node_analytics()
        
        # Aggregate statistics
        total_errors = 0
        total_remediations = 0
        error_categories = {}
        error_severities = {}
        nodes_with_errors = 0
        
        for node_id, analytics in node_analytics.items():
            if "error" in analytics:
                continue  # Skip nodes without analytics
                
            if "total_errors" in analytics:
                total_errors += analytics["total_errors"]
                nodes_with_errors += 1
                
                # Aggregate categories
                for category, count in analytics.get("error_categories", {}).items():
                    error_categories[category] = error_categories.get(category, 0) + count
                
                # Aggregate severities
                for severity, count in analytics.get("error_severities", {}).items():
                    error_severities[severity] = error_severities.get(severity, 0) + count
                
                # Aggregate remediation stats
                remediation_stats = analytics.get("remediation_stats", {})
                total_remediations += remediation_stats.get("total_remediations", 0)
        
        return {
            "graph_summary": {
                "total_nodes": len(self.nodes),
                "nodes_with_errors": nodes_with_errors,
                "total_errors": total_errors,
                "total_remediations": total_remediations
            },
            "error_distribution": {
                "categories": error_categories,
                "severities": error_severities
            },
            "node_analytics": node_analytics,
            "configuration": {
                "enable_gemini_remediation": self.enable_gemini_remediation,
                "auto_apply_fixes": self.auto_apply_fixes,
                "log_remediation": self.log_remediation,
                "state_schema": self.state_schema.__name__
            }
        }

    def clear_analytics(self):
        """Clear analytics data for all nodes"""
        for node_id, node in self.nodes.items():
            if hasattr(node, 'error_history'):
                node.error_history.clear()
            if hasattr(node, 'remediation_history'):
                node.remediation_history.clear()
        
        self.node_analytics.clear()
        print("✅ Analytics data cleared for all nodes")

# Import time module for timestamps
import time
import os
import json