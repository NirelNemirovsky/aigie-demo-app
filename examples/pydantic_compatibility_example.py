"""
Pydantic Compatibility Example for Aigie

This example demonstrates how to use Aigie with Pydantic models, solving the
state schema incompatibility issue that customers face when migrating from LangGraph.

The example shows:
1. How to define Pydantic models for workflow state
2. How to use the Pydantic-compatible Aigie graph
3. How to convert between Pydantic models and dictionaries
4. How to handle workflow-specific state management
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid

# Import Aigie components
from aigie import (
    PydanticCompatibleAigieGraph,
    WorkflowCompatibleAigieGraph,
    create_workflow_compatible_graph,
    pydantic_to_dict,
    dict_to_pydantic,
    validate_workflow_state
)


# Define workflow step enum
class WorkflowStep(str, Enum):
    TICKET_RECEPTION = "ticket_reception"
    INTENT_ANALYSIS = "intent_analysis"
    SOLUTION_GENERATION = "solution_generation"
    QUALITY_CHECK = "quality_check"
    RESPONSE_SENT = "response_sent"
    FOLLOW_UP_SCHEDULED = "follow_up_scheduled"


# Define ticket priority enum
class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# Define Pydantic models for the workflow state
class CustomerTicket(BaseModel):
    """Customer ticket model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    priority: TicketPriority = TicketPriority.MEDIUM
    customer_email: str
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)


class IntentAnalysis(BaseModel):
    """Intent analysis result"""
    intent: str
    confidence: float
    entities: List[str] = Field(default_factory=list)
    sentiment: str = "neutral"
    urgency_score: float = 0.5


class GeneratedSolution(BaseModel):
    """Generated solution"""
    solution_text: str
    confidence: float
    requires_human_review: bool = False
    estimated_resolution_time: Optional[int] = None  # in minutes


class QualityCheck(BaseModel):
    """Quality check result"""
    passed: bool
    score: float
    issues: List[str] = Field(default_factory=list)
    reviewer_notes: Optional[str] = None


class WorkflowState(BaseModel):
    """Main workflow state model - this is what the customer was using with LangGraph"""
    ticket_id: str
    current_step: WorkflowStep
    ticket: CustomerTicket
    intent_analysis: Optional[IntentAnalysis] = None
    generated_solution: Optional[GeneratedSolution] = None
    quality_check: Optional[QualityCheck] = None
    response_sent: bool = False
    follow_up_scheduled: bool = False
    error_message: Optional[str] = None
    workflow_started_at: datetime = Field(default_factory=datetime.now)
    last_updated_at: datetime = Field(default_factory=datetime.now)


def ticket_reception_node(state: WorkflowState) -> WorkflowState:
    """Node for receiving and validating tickets"""
    print(f"📋 Processing ticket: {state.ticket.title}")
    
    # Update the state
    state.current_step = WorkflowStep.INTENT_ANALYSIS
    state.last_updated_at = datetime.now()
    
    return state


def intent_analysis_node(state: WorkflowState) -> WorkflowState:
    """Node for analyzing customer intent"""
    print(f"🧠 Analyzing intent for ticket: {state.ticket.title}")
    
    # Simulate intent analysis
    intent_analysis = IntentAnalysis(
        intent="technical_support",
        confidence=0.85,
        entities=["API", "authentication"],
        sentiment="frustrated",
        urgency_score=0.7
    )
    
    state.intent_analysis = intent_analysis
    state.current_step = WorkflowStep.SOLUTION_GENERATION
    state.last_updated_at = datetime.now()
    
    return state


def solution_generation_node(state: WorkflowState) -> WorkflowState:
    """Node for generating solutions"""
    print(f"💡 Generating solution for: {state.intent_analysis.intent}")
    
    # Simulate solution generation
    solution = GeneratedSolution(
        solution_text="Please check your API key configuration and ensure it's properly set in the headers.",
        confidence=0.9,
        requires_human_review=False,
        estimated_resolution_time=15
    )
    
    state.generated_solution = solution
    state.current_step = WorkflowStep.QUALITY_CHECK
    state.last_updated_at = datetime.now()
    
    return state


def quality_check_node(state: WorkflowState) -> WorkflowState:
    """Node for quality checking solutions"""
    print(f"✅ Quality checking solution...")
    
    # Simulate quality check
    quality_check = QualityCheck(
        passed=True,
        score=0.95,
        issues=[],
        reviewer_notes="Solution looks comprehensive and actionable"
    )
    
    state.quality_check = quality_check
    state.current_step = WorkflowStep.RESPONSE_SENT
    state.last_updated_at = datetime.now()
    
    return state


def response_sent_node(state: WorkflowState) -> WorkflowState:
    """Node for sending responses"""
    print(f"📧 Sending response to customer...")
    
    state.response_sent = True
    state.current_step = WorkflowStep.FOLLOW_UP_SCHEDULED
    state.last_updated_at = datetime.now()
    
    return state


def follow_up_scheduled_node(state: WorkflowState) -> WorkflowState:
    """Node for scheduling follow-ups"""
    print(f"📅 Scheduling follow-up...")
    
    state.follow_up_scheduled = True
    state.last_updated_at = datetime.now()
    
    return state


def main():
    """Main example demonstrating Pydantic compatibility"""
    print("🚀 Aigie Pydantic Compatibility Example")
    print("=" * 50)
    
    # Create a sample ticket
    ticket = CustomerTicket(
        title="API Authentication Issue",
        description="I'm getting 401 errors when trying to access the API endpoints",
        priority=TicketPriority.HIGH,
        customer_email="customer@example.com",
        tags=["api", "authentication", "urgent"]
    )
    
    # Create initial workflow state (this is what the customer was doing with LangGraph)
    initial_state = WorkflowState(
        ticket_id=ticket.id,
        current_step=WorkflowStep.TICKET_RECEPTION,
        ticket=ticket
    )
    
    print(f"📋 Initial state created with ticket: {initial_state.ticket.title}")
    print(f"🔧 Current step: {initial_state.current_step}")
    print()
    
    # Method 1: Using WorkflowCompatibleAigieGraph (Recommended for workflows)
    print("🔧 Method 1: Using WorkflowCompatibleAigieGraph")
    print("-" * 40)
    
    workflow_graph = create_workflow_compatible_graph(
        WorkflowState,
        enable_gemini_remediation=True,
        auto_apply_fixes=False,
        log_remediation=True
    )
    
    # Add workflow nodes
    workflow_graph.add_workflow_node("ticket_reception", ticket_reception_node)
    workflow_graph.add_workflow_node("intent_analysis", intent_analysis_node)
    workflow_graph.add_workflow_node("solution_generation", solution_generation_node)
    workflow_graph.add_workflow_node("quality_check", quality_check_node)
    workflow_graph.add_workflow_node("response_sent", response_sent_node)
    workflow_graph.add_workflow_node("follow_up_scheduled", follow_up_scheduled_node)
    
    # Set up the workflow flow
    workflow_graph.set_entry_point("ticket_reception")
    workflow_graph.add_edge("ticket_reception", "intent_analysis")
    workflow_graph.add_edge("intent_analysis", "solution_generation")
    workflow_graph.add_edge("solution_generation", "quality_check")
    workflow_graph.add_edge("quality_check", "response_sent")
    workflow_graph.add_edge("response_sent", "follow_up_scheduled")
    workflow_graph.set_finish_point("follow_up_scheduled")
    
    # Execute the workflow with Pydantic model
    print("🔄 Executing workflow...")
    final_state = workflow_graph.invoke(initial_state)
    
    print(f"✅ Workflow completed!")
    print(f"📋 Final step: {final_state.current_step}")
    print(f"📧 Response sent: {final_state.response_sent}")
    print(f"📅 Follow-up scheduled: {final_state.follow_up_scheduled}")
    print()
    
    # Method 2: Manual conversion using utility functions
    print("🔧 Method 2: Manual conversion using utility functions")
    print("-" * 40)
    
    # Convert Pydantic model to dictionary
    state_dict = pydantic_to_dict(initial_state)
    print(f"📋 Converted to dictionary: {list(state_dict.keys())}")
    
    # Convert back to Pydantic model
    converted_state = dict_to_pydantic(state_dict, WorkflowState)
    print(f"📋 Converted back to Pydantic: {type(converted_state)}")
    print(f"🔧 Ticket title: {converted_state.ticket.title}")
    print()
    
    # Method 3: Validation utilities
    print("🔧 Method 3: Validation utilities")
    print("-" * 40)
    
    # Validate workflow state
    is_valid = validate_workflow_state(initial_state)
    print(f"✅ State validation: {is_valid}")
    
    # Validate with custom fields
    custom_fields = {"ticket_id", "current_step", "ticket"}
    is_valid_custom = validate_workflow_state(initial_state, custom_fields)
    print(f"✅ Custom validation: {is_valid_custom}")
    print()
    
    # Method 4: Using PydanticCompatibleAigieGraph (General purpose)
    print("🔧 Method 4: Using PydanticCompatibleAigieGraph")
    print("-" * 40)
    
    from aigie import PydanticCompatibleAigieGraph
    
    general_graph = PydanticCompatibleAigieGraph(
        WorkflowState,
        enable_gemini_remediation=True
    )
    
    # Add a simple node
    def simple_node(state: WorkflowState) -> WorkflowState:
        print(f"🔧 Simple node processing: {state.ticket.title}")
        state.ticket.tags.append("processed")
        return state
    
    general_graph.add_node("simple_processing", simple_node)
    general_graph.set_entry_point("simple_processing")
    general_graph.set_finish_point("simple_processing")
    
    # Execute
    result = general_graph.invoke(initial_state)
    print(f"✅ Simple processing completed!")
    print(f"🏷️ Tags: {result.ticket.tags}")
    print()
    
    # Show analytics
    print("📊 Analytics")
    print("-" * 40)
    analytics = workflow_graph.get_graph_analytics()
    print(f"📈 Total nodes: {analytics['graph_summary']['total_nodes']}")
    print(f"🔧 Configuration: {analytics['configuration']}")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Key Benefits:")
    print("   • Seamless integration with existing Pydantic models")
    print("   • No need to rewrite existing LangGraph workflows")
    print("   • Automatic state conversion and validation")
    print("   • Enhanced error handling with Gemini AI")
    print("   • Workflow-specific optimizations")


if __name__ == "__main__":
    main()
