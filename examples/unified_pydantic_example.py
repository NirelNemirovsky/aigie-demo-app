"""
Unified Pydantic Example for Aigie

This example demonstrates the new unified approach where Aigie natively supports
Pydantic models as the single standard for state management.

No more multiple versions or compatibility layers - just one clean, standard approach!
"""

from typing import Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid

# Import the unified Aigie components
from aigie import AigieStateGraph


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
    """Main workflow state model - this is the single standard!"""
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


# Define workflow nodes that work directly with Pydantic models
def ticket_reception_node(state: WorkflowState) -> WorkflowState:
    """Node for receiving and validating tickets"""
    print(f"📋 Processing ticket: {state.ticket.title}")
    
    # Update the state directly - no conversion needed!
    state.current_step = WorkflowStep.INTENT_ANALYSIS
    state.last_updated_at = datetime.now()
    
    return state


def intent_analysis_node(state: WorkflowState) -> WorkflowState:
    """Node for analyzing customer intent"""
    print(f"🧠 Analyzing intent for ticket: {state.ticket.title}")
    
    # Create and assign Pydantic models directly
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
    
    # Create and assign Pydantic models directly
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
    
    # Create and assign Pydantic models directly
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
    """Main example demonstrating the unified Pydantic approach"""
    print("🚀 Aigie Unified Pydantic Example")
    print("=" * 50)
    print("✨ Single Standard: Pydantic Models Only!")
    print("🔧 No more multiple versions or compatibility layers")
    print()
    
    # Create a sample ticket
    ticket = CustomerTicket(
        title="API Authentication Issue",
        description="I'm getting 401 errors when trying to access the API endpoints",
        priority=TicketPriority.HIGH,
        customer_email="customer@example.com",
        tags=["api", "authentication", "urgent"]
    )
    
    # Create initial workflow state - this is the single standard!
    initial_state = WorkflowState(
        ticket_id=ticket.id,
        current_step=WorkflowStep.TICKET_RECEPTION,
        ticket=ticket
    )
    
    print(f"📋 Initial state created with ticket: {initial_state.ticket.title}")
    print(f"🔧 Current step: {initial_state.current_step}")
    print(f"✅ Type: {type(initial_state)} (Pydantic model)")
    print()
    
    # Create the Aigie graph with Pydantic schema
    print("🔧 Creating Aigie graph with Pydantic schema...")
    workflow_graph = AigieStateGraph(
        state_schema=WorkflowState,  # Pass the Pydantic model class directly!
        enable_gemini_remediation=True,
        auto_apply_fixes=False,
        log_remediation=True
    )
    
    # Add nodes - they work directly with Pydantic models
    workflow_graph.add_node("ticket_reception", ticket_reception_node)
    workflow_graph.add_node("intent_analysis", intent_analysis_node)
    workflow_graph.add_node("solution_generation", solution_generation_node)
    workflow_graph.add_node("quality_check", quality_check_node)
    workflow_graph.add_node("response_sent", response_sent_node)
    workflow_graph.add_node("follow_up_scheduled", follow_up_scheduled_node)
    
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
    compiled_graph = workflow_graph.compile()
    final_state = compiled_graph.invoke(initial_state)
    
    print(f"✅ Workflow completed!")
    print(f"📋 Final step: {final_state.current_step}")
    print(f"📧 Response sent: {final_state.response_sent}")
    print(f"📅 Follow-up scheduled: {final_state.follow_up_scheduled}")
    print(f"✅ Final state type: {type(final_state)} (Still a Pydantic model!)")
    print()
    
    # Demonstrate built-in validation
    print("🔧 Built-in Pydantic validation...")
    is_valid = workflow_graph.validate_state(final_state)
    print(f"✅ State validation: {is_valid}")
    
    # Demonstrate conversion utilities (if needed)
    print("🔄 Conversion utilities (for external systems)...")
    state_dict = workflow_graph.to_dict(final_state)
    print(f"📋 Converted to dict: {list(state_dict.keys())}")
    
    converted_back = workflow_graph.from_dict(state_dict)
    print(f"📋 Converted back to Pydantic: {type(converted_back)}")
    print(f"🔧 Ticket title: {converted_back.ticket.title}")
    print()
    
    # Show analytics
    print("📊 Analytics")
    print("-" * 40)
    analytics = workflow_graph.get_graph_analytics()
    print(f"📈 Total nodes: {analytics['graph_summary']['total_nodes']}")
    print(f"🔧 State schema: {analytics['configuration']['state_schema']}")
    print(f"🤖 Gemini remediation: {analytics['configuration']['enable_gemini_remediation']}")
    
    print("\n🎉 Unified example completed successfully!")
    print("\n💡 Key Benefits of the Unified Approach:")
    print("   • Single standard: Pydantic models only")
    print("   • No compatibility layers or multiple versions")
    print("   • Native type safety and validation")
    print("   • Seamless integration with existing LangGraph workflows")
    print("   • Enhanced error handling with Gemini AI")
    print("   • Clean, maintainable code")
    print("\n🚫 What we eliminated:")
    print("   • Multiple state formats (dict, TypedDict, Pydantic)")
    print("   • Compatibility adapters and wrappers")
    print("   • Conversion utilities for basic operations")
    print("   • Confusion about which approach to use")


if __name__ == "__main__":
    main()
