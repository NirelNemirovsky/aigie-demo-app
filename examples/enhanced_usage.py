"""
Enhanced Usage Example for Aigie with Proper Pydantic Model Conversion

This example demonstrates the correct way to use Aigie with Pydantic models,
addressing the specific issue where Aigie expects plain dictionaries but
we want to work with structured Pydantic models.

The key insight: Aigie internally uses plain dictionaries, but we provide
a seamless interface that handles conversion automatically.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid
import os

# Import the enhanced Aigie components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.aigie.aigie_state_graph import AigieStateGraph


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
    customer_id: str
    subject: str
    message: str
    email: str
    created_at: datetime = Field(default_factory=datetime.now)
    attachments: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


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
    """Main workflow state model - this is what we work with in our code"""
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
    
    # AIGIE-specific fields for enhanced error handling and remediation
    error_details: Optional[Dict[str, Any]] = None
    gemini_remediation_details: Optional[Dict[str, Any]] = None
    aigie_metadata: Optional[Dict[str, Any]] = None
    last_node: Optional[str] = None


# Define workflow nodes that work directly with Pydantic models
def ticket_reception_node(state: WorkflowState) -> WorkflowState:
    """Node for receiving and validating tickets"""
    print(f"📋 Processing ticket: {state.ticket.subject}")
    print(f"   Customer: {state.ticket.customer_id}")
    print(f"   Email: {state.ticket.email}")
    
    # Update the state directly - no conversion needed!
    state.current_step = WorkflowStep.INTENT_ANALYSIS
    state.last_updated_at = datetime.now()
    
    return state


def intent_analysis_node(state: WorkflowState) -> WorkflowState:
    """Node for analyzing customer intent"""
    print(f"🧠 Analyzing intent for ticket: {state.ticket.subject}")
    
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


def demonstrate_native_pydantic():
    """Demonstrate that Pydantic models work natively with LangGraph"""
    print("\n🔄 DEMONSTRATING NATIVE PYDANTIC SUPPORT")
    print("=" * 50)
    
    # Create a sample workflow state
    ticket = CustomerTicket(
        customer_id="aigie-0.3.0-test",
        subject="Testing Aigie Version 0.3.0",
        message="This is a test ticket to verify native Pydantic support.",
        email="v0.3.0@aigie.com"
    )
    
    initial_state = WorkflowState(
        ticket_id=ticket.id,
        current_step=WorkflowStep.TICKET_RECEPTION,
        ticket=ticket
    )
    
    print("📋 Pydantic Model:")
    print(f"   Type: {type(initial_state)}")
    print(f"   current_step: {initial_state.current_step} (type: {type(initial_state.current_step)})")
    print(f"   ticket: {type(initial_state.ticket)}")
    print(f"   workflow_started_at: {type(initial_state.workflow_started_at)}")
    
    print("\n✅ No conversion needed!")
    print("   • LangGraph supports Pydantic models natively")
    print("   • Enum values are preserved as-is")
    print("   • Datetime objects work directly")
    print("   • Nested Pydantic models are supported")
    print("   • No conversion overhead or errors")


def main():
    """Main example demonstrating the enhanced Aigie usage"""
    print("🚀 Enhanced Aigie Usage Example")
    print("=" * 50)
    print("✨ Proper Pydantic Model to Dictionary Conversion!")
    print("🔧 Seamless integration with Aigie's dictionary-based system")
    print()
    
    # Demonstrate native Pydantic support
    demonstrate_native_pydantic()
    
    # Create a sample ticket
    ticket = CustomerTicket(
        customer_id="aigie-0.3.0-test",
        subject="API Authentication Issue",
        message="I'm getting 401 errors when trying to access the API endpoints",
        email="customer@example.com"
    )
    
    # Create initial workflow state - this is what we work with!
    initial_state = WorkflowState(
        ticket_id=ticket.id,
        current_step=WorkflowStep.TICKET_RECEPTION,
        ticket=ticket
    )
    
    print(f"\n📋 Initial state created with ticket: {initial_state.ticket.subject}")
    print(f"🔧 Current step: {initial_state.current_step}")
    print(f"✅ Type: {type(initial_state)} (Pydantic model)")
    print()
    
    # Create the Aigie graph with Pydantic schema
    print("🔧 Creating Aigie graph with Pydantic schema...")
    workflow_graph = AigieStateGraph(
        state_schema=WorkflowState,  # Pass the Pydantic model class!
        enable_gemini_remediation=True,
        gemini_project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),  # Use latest Google Gen AI standards
        auto_apply_fixes=False,
        log_remediation=True
    )
    
    # Add nodes - they work directly with Pydantic models
    workflow_graph.add_node("ticket_reception", ticket_reception_node)
    workflow_graph.add_node("analyze_intent", intent_analysis_node)
    workflow_graph.add_node("generate_solution", solution_generation_node)
    workflow_graph.add_node("check_quality", quality_check_node)
    workflow_graph.add_node("send_response", response_sent_node)
    workflow_graph.add_node("schedule_followup", follow_up_scheduled_node)
    
    # Set up the workflow flow
    workflow_graph.set_entry_point("ticket_reception")
    workflow_graph.add_edge("ticket_reception", "analyze_intent")
    workflow_graph.add_edge("analyze_intent", "generate_solution")
    workflow_graph.add_edge("generate_solution", "check_quality")
    workflow_graph.add_edge("check_quality", "send_response")
    workflow_graph.add_edge("send_response", "schedule_followup")
    workflow_graph.set_finish_point("schedule_followup")
    
    # Execute the workflow with Pydantic model (handles conversion automatically)
    print("🔄 Executing workflow...")
    final_state = workflow_graph.invoke(initial_state)
    
    print(f"\n✅ Workflow completed!")
    print(f"📋 Final step: {final_state.current_step}")
    print(f"📧 Response sent: {final_state.response_sent}")
    print(f"📅 Follow-up scheduled: {final_state.follow_up_scheduled}")
    print(f"✅ Final state type: {type(final_state)} (Still a Pydantic model!)")
    print()
    
    # Demonstrate built-in validation
    print("🔧 Built-in Pydantic validation...")
    is_valid = workflow_graph.validate_state(final_state)
    print(f"✅ State validation: {is_valid}")
    
    # Demonstrate conversion utilities
    print("🔄 Conversion utilities...")
    state_dict = workflow_graph.to_dict(final_state)
    print(f"📋 Converted to dict: {list(state_dict.keys())}")
    
    converted_back = workflow_graph.from_dict(state_dict)
    print(f"📋 Converted back to Pydantic: {type(converted_back)}")
    print(f"🔧 Ticket subject: {converted_back.ticket.subject}")
    print()
    
    # Show analytics
    print("📊 Analytics")
    print("-" * 40)
    analytics = workflow_graph.get_graph_analytics()
    print(f"📈 Total nodes: {analytics['graph_summary']['total_nodes']}")
    print(f"🔧 State schema: {analytics['configuration']['state_schema']}")
    print(f"🤖 Gemini remediation: {analytics['configuration']['enable_gemini_remediation']}")
    
    print("\n🎉 Enhanced example completed successfully!")
    print("\n💡 Key Benefits of This Approach:")
    print("   • Work with Pydantic models in your code")
    print("   • Native Pydantic support with LangGraph")
    print("   • Type safety and validation throughout")
    print("   • Seamless integration with existing LangGraph workflows")
    print("   • Enhanced error handling with Gemini AI")
    print("   • No conversion needed")
    print("\n🔧 What Happens Internally:")
    print("   • Your code works with Pydantic models")
    print("   • LangGraph handles Pydantic models natively")
    print("   • No conversion overhead or errors")
    print("   • Direct Pydantic model support")


if __name__ == "__main__":
    main()
