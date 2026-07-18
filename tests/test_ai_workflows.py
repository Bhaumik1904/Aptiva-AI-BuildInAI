import pytest
from app import load_config
from core.models import JobDescription

# Ensure we use exactly the existing architecture
from agents.jd_agent import JDIntelligenceAgent
from agents.resume_agent import ResumeIntelligenceAgent
from agents.comparison_agent import ComparisonIntelligenceAgent
from agents.memory_agent import RecruiterMemoryAgent
from services.gnani_service import GnaniService

# Use real configuration for Keploy to record genuine traffic
CONFIG = load_config()

@pytest.fixture
def mock_pdf_parser(monkeypatch):
    """
    Mock the PDF parser so we can test the AI agent without needing a valid PDF file blob.
    This preserves the business logic of analyze() while feeding it a predictable string.
    """
    def mock_extract(file_bytes):
        return "John Doe. Senior Software Engineer. Skills: Python, AWS, Docker. 5 years experience."
    monkeypatch.setattr("agents.resume_agent.extract_text_pdf", mock_extract)

def test_e2e_ai_recruitment_workflow(mock_pdf_parser):
    """
    Workflow: JD Extraction -> Resume Parsing -> Candidate Comparison
    """
    # 1. JD Extraction
    jd_agent = JDIntelligenceAgent(CONFIG)
    jd_text = "Looking for a Senior Python Developer with FastAPI and AWS experience."
    jd_result, steps = jd_agent.analyze(jd_text)
    
    assert isinstance(jd_result, JobDescription)

    # 2. Resume Parsing (Mocking the binary extraction, testing the AI)
    resume_agent = ResumeIntelligenceAgent(CONFIG)
    resume_result_a, steps_a = resume_agent.analyze(b"dummy_bytes", "dummy.pdf")
    
    assert isinstance(resume_result_a, dict)
    
    # Create a second distinct candidate for realistic comparison
    resume_result_b = resume_result_a.copy()
    resume_result_b["name"] = "Alice Candidate"
    resume_result_a["name"] = "Bob Candidate"

    # 3. AI Comparison
    # The true signature is (self, jd, candidate_a, candidate_b, components_a, components_b)
    comp_agent = ComparisonIntelligenceAgent(CONFIG)
    comparison_result = comp_agent.compare(
        jd=jd_result, 
        candidate_a=resume_result_a, 
        candidate_b=resume_result_b, 
        components_a={}, 
        components_b={}
    )
    
    assert isinstance(comparison_result, dict)
    assert "hiring_recommendation" in comparison_result

def test_voice_and_memory_workflow():
    """
    Workflow: Voice Synthesizer -> Recruiter Memory Storage
    """
    # 1. Gnani Voice AI
    gnani = GnaniService(CONFIG)
    if not gnani.enabled:
        pytest.skip("Gnani is not configured in secrets; skipping test.")

    audio_bytes = gnani.synthesize("Candidate matched successfully.")
    assert isinstance(audio_bytes, bytes)

    # 2. Recruiter Memory (Mem0)
    # The true methods are store_jd_saved() and recall_all()
    memory = RecruiterMemoryAgent(CONFIG)
    if not memory.is_configured():
        pytest.skip("Mem0 is not configured in secrets; skipping test.")

    jd_mock = JobDescription(title="Python Dev", core_skills=["Python"])
    memory.store_jd_saved(jd=jd_mock)
    
    memories = memory.recall_all()
    assert isinstance(memories, list)

def test_keploy_replay_regression():
    """
    Keploy Replay Regression Test.
    This test verifies that Keploy's egress proxying works.
    When running `keploy test`, Keploy intercepts the outbound call to Gemini 
    and replays the recorded mock, allowing this to pass deterministically.
    """
    jd_agent = JDIntelligenceAgent(CONFIG)
    jd_result, _ = jd_agent.analyze("Looking for a fullstack developer.")
    assert isinstance(jd_result, JobDescription)
