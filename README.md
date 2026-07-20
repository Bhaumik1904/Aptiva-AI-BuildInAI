<div align="center">

# ⬡ APTIVA AI
### Intelligent Candidate Discovery & Ranking
**Build in AI for India Hackathon**

*An AI-powered recruitment intelligence platform that combines deterministic candidate ranking with explainable AI, recruiter memory, voice summaries, and AI-generated interview kits.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## Built With

- **Python**
- **Streamlit**
- **Google Gemini 2.5 Flash**
- **Mem0**
- **Gnani AI**
- **scikit-learn**
- **Plotly**

---

## Highlights

- **Explainable AI**
- **Deterministic Ranking**
- **AI Job Description Analysis**
- **Candidate Intelligence**
- **Voice AI (Gnani AI)**
- **Recruiter Memory (Mem0)**
- **AI Interview Kit**
- **Candidate Comparison**
- **Judge Mode**
- **Analytics Dashboard**

---

## Live Demo

🌐 **Streamlit Cloud**  
https://aptiva-ai-buildinai.streamlit.app/

💻 **GitHub Repository**  
https://github.com/Bhaumik1904/Aptiva-AI

---

## Demo Information

- **No login required.**
- **Hosted on Streamlit Community Cloud.**
- **Sample candidate dataset already loaded.** *(Note: The candidate dataset is preloaded only for convenience. The Job Description is intentionally created manually by the recruiter/judge so the AI can demonstrate real-time Job Description analysis).*
- **Judges only need to create or edit a Job Description.**
- **No manual dataset upload is required for evaluation.**
- **The workflow has been optimized for quick exploration.**

---

## Quick Start

Experience the APTIVA AI platform in under one minute:

1. **Open the Streamlit Demo** using the link above.
2. **Create a Hiring Project** and provide a Job Description (manually or upload).
3. **Select Candidate Sources** (use the preloaded Demo Dataset).
4. **Click Run Ranking** to process the candidate pool.
5. **Open AI Analysis** for any top candidate to see a 15-dimension breakdown and listen to **Voice AI summaries powered by Gnani AI**.
6. **Explore Judge Mode** to view the simulated recruiter verdict.
7. **Generate an AI Interview Kit** to prepare technical evaluation rubrics.
8. **Compare Candidates** side-by-side using the comparison tool.

---

## Product Overview

Aptiva AI is an advanced, AI-powered recruitment intelligence platform designed to help recruiters discover, evaluate, compare, and interview candidates with unprecedented efficiency. By combining the speed and fairness of deterministic ranking algorithms with the rich, contextual insights of generative AI, Aptiva AI transforms the traditional candidate screening process into an interactive and highly explainable experience.

Rather than relying on opaque, black-box AI screening systems, Aptiva AI grounds every recommendation in factual data. Our hybrid approach ensures algorithmic fairness while leveraging specialized AI agents to generate tailored interview kits, spoken recruiter summaries, and semantic skills matching—keeping the human recruiter firmly in the loop.

Aptiva AI was developed as a flagship submission for the **Build in AI for India Hackathon**, demonstrating how modern generative AI primitives can be orchestrated to solve real-world hiring challenges at scale.

---

## Why Aptiva AI?

Unlike traditional Applicant Tracking Systems (ATS) that often rely on opaque matching algorithms and tedious manual workflows, Aptiva AI is built for transparency, efficiency, and fairness:

- **Explainable AI:** Eliminates black-box recommendations by providing clear, fact-grounded reasoning for every candidate's ranking.
- **Deterministic Ranking with AI Reasoning:** Ensures algorithmic fairness by using a strict mathematical scoring pipeline, complemented by AI-generated insights.
- **Human-in-the-Loop Hiring:** Empowers recruiters with actionable data and suggestions while keeping the final hiring verdict firmly in their hands.
- **Personalized Recruiter Memory:** Learns individual recruiter preferences and patterns over time using **Mem0**, curating context-aware recommendations across multiple sessions.
- **Voice-Enabled Workflow:** Allows recruiters to listen to spoken candidate briefings on the go through **Gnani AI**, significantly speeding up candidate screening.
- **AI-Generated Interview Kits:** Automatically crafts technical questions and evaluation rubrics tailored specifically to the candidate's unique profile and the job description.
- **Fair Evaluation through Judge Mode:** Simulates comprehensive evaluations to identify risk factors and hidden strengths, ensuring an unbiased and robust assessment.
- **End-to-End Recruitment Workflow:** Seamlessly manages the entire process—from Job Description analysis and candidate ingestion to side-by-side comparison and interview prep—in one unified platform.

---

## Our AI Philosophy

- **Aptiva AI combines deterministic ranking with Generative AI.**
- **Ranking remains deterministic.**
- **AI never decides rankings.**
- **AI explains rankings.**
- **AI generates interview kits.**
- **AI summarizes candidates.**
- **AI personalizes recruiter workflows through Mem0.**
- **Human recruiters always make the final decision.**

---

## Key Features

- **Hiring Projects**
- **AI Job Description Analysis**
- **Candidate Dataset Management**
- **Deterministic Candidate Ranking**
- **Explainable AI Insights**
- **Voice AI Recruiter Briefs**
- **AI Interview Kit**
- **Candidate Comparison**
- **Judge Mode**
- **Recruiter Memory**
- **Analytics Dashboard**

---

## Application Preview

### Rankings Dashboard
![Rankings Dashboard](docs/images/rankings.png)

### AI Analysis
![AI Analysis](docs/images/ai_analysis.png)

### Candidate Profile
![Candidate Profile](docs/images/candidate_profile.png)

### Judge Mode
![Judge Mode](docs/images/judge_mode.png)

### Analytics Dashboard
![Analytics Dashboard](docs/images/analytics.png)

---

## System Architecture

APTIVA AI follows a sequential intelligence pipeline from project initialization to advanced candidate evaluation:

```text
Recruiter
↓
Hiring Project
↓
Job Description Intelligence
↓
Resume Intelligence
↓
Deterministic Ranking Engine
↓
Matching Agent
↓
Candidate Profile
├── Voice AI (Gnani AI)
├── Recruiter Memory (Mem0)
├── Interview Kit
├── Comparison
├── Judge Mode
└── Analytics
```

---

## Tech Stack

The Aptiva AI platform is built upon a robust and modern stack, separating logic and AI integrations clearly:

- **Frontend:** Streamlit
- **Backend:** Python
- **AI Models:** Google Gemini 2.5 Flash
- **Memory:** Mem0
- **Voice:** Gnani AI
- **Ranking:** Deterministic TF-IDF Engine
- **Deployment:** Streamlit Community Cloud

---

## AI Workflow & Agents

APTIVA AI leverages multiple specialized AI Agents to assist the recruiter throughout the workflow:

1. **Memory Agent (Mem0):** Learns and remembers recruiter preferences (e.g., "I prefer candidates with startup experience") and applies them to future shortlists.
2. **Matching Agent:** Analyzes the gap between the candidate's parsed skills and the Job Description to generate a semantic fit score.
3. **Shortlist Agent:** Recommends the absolute best candidates for the specific Hiring Project, highlighting why they stand out.
4. **Interview Agent:** Dynamically creates a customized Interview Kit for the recruiter, complete with technical questions and a grading rubric based on the candidate's claimed skills and identified weak points.
5. **Voice AI Integration (Gnani AI):** APTIVA AI synthesizes a professional spoken briefing about the candidate, allowing recruiters to get up to speed while multitasking.

---

## Recruiter Workflow

Aptiva AI is designed to seamlessly integrate into the recruiter's daily routine, providing end-to-end support from candidate ingestion to final evaluation:

1. **Create Hiring Project:** Initialize a dedicated workspace for a specific open role or recruitment drive.
2. **Define Job Description:** Upload or manually type the job description for AI to automatically parse and extract key requirements. *(Note: While the candidate dataset is preloaded for Demo Mode convenience, the JD is intentionally created manually to demonstrate real-time AI analysis).*
3. **Configure Skills and Experience:** Review and adjust the AI-extracted core skills, bonus skills, and ideal years of experience.
4. **Upload Candidate Dataset:** Ingest candidates via bulk CSV upload or by parsing PDF resumes directly into the platform.
5. **AI Candidate Ranking:** Run the deterministic ranking engine to score candidates across 7 dimensions and surface the top matches.
6. **Candidate Profile Analysis:** Deep dive into individual candidate profiles using the 15-dimension AI breakdown and explainable insights.
7. **Voice AI Summary:** Listen to a generated, professional spoken briefing by Gnani AI to quickly understand a candidate's fit on the go.
8. **AI Interview Kit Generation:** Automatically generate technical interview questions and scoring rubrics tailored specifically to the candidate and the JD.
9. **Candidate Comparison:** Place top candidates side-by-side using radar charts and head-to-head metric tables to spot key differences.
10. **Judge Mode Evaluation:** Review a simulated recruiter verdict (Strong Hire, Hire, Maybe, Pass) alongside potential risk factors.
11. **Analytics Dashboard:** Analyze the overall candidate pool demographics, skill distributions, and experience levels to refine the sourcing strategy.

---

## Repository Structure

```text
APTIVA AI/
├── app.py                # Streamlit application entry point
├── agents/               # AI Agents (Memory, Shortlist, Interview, Matching)
├── core/                 # Core scoring engine and algorithms
├── ui/                   # Streamlit UI pages and components
├── data/                 # Dataset directory
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

### Detailed Architecture

```text
APTIVA AI/
│
├── agents/                   AI Agents
│   ├── interview_agent.py    AI Interview generation
│   ├── matching_agent.py     Candidate matching
│   ├── memory_agent.py       Recruiter preferences memory
│   ├── shortlist_agent.py    AI shortlisting
│   ├── report_agent.py       Reporting intelligence
│   ├── resume_agent.py       Resume intelligence
│   └── skill_gap_agent.py    Skill gap intelligence
│
├── core/                     Scoring engine
│   ├── jd_config.py          JD feature vector — single source of truth
│   ├── scorer.py             7 component scoring functions + final combiner
│   ├── csv_loader.py         CSV dataset parser
│   ├── zip_loader.py         ZIP archive extraction
│   ├── data_ingestion.py     Data pipeline entry point
│   ├── behavioral.py         15-signal behavioral multiplier
│   ├── honeypot.py           Adversarial profile detection
│   ├── hireability.py        Hireability Index™ computation
│   ├── skill_gap.py          Core/Missing/Bonus skill classification
│   ├── reasoning.py          Ranking reasoning + AI insights generation
│   ├── judge_mode.py         Judge Mode verdict generation
│   └── gemini_enricher.py    Offline Gemini reasoning enrichment
│
├── ui/                       Streamlit demo pages
│   ├── styles.py             Apple-inspired CSS design system
│   ├── components.py         Reusable UI components
│   ├── charts.py             Plotly chart builders
│   └── pages/
│       ├── home.py           Rankings Dashboard (entry point)
│       ├── ai_analysis.py    AI Analysis — 15-dimension candidate view
│       ├── candidate_profile.py  Candidate deep-dive
│       ├── comparison.py     Side-by-side candidate comparison
│       ├── judge_mode_page.py    Judge Mode — recruiter verdict simulation
│       ├── analytics.py      Dataset-wide analytics dashboard
│       ├── projects.py       Hiring Projects management
│       ├── candidate_sources.py  Candidate ingestion (CSV/Resume)
│       └── interview.py      AI Interview Kit generation
│
└── data/                     Dataset directory (not committed)
```

---

## Deterministic Ranking Pipeline

The core ranking engine is 100% deterministic, ensuring algorithmic fairness and explainability.

### Final Score Formula

```text
Final Score = min(1.0, Base Score × Behavioral Multiplier)
```

### Why Deterministic Ranking?

Relying entirely on LLMs for sorting hundreds of candidates is often opaque, unpredictable, and prone to hallucinations. Aptiva AI uses a **7-component mathematical pipeline** to rank candidates objectively based on pure qualifications. 

Because the ranking formula is deterministic, the AI layer can effortlessly and accurately **explain exactly why** a candidate received their score. The generative AI agents read the deterministic math and generate transparent, fact-grounded insights for the recruiter.

---

## Partner Integrations

### Mem0
Mem0 provides powerful recruiter memory by tracking and remembering recruiter preferences, historical hiring patterns, preferred skills, and contextual feedback across hiring sessions. This continuous learning enables highly personalized candidate recommendations tailored to the specific recruiter, while maintaining the integrity of the underlying deterministic ranking engine.

### Gnani AI
Gnani AI powers the Voice AI feature by converting our AI-generated candidate summaries into natural, professional speech. This audio-first approach allows recruiters to quickly listen to candidate profiles and key insights on the go, accelerating the screening process without the need to read long reports.

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### Install

```bash
git clone https://github.com/Bhaumik1904/Aptiva-AI.git
cd Aptiva-AI
pip install -r requirements.txt
```

### Dataset Setup

Place your candidate dataset in the `data/` directory or upload it via the **Candidate Sources** UI in the app. The application auto-extracts and loads the data dynamically.

---

## Usage

### Run the Streamlit Demo

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### External AI Configuration

To enable advanced GenAI features (Interview Kit, Voice Summaries, Memory), configure your API keys:

```bash
# Windows
set GEMINI_API_KEY=your_key_here
set MEM0_API_KEY=your_key_here

# macOS / Linux
export GEMINI_API_KEY=your_key_here
export MEM0_API_KEY=your_key_here
```

---

## Future Roadmap

| Enhancement | Description |
|---|---|
| **Graph Neural Networks** | Model the candidate pool as a knowledge graph to find hidden relational patterns (university → company → skill correlations). |
| **Career Trajectory Forecasting** | Sequence models (Transformers) on chronological career history to predict candidate fit beyond current title. |
| **Multi-agent RAG Outreach** | LLM agents (LangChain/AutoGen) to automate recruiter email drafting post-ranking. |
| **Real-time Streaming** | Kafka + Flink pipeline to update rankings as new candidates enter the platform. |

---

## Dependencies

```text
streamlit>=1.32.0       # Interactive demo
pandas>=2.0.0           # Data handling
numpy>=1.24.0           # Numerical operations
scikit-learn>=1.3.0     # TF-IDF vectorization
plotly>=5.18.0          # Interactive charts
pyyaml>=6.0.0           # Config file parsing
google-generativeai>=0.7.0  # AI Agents & Explanations
mem0ai>=0.1.0           # Recruiter Memory
```

---

## Acknowledgements

Built for the **Build in AI for India Hackathon**.

---

<div align="center">

**Built for the Build in AI for India Hackathon**

*APTIVA AI · Intelligent Candidate Discovery · 2026*

</div>

---

## CHANGELOG
- **Repository Structure:** Verified local directories. Removed non-existent legacy files (`rank.py`, `evaluate.py`, `evaluation/`, `tfidf_engine.py`, `dataset_loader.py`) and synced structure to accurately reflect current modules (`data_ingestion.py`, `csv_loader.py`, `zip_loader.py`, `resume_agent.py`, etc.).
- **Technical Deep Dive:** Simplified the "Deterministic Ranking Pipeline" section, removing overly complex algorithmic weighting tables while preserving the core philosophy, final score formula, and explainability reasoning.
- **Demo Information:** Replaced "Demo Experience" with a structured "Demo Information" section explicitly detailing the no-login, cloud-hosted setup.
- **Job Description Context:** Added targeted notes inside the Demo Information and Recruiter Workflow clarifying that the JD is created manually to showcase real-time AI parsing despite the preloaded candidate dataset.
- **AI Philosophy:** Added an "Our AI Philosophy" section to reinforce the strict boundary between deterministic ranking logic and AI-driven insights/explainability.
- **Built With & Highlights:** Appended "Built With" and "Highlights" sections to the top of the document for rapid technological and feature overview.
- **Keploy Egress Testing:** Checked repository. The framework is no longer active in the current stack, so the Keploy documentation section was completely removed.
- **Final Consistency:** Verified that all content accurately represents Aptiva AI natively as a "Build in AI for India Hackathon" submission without any remaining legacy wording or artifacts.
