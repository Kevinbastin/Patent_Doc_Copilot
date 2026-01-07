"""
CrewAI Patent Verification System
==================================
Uses CrewAI with 5 specialized agents for IPO compliance verification.
Configured to use OpenRouter API.
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Ensure CrewAI uses OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from crewai import Agent, Task, Crew, Process
from typing import Dict


def verify_patent_5_sections(patent_sections: Dict[str, str]) -> str:
    """
    5-Agent Patent Verification using CrewAI
    Uses OpenRouter API with Qwen 2.5 model
    """
    
    try:
        print("🔍 CREWAI PATENT VERIFICATION SYSTEM")
        print("=" * 60)
        print("📋 Indian Patent Office (IPO) Compliance Check")
        print("🤖 5 CrewAI Agents")
        print("=" * 60)
        
        # Agent 1: Title & Abstract Validator
        agent1 = Agent(
            role="Title & Abstract Examiner",
            goal="Verify title and abstract comply with Indian Patent Office standards",
            backstory="Senior IPO examiner with 15 years experience in title and abstract evaluation according to Patents Act, 1970.",
            verbose=False,
            allow_delegation=False,
            llm="openrouter/qwen/qwen-2.5-7b-instruct"
        )
        
        task1 = Task(
            description=f"""Analyze this patent title and abstract for IPO compliance:

TITLE: {patent_sections.get('title', 'Not provided')}

ABSTRACT: {patent_sections.get('abstract', 'Not provided')}

Verify:
1. Title is 10-15 words
2. No forbidden starting words (A, An, The)
3. Abstract is ≤150 words
4. Both are clear and technical

Provide: Compliance Status (PASS/FAIL), Score (X/10), Issues, Recommendations.""",
            expected_output="Structured compliance report",
            agent=agent1
        )
        
        # Agent 2: Claims Expert
        agent2 = Agent(
            role="Claims Attorney",
            goal="Verify claims structure and legal compliance",
            backstory="Patent attorney specializing in Indian patent claims with expertise in single-sentence requirement.",
            verbose=False,
            allow_delegation=False,
            llm="openrouter/qwen/qwen-2.5-7b-instruct"
        )
        
        task2 = Task(
            description=f"""Analyze these patent claims:

CLAIMS: {patent_sections.get('claims', 'Not provided')}

CRITICAL IPO REQUIREMENTS:
- Each claim MUST be a SINGLE SENTENCE
- Claims numbered 1., 2., 3., etc.
- Independent claim defines invention broadly
- Dependent claims reference parent properly

Provide: Compliance Status, Score (X/10), Issues, Recommendations.""",
            expected_output="Claims compliance report",
            agent=agent2
        )
        
        # Agent 3: Background Analyst
        agent3 = Agent(
            role="Prior Art Specialist",
            goal="Verify background section quality",
            backstory="Technical writer with expertise in patent background and prior art analysis.",
            verbose=False,
            allow_delegation=False,
            llm="openrouter/qwen/qwen-2.5-7b-instruct"
        )
        
        task3 = Task(
            description=f"""Analyze this background section:

BACKGROUND: {patent_sections.get('background', 'Not provided')}

Verify:
- Describes technical field
- Mentions prior art limitations
- Does NOT disparage specific products
- 2-3 paragraphs

Provide: Compliance Status, Score (X/10), Issues, Recommendations.""",
            expected_output="Background compliance report",
            agent=agent3
        )
        
        # Agent 4: Summary Expert
        agent4 = Agent(
            role="Summary Specialist",
            goal="Verify summary follows IPO format",
            backstory="Expert in patent summary drafting for Indian Complete Specification applications.",
            verbose=False,
            allow_delegation=False,
            llm="openrouter/qwen/qwen-2.5-7b-instruct"
        )
        
        task4 = Task(
            description=f"""Analyze this summary section:

SUMMARY: {patent_sections.get('summary', 'Not provided')}

IPO REQUIREMENTS:
- Should start with "Thus according to the present invention..."
- Use formal technical language
- Cover all invention aspects

Provide: Compliance Status, Score (X/10), Issues, Recommendations.""",
            expected_output="Summary compliance report",
            agent=agent4
        )
        
        # Agent 5: Filing Coordinator
        agent5 = Agent(
            role="Filing Director",
            goal="Compile final verification report",
            backstory="Deputy Controller at IPO with 25 years experience in final patent review.",
            verbose=False,
            allow_delegation=False,
            llm="openrouter/qwen/qwen-2.5-7b-instruct"
        )
        
        task5 = Task(
            description="""Based on all previous analyses, compile a FINAL REPORT with:
1. Overall PASS/FAIL status
2. Overall Score (X/10)
3. Priority issues to fix
4. Filing readiness assessment""",
            expected_output="Final consolidated verification report",
            agent=agent5,
            context=[task1, task2, task3, task4]
        )
        
        # Create and run crew
        print("\n� Starting CrewAI verification...")
        
        crew = Crew(
            agents=[agent1, agent2, agent3, agent4, agent5],
            tasks=[task1, task2, task3, task4, task5],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Format report
        report = []
        report.append("=" * 70)
        report.append("🔍 CREWAI PATENT VERIFICATION REPORT")
        report.append("   Indian Patent Office (IPO) Compliance")
        report.append("   5-Agent Analysis Complete")
        report.append("=" * 70)
        report.append("")
        report.append(str(result))
        report.append("")
        report.append("=" * 70)
        
        print("✅ CrewAI Verification Complete!")
        
        return "\n".join(report)
        
    except Exception as e:
        import traceback
        return f"❌ CrewAI Error:\n{traceback.format_exc()}"


if __name__ == "__main__":
    test_sections = {
        'title': 'SMART MONITORING SYSTEM FOR INDUSTRIAL ENVIRONMENTS',
        'abstract': 'A smart monitoring system comprising sensors and a processing hub.',
        'claims': '1. A smart monitoring system comprising sensor units and a hub.',
        'background': 'Industrial environments need monitoring for safety.',
        'summary': 'Thus according to the present invention, there is provided a monitoring system.'
    }
    
    print("\nTesting CrewAI Patent Verifier...")
    result = verify_patent_5_sections(test_sections)
    print(result)
