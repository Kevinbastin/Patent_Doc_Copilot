"""
ULTIMATE Patent Verification System
CrewAI + Qwen 2.5 7B + 5 Agents
UPDATED: Using local Qwen 2.5 model via llm_runtime.py
"""


from crewai import Agent, Task, Crew, Process
from langchain.llms.base import LLM
from typing import Optional, List, Any
from llm_runtime import llm_generate



class QwenLLM(LLM):
    """Custom LangChain LLM wrapper for Qwen 2.5 model"""
    
    temperature: float = 0.3
    max_tokens: int = 400
    top_p: float = 0.9
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Qwen model through llm_runtime.py"""
        try:
            response = llm_generate(
                prompt=prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=1.2,
                stop_strings=stop if stop else ["\n\n"]
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"



def verify_patent_5_sections(patent_sections):
    """5-Agent Patent Verification with Qwen 2.5 7B"""
   
    try:
        # Initialize Qwen 2.5 model wrapper
        local_llm = QwenLLM(
            temperature=0.3,
            max_tokens=400,
            top_p=0.9
        )
       
        print("✅ Qwen 2.5 7B initialized (local model)")
        print("\n🤖 CrewAI 5-Agent Patent Verification")
        print("="*60)
       
        # Agent 1: Title & Abstract Validator
        agent1 = Agent(
            role="Title & Abstract Validator",
            goal="Verify title and abstract USPTO compliance",
            backstory="USPTO patent examiner with 15 years experience specializing in title and abstract evaluation.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task1 = Task(
            description=f"""Analyze the following patent title and abstract for USPTO compliance:


TITLE: {patent_sections['title']}


ABSTRACT: {patent_sections['abstract']}


Provide a structured analysis with:
1. Title word count (requirement: 10-15 words optimal)
2. Title verdict: PASS/FAIL
3. Abstract word count (requirement: ≤150 words)
4. Abstract clarity rating: EXCELLENT/GOOD/FAIR/POOR
5. Abstract verdict: PASS/FAIL


Be concise and specific in your evaluation.""",
            expected_output="Structured title and abstract analysis with pass/fail verdicts",
            agent=agent1
        )
       
        # Agent 2: Claims Analyzer
        agent2 = Agent(
            role="Claims Analyzer",
            goal="Validate claims structure and dependencies",
            backstory="Senior patent examiner with 20 years experience in claims analysis and patent prosecution.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task2 = Task(
            description=f"""Analyze the following patent claims for structure and compliance:


CLAIMS:
{patent_sections['claims']}


Provide:
1. Total claims count
2. Independent vs dependent claims breakdown
3. Numbering assessment: PROPER/IMPROPER
4. Overall quality: EXCELLENT/GOOD/FAIR/POOR
5. Critical issues (if any)


Focus on structural compliance and clarity.""",
            expected_output="Detailed claims structure analysis with quality assessment",
            agent=agent2
        )
       
        # Agent 3: Background Reviewer
        agent3 = Agent(
            role="Background Reviewer",
            goal="Evaluate background section and prior art discussion",
            backstory="Patent attorney specializing in prior art analysis and background section evaluation.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task3 = Task(
            description=f"""Review the background section for completeness and compliance:


BACKGROUND:
{patent_sections['background']}


Provide:
1. Technical field identified: YES/NO
2. Problem statement clarity: EXCELLENT/GOOD/FAIR/POOR
3. Prior art discussed: YES/NO
4. Overall quality: EXCELLENT/GOOD/FAIR/POOR


Evaluate if the background properly sets up the invention.""",
            expected_output="Background section review with quality metrics",
            agent=agent3
        )
       
        # Agent 4: Summary Evaluator
        agent4 = Agent(
            role="Summary Evaluator",
            goal="Assess summary of invention completeness",
            backstory="Patent examiner specializing in evaluating invention summaries for clarity and completeness.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task4 = Task(
            description=f"""Evaluate the summary of invention:


SUMMARY:
{patent_sections['summary']}


Provide:
1. Completeness: COMPLETE/PARTIAL/INCOMPLETE
2. Invention clearly described: YES/NO
3. Key features mentioned: YES/NO
4. Overall quality: EXCELLENT/GOOD/FAIR/POOR


Assess if the summary adequately describes the invention.""",
            expected_output="Summary evaluation with completeness assessment",
            agent=agent4
        )
       
        # Agent 5: Quality Judge
        agent5 = Agent(
            role="Quality Judge",
            goal="Provide final assessment and overall quality score",
            backstory="Senior patent partner with 25 years experience in patent prosecution and quality assessment.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )
       
        task5 = Task(
            description=f"""Provide a comprehensive final assessment for the patent application:


PATENT TITLE: {patent_sections['title']}


Based on all previous analyses, provide:


1. Consistency score: 0-100 (how well all sections align)
2. Critical issues (list top 3 most important)
3. Strengths (list top 2)
4. Overall quality score breakdown (total: 0-100):
   - Title/Abstract: X/25 points
   - Claims: X/30 points
   - Background: X/20 points
   - Summary: X/15 points
   - Consistency: X/10 points
5. Filing ready: YES/NO
6. Priority actions (if not ready, list specific improvements needed)


Provide actionable recommendations.""",
            expected_output="Comprehensive final quality assessment with scoring and recommendations",
            agent=agent5,
            context=[task1, task2, task3, task4]
        )
       
        # Create crew with all agents and tasks
        crew = Crew(
            agents=[agent1, agent2, agent3, agent4, agent5],
            tasks=[task1, task2, task3, task4, task5],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=False
        )
       
        print("✅ Created 5 specialized agents")
        print("✅ Created 5 verification tasks")
        print("🚀 Running verification pipeline...\n")
       
        result = crew.kickoff()
       
        print("\n✅ Verification Complete!")
        print("="*60)
       
        return f"""
╔═══════════════════════════════════════════════════════════╗
║     5-AGENT PATENT VERIFICATION REPORT                    ║
╚═══════════════════════════════════════════════════════════╝


{result}


╔═══════════════════════════════════════════════════════════╗
║ System: CrewAI + Qwen 2.5 7B (5 Specialized Agents)      ║
║ Model: /workspace/patentdoc-copilot/models/Qwen2.5-7B    ║
╚═══════════════════════════════════════════════════════════╝
"""
       
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"\n❌ ERROR in verification process:\n{error_msg}")
        return f"❌ Error: {error_msg}"



if __name__ == "__main__":
    # Test patent sections
    test_patent = {
        'title': 'Smart Agricultural Monitoring System Using IoT Sensors and Machine Learning',
        'abstract': '''A comprehensive smart agricultural monitoring system that integrates
        IoT sensors with machine learning algorithms to provide real-time monitoring and
        predictive analytics for crop management. The system comprises soil moisture sensors,
        temperature sensors, humidity sensors, and a processing unit that analyzes data using
        advanced algorithms to predict irrigation requirements and optimize water usage.''',
        'claims': '''1. A smart agricultural monitoring system comprising:
   a) a plurality of soil moisture sensors configured to detect moisture at multiple depths;
   b) temperature sensors distributed across a monitored area;
   c) a central processing unit operatively connected to said sensors;
   d) a wireless communication module for data transmission; and
   e) a machine learning module configured to analyze sensor data and predict irrigation requirements.


2. The system of claim 1, wherein the soil moisture sensors detect moisture at depths of 10cm, 20cm, and 30cm.


3. The system of claim 1, wherein the machine learning module employs a neural network trained on historical agricultural data.


4. The system of claim 1, further comprising a user interface for displaying real-time monitoring data and predictions.''',
        'background': '''Traditional farming methods rely heavily on manual monitoring and scheduled irrigation,
        often leading to water wastage and suboptimal crop yields. Farmers typically lack access to real-time data
        about soil conditions and environmental factors. Prior art includes basic sensor systems (US Patent 9,123,456)
        that provide simple moisture readings, but these lack integrated machine learning capabilities for predictive
        analytics. Other systems (US Patent 8,765,432) focus on weather prediction but do not incorporate soil-level
        monitoring. There remains a need for an integrated system combining multi-parameter sensing with intelligent
        predictive analytics.''',
        'summary': '''The present invention provides a comprehensive smart agricultural monitoring system that addresses
        the limitations of existing solutions. By integrating multiple types of sensors with machine learning algorithms,
        the system provides accurate real-time monitoring and predictive analytics for optimized crop management. The
        system enables farmers to make data-driven decisions about irrigation scheduling, reducing water waste while
        maximizing crop yields. Key features include multi-depth soil moisture sensing, environmental parameter monitoring,
        wireless data transmission, and machine learning-based prediction of irrigation requirements.'''
    }
   
    print("="*60)
    print("Testing 5-Agent Patent Verification with Qwen 2.5 7B")
    print("="*60)
    print("\n📋 Patent Title:", test_patent['title'])
    print("\n⏳ Starting verification process...\n")
    
    result = verify_patent_5_sections(test_patent)
    print("\n" + "="*60)
    print("FINAL REPORT:")
    print("="*60)
    print(result)
