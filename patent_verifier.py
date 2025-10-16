""" 
ULTIMATE Patent Verification System
CrewAI + Llama 3.1 8B + 6 Agents
FIXED: ollama/ prefix for LiteLLM
"""

from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

def verify_patent_description_scoring(patent_sections):
    """6-Agent Patent Description Scoring with Llama 3.1 8B"""

    try:
        # FIXED: ollama/ prefix for LiteLLM routing
        local_llm = Ollama(
            model="ollama/llama3.1:8b",  # ✅ FIXED!
            base_url="http://localhost:11434",
            temperature=0.3,
            num_ctx=8192,
            num_predict=400,
            top_p=0.9
        )

        print("✅ Llama 3.1 8B initialized (128K context)")
        print("\n🤖 CrewAI 6-Agent Patent Description Scoring")
        print("=" * 60)

        # Agent 1: Language Style Evaluator
        agent1 = Agent(
            role="Language Style Evaluator",
            goal=(
                "Evaluate the patent description for dry, factual, legally appropriate "
                "language style, avoiding advocacy, promotional phrases, patent profanity, or direct claim references."
            ),
            backstory="Experienced patent attorney expert in legal patent drafting style.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )

        task1 = Task(
            description=f"""Evaluate the following patent description's language style:

{patent_sections['description']}

Provide:
- Language Style Score (1-5)
- Brief rationale for the score""",
            expected_output="Language style evaluation score and rationale",
            agent=agent1
        )

        # Agent 2: Elaboration Assessor
        agent2 = Agent(
            role="Elaboration Assessor",
            goal=(
                "Assess how well the description elaborates beyond claims by explaining "
                "key technical concepts without merely repeating claim language."
            ),
            backstory="Patent expert focused on technical clarity and elaboration.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )

        task2 = Task(
            description=f"""Analyze elaboration of the following patent description:

{patent_sections['description']}

Provide:
- Elaboration Score (1-5)
- Brief rationale for the score""",
            expected_output="Elaboration score and rationale",
            agent=agent2
        )

        # Agent 3: Diversity Checker
        agent3 = Agent(
            role="Diversity Checker",
            goal=(
                "Evaluate the diversity of language and content, spotting unnecessary repetition "
                "or restatements in the patent description."
            ),
            backstory="Patent drafter experienced in maintaining linguistic variety.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )

        task3 = Task(
            description=f"""Evaluate language and content diversity in the patent description:

{patent_sections['description']}

Provide:
- Diversity Score (1-5)
- Brief rationale for the score""",
            expected_output="Diversity score and rationale",
            agent=agent3
        )

        # Agent 4: Factual Accuracy Auditor
        agent4 = Agent(
            role="Factual Accuracy Auditor",
            goal=(
                "Identify hallucinations or factual inaccuracies in the description, ensuring "
                "content is true to the invention without erroneous references."
            ),
            backstory="Senior patent reviewer specialized in factual correctness.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )

        task4 = Task(
            description=f"""Assess factual accuracy of the patent description:

{patent_sections['description']}

Provide:
- Factual Accuracy Score (1-5)
- List any factual issues identified""",
            expected_output="Factual accuracy score and issues",
            agent=agent4
        )

        # Agent 5: Claims Coverage Evaluator
        agent5 = Agent(
            role="Claims Coverage Evaluator",
            goal=(
                "Evaluate how completely the description covers the content of all claims "
                "without significant omissions."
            ),
            backstory="Patent examiner expert in claims and description consistency.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )

        task5 = Task(
            description=f"""Analyze claims coverage in description:

Claims:
{patent_sections['claims']}

Description:
{patent_sections['description']}

Provide:
- Claims Coverage Score (1-5)
- List any missing or inadequately covered claims""",
            expected_output="Claims coverage score and remarks",
            agent=agent5
        )

        # Agent 6: Final Ranking Agent
        agent6 = Agent(
            role="Final Ranking Agent",
            goal=(
                "Aggregate scores from other agents, rank description quality, "
                "highlight critical issues and strengths, and provide overall readiness."
            ),
            backstory="Senior patent partner with 30 years experience in patent quality assessment.",
            llm=local_llm,
            verbose=False,
            allow_delegation=False
        )

        task6 = Task(
            description=f"""Final evaluation based on aggregated scores:

- Language Style Score from Task 1
- Elaboration Score from Task 2
- Diversity Score from Task 3
- Factual Accuracy Score from Task 4
- Claims Coverage Score from Task 5

Provide:
1. Overall rank (1=best, 3=worst)
2. Summary of critical issues (top 3)
3. Summary of strengths (top 2)
4. Filing readiness: YES/NO
5. Priority actions if not ready""",
            expected_output="Overall rank and final assessment",
            agent=agent6,
            context=[task1, task2, task3, task4, task5]
        )

        crew = Crew(
            agents=[agent1, agent2, agent3, agent4, agent5, agent6],
            tasks=[task1, task2, task3, task4, task5, task6],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=False
        )

        print("✅ Created 6 agents")
        print("✅ Created 6 tasks")
        print("🚀 Running verification...\n")

        result = crew.kickoff()

        print("\n✅ Complete!")
        print("=" * 60)

        return f"""
╔═══════════════════════════════════════════════════════════╗
║     6-AGENT PATENT DESCRIPTION SCORING REPORT             ║
╚═══════════════════════════════════════════════════════════╝

{result}

╔═══════════════════════════════════════════════════════════╗
║ System: CrewAI + Llama 3.1 8B (6 Agents)                 ║
╚═══════════════════════════════════════════════════════════╝
"""

    except Exception as e:
        import traceback
        return f"❌ Error: {traceback.format_exc()}"

if __name__ == "__main__":
    test = {
        'title': 'Smart Agricultural Monitoring System Using IoT Sensors and Machine Learning',
        'abstract': '''A comprehensive smart agricultural monitoring system that integrates
        IoT sensors with machine learning algorithms to provide real-time monitoring and
        predictive analytics for crop management. The system comprises soil moisture sensors,
        temperature sensors, humidity sensors, and a processing unit that analyzes data.''',
        'claims': '''1. A smart agricultural monitoring system comprising:
       a) soil moisture sensors;
       b) temperature sensors;
       c) a central processing unit;
       d) a communication module; and
       e) a machine learning module.
    2. The system of claim 1, wherein sensors detect moisture at multiple depths.
    3. The system of claim 1, wherein the module predicts irrigation requirements.''',
        'background': '''Traditional farming relies on manual monitoring and scheduled irrigation,
        leading to water wastage. Farmers lack real-time data. Prior art includes basic sensors
        (US Patent 9,123,456) but these lack integrated machine learning.''',
        'summary': '''The invention provides a comprehensive smart monitoring system that addresses
        limitations of existing solutions. By integrating sensors with machine learning, the system
        provides accurate real-time monitoring and predictive analytics.''',
        'description': '''The invention relates to a smart agricultural monitoring system integrating
        IoT sensors with advanced machine learning algorithms, enabling real-time data collection
        and predictive analysis to optimize crop irrigation and management. This comprehensive
        solution includes multi-layer soil moisture sensing, environmental condition monitoring,
        and adaptive decision-making based on learned data patterns. Unlike prior art,
        this system enhances accuracy and responsiveness, reducing resource wastage while
        improving crop yield effectively.'''
    }

    print("Testing 6-Agent Patent Description Scoring with Llama 3.1 8B...\n")
    result = verify_patent_description_scoring(test)
    print(result)
