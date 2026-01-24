import ollama
from crewai import Agent

class WorkerAgent(Agent):
    model_name: str = "mistral"

    def __init__(self, model="mistral"):
        super().__init__(
            role="Worker Agent (Execution & Computation)",
            goal="Execute analysis tasks, perform computations, and extract structured insights.",
            backstory="""You are the Worker Agent.

Your task:
- Execute each task defined by the Planner
- Perform computations, heuristics, and pattern extraction
- Use statistics, rules, and reasoning
- Extract structured insights (NOT raw tables)

Capabilities:
- Aggregations & growth calculations
- Trend & anomaly detection
- Segmentation & ranking
- Keyword & regex-based text analysis (for resumes)
- Heuristic scoring when needed

Output:
- Structured findings per task
- Clear reasoning for each insight""",
            verbose=True,
            allow_delegation=False,
            # llm=f"ollama/{model}"
        )
        self.model_name = model

    def execute_task(self, task):
        """
        Execute task using LLM + Tools (Simulated tool use via LLM reasoning for now)
        """
        try:
            prompt = f"""
            Execute the following analysis task.
            
            TASK:
            {task}
            
            Provide structured findings and clear reasoning.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.backstory},
                {'role': 'user', 'content': prompt}
            ])
            
            return response['message']['content']
        except Exception as e:
            return f"Error executing task: {str(e)}"
