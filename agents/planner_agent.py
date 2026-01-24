import ollama
import json
from crewai import Agent

class PlannerAgent(Agent):
    model_name: str = "llama3"

    def __init__(self, model="llama3"):
        super().__init__(
            role="Planner Agent (Strategy & Reasoning)",
            goal="Analyze dataset metadata, identify the domain, and create a structured analysis plan.",
            backstory="""You are the Planner Agent.
Your task:
- Analyze dataset metadata
- Identify the dataset domain
- Decide the most valuable analytical tasks
- Break the problem into executable subtasks for the Worker Agent

You should output:
- Detected domain
- Key business questions the data can answer
- Ordered task list (max 5–7 tasks)

Examples:
- "Identify top-performing categories"
- "Detect anomalies or outliers"
- "Rank candidates based on skill relevance"
- "Forecast trends if time-series exists"

Focus on IMPACT, not volume.""",
            verbose=True,
            allow_delegation=False,
            # llm=f"ollama/{model}"  # Removed to avoid LiteLLM dependency issues, using manual ollama call
        )
        self.model_name = model

    def plan(self, dataset_info):
        """
        Generate a plan based on dataset info using LLM.
        """
        try:
            # Convert info to string if it's a dict
            if isinstance(dataset_info, dict):
                context = json.dumps(dataset_info, indent=2, default=str)
            else:
                context = str(dataset_info)
                
            prompt = f"""
            Analyze the following dataset metadata and Generate an Analysis Plan.
            
            DATASET METADATA:
            {context}
            
            Provide your response as a structured plan.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.backstory},
                {'role': 'user', 'content': prompt}
            ])
            
            return response['message']['content']
        except Exception as e:
            return f"Error generating plan: {str(e)}"
