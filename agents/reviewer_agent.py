import ollama
import json
from crewai import Agent

class ReviewerAgent(Agent):
    model_name: str = "qwen2.5"

    def __init__(self, model="qwen2.5"):
        super().__init__(
            role="Reviewer Agent (Validation & Executive Summary)",
            goal="Validate findings, remove noise, and generate a strategic executive summary.",
            backstory="""You are the Reviewer Agent.

Your task:
- Validate Worker findings
- Remove weak or noisy insights
- Prioritize what matters most to decision-makers

Tone:
- Clear
- Strategic
- Business-friendly

IMPORTANT: Your final output MUST follow this exact structure:

📊 DATASET OVERVIEW
- Rows: [Count]
- Columns: [Count]
- Detected Domain: [Domain]

🧠 KEY INSIGHTS
- [Insight 1]
- [Insight 2]
- [Insight 3]

📈 NOTABLE PATTERNS / ANOMALIES
- [Pattern or risk detected]

🎯 RECOMMENDATIONS
- [Action 1]
- [Action 2]

⚠️ LIMITATIONS
- [Any data or model constraints]

💬 ASK ME NEXT
- [Suggested follow-up question 1]
- [Suggested follow-up question 2]""",
            verbose=True,
            allow_delegation=False,
            # llm=f"ollama/{model}"
        )
        self.model_name = model

    def review(self, analysis_results):
        """
        Generates the executive summary using the LLM.
        """
        try:
            # Convert results to string if it's a dict
            if isinstance(analysis_results, dict):
                context = json.dumps(analysis_results, indent=2, default=str)
            else:
                context = str(analysis_results)
                
            prompt = f"""
            Analyze the following worker findings and Generate the Executive Summary.
            
            WORKER FINDINGS:
            {context}
            
            Remember to follow the REQUIRED OUTPUT FORMAT strictly.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.backstory},
                {'role': 'user', 'content': prompt}
            ])
            
            return response['message']['content']
        except Exception as e:
            return f"Error generating review: {str(e)}"
