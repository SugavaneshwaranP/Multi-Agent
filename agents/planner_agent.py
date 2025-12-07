from crewai import Agent

class PlannerAgent(Agent):
    def __init__(self, model="llama3"):
        super().__init__(
            role="Planner",
            goal="Understand user input and break tasks into steps",
            backstory="I am a strategic planner who analyzes tasks and creates plans.",
            llm=f"ollama/{model}"
        )

    def plan(self, task):
        if "sales" in task.lower():
            return "Analyze sales data"
        elif "resume" in task.lower():
            return "Screen resume"
        else:
            return "General task"

planner_agent = Agent(
    role="Planner",
    goal="Understand user input and break tasks into steps",
    backstory="I am a strategic planner who analyzes tasks and creates plans.",
    llm="ollama/llama3"
)

def plan(task):
    if "sales" in task.lower():
        return "Analyze sales data"
    elif "resume" in task.lower():
        return "Screen resume"
    else:
        return "General task"
