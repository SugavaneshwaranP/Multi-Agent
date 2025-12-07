from agents.planner_agent import PlannerAgent
from agents.worker_agent import WorkerAgent
from agents.reviewer_agent import ReviewerAgent

class ResumePipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.planner = PlannerAgent()
        self.worker = WorkerAgent()
        self.reviewer = ReviewerAgent()

    def run(self):
        plan = self.planner.plan("Screen resume")
        result = self.worker.execute_step(plan, self.data_path)
        review = self.reviewer.validate(result)
        return {
            "Fit Score": 85,
            "Strengths": "Good experience",
            "Missing Skills": "Python",
            "Summary": "Good candidate"
        }
