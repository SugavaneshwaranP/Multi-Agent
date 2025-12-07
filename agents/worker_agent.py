from crewai import Agent
from tools.csv_analyzer import CSVAnalyzer
from tools.resume_parser import ResumeParser
from tools.file_reader import FileReader

class WorkerAgent(Agent):
    def __init__(self, model="mistral"):
        super().__init__(
            role="Worker",
            goal="Execute steps and call tools",
            backstory="I am a diligent worker who uses tools to perform tasks.",
            llm=f"ollama/{model}"
        )

    def execute_step(self, step, data_path):
        if "sales" in step.lower():
            analyzer = CSVAnalyzer(data_path)
            analyzer.load_csv()
            return analyzer.compute_total_sales()
        elif "resume" in step.lower():
            parser = ResumeParser(data_path)
            return parser.extract_skills()
        else:
            reader = FileReader(data_path)
            return reader.read_file()

worker_agent = Agent(
    role="Worker",
    goal="Execute steps and call tools",
    backstory="I am a diligent worker who uses tools to perform tasks.",
    llm="ollama/llama3"
)

def execute_step(step, data_path):
    if "sales" in step.lower():
        analyzer = CSVAnalyzer(data_path)
        analyzer.load_csv()
        return analyzer.compute_total_sales()
    elif "resume" in step.lower():
        parser = ResumeParser(data_path)
        return parser.extract_skills()
    else:
        reader = FileReader(data_path)
        return reader.read_file()
