from crewai import Agent

class ReviewerAgent(Agent):
    def __init__(self, model="qwen2.5"):
        super().__init__(
            role="Reviewer",
            goal="Validate Worker output and send corrections",
            backstory="I am a critical reviewer who ensures quality and finalizes insights.",
            llm=f"ollama/{model}"
        )

    def validate(self, output):
        return f"Validated: {output}"

def validate(output):
    return f"Validated: {output}"
