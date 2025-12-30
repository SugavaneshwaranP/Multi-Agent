import pandas as pd
import ollama

class ChatAgent:
    def __init__(self, data: pd.DataFrame, model="llama3"):
        self.data = data
        self.model = model
        
    def query(self, question):
        """
        Analyze the dataframe and answer the question.
        """
        if self.data is None or self.data.empty:
            return "No data available to analyze."

        # Create a context summary
        # We truncate to avoid exceeding token limits
        columns = list(self.data.columns)
        head_data = self.data.head(5).to_markdown()
        stats = self.data.describe().to_markdown()
        
        prompt = f"""
        You are an expert Data Analyst inside the CognifyX platform.
        You have access to a dataset with the following structure:
        
        COLUMNS: {columns}
        
        SAMPLE DATA (First 5 rows):
        {head_data}
        
        STATISTICAL SUMMARY:
        {stats}
        
        USER QUESTION: {question}
        
        INSTRUCTIONS:
        1. Answer the question based ONLY on the provided data context.
        2. If you cannot answer based on the summary, explain why.
        3. Be concise and professional.
        4. If the user asks for specific values not in the summary, explain that you only have access to a summary view.
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': 'You are a helpful data analyst assistant.'},
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"I encountered an error connecting to the AI model ({self.model}). Please ensure Ollama is running.\nDetails: {str(e)}"
