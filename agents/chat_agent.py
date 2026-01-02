import pandas as pd
import ollama
import re
import traceback
import sys
import io

class ChatAgent:
    def __init__(self, data: pd.DataFrame, model="llama3"):
        self.data = data
        self.model = model
        
    def query(self, question):
        """
        Analyze the dataframe and answer the question using Python code generation.
        """
        if self.data is None or self.data.empty:
            return "No data available to analyze."

        # Prepare schema description
        buffer = io.StringIO()
        self.data.info(buf=buffer)
        schema_info = buffer.getvalue()
        
        # Initial code generation prompt
        prompt = f"""
        You are an expert Python Data Analyst.
        You have a pandas DataFrame named 'df'.
        
        DF INFO:
        {schema_info}
        
        SAMPLE DATA:
        {self.data.head(3).to_markdown()}
        
        USER QUESTION: {question}
        
        TASK:
        Write Python code to answer the question. 
        1. Access the dataframe using the variable 'df'.
        2. Perform necessary filtering, aggregation, or calculation.
        3. Store the FINAL ANSWER in a variable named 'result'.
        4. If the answer is a plot, using 'result = "Plot generated"' is fine, but focus on text data analysis first.
        5. Wrap your code strictly in ```python ... ```.
        6. Do NOT print the result, just assign it to 'result'.
        7. Use efficient pandas operations.
        
        Example:
        ```python
        # Calculate average sales
        avg_sales = df['sales'].mean()
        result = f"The average sales is {{avg_sales:.2f}}"
        ```
        """
        
        # Validation/Retry Loop
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # 1. Generate Logic
                response = ollama.chat(model=self.model, messages=[
                    {'role': 'system', 'content': 'You are a Python Data Analysis expert. Write code to solve the user problem.'},
                    {'role': 'user', 'content': prompt}
                ])
                llm_output = response['message']['content']
                
                # 2. Extract Code
                code_match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
                if not code_match:
                    if "```" in llm_output:
                        code_match = re.search(r"```(.*?)```", llm_output, re.DOTALL)
                
                if not code_match:
                    return llm_output # Return text if no code generated
                
                code_block = code_match.group(1).strip()
                
                # 3. Execute Code
                local_env = {'df': self.data, 'pd': pd, 'result': None}
                
                # Capture stdout just in case
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                try:
                    exec(code_block, {}, local_env)
                    execution_output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                result = local_env.get('result')
                
                # 4. Final Formatting (if result is raw data, make it readable)
                if result is not None:
                    # Provide context back to LLM to humanize
                    explain_prompt = f"""
                    User Question: {question}
                    Analysis Code Result: {result}
                    Execution Output: {execution_output}
                    
                    Explain this result to the user clearly and professionally.
                    Answer:
                    """
                    final_response = ollama.chat(model=self.model, messages=[
                        {'role': 'user', 'content': explain_prompt}
                    ])
                    return final_response['message']['content']
                else:
                    return f"Executed code but no 'result' variable was found.\nOutput: {execution_output}"
                    
            except Exception as e:
                error_msg = traceback.format_exc()
                if attempt < max_retries:
                    prompt += f"\n\nPREVIOUS CODE FAILED:\n{code_block}\n\nERROR:\n{error_msg}\n\nPlease fix the code and try again."
                    continue
                else:
                    return f"I failed to analyze the data after multiple attempts.\nLast Error: {str(e)}"
