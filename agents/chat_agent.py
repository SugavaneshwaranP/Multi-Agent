import pandas as pd
import ollama
import re
import traceback
import sys
import io

class ChatAgent:
    def __init__(self, data: pd.DataFrame, model="llama3", engine=None):
        self.data = data
        self.model = model
        self.engine = engine
        
    def query(self, question):
        """
        Analyze the dataframe and answer the question using Autonomous Agentic Reasoning.
        """
        if self.data is None or self.data.empty:
            return "No data available to analyze. Please upload a dataset first."

        # Prepare schema description
        buffer = io.StringIO()
        self.data.info(buf=buffer)
        schema_info = buffer.getvalue()
        
        # Describe available specialized tools if engine exists
        tools_description = ""
        if self.engine:
            tools_description = """
            SPECIALIZED TOOLS (Accessible via 'engine' object):
            - engine.llm_reasoning_forecast(): For predicting future trends.
            - engine.llm_customer_segmentation(): For intent/entity clustering.
            - engine.llm_anomaly_detection(): For identifying gaps or outliers.
            - engine.ecommerce_price_intelligence(): For pricing/discount analysis (if applicable).
            - engine.ecommerce_stock_prediction(): For inventory insights (if applicable).
            """

        # Balanced Speed/Intelligence Prompt
        prompt = f"""
        Analyze this DataFrame 'df' and answer: "{question}"
        {tools_description}
        SCHEMA: {schema_info}
        
        TASK:
        1. [THOUGHT] Plan logic.
        2. [CODE] Write python code. Assign final info to 'result'.
        3. [EXPLANATION] Provide a 1-sentence quick summary of the expected finding.
        
        Format strictly with [THOUGHT], [CODE], [EXPLANATION] tags.
        ```python
        # code
        ```
        """
        
        # Validation/Retry Loop
        max_retries = 1 # Reduced for speed
        for attempt in range(max_retries + 1):
            code_block = ""
            try:
                # 1. Single-Shot Generation
                response = ollama.chat(model=self.model, messages=[
                    {'role': 'system', 'content': 'You are CognifyX Fast Intelligence. Perform analysis and explain in one shot.'},
                    {'role': 'user', 'content': prompt}
                ])
                llm_output = response['message']['content']
                
                # 2. Parallel Extraction
                thought_match = re.search(r"\[THOUGHT\](.*?)(\[CODE\]|```)", llm_output, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else "Analyzing..."
                
                expl_match = re.search(r"\[EXPLANATION\](.*?)$", llm_output, re.DOTALL)
                prelim_expl = expl_match.group(1).strip() if expl_match else ""
                
                code_match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
                if not code_match: continue
                code_block = code_match.group(1).strip()
                
                # 3. Execution
                local_env = {'df': self.data, 'pd': pd, 'engine': self.engine, 'result': None}
                exec(code_block, {}, local_env)
                result = local_env.get('result')
                
                if result is not None:
                    # SMART SPEED: If result is small/string, skip final LLM call
                    if isinstance(result, str) and len(result) < 200:
                        final_answer = f"{prelim_expl}\n\n**Result:** {result}"
                    else:
                        # Only humanize if complex
                        final_response = ollama.chat(model=self.model, messages=[
                            {'role': 'user', 'content': f"Context: {thought}\nResult: {result}\nSummarize briefly."}
                        ])
                        final_answer = final_response['message']['content']
                        
                    return {
                        "thought": thought,
                        "code": code_block,
                        "answer": final_answer
                    }
                
            except Exception as e:
                if attempt == max_retries:
                    return f"Neural bottleneck. Last Error: {str(e)}"

    def discover_tasks(self):
        """
        Analyze the dataframe and suggest automated tasks for the agent.
        """
        if self.data is None or self.data.empty:
            return []

        # Prepare schema description
        import io
        import json
        buffer = io.StringIO()
        self.data.info(buf=buffer)
        schema_info = buffer.getvalue()
        
        prompt = f"""
        Analyze this dataset schema and suggest 5 high-value automated analysis tasks.
        
        SCHEMA:
        {schema_info}
        
        Return ONLY a JSON list of strings. Each string should be a short, actionable instruction.
        Example: ["Forecast sales trends", "Identify anomalous records", "Segment customers by value"]
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': 'You are a task discovery agent. Output ONLY a valid JSON list of 5 strings.'},
                {'role': 'user', 'content': prompt}
            ])
            llm_output = response['message']['content']
            
            # Clean JSON
            json_match = re.search(r"(\[.*\])", llm_output, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group(1))
                return tasks[:5]
            return ["Analyze core metrics", "Detect outliers", "Predict future trends", "Segment analysis", "Quality assessment"]
        except Exception:
            return ["Analyze core metrics", "Detect outliers", "Predict future trends", "Segment analysis", "Quality assessment"]
