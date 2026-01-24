
import sys
import os

# Mock OpenAI Key to bypass CrewAI validation
os.environ["OPENAI_API_KEY"] = "NA"

import pandas as pd
from tools.cognifyx_engine import CognifyXEngine
from tools.resume_analyzer import ResumeAnalyzer

def test_engine():
    print("Testing CognifyXEngine...")
    try:
        # Create a dummy CSV
        df = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=10),
            'sales': [100, 120, 130, 110, 150, 160, 170, 140, 180, 200],
            'category': ['A', 'B'] * 5
        })
        df.to_csv('test_data.csv', index=False)
        
        # Instantiate
        engine = CognifyXEngine('test_data.csv')
        print("Engine instantiated.")
        
        # Load
        engine.load_and_preprocess()
        print("Data loaded.")
        
        # Run specific method that uses Worker Agent
        metrics = engine.get_basic_metrics()
        print(f"Metrics: {metrics['dataset_info']}")
        
        # Note: We won't run full analysis as it calls LLM (slow/costly), 
        # but just importing and instantiating proves syntax is likely OK.
        # We can try one simple method if we want to test LLM connection, 
        # but syntax check is the main goal here.
        
    except Exception as e:
        print(f"Engine Test Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

def test_resume():
    print("\nTesting ResumeAnalyzer...")
    try:
        # Create dummy resume entries if possible, or just instantiate
        analyzer = ResumeAnalyzer('dummy_path')
        print("ResumeAnalyzer instantiated.")
        
    except Exception as e:
        print(f"ResumeAnalyzer Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
    test_resume()
