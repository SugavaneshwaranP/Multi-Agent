"""CSV Analysis Tool"""
import pandas as pd

class CSVAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_csv(self):
        """Load CSV file"""
        self.data = pd.read_csv(self.file_path, encoding='latin-1')
        return self.data
    
    def compute_total_sales(self):
        """Calculate total sales"""
        if self.data is None:
            self.load_csv()
        return float(self.data['Sales'].sum()) if 'Sales' in self.data.columns else 0
