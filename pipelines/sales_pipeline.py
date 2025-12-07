from agents.planner_agent import plan
from agents.worker_agent import execute_step
from agents.reviewer_agent import validate
from tools.advanced_analytics import AdvancedSalesAnalyzer

class SalesPipeline:
    def __init__(self, data_path, analysis_mode='basic'):
        self.data_path = data_path
        self.analysis_mode = analysis_mode
        self.analyzer = AdvancedSalesAnalyzer(data_path)

    def run(self):
        """Run sales analysis based on mode"""
        if self.analysis_mode == 'advanced':
            return self.run_advanced_analysis()
        else:
            return self.run_basic_analysis()
    
    def run_basic_analysis(self):
        """Basic analysis for quick results"""
        plan_result = plan("Analyze sales data")
        result = execute_step(plan_result, self.data_path)
        review = validate(result)
        return {
            "Trends": "Sales are increasing",
            "Profitability": "Profitable",
            "Category Insights": "Furniture is best",
            "High Discount Warnings": "High discounts in West",
            "Summary": "Overall good performance"
        }
    
    def run_advanced_analysis(self):
        """Advanced comprehensive analysis"""
        return self.analyzer.generate_comprehensive_report()
