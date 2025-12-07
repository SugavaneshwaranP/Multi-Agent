"""
Quick test script to run CognifyX Intelligence Engine
with your Ollama models: llama3, mistral, qwen2.5
"""

from tools.cognifyx_engine import CognifyXEngine
import json

def main():
    print("🚀 Starting CognifyX Intelligence Engine...")
    print("=" * 70)
    print("Multi-Agent Configuration:")
    print("  🤖 Planner: llama3")
    print("  🤖 Worker: mistral")
    print("  🤖 Reviewer: qwen2.5")
    print("=" * 70)
    print()
    
    # Initialize CognifyX with your Ollama models
    engine = CognifyXEngine(
        file_path='datasets/sales/Sample - Superstore.csv',
        planner_model='llama3',
        worker_model='mistral',
        reviewer_model='qwen2.5'
    )
    
    print("📊 Loading and preprocessing data...")
    engine.load_and_preprocess()
    print(f"✅ Loaded {len(engine.data):,} transactions")
    print()
    
    print("🔍 Running CognifyX Analysis...")
    print("-" * 70)
    
    # Basic metrics
    print("\n1️⃣ Basic Metrics")
    metrics = engine.get_basic_metrics()
    print(f"   Total Sales: ${metrics['total_sales']:,.2f}")
    print(f"   Total Profit: ${metrics['total_profit']:,.2f}")
    print(f"   Customers: {metrics['total_customers']:,}")
    print(f"   Orders: {metrics['total_orders']:,}")
    
    # Forecasting with LLM reasoning
    print("\n2️⃣ Sales Forecasting (LLM-based)")
    forecast = engine.llm_reasoning_forecast(periods=6)
    print(f"   Trend: {forecast['trend']}")
    print(f"   Monthly Growth: ${forecast['monthly_growth_rate']:,.2f}")
    print(f"   Next 6 months: {[f'${x:,.0f}' for x in forecast['forecasted_sales']]}")
    
    # Customer segmentation with LLM
    print("\n3️⃣ Customer Segmentation (LLM-based)")
    segments = engine.llm_customer_segmentation()
    for seg_id, seg_data in segments.items():
        print(f"   Segment {seg_id} - {seg_data['Label']}: {seg_data['Customer ID']} customers")
    
    # Anomaly detection with LLM
    print("\n4️⃣ Anomaly Detection (LLM-based)")
    anomalies = engine.llm_anomaly_detection()
    print(f"   Sales Outliers: {anomalies['sales_outliers_count']}")
    print(f"   Negative Profit Orders: {anomalies['negative_profit_orders']}")
    print(f"   Risk Level: {anomalies['risk_level']}")
    
    # Product intelligence
    print("\n5️⃣ Product Intelligence (LLM-based)")
    products = engine.llm_product_intelligence()
    print(f"   Categories Analyzed: {len(products['category_performance'])}")
    print(f"   Top Subcategories: {len(products['top_subcategories'])}")
    
    # Generate Executive Summary with multi-agent collaboration
    print("\n" + "=" * 70)
    print("🎯 Generating Executive Summary (Multi-Agent Collaboration)")
    print("=" * 70)
    print()
    
    summary_result = engine.generate_executive_summary()
    print(summary_result['executive_summary'])
    
    print("\n" + "=" * 70)
    print("✅ CognifyX Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
