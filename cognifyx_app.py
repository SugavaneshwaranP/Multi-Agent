"""
CognifyX Agentic UI - Multi-Agent Intelligence Platform
Interactive Streamlit interface for CognifyX Engine
Supports: Sales Data, Resume Analysis, Generic CSV
"""

import os
# Mock OpenAI Key to bypass CrewAI default validation (running purely local)
os.environ["OPENAI_API_KEY"] = "NA"

import streamlit as st
import pandas as pd

import json
from datetime import datetime
from tools.cognifyx_engine import CognifyXEngine
from tools.resume_analyzer import ResumeAnalyzer
import time
import os
import glob
from ui.visualizations import (
    create_sales_trend_chart,
    create_customer_segment_chart,
    create_category_performance_chart,
    create_skills_chart,
    create_experience_chart,
    create_education_chart,
    create_category_distribution_chart
)
from ui.components import (
    display_agent_card,
    display_metric_card,
    log_agent_activity,
    display_agent_logs
)
from ui.chat_ui import render_chat_tab
from ui.agentic_ui import (
    inject_premium_css,
    render_agent_status_mirror,
    render_premium_chat,
    render_live_data_feed
)

st.set_page_config(
    page_title="CognifyX Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agentic look
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: #0e1117;
    }
    .agent-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #60a5fa;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #374151;
        text-align: center;
    }
    .insight-box {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 10px 0;
    }
    .agent-thinking {
        background: #1f2937;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #60a5fa;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #60a5fa;
        margin: 5px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    h1, h2, h3 {
        color: #60a5fa !important;
        font-weight: 700 !important;
    }
    .status-running {
        color: #fbbf24;
        animation: pulse 2s infinite;
    }
    .status-complete {
        color: #10b981;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    if 'start_analysis' not in st.session_state:
        st.session_state.start_analysis = False
    
    # Store models in session state for UI
    if 'planner_model' not in st.session_state:
        st.session_state.planner_model = "llama3"
    if 'worker_model' not in st.session_state:
        st.session_state.worker_model = "mistral"
    if 'reviewer_model' not in st.session_state:
        st.session_state.reviewer_model = "qwen2.5"



def detect_dataset_type(file_path):
    """Auto-detect if dataset is Sales, Resume, E-commerce or Generic"""
    try:
        # Check if it's a directory (PDF resumes)
        if os.path.isdir(file_path):
            return 'resume'
        
        # Read first few rows (support CSV and Excel)
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, nrows=100)
        else:
            df = pd.read_csv(file_path, nrows=100)
        columns = [col.lower() for col in df.columns]
        
        # Resume indicators
        resume_keywords = ['resume', 'skill', 'experience', 'education', 'degree', 'qualification', 'candidate', 'cv', 'job']
        resume_score = sum(1 for col in columns if any(kw in col for kw in resume_keywords))
        
        # E-commerce indicators (more specific)
        ecommerce_keywords = ['seller', 'brand', 'category', 'sub_category', 'actual_price', 'selling_price', 
                            'out_of_stock', 'rating', 'title', 'description', 'crawled']
        ecommerce_score = sum(1 for col in columns if any(kw in col for kw in ecommerce_keywords))
        
        # Sales indicators
        sales_keywords = ['sales', 'revenue', 'profit', 'customer', 'order', 'quantity', 'ship', 'region']
        sales_score = sum(1 for col in columns if any(kw in col for kw in sales_keywords))
        
        if resume_score > max(sales_score, ecommerce_score) and resume_score >= 2:
            return 'resume'
        elif ecommerce_score >= 4:  # E-commerce has many specific columns
            return 'ecommerce'
        elif sales_score >= 2:
            return 'sales'
        else:
            return 'generic'
            
    except:
        return 'generic'

def main():
    """Main application"""
    init_session_state()
    inject_premium_css()
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>
            🤖 CognifyX Intelligence Platform
        </h1>
        <p style='text-align: center; color: #9ca3af; font-size: 18px;'>
            Multi-Agent AI Analytics Engine
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Agent Configuration")
        
        st.markdown("### 🤖 Multi-Agent System")
        
        planner_model = st.selectbox(
            "Planner Agent",
            ["llama3", "mistral", "qwen2.5"],
            index=0,
            help="Strategic planning and task decomposition"
        )
        
        worker_model = st.selectbox(
            "Worker Agent", 
            ["llama3", "mistral", "qwen2.5"],
            index=1,
            help="Data analysis and execution"
        )
        
        reviewer_model = st.selectbox(
            "Reviewer Agent",
            ["llama3", "mistral", "qwen2.5"],
            index=2,
            help="Quality validation and review"
        )
        
        # Update session state models
        st.session_state.planner_model = planner_model
        st.session_state.worker_model = worker_model
        st.session_state.reviewer_model = reviewer_model
        
        st.markdown(f"""
            <div style='background: #1f2937; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <p style='margin: 0; font-size: 12px;'>
                    <strong>Pipeline:</strong><br>
                    {planner_model} → {worker_model} → {reviewer_model}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Data Source")
        dataset = st.selectbox(
            "Select Dataset",
            ["E-Commerce Products (30K)", "Resume PDFs (500 samples)", "Resume Dataset (CSV)", "Upload Custom"],
            index=0
        )
        
        if dataset == "Upload Custom":
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
            if uploaded_file:
                os.makedirs("datasets/uploads", exist_ok=True)
                file_path = f"datasets/uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")
                # Detect dataset type
                dataset_type = detect_dataset_type(file_path)
                st.info(f"📊 Detected: {dataset_type.upper()} dataset")
                # Trigger agent welcome for new upload
                if 'p_messages' in st.session_state:
                    st.session_state.p_messages.append({
                        "role": "assistant",
                        "content": f"I see you've uploaded a new {dataset_type} dataset. I'm ready to analyze it using my multi-agent team. Click 'Start Analysis' to begin the neural processing.",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            else:
                file_path = None
        elif dataset == "E-Commerce Products (30K)":
            file_path = "datasets/sales/output.xlsx"
            if os.path.exists(file_path):
                st.success("✅ E-commerce dataset ready (30,000 products)")
                st.info("📊 Includes: prices, ratings, brands, categories, discounts, stock data")
            else:
                st.error("❌ E-commerce dataset not found")
                file_path = None
        elif dataset == "Resume PDFs (500 samples)":
            file_path = "datasets/resumes/data/data"
            if os.path.exists(file_path):
                pdf_count = len(glob.glob(os.path.join(file_path, '**', '*.pdf'), recursive=True))
                st.success(f"✅ Found {pdf_count} PDF resumes in {len(os.listdir(file_path))} categories")
                st.info("📁 Will process first 500 PDFs for performance")
            else:
                st.error("❌ Resume PDF folder not found")
                file_path = None
        elif dataset == "Resume Dataset (CSV)":
            file_path = "datasets/resumes/Resume/Resume.csv"
            if os.path.exists(file_path):
                st.success("✅ Resume CSV dataset ready")
            else:
                st.error("❌ Resume dataset not found")
                file_path = None
        
        st.markdown("---")
        
        # Action Button
        if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
            if file_path:
                st.session_state.analysis_complete = False
                st.session_state.agent_logs = []
                st.session_state.start_analysis = True
                st.rerun()
            else:
                st.error("⚠️ Please select or upload a dataset first!")
    
    # Main Content Area
    if st.session_state.get('start_analysis', False) and not st.session_state.analysis_complete and file_path:
        st.session_state.start_analysis = False
        # Agent Status Dashboard
        st.markdown("## 🤖 Multi-Agent Intelligence System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_agent_card("Planner", planner_model, "idle", "Strategic Planning")
        with col2:
            display_agent_card("Worker", worker_model, "idle", "Data Analysis")
        with col3:
            display_agent_card("Reviewer", reviewer_model, "idle", "Quality Validation")
        
        st.markdown("---")
        
        # Analysis Progress
        st.markdown("## 📊 Running Analysis...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Agent Activity Log
        log_placeholder = st.empty()
        
        try:
            # Detect dataset type
            dataset_type = detect_dataset_type(file_path)
            st.session_state.dataset_type = dataset_type
            
            # Initialize appropriate engine
            status_text.text("🔧 Initializing CognifyX Engine...")
            log_agent_activity("SYSTEM", f"Initializing multi-agent system for {dataset_type.upper()} data")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            if dataset_type == 'resume':
                analyzer = ResumeAnalyzer(file_path, planner_model, worker_model, reviewer_model)
                engine = None
            else:
                engine = CognifyXEngine(file_path, planner_model, worker_model, reviewer_model)
                analyzer = None
            
            # Load Data
            status_text.text("📥 Loading and preprocessing data...")
            log_agent_activity("SYSTEM", f"Loading dataset: {file_path}")
            progress_bar.progress(20)
            
            if dataset_type == 'resume':
                load_result = analyzer.load_and_preprocess()
                if load_result.get('success'):
                    log_agent_activity("SYSTEM", f"Loaded {load_result['total_resumes']:,} resumes")
                else:
                    raise Exception(load_result.get('error'))
            else:
                engine.load_and_preprocess()
                log_agent_activity("SYSTEM", f"Loaded {len(engine.data):,} records")
            time.sleep(0.5)
            
            # Planner Agent
            status_text.text(f"🤖 {planner_model.upper()} (Planner) analyzing strategy...")
            log_agent_activity(planner_model, "Planning analysis strategy")
            progress_bar.progress(30)
            time.sleep(0.5)
            
            if dataset_type == 'resume':
                # RESUME ANALYSIS PIPELINE
                # Skills Extraction
                status_text.text(f"🤖 {worker_model.upper()} (Worker) extracting skills...")
                log_agent_activity(worker_model, "Analyzing skills from resumes")
                progress_bar.progress(45)
                skills = analyzer.extract_skills()
                if skills.get('available'):
                    log_agent_activity(worker_model, f"Found {skills['total_skills_found']} distinct skills")
                time.sleep(0.5)
                
                # Experience Analysis
                status_text.text(f"🤖 {worker_model.upper()} analyzing experience...")
                log_agent_activity(worker_model, "Analyzing candidate experience")
                progress_bar.progress(60)
                experience = analyzer.analyze_experience()
                if experience.get('available'):
                    log_agent_activity(worker_model, f"Avg experience: {experience['average_experience']:.1f} years")
                time.sleep(0.5)
                
                # Education Analysis
                status_text.text(f"🤖 {worker_model.upper()} analyzing education...")
                log_agent_activity(worker_model, "Analyzing education qualifications")
                progress_bar.progress(75)
                education = analyzer.analyze_education()
                if education.get('available'):
                    log_agent_activity(worker_model, f"Classified {education['classified']} candidates")
                time.sleep(0.5)
                
                # Candidate Ranking
                status_text.text(f"🤖 {worker_model.upper()} ranking candidates...")
                log_agent_activity(worker_model, "Performing candidate ranking")
                progress_bar.progress(90)
                ranking = analyzer.candidate_ranking()
                if ranking.get('available'):
                    log_agent_activity(worker_model, f"Ranked {ranking['total_candidates']} candidates")
                time.sleep(0.5)
                
                # Generate Report
                status_text.text(f"🤖 Multi-agent collaboration: Generating report...")
                log_agent_activity(planner_model, "Coordinating resume report")
                summary = analyzer.generate_resume_report(
                    skills=skills, 
                    experience=experience, 
                    education=education, 
                    ranking=ranking
                )
                log_agent_activity(reviewer_model, "Validating analysis quality")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Store results
                st.session_state.results = {
                    'skills': skills,
                    'experience': experience,
                    'education': education,
                    'ranking': ranking,
                    'summary': summary
                }
                st.session_state.analyzer = analyzer
                
            elif dataset_type == 'ecommerce':
                # E-COMMERCE SPECIFIC ANALYSIS PIPELINE
                # Basic Metrics
                status_text.text("📊 Extracting e-commerce metrics...")
                log_agent_activity(worker_model, "Computing product metrics")
                progress_bar.progress(40)
                metrics = engine.get_basic_metrics()
                if 'numeric_summary' in metrics and metrics['numeric_summary']:
                    first_col = list(metrics['numeric_summary'].keys())[0]
                    log_agent_activity(worker_model, f"{first_col} analysis complete")
                time.sleep(0.5)
                
                # Price Intelligence
                status_text.text(f"🤖 {worker_model.upper()} analyzing prices...")
                log_agent_activity(worker_model, "Running price intelligence analysis")
                progress_bar.progress(50)
                price_intel = engine.ecommerce_price_intelligence()
                if price_intel.get('available'):
                    log_agent_activity(worker_model, f"Price analysis: ₹{price_intel['price_stats'].get('avg_price', 0):,.0f} avg")
                else:
                    log_agent_activity(worker_model, f"Price: {price_intel.get('message', 'N/A')}")
                time.sleep(0.5)
                
                # Stock Prediction
                status_text.text(f"🤖 {worker_model.upper()} predicting stock levels...")
                log_agent_activity(worker_model, "Running stock prediction")
                progress_bar.progress(60)
                stock_pred = engine.ecommerce_stock_prediction()
                if stock_pred.get('available'):
                    log_agent_activity(worker_model, f"Stock analysis: {stock_pred['stock_stats'].get('out_of_stock', 0)} out of stock")
                else:
                    log_agent_activity(worker_model, f"Stock: {stock_pred.get('message', 'N/A')}")
                time.sleep(0.5)
                
                # Seller Trust Analysis
                status_text.text(f"🤖 {worker_model.upper()} analyzing seller trust...")
                log_agent_activity(worker_model, "Computing seller trust scores")
                progress_bar.progress(70)
                seller_trust = engine.ecommerce_seller_trust()
                if seller_trust.get('available'):
                    log_agent_activity(worker_model, f"Trust analysis: {seller_trust.get('seller_stats', {}).get('total_sellers', 0)} sellers")
                else:
                    log_agent_activity(worker_model, f"Sellers: {seller_trust.get('message', 'N/A')}")
                time.sleep(0.5)
                
                # Brand Analysis
                status_text.text(f"🤖 {worker_model.upper()} analyzing brands...")
                log_agent_activity(worker_model, "Running brand performance analysis")
                progress_bar.progress(80)
                brand_analysis = engine.ecommerce_brand_analysis()
                if brand_analysis.get('available'):
                    log_agent_activity(worker_model, f"Brand analysis: {brand_analysis['brand_stats'].get('total_brands', 0)} brands")
                else:
                    log_agent_activity(worker_model, f"Brands: {brand_analysis.get('message', 'N/A')}")
                time.sleep(0.5)
                
                # Fraud Detection
                status_text.text(f"🤖 {worker_model.upper()} detecting fraud patterns...")
                log_agent_activity(worker_model, "Scanning for fraudulent listings")
                progress_bar.progress(90)
                fraud_detection = engine.ecommerce_fraud_detection()
                if fraud_detection.get('available'):
                    log_agent_activity(worker_model, f"Fraud scan: {fraud_detection.get('suspicious_count', 0)} suspicious items")
                else:
                    log_agent_activity(worker_model, "Fraud detection complete")
                time.sleep(0.5)
                
                # Executive Summary
                status_text.text(f"🤖 Multi-agent collaboration: Generating summary...")
                log_agent_activity(planner_model, "Coordinating executive summary")
                summary = engine.generate_executive_summary()
                log_agent_activity(reviewer_model, "Validating analysis quality ✅")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Store E-commerce specific results
                st.session_state.results = {
                    'metrics': metrics,
                    'price_intel': price_intel,
                    'stock_pred': stock_pred,
                    'seller_trust': seller_trust,
                    'brand_analysis': brand_analysis,
                    'fraud_detection': fraud_detection,
                    'summary': summary
                }
                st.session_state.engine = engine
            
            else:
                # SALES/GENERIC ANALYSIS PIPELINE  
                # Basic Metrics
                status_text.text("📊 Extracting business metrics...")
                log_agent_activity(worker_model, "Computing basic metrics")
                progress_bar.progress(40)
                metrics = engine.get_basic_metrics()
                if 'numeric_summary' in metrics and metrics['numeric_summary']:
                    first_col = list(metrics['numeric_summary'].keys())[0]
                    log_agent_activity(worker_model, f"{first_col}: {metrics['numeric_summary'][first_col]['sum']:,.2f}")
                else:
                    log_agent_activity(worker_model, "Metrics extraction complete")
                time.sleep(0.5)
                
                # Forecasting
                status_text.text(f"🤖 {worker_model.upper()} (Worker) generating forecast...")
                log_agent_activity(worker_model, "Running LLM-based forecasting")
                progress_bar.progress(50)
                forecast = engine.llm_reasoning_forecast()
                if forecast.get('available'):
                    log_agent_activity(worker_model, f"Forecast complete - Trend: {forecast['trend']}")
                else:
                    log_agent_activity(worker_model, f"Forecast: {forecast.get('message', 'Not available')}")
                time.sleep(0.5)
                
                # Customer Segmentation
                status_text.text(f"🤖 {worker_model.upper()} analyzing entity segments...")
                log_agent_activity(worker_model, "Performing entity segmentation")
                progress_bar.progress(60)
                segments = engine.llm_customer_segmentation()
                if segments.get('available'):
                    seg_count = len(segments.get('segments', {}))
                    log_agent_activity(worker_model, f"Identified {seg_count} segments")
                else:
                    log_agent_activity(worker_model, f"Segmentation: {segments.get('message', 'Not available')}")
                time.sleep(0.5)
                
                # Anomaly Detection
                status_text.text(f"🤖 {worker_model.upper()} detecting anomalies...")
                log_agent_activity(worker_model, "Running anomaly detection")
                progress_bar.progress(70)
                anomalies = engine.llm_anomaly_detection()
                if anomalies.get('available'):
                    anomaly_count = anomalies.get('sales_outliers_count', 0)
                    log_agent_activity(worker_model, f"Found {anomaly_count} anomalies")
                else:
                    log_agent_activity(worker_model, "Anomaly detection complete")
                time.sleep(0.5)
                
                # Product Intelligence
                status_text.text(f"🤖 {worker_model.upper()} analyzing products...")
                log_agent_activity(worker_model, "Generating product intelligence")
                progress_bar.progress(80)
                products = engine.llm_product_intelligence()
                log_agent_activity(worker_model, "Product analysis complete")
                time.sleep(0.5)
                
                # Executive Summary
                status_text.text(f"🤖 Multi-agent collaboration: Generating executive summary...")
                log_agent_activity(planner_model, "Coordinating executive summary")
                log_agent_activity(worker_model, "Compiling insights")
                progress_bar.progress(90)
                summary = engine.generate_executive_summary()
                log_agent_activity(reviewer_model, "Validating analysis quality")
                time.sleep(0.5)
                
                # Review
                status_text.text(f"🤖 {reviewer_model.upper()} (Reviewer) validating results...")
                log_agent_activity(reviewer_model, "Quality check complete ✅")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Store results
                st.session_state.results = {
                    'metrics': metrics,
                    'forecast': forecast,
                    'segments': segments,
                    'anomalies': anomalies,
                    'products': products,
                    'summary': summary
                }
                st.session_state.engine = engine
            
            st.session_state.analysis_complete = True
            
            # Display logs
            with log_placeholder.container():
                display_agent_logs()
            
            status_text.text("✅ Analysis Complete!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            log_agent_activity("SYSTEM", f"Error: {str(e)}")
    
    elif st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        dataset_type = st.session_state.get('dataset_type', 'sales')
        
        # Agent Status - Complete
        st.markdown("## 🤖 Multi-Agent Intelligence System")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_agent_card("Planner", planner_model, "complete", "Strategic Planning")
        with col2:
            display_agent_card("Worker", worker_model, "complete", "Data Analysis")
        with col3:
            display_agent_card("Reviewer", reviewer_model, "complete", "Quality Validation")
        
        st.markdown("---")
        
        # Check dataset type and display appropriate UI
        if dataset_type == 'resume':
            # RESUME ANALYTICS DASHBOARD
            st.markdown("## 👥 Resume Intelligence Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            skills = results.get('skills', {})
            experience = results.get('experience', {})
            education = results.get('education', {})
            ranking = results.get('ranking', {})
            
            with col1:
                display_metric_card("Total Resumes", 
                                  ranking.get('total_candidates', 0) if ranking.get('available') else 'N/A')
            with col2:
                display_metric_card("Skills Found", 
                                  skills.get('total_skills_found', 0) if skills.get('available') else 'N/A')
            with col3:
                display_metric_card("Avg Experience", 
                                  f"{experience.get('average_experience', 0):.1f} yrs" if experience.get('available') else 'N/A')
            with col4:
                display_metric_card("Qualified", 
                                  f"{education.get('classified', 0)}" if education.get('available') else 'N/A')
            
            st.markdown("---")
            
            # Resume Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔧 Skills Analysis",
                "💼 Experience",
                "🎓 Education",
                "🏆 Top Candidates",
                "🤖 Agent Intelligence"
            ])
            
            with tab1:
                st.markdown("### 🔧 Skills Distribution")
                if skills.get('available'):
                    fig = create_skills_chart(skills)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show category distribution if available
                    if skills.get('category_distribution'):
                        st.markdown("### 📂 Resume Categories")
                        cat_fig = create_category_distribution_chart(skills['category_distribution'])
                        st.plotly_chart(cat_fig, use_container_width=True)
                    
                    st.markdown("### 📊 Skill Insights")
                    st.info(skills.get('insights', 'No insights available'))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 💻 Top Technical Skills")
                        if skills.get('top_technical_skills'):
                            for skill, count in skills['top_technical_skills'][:10]:
                                st.markdown(f"- **{skill}**: {count} resumes")
                    
                    with col2:
                        st.markdown("### 🤝 Top Soft Skills")
                        if skills.get('top_soft_skills'):
                            for skill, count in skills['top_soft_skills'][:10]:
                                st.markdown(f"- **{skill}**: {count} resumes")
                else:
                    st.warning("Skills analysis not available")
            
            with tab2:
                st.markdown("### 💼 Experience Distribution")
                if experience.get('available'):
                    fig = create_experience_chart(experience)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Experience", f"{experience['average_experience']:.1f} years")
                        st.metric("Minimum", f"{experience['min_experience']:.0f} years")
                    with col2:
                        st.metric("Median Experience", f"{experience['median_experience']:.1f} years")
                        st.metric("Maximum", f"{experience['max_experience']:.0f} years")
                    
                    st.info(experience.get('insights', 'No insights available'))
                else:
                    st.warning("Experience analysis not available")
            
            with tab3:
                st.markdown("### 🎓 Education Qualifications")
                if education.get('available'):
                    fig = create_education_chart(education)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Distribution Details")
                    for qual, count in education.get('distribution', {}).items():
                        st.markdown(f"- **{qual}**: {count} candidates")
                    
                    st.info(education.get('insights', 'No insights available'))
                else:
                    st.warning("Education analysis not available")
            
            with tab4:
                st.markdown("### 🏆 Top Ranked Candidates")
                if ranking.get('available'):
                    st.markdown(f"**Total Candidates:** {ranking['total_candidates']}")
                    st.markdown(f"**Scoring Criteria:** {', '.join(ranking['scoring_criteria'][:5])}")
                    
                    st.info(ranking.get('insights', 'No insights available'))
                    
                    st.markdown("### Top 10 Candidates")
                    if ranking.get('top_10_candidates'):
                        for idx, candidate in enumerate(ranking['top_10_candidates'][:10], 1):
                            with st.expander(f"#{idx} - Score: {candidate.get('score', 0):.2f}"):
                                for key, value in candidate.items():
                                    if key != 'score' and len(str(value)) < 200:
                                        st.markdown(f"**{key}:** {value}")
                else:
                    st.warning("Candidate ranking not available")

            with tab5:
                # Premium Agent Intelligence Tab
                render_agent_status_mirror()
                
                col_chat, col_feed = st.columns([3, 2])
                with col_chat:
                    render_premium_chat(st.session_state.analyzer, planner_model)
                with col_feed:
                    render_live_data_feed(st.session_state.analyzer.data if hasattr(st.session_state.analyzer, 'data') else None)
            
            # Executive Summary with Recommendations
            st.markdown("---")
            st.markdown("## 📋 Executive Summary & Recommendations")
            summary = results.get('summary', {})
            
            # Display full agent-generated report first
            if summary.get('executive_summary'):
                st.markdown(summary['executive_summary'])
                
            elif isinstance(summary, str):
                st.markdown(summary)
            
            # Optional: Display structured widgets if available (legacy support)
            if isinstance(summary, dict) and summary.get('recommendations'):
                st.markdown("### ⚡ Strategic Actions (Auto-Generated)")
                for rec in summary['recommendations']:
                    priority_color = '#ef4444' if rec.get('priority') == 'HIGH' else '#f59e0b' if rec.get('priority') == 'MEDIUM' else '#3b82f6'
                    st.markdown(f"""
                        <div style='background: #1f2937; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {priority_color};'>
                            <h4 style='margin: 0; color: {priority_color};'>[{rec.get('priority', 'INFO')}] {rec.get('category', 'Recommendation')}</h4>
                            <p style='margin: 10px 0;'><strong>💡 Insight:</strong> {rec.get('insight', '')}</p>
                            <p style='margin: 5px 0;'><strong>✅ Action:</strong> {rec.get('action', '')}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            if not summary:
                 st.warning("Summary not available")
            
        elif dataset_type == 'ecommerce':
            # E-COMMERCE ANALYTICS DASHBOARD
            st.markdown("## 🛒 E-Commerce Intelligence Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            price_intel = results.get('price_intel', {})
            stock_pred = results.get('stock_pred', {})
            seller_trust = results.get('seller_trust', {})
            brand_analysis = results.get('brand_analysis', {})
            fraud_detection = results.get('fraud_detection', {})
            metrics = results.get('metrics', {})
            
            with col1:
                if price_intel.get('available') and price_intel.get('price_stats'):
                    display_metric_card(
                        "Avg Price",
                        f"₹{price_intel['price_stats'].get('avg_price', 0):,.0f}",
                        f"Range: ₹{price_intel['price_stats'].get('min_price', 0):,.0f} - ₹{price_intel['price_stats'].get('max_price', 0):,.0f}"
                    )
                else:
                    display_metric_card("Price Data", "N/A", "")
            
            with col2:
                if stock_pred.get('available') and stock_pred.get('stock_stats'):
                    display_metric_card(
                        "Stock Status",
                        f"{stock_pred['stock_stats'].get('in_stock', 0):,} In Stock",
                        f"❌ {stock_pred['stock_stats'].get('out_of_stock', 0):,} Out of Stock"
                    )
                else:
                    display_metric_card("Stock Data", "N/A", "")
            
            with col3:
                if seller_trust.get('available') and seller_trust.get('rating_stats'):
                    display_metric_card(
                        "Avg Rating",
                        f"⭐ {seller_trust['rating_stats'].get('avg_rating', 0):.2f}/5.0",
                        f"{seller_trust['rating_stats'].get('high_rated', 0):,} highly rated"
                    )
                else:
                    display_metric_card("Rating Data", "N/A", "")
            
            with col4:
                if fraud_detection.get('available'):
                    risk_color = '🔴' if fraud_detection.get('risk_level') == 'HIGH' else '🟡' if fraud_detection.get('risk_level') == 'MEDIUM' else '🟢'
                    display_metric_card(
                        "Risk Level",
                        f"{risk_color} {fraud_detection.get('risk_level', 'LOW')}",
                        f"{fraud_detection.get('suspicious_count', 0):,} suspicious items"
                    )
                else:
                    display_metric_card("Risk Status", "N/A", "")
            
            st.markdown("---")
            
            # E-commerce Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🤖 Agent Intelligence",
                "💲 Price Intelligence",
                "📦 Stock Prediction",
                "⭐ Seller Trust",
                "🏷️ Brand Analysis",
                "🚨 Fraud Detection",
                "📋 AI Insights"
            ])
            
            with tab1:
                # Premium Agent Intelligence Tab
                render_agent_status_mirror()
                
                col_chat, col_feed = st.columns([3, 2])
                with col_chat:
                    render_premium_chat(st.session_state.engine, planner_model)
                with col_feed:
                    render_live_data_feed(st.session_state.engine.data if hasattr(st.session_state.engine, 'data') else None)

            with tab2:
                st.markdown("### 💲 Price Intelligence & Dynamic Pricing")
                if price_intel.get('available'):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Price tier distribution chart
                        if price_intel.get('price_tiers'):
                            import plotly.express as px
                            tiers_df = pd.DataFrame({
                                'Tier': list(price_intel['price_tiers'].keys()),
                                'Products': list(price_intel['price_tiers'].values())
                            })
                            fig = px.pie(tiers_df, values='Products', names='Tier', 
                                        title='Price Tier Distribution',
                                        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Discount analysis chart
                        if price_intel.get('discount_analysis'):
                            disc = price_intel['discount_analysis']
                            disc_df = pd.DataFrame({
                                'Category': ['With Discount', 'No Discount', 'High (>50%)', 'Suspicious (>80%)'],
                                'Count': [disc.get('products_with_discount', 0), disc.get('no_discount', 0),
                                         disc.get('high_discount_count', 0), disc.get('suspicious_discounts', 0)]
                            })
                            fig2 = px.bar(disc_df, x='Category', y='Count', title='Discount Distribution',
                                         color='Category', color_discrete_sequence=['#10b981', '#6b7280', '#f59e0b', '#ef4444'])
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Price Statistics")
                        stats = price_intel.get('price_stats', {})
                        st.metric("Average Price", f"₹{stats.get('avg_price', 0):,.2f}")
                        st.metric("Median Price", f"₹{stats.get('median_price', 0):,.2f}")
                        st.metric("Price Range", f"₹{stats.get('price_range', 0):,.2f}")
                        st.metric("Std Deviation", f"₹{stats.get('std_dev', 0):,.2f}")
                        
                        if price_intel.get('price_anomalies'):
                            st.warning(f"⚠️ {price_intel['price_anomalies'].get('count', 0)} price anomalies detected")
                    
                    st.markdown("### 💡 Price Intelligence Insights")
                    st.info(price_intel.get('insights', 'No insights available'))
                else:
                    st.warning("⚠️ Price data not available")
                    st.info("💡 Upload data with price columns for analysis")
            
            with tab2:
                st.markdown("### 📦 Stock Prediction & Inventory Intelligence")
                if stock_pred.get('available'):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Stock status pie chart
                        stats = stock_pred.get('stock_stats', {})
                        stock_df = pd.DataFrame({
                            'Status': ['In Stock', 'Out of Stock'],
                            'Count': [stats.get('in_stock', 0), stats.get('out_of_stock', 0)]
                        })
                        fig = px.pie(stock_df, values='Count', names='Status',
                                    title='Stock Status Distribution',
                                    color_discrete_sequence=['#10b981', '#ef4444'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Stock Metrics")
                        st.metric("Total Products", f"{stats.get('total_products', 0):,}")
                        st.metric("In Stock", f"{stats.get('in_stock', 0):,}", f"{stats.get('stock_rate', 0):.1f}%")
                        st.metric("Out of Stock", f"{stats.get('out_of_stock', 0):,}", f"{100 - stats.get('stock_rate', 0):.1f}%")
                        
                        # Urgent restock alert
                        if stock_pred.get('restock_urgency'):
                            st.markdown("### 🚨 Urgent Restock Categories")
                            for cat in stock_pred['restock_urgency'][:5]:
                                st.error(f"📦 {cat}")
                    
                    st.markdown("### 💡 Stock Prediction Insights")
                    st.info(stock_pred.get('insights', 'No insights available'))
                else:
                    st.warning("⚠️ Stock data not available")
                    st.info("💡 Upload data with out_of_stock column for prediction")
            
            with tab3:
                st.markdown("### ⭐ Seller Trust Score & Rating Analysis")
                if seller_trust.get('available'):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Rating distribution chart
                        if seller_trust.get('rating_distribution'):
                            rating_df = pd.DataFrame({
                                'Rating': list(seller_trust['rating_distribution'].keys()),
                                'Count': list(seller_trust['rating_distribution'].values())
                            })
                            fig = px.bar(rating_df, x='Rating', y='Count', title='Rating Distribution',
                                        color='Count', color_continuous_scale='Greens')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Rating Statistics")
                        rating_stats = seller_trust.get('rating_stats', {})
                        st.metric("Average Rating", f"⭐ {rating_stats.get('avg_rating', 0):.2f}/5.0")
                        st.metric("Highly Rated (≥4.0)", f"{rating_stats.get('high_rated', 0):,}")
                        st.metric("Low Rated (<3.0)", f"{rating_stats.get('low_rated', 0):,}")
                        st.metric("Unrated Products", f"{rating_stats.get('unrated', 0):,}")
                        
                        seller_stats = seller_trust.get('seller_stats', {})
                        if seller_stats:
                            st.markdown("### 🏪 Seller Stats")
                            st.metric("Total Sellers", f"{seller_stats.get('total_sellers', 0):,}")
                            st.metric("Top Seller", seller_stats.get('top_seller', 'N/A')[:30])
                    
                    # Flagged sellers
                    if seller_trust.get('flagged_sellers'):
                        st.markdown("### 🚨 Flagged Sellers (Low Rating + High Volume)")
                        for seller in seller_trust['flagged_sellers'][:5]:
                            st.error(f"⚠️ {seller}")
                    
                    # Trusted sellers
                    if seller_trust.get('trusted_sellers'):
                        st.markdown("### ✅ Trusted Sellers (High Rating)")
                        for seller in seller_trust['trusted_sellers'][:5]:
                            st.success(f"⭐ {seller}")
                    
                    st.markdown("### 💡 Seller Trust Insights")
                    st.info(seller_trust.get('insights', 'No insights available'))
                else:
                    st.warning("⚠️ Seller/rating data not available")
                    st.info("💡 Upload data with seller and rating columns")
            
            with tab4:
                st.markdown("### 🏷️ Brand & Category Intelligence")
                if brand_analysis.get('available'):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Top brands chart
                        if brand_analysis.get('top_brands'):
                            brands_df = pd.DataFrame(brand_analysis['top_brands'], columns=['Brand', 'Products'])
                            fig = px.bar(brands_df, x='Brand', y='Products', title='Top 10 Brands',
                                        color='Products', color_continuous_scale='Blues')
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Category distribution
                        if brand_analysis.get('category_distribution'):
                            cat_df = pd.DataFrame({
                                'Category': list(brand_analysis['category_distribution'].keys()),
                                'Products': list(brand_analysis['category_distribution'].values())
                            })
                            fig2 = px.pie(cat_df, values='Products', names='Category',
                                         title='Category Distribution')
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Brand Statistics")
                        brand_stats = brand_analysis.get('brand_stats', {})
                        st.metric("Total Brands", f"{brand_stats.get('total_brands', 0):,}")
                        st.metric("Leading Brand", brand_stats.get('top_brand', 'N/A')[:25])
                        st.metric("Top Brand Products", f"{brand_stats.get('top_brand_products', 0):,}")
                        
                        cat_stats = brand_analysis.get('category_stats', {})
                        if cat_stats:
                            st.markdown("### 📂 Category Stats")
                            st.metric("Total Categories", f"{cat_stats.get('total_categories', 0):,}")
                            st.metric("Top Category", cat_stats.get('top_category', 'N/A')[:25])
                        
                        # Best rated brands
                        if brand_analysis.get('best_rated_brands'):
                            st.markdown("### 🏆 Best Rated Brands")
                            for brand in brand_analysis['best_rated_brands'][:5]:
                                st.success(f"⭐ {brand}")
                    
                    st.markdown("### 💡 Brand Intelligence Insights")
                    st.info(brand_analysis.get('insights', 'No insights available'))
                else:
                    st.warning("⚠️ Brand data not available")
                    st.info("💡 Upload data with brand/category columns")
            
            with tab5:
                st.markdown("### 🚨 Fraud Detection & Anomaly Alerts")
                if fraud_detection.get('available'):
                    # Risk level banner
                    risk_level = fraud_detection.get('risk_level', 'LOW')
                    if risk_level == 'HIGH':
                        st.error(f"⚠️ HIGH RISK LEVEL - {fraud_detection.get('suspicious_count', 0):,} suspicious items detected!")
                    elif risk_level == 'MEDIUM':
                        st.warning(f"⚡ MEDIUM RISK LEVEL - {fraud_detection.get('suspicious_count', 0):,} items need review")
                    else:
                        st.success(f"✅ LOW RISK LEVEL - Marketplace appears healthy")
                    
                    # Fraud signals breakdown
                    if fraud_detection.get('fraud_signals'):
                        st.markdown("### 🔍 Fraud Signals Detected")
                        for signal in fraud_detection['fraud_signals']:
                            severity = signal.get('severity', 'LOW')
                            color = '#ef4444' if severity == 'HIGH' else '#f59e0b' if severity == 'MEDIUM' else '#3b82f6'
                            st.markdown(f"""
                                <div style='background: #1f2937; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {color};'>
                                    <h4 style='margin: 0; color: {color};'>[{severity}] {signal.get('type', 'Unknown')}</h4>
                                    <p style='margin: 10px 0;'><strong>Count:</strong> {signal.get('count', 0):,} items</p>
                                    <p style='margin: 5px 0;'>{signal.get('description', '')}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Summary chart
                        signals_df = pd.DataFrame(fraud_detection['fraud_signals'])
                        if not signals_df.empty:
                            fig = px.bar(signals_df, x='type', y='count', color='severity',
                                        title='Fraud Signal Distribution',
                                        color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#3b82f6'})
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No significant fraud signals detected!")
                    
                    st.markdown("### 💡 Fraud Detection Insights")
                    st.info(fraud_detection.get('insights', 'No insights available'))
                else:
                    st.warning("⚠️ Fraud detection not available")
                    st.info("💡 Upload data with price, discount, rating columns")
            
            with tab6:
                st.markdown("### 📋 AI-Powered Executive Summary")
                summary = results.get('summary', {})
                
                if summary.get('available'):
                    # Recommendations
                    if summary.get('recommendations'):
                        st.markdown("### ⚡ Strategic Recommendations")
                        for rec in summary['recommendations']:
                            priority_color = '#ef4444' if rec['priority'] == 'HIGH' else '#f59e0b' if rec['priority'] == 'MEDIUM' else '#3b82f6'
                            st.markdown(f"""
                                <div style='background: #1f2937; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {priority_color};'>
                                    <h4 style='margin: 0; color: {priority_color};'>[{rec['priority']}] {rec['category']}</h4>
                                    <p style='margin: 10px 0;'><strong>💡 Insight:</strong> {rec['insight']}</p>
                                    <p style='margin: 5px 0;'><strong>✅ Action:</strong> {rec['action']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Use cases
                    if summary.get('use_cases'):
                        st.markdown("### 🎯 Business Use Cases")
                        cols = st.columns(3)
                        for idx, use_case in enumerate(summary['use_cases']):
                            with cols[idx % 3]:
                                st.info(f"✓ {use_case}")
                    
                    st.markdown("### 📊 Full Analysis Report")
                    st.markdown(summary.get('executive_summary', 'Summary not available'))
                else:
                    st.warning("Summary generation not available")
                    
                # Show all analysis insights together
                st.markdown("### 🔍 Combined Analysis Insights")
                all_insights = []
                if price_intel.get('insights'):
                    all_insights.append(("Price Intelligence", price_intel['insights']))
                if stock_pred.get('insights'):
                    all_insights.append(("Stock Prediction", stock_pred['insights']))
                if seller_trust.get('insights'):
                    all_insights.append(("Seller Trust", seller_trust['insights']))
                if brand_analysis.get('insights'):
                    all_insights.append(("Brand Analysis", brand_analysis['insights']))
                if fraud_detection.get('insights'):
                    all_insights.append(("Fraud Detection", fraud_detection['insights']))
                
                for title, insight in all_insights:
                    with st.expander(f"📊 {title}"):
                        st.markdown(f"```\n{insight}\n```")
            
            # Legacy chat removed
        
        else:
            # SALES/GENERIC ANALYTICS DASHBOARD
            st.markdown("## 📊 Business Intelligence Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = results.get('metrics', {})
            
            with col1:
                if 'dataset_info' in metrics:
                    display_metric_card(
                        "Dataset Size",
                        f"{metrics['dataset_info']['rows']:,} rows",
                        f"{metrics['dataset_info']['columns']} columns"
                    )
                else:
                    display_metric_card("Data", "N/A", "")
            
            with col2:
                if 'numeric_summary' in metrics and metrics['numeric_summary']:
                    first_col = list(metrics['numeric_summary'].keys())[0]
                    display_metric_card(
                        first_col,
                        f"{metrics['numeric_summary'][first_col]['sum']:,.0f}",
                        f"Avg: {metrics['numeric_summary'][first_col]['mean']:,.0f}"
                    )
                else:
                    display_metric_card("Numeric Data", "N/A", "")
            
            with col3:
                if 'entity_metrics' in metrics:
                    display_metric_card(
                        "Entities",
                        f"{metrics['entity_metrics']['total_entities']:,}",
                        f"Avg: {metrics['entity_metrics']['avg_value_per_entity']:,.0f}"
                    )
                elif results.get('segments', {}).get('available'):
                    display_metric_card(
                        "Segments",
                        f"{results['segments']['total_entities']:,}",
                        f"{len(results['segments'].get('segments', {}))} groups"
                    )
                else:
                    display_metric_card("Entities", "N/A", "")
            
            with col4:
                if 'data_quality' in metrics:
                    missing_pct = float(metrics['data_quality']['missing_percentage'].rstrip('%'))
                    display_metric_card(
                        "Data Quality",
                        f"{100-missing_pct:.1f}%",
                        f"{metrics['data_quality']['missing_values']} missing"
                    )
                else:
                    display_metric_card("Quality", "N/A", "")
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🤖 Agent Intelligence",
                "📈 Forecast & Trends", 
                "👥 Segmentation", 
                "🎯 Products/Categories", 
                "🚨 Anomalies",
                "📊 Metrics", 
                "📋 AI Summary"
            ])
            
            with tab1:
                # Premium Agent Intelligence Tab
                render_agent_status_mirror()
                
                col_chat, col_feed = st.columns([3, 2])
                with col_chat:
                    render_premium_chat(st.session_state.engine, planner_model)
                with col_feed:
                    render_live_data_feed(st.session_state.engine.data if hasattr(st.session_state.engine, 'data') else None)

            with tab2:
                st.markdown("### 📈 Trend Analysis & Forecasting")
                
                if results.get('forecast', {}).get('available'):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        forecast_data = results['forecast']['forecasted_sales']
                        fig = create_sales_trend_chart(forecast_data)
                        fig.update_layout(title='Trend Forecast')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="insight-box">
                                <h4>📊 Forecast Analysis</h4>
                                <p><strong>Metric:</strong> {results['forecast']['column']}</p>
                                <p><strong>Trend:</strong> {results['forecast']['trend'].upper()}</p>
                                <p><strong>Growth Rate:</strong> {results['forecast']['monthly_growth_rate']:+,.2f}%</p>
                                <p><strong>Confidence:</strong> {results['forecast']['confidence']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🎯 Recommendations")
                        trend = results['forecast'].get('trend', 'stable')
                        if trend == 'increasing':
                            st.success("📈 **Growing Trend**: Continue current strategies")
                        elif trend == 'decreasing':
                            st.warning("📉 **Declining Trend**: Review and adjust approach")
                        else:
                            st.info("📊 **Stable Trend**: Maintain performance")
                else:
                    st.warning("⚠️ Forecast data not available")
                    st.info("💡 Upload data with date/time columns for forecasting")
            
            with tab3:
                st.markdown("### 👥 Entity Segmentation Analysis")
                
                if results.get('segments', {}).get('available'):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig = create_customer_segment_chart(results['segments']['segments'])
                        fig.update_layout(title='Segment Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Segment Details")
                        for seg_id, seg_data in results['segments']['segments'].items():
                            label = seg_data.get('Label', seg_id)
                            st.markdown(f"""
                                <div class="insight-box">
                                    <h4>{label}</h4>
                                    <p><strong>Details:</strong> Segment {seg_id}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("### 💡 Segmentation Insights")
                    st.info(results['segments'].get('insights', 'No insights available'))
                else:
                    st.warning("⚠️ Segmentation not available")
                    st.info("💡 Upload data with entity columns for segmentation")
            
            with tab4:
                st.markdown("### 🎯 Category/Product Intelligence")
                
                if results.get('products', {}).get('available'):
                    category_data = results['products']['category_performance']
                    
                    if category_data:
                        fig = create_category_performance_chart(category_data)
                        fig.update_layout(title='Category Performance')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Grouping:** {results['products'].get('grouping_column', 'N/A')}")
                        with col2:
                            st.info(f"**Metric:** {results['products'].get('value_column', 'N/A')}")
                    
                    st.markdown("### 📈 Category Insights")
                    st.markdown(f"""
                        <div class="insight-box">
                            {results['products']['insights'].replace(chr(10), '<br>')}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Category data not available")
                    st.info("💡 Upload data with categorical columns")
            
            with tab5:
                st.markdown("### 🚨 Anomaly Detection")
                
                if results.get('anomalies', {}).get('available'):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="warning-box">
                                <h4>🚨 Anomalies Detected</h4>
                                <h2>{results['anomalies'].get('sales_outliers_count', 0)}</h2>
                                <p>{results['anomalies'].get('outlier_percentage', 0):.2f}% of data</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="insight-box">
                                <h4>📊 Detection Summary</h4>
                                <p>Outliers found in numeric columns</p>
                                <p>Risk Level: {results['anomalies'].get('risk_level', 'UNKNOWN')}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    if results['anomalies'].get('findings'):
                        st.markdown("### 📋 Anomaly Details")
                        for finding in results['anomalies']['findings'][:10]:
                            severity_color = '#ef4444' if finding.get('severity') == 'CRITICAL' else '#f59e0b'
                            st.markdown(f"""
                                <div style='background: #1f2937; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid {severity_color};'>
                                    <strong>{finding.get('column', 'N/A')}</strong> - {finding.get('count', 0)} outliers
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if results['anomalies'].get('reasoning'):
                        st.markdown("### 🤖 AI Analysis")
                        st.warning(results['anomalies']['reasoning'])
                else:
                    st.info("🔍 Anomaly detection not available")
            
            with tab6:
                st.markdown("### 📊 Dataset Metrics Overview")
                
                metrics = results.get('metrics', {})
                
                if metrics.get('numeric_summary'):
                    st.markdown("### 📈 Numeric Column Statistics")
                    for col_name, stats in metrics['numeric_summary'].items():
                        with st.expander(f"📊 {col_name}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sum", f"{stats.get('sum', 0):,.2f}")
                                st.metric("Mean", f"{stats.get('mean', 0):,.2f}")
                            with col2:
                                st.metric("Min", f"{stats.get('min', 0):,.2f}")
                                st.metric("Max", f"{stats.get('max', 0):,.2f}")
                            with col3:
                                st.metric("Median", f"{stats.get('median', 0):,.2f}")
                                st.metric("Std Dev", f"{stats.get('std', 0):,.2f}")
                
                if metrics.get('categorical_summary'):
                    st.markdown("### 📂 Categorical Columns")
                    for col_name, stats in metrics['categorical_summary'].items():
                        st.info(f"**{col_name}**: {stats.get('unique_values', 0)} unique values, Top: {stats.get('top_value', 'N/A')}")
                
                if metrics.get('data_quality'):
                    st.markdown("### ✅ Data Quality")
                    st.success(f"Data completeness: {100 - float(metrics['data_quality']['missing_percentage'].rstrip('%')):.1f}%")
            
            with tab7:
                st.markdown("### 📋 AI-Powered Executive Summary")
                
                summary = results.get('summary', {})
                
                # Display full agent-generated report first
                if summary.get('executive_summary'):
                    st.markdown(summary['executive_summary'])
                elif isinstance(summary, str):
                    st.markdown(summary)
                    
                # Show recommendations if available (Legacy support)
                if isinstance(summary, dict) and summary.get('recommendations'):
                    st.markdown("### ⚡ Strategic Actions (Auto-Generated)")
                    for rec in summary['recommendations']:
                        priority_color = '#ef4444' if rec.get('priority') == 'HIGH' else '#f59e0b' if rec.get('priority') == 'MEDIUM' else '#3b82f6'
                        st.markdown(f"""
                            <div style='background: #1f2937; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {priority_color};'>
                                <h4 style='margin: 0; color: {priority_color};'>[{rec.get('priority', 'INFO')}] {rec.get('category', 'Recommendation')}</h4>
                                <p style='margin: 10px 0;'><strong>💡 Insight:</strong> {rec.get('insight', '')}</p>
                                <p style='margin: 5px 0;'><strong>✅ Action:</strong> {rec.get('action', '')}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📄 Download Summary (TXT)",
                        data=summary.get('executive_summary', 'No summary'),
                        file_name=f"cognifyx_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                # Action buttons
                st.markdown("---")
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col2:
                    json_data = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        "📊 Download Data (JSON)",
                        data=json_data,
                        file_name=f"cognifyx_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with action_col3:
                    if st.button("🔄 New Analysis", use_container_width=True):
                        st.session_state.analysis_complete = False
                        st.session_state.results = None
                        st.rerun()
    
    else:
        # Welcome Screen
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h2>👋 Welcome to CognifyX Intelligence Platform</h2>
                <p style='font-size: 18px; color: #9ca3af;'>
                    Your Universal Multi-Agent Analytics Engine
                </p>
                <p style='font-size: 16px; color: #6b7280; margin-top: 20px;'>
                    Upload ANY CSV dataset and let our AI agents analyze it intelligently
                </p>
                <br>
                <div style='display: flex; justify-content: center; gap: 30px; margin-top: 30px;'>
                    <div class='agent-card' style='width: 250px;'>
                        <h3>🧠 Planner Agent</h3>
                        <p><strong>Role:</strong> Strategic planning</p>
                        <p>Analyzes dataset structure and creates analysis strategy</p>
                    </div>
                    <div class='agent-card' style='width: 250px;'>
                        <h3>⚙️ Worker Agent</h3>
                        <p><strong>Role:</strong> Data analysis</p>
                        <p>Executes analysis and generates insights</p>
                    </div>
                    <div class='agent-card' style='width: 250px;'>
                        <h3>✅ Reviewer Agent</h3>
                        <p><strong>Role:</strong> Quality validation</p>
                        <p>Reviews and validates all findings</p>
                    </div>
                </div>
                <br><br>
                <div style='background: #1f2937; padding: 20px; border-radius: 10px; margin-top: 30px; max-width: 600px; margin-left: auto; margin-right: auto;'>
                    <h3 style='color: #10b981;'>✨ What CognifyX Can Analyze:</h3>
                    <ul style='text-align: left; color: #9ca3af;'>
                        <li>📈 Sales & Revenue Data</li>
                        <li>👥 Customer Analytics</li>
                        <li>📦 Inventory & Products</li>
                        <li>💰 Financial Transactions</li>
                        <li>📊 Any CSV Dataset!</li>
                    </ul>
                </div>
                <br>
                <p style='font-size: 14px; color: #6b7280;'>
                    👈 Configure agents in the sidebar and click <strong>"🚀 Start Analysis"</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
