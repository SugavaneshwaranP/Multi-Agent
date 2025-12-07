"""
CognifyX Agentic UI - Multi-Agent Intelligence Platform
Interactive Streamlit interface for CognifyX Engine
Supports: Sales Data, Resume Analysis, Generic CSV
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from tools.cognifyx_engine import CognifyXEngine
from tools.resume_analyzer import ResumeAnalyzer
import time
import os
import glob

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

def display_agent_card(agent_name, model, status, role):
    """Display agent status card"""
    status_icon = "🔄" if status == "running" else "✅" if status == "complete" else "⏸️"
    status_class = "status-running" if status == "running" else "status-complete" if status == "complete" else ""
    
    st.markdown(f"""
        <div class="agent-card">
            <h3>{status_icon} {agent_name}</h3>
            <p><strong>Model:</strong> {model}</p>
            <p><strong>Role:</strong> {role}</p>
            <p class="{status_class}"><strong>Status:</strong> {status.upper()}</p>
        </div>
    """, unsafe_allow_html=True)

def display_metric_card(label, value, delta=None):
    """Display metric card"""
    delta_html = f"<p style='color: #10b981; font-size: 14px;'>{delta}</p>" if delta else ""
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #9ca3af; font-size: 14px; margin: 0;">{label}</h4>
            <h2 style="color: #ffffff; margin: 10px 0;">{value}</h2>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def log_agent_activity(agent, message):
    """Log agent activity"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {agent}: {message}"
    st.session_state.agent_logs.append(log_entry)
    
def display_agent_logs():
    """Display agent activity logs"""
    st.markdown("### 🔍 Agent Activity Log")
    log_container = st.container()
    with log_container:
        for log in st.session_state.agent_logs[-10:]:  # Show last 10 logs
            st.markdown(f'<div class="agent-thinking">{log}</div>', unsafe_allow_html=True)

def create_sales_trend_chart(data):
    """Create interactive sales trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines+markers',
        name='Sales',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, color='#60a5fa'),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        title='Sales Trend Analysis',
        xaxis_title='Period',
        yaxis_title='Sales ($)',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_customer_segment_chart(segments):
    """Create customer segmentation visualization"""
    labels = [seg['Label'] for seg in segments.values()]
    values = [seg['Customer ID'] for seg in segments.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']),
        textinfo='label+percent',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title='Customer Segmentation',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_category_performance_chart(data, value_col=None):
    """Create category performance chart - dynamic"""
    if not data or len(data) == 0:
        # Return empty chart
        fig = go.Figure()
        fig.update_layout(
            title='No categorical data available',
            template='plotly_dark',
            height=400
        )
        return fig
    
    categories = list(data.keys())
    
    # Get first numeric column from data
    first_cat_data = data[categories[0]]
    numeric_cols = [k for k, v in first_cat_data.items() if isinstance(v, (int, float))]
    
    if len(numeric_cols) == 0:
        fig = go.Figure()
        fig.update_layout(title='No numeric data', template='plotly_dark', height=400)
        return fig
    
    # Use first two numeric columns or duplicate if only one
    col1 = numeric_cols[0]
    col2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    
    values1 = [data[cat].get(col1, 0) for cat in categories]
    values2 = [data[cat].get(col2, 0) for cat in categories]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{col1} by Category', f'{col2} by Category'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=values1, name=col1, marker_color='#3b82f6'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=values2, name=col2, marker_color='#10b981'),
        row=1, col=2
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def create_skills_chart(skills_data):
    """Create skills frequency chart"""
    if not skills_data or not skills_data.get('available'):
        fig = go.Figure()
        fig.update_layout(title='No skills data', template='plotly_dark', height=400)
        return fig
    
    tech_skills = skills_data.get('top_technical_skills', [])
    soft_skills = skills_data.get('top_soft_skills', [])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top Technical Skills', 'Top Soft Skills'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    if tech_skills:
        skills, counts = zip(*tech_skills)
        fig.add_trace(
            go.Bar(x=list(counts), y=list(skills), orientation='h', marker_color='#3b82f6', name='Technical'),
            row=1, col=1
        )
    
    if soft_skills:
        skills, counts = zip(*soft_skills)
        fig.add_trace(
            go.Bar(x=list(counts), y=list(skills), orientation='h', marker_color='#10b981', name='Soft Skills'),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Skills Analysis',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def create_experience_chart(exp_data):
    """Create experience distribution chart"""
    if not exp_data or not exp_data.get('available'):
        fig = go.Figure()
        fig.update_layout(title='No experience data', template='plotly_dark', height=400)
        return fig
    
    distribution = exp_data.get('distribution', {})
    categories = list(distribution.keys())
    values = list(distribution.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=0.4,
        marker=dict(colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']),
        textinfo='label+value',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title=f'Experience Distribution (Avg: {exp_data.get("average_experience", 0):.1f} years)',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_education_chart(edu_data):
    """Create education distribution chart"""
    if not edu_data or not edu_data.get('available'):
        fig = go.Figure()
        fig.update_layout(title='No education data', template='plotly_dark', height=400)
        return fig
    
    distribution = edu_data.get('distribution', {})
    categories = list(distribution.keys())
    values = list(distribution.values())
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker=dict(color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']),
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Education Qualifications',
        template='plotly_dark',
        height=400,
        xaxis_title='Qualification',
        yaxis_title='Number of Candidates'
    )
    
    return fig

def create_category_distribution_chart(category_data):
    """Create resume category distribution chart"""
    if not category_data or len(category_data) == 0:
        fig = go.Figure()
        fig.update_layout(title='No category data', template='plotly_dark', height=400)
        return fig
    
    categories = list(category_data.keys())[:15]  # Top 15
    values = [category_data[cat] for cat in categories]
    
    fig = go.Figure(data=[go.Bar(
        y=categories,
        x=values,
        orientation='h',
        marker=dict(color=values, colorscale='Viridis'),
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Resume Categories Distribution',
        template='plotly_dark',
        height=500,
        xaxis_title='Number of Resumes',
        yaxis_title='Category'
    )
    
    return fig

def detect_dataset_type(file_path):
    """Auto-detect if dataset is Sales, Resume, or Generic"""
    try:
        # Check if it's a directory (PDF resumes)
        if os.path.isdir(file_path):
            return 'resume'
        
        # Read first few rows of CSV
        df = pd.read_csv(file_path, nrows=100)
        columns = [col.lower() for col in df.columns]
        
        # Resume indicators
        resume_keywords = ['resume', 'skill', 'experience', 'education', 'degree', 'qualification', 'candidate', 'cv', 'job']
        resume_score = sum(1 for col in columns if any(kw in col for kw in resume_keywords))
        
        # Sales indicators
        sales_keywords = ['sales', 'revenue', 'profit', 'customer', 'product', 'order', 'discount', 'quantity']
        sales_score = sum(1 for col in columns if any(kw in col for kw in sales_keywords))
        
        if resume_score > sales_score and resume_score >= 2:
            return 'resume'
        elif sales_score >= 2:
            return 'sales'
        else:
            return 'generic'
            
    except:
        return 'generic'

def main():
    """Main application"""
    init_session_state()
    
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
            ["Sample - Superstore.csv", "Resume PDFs (500 samples)", "Resume Dataset (CSV)", "Upload Custom"],
            index=0
        )
        
        if dataset == "Upload Custom":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                os.makedirs("datasets/uploads", exist_ok=True)
                file_path = f"datasets/uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")
                # Detect dataset type
                dataset_type = detect_dataset_type(file_path)
                st.info(f"📊 Detected: {dataset_type.upper()} dataset")
            else:
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
        else:
            file_path = "datasets/sales/Sample - Superstore.csv"
            st.success("✅ Superstore dataset ready")
        
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
                summary = analyzer.generate_resume_report()
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
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔧 Skills Analysis",
                "💼 Experience",
                "🎓 Education",
                "🏆 Top Candidates"
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
            
            # Executive Summary with Recommendations
            st.markdown("---")
            st.markdown("## 📋 Executive Summary & Recommendations")
            summary = results.get('summary', {})
            if summary.get('available'):
                # Show recommendations prominently
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
                
                # Show use cases
                if summary.get('use_cases'):
                    st.markdown("### 🎯 Business Use Cases")
                    cols = st.columns(3)
                    for idx, use_case in enumerate(summary['use_cases']):
                        with cols[idx % 3]:
                            st.info(f"✓ {use_case}")
                
                st.markdown("### 📊 Full Analysis Report")
                st.markdown(summary.get('executive_summary', 'Summary not available'))
            else:
                st.warning("Summary not available")
            
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
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Forecasting", 
                "👥 Customers", 
                "📦 Products", 
                "⚠️ Anomalies", 
                "📋 Executive Summary"
            ])
            
            with tab1:
                st.markdown("### 🔮 Forecasting Analysis (LLM-Powered)")
                
                if results.get('forecast', {}).get('available'):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Forecast chart
                        forecast_data = results['forecast']['forecasted_sales']
                        fig = create_sales_trend_chart(forecast_data)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="insight-box">
                                <h4>📊 Forecast Insights</h4>
                                <p><strong>Column:</strong> {results['forecast']['column']}</p>
                                <p><strong>Trend:</strong> {results['forecast']['trend'].upper()}</p>
                                <p><strong>Growth Rate:</strong> {results['forecast']['monthly_growth_rate']:,.2f}</p>
                                <p><strong>Confidence:</strong> {results['forecast']['confidence']}</p>
                                <p><strong>Next Period:</strong> {results['forecast']['forecasted_sales'][0]:,.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🤖 AI Reasoning")
                        st.info(results['forecast']['reasoning'])
                else:
                    st.warning(f"⚠️ {results.get('forecast', {}).get('message', 'Not available')}")
                    st.info(f"💡 {results.get('forecast', {}).get('suggestion', 'Upload a dataset with date and numeric columns')}")
            
            with tab2:
                st.markdown("### 👥 Entity Segmentation (LLM-Powered)")
                
                if results.get('segments', {}).get('available'):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig = create_customer_segment_chart(results['segments']['segments'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Segment Details")
                        st.info(f"**Entity Column:** {results['segments']['entity_column']}")
                        st.info(f"**Value Column:** {results['segments']['value_column']}")
                        
                        for seg_id, seg_data in results['segments']['segments'].items():
                            st.markdown(f"""
                                <div class="insight-box">
                                    <h4>{seg_data['Label']}</h4>
                                    <p><strong>Count:</strong> {seg_data['Customer ID']}</p>
                                    <p><strong>Avg Value:</strong> {seg_data['Sales']:,.2f}</p>
                                    <p><strong>Insight:</strong> {seg_data['insight']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ {results.get('segments', {}).get('message', 'Not available')}")
                    st.info(f"💡 {results.get('segments', {}).get('suggestion', 'Upload a dataset with entity and value columns')}")
            
            with tab3:
                st.markdown("### 📦 Categorical Analysis")
                
                if results.get('products', {}).get('available'):
                    category_data = results['products']['category_performance']
                    
                    if category_data:
                        fig = create_category_performance_chart(category_data)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Grouping Column:** {results['products'].get('grouping_column', 'N/A')}")
                        with col2:
                            st.info(f"**Value Column:** {results['products'].get('value_column', 'N/A')}")
                    else:
                        st.info("No categorical data to display")
                    
                    st.markdown("### 💡 Analysis Insights")
                    st.markdown(f"""
                        <div class="insight-box">
                            {results['products']['insights'].replace('\n', '<br>')}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ {results.get('products', {}).get('message', 'Not available')}")
                    st.info(f"💡 {results.get('products', {}).get('suggestion', 'Upload a dataset with categorical columns')}")
            
            with tab4:
                st.markdown("### ⚠️ Anomaly Detection (LLM-Powered)")
                
                if results.get('anomalies', {}).get('available'):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="warning-box">
                                <h4>🚨 Outliers</h4>
                                <h2>{results['anomalies'].get('sales_outliers_count', 0)}</h2>
                                <p>{results['anomalies'].get('outlier_percentage', 0):.2f}% of total</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        total_findings = sum(f.get('count', 0) for f in results['anomalies'].get('findings', []))
                        st.markdown(f"""
                            <div class="warning-box">
                                <h4>🔍 Total Findings</h4>
                                <h2>{total_findings}</h2>
                                <p>{len(results['anomalies'].get('findings', []))} types detected</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                            <div class="warning-box">
                                <h4>⚡ Risk Level</h4>
                                <h2>{results['anomalies'].get('risk_level', 'UNKNOWN')}</h2>
                                <p>Overall assessment</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed findings
                    if results['anomalies'].get('findings'):
                        st.markdown("### 📋 Detailed Findings")
                        for finding in results['anomalies']['findings'][:10]:  # Show top 10
                            severity_color = '#ef4444' if finding.get('severity') == 'CRITICAL' else '#f59e0b' if finding.get('severity') == 'HIGH' else '#3b82f6'
                            st.markdown(f"""
                                <div style='background: #1f2937; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid {severity_color};'>
                                    <strong>{finding.get('column', 'N/A')}</strong> - {finding.get('type', 'unknown')}
                                    <br>Count: {finding.get('count', 0)} | Severity: {finding.get('severity', 'MEDIUM')}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("### 🔍 Anomaly Analysis")
                    if results['anomalies'].get('reasoning'):
                        st.warning(results['anomalies']['reasoning'])
                else:
                    st.info("No anomaly detection results available")
                    if results.get('anomalies', {}).get('message'):
                        st.warning(results['anomalies']['message'])
            
            with tab5:
                st.markdown("### 📋 Executive Summary")
                
                st.markdown(f"""
                    <div style='background: #1f2937; padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                        <pre style='color: #e5e7eb; font-size: 13px; white-space: pre-wrap; font-family: monospace;'>
{results.get('summary', {}).get('executive_summary', 'No summary available')}
                        </pre>
                    </div>
                """, unsafe_allow_html=True)
                
                # Download options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📄 Download Summary (TXT)",
                        data=results.get('summary', {}).get('executive_summary', 'No summary'),
                        file_name=f"cognifyx_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    json_data = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        "📊 Download Data (JSON)",
                        data=json_data,
                        file_name=f"cognifyx_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col3:
                    if st.button("🔄 New Analysis"):
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
