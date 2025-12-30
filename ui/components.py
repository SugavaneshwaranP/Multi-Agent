import streamlit as st
from datetime import datetime

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
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    st.session_state.agent_logs.append(log_entry)
    
def display_agent_logs():
    """Display agent activity logs"""
    st.markdown("### 🔍 Agent Activity Log")
    log_container = st.container()
    with log_container:
        if 'agent_logs' in st.session_state:
            for log in st.session_state.agent_logs[-10:]:  # Show last 10 logs
                st.markdown(f'<div class="agent-thinking">{log}</div>', unsafe_allow_html=True)
