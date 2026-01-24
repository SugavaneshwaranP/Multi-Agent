import streamlit as st
import time
from datetime import datetime
import pandas as pd
from agents.chat_agent import ChatAgent

def inject_premium_css():
    """Inject premium CSS for the agentic UI"""
    st.markdown("""
        <style>
        /* Glassmorphism containers */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        .agent-pulse {
            width: 12px;
            height: 12px;
            background: #60a5fa;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
            box-shadow: 0 0 0 0 rgba(96, 165, 250, 1);
            animation: pulse-blue 2s infinite;
        }
        
        @keyframes pulse-blue {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 10px rgba(96, 165, 250, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(96, 165, 250, 0);
            }
        }
        
        .agent-status-tag {
            background: rgba(96, 165, 250, 0.1);
            color: #60a5fa;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            border: 1px solid rgba(96, 165, 250, 0.3);
        }
        
        .chat-timestamp {
            font-size: 10px;
            color: #9ca3af;
            margin-bottom: 5px;
        }
        
        .thinking-process {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 11px;
            color: #10b981;
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 8px;
            border-left: 2px solid #10b981;
            margin: 10px 0;
        }
        
        /* Premium Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(96, 165, 250, 0.3);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(96, 165, 250, 0.5);
        }
        
        .glow-text {
            color: #fff;
            text-shadow: 0 0 10px rgba(96, 165, 250, 0.8), 0 0 20px rgba(96, 165, 250, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)

def render_agent_status_mirror():
    """Render a live status of the 3 agents"""
    st.markdown("<h3 class='glow-text'>🧠 Neural Processing Core</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    agents = [
        {"name": "Planner", "model": st.session_state.get('planner_model', 'llama3'), "role": "Strategy"},
        {"name": "Worker", "model": st.session_state.get('worker_model', 'mistral'), "role": "Analysis"},
        {"name": "Reviewer", "model": st.session_state.get('reviewer_model', 'qwen2.5'), "role": "Validation"}
    ]
    
    for i, agent in enumerate(agents):
        with [col1, col2, col3][i]:
            st.markdown(f"""
                <div class="glass-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-weight: bold; color: #fff;">{agent['name']}</span>
                        <span class="agent-pulse"></span>
                    </div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 10px;">
                        Model: <span style="color: #60a5fa;">{agent['model']}</span><br>
                        Role: {agent['role']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_premium_chat(data_source_obj, model_name="llama3"):
    """Enhanced chat interface with premium look and real-time feel"""
    st.markdown("<h3 class='glow-text'>💬 Agentic Reasoning Terminal</h3>", unsafe_allow_html=True)
    
    # Session state for premium chat
    if "p_messages" not in st.session_state:
        st.session_state.p_messages = []
        # Initial greeting from agent
        st.session_state.p_messages.append({
            "role": "assistant", 
            "content": f"Neural Core initialized with {model_name}. I have loaded the dataset and I'm ready for real-time analysis. How can I assist you today?",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

    # Container for messages
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.p_messages:
            with st.chat_message(msg["role"]):
                st.markdown(f"<div class='chat-timestamp'>{msg['timestamp']}</div>", unsafe_allow_html=True)
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Command the agent..."):
        # Log user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.p_messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(f"<div class='chat-timestamp'>{timestamp}</div>", unsafe_allow_html=True)
                st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(f"""
                <div class="thinking-process">
                    [SYSTEM] Routing query to {model_name}...<br>
                    [ANALYSIS] Scanning dataset indices...<br>
                    [PLANNING] Decomposing request into pandas operations...
                </div>
            """, unsafe_allow_html=True)
            
            try:
                # Execution
                if hasattr(data_source_obj, 'data'):
                    df = data_source_obj.data
                else:
                    df = None
                
                start_time = time.time()
                agent = ChatAgent(df, model=model_name)
                response = agent.query(prompt)
                end_time = time.time()
                
                thinking_placeholder.empty()
                
                final_timestamp = datetime.now().strftime("%H:%M:%S")
                lat = end_time - start_time
                
                st.markdown(f"<div class='chat-timestamp'>{final_timestamp} | Latency: {lat:.2f}s</div>", unsafe_allow_html=True)
                st.markdown(response)
                
                st.session_state.p_messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "timestamp": final_timestamp
                })
                
            except Exception as e:
                thinking_placeholder.error(f"Execution Error: {str(e)}")

def render_live_data_feed(data):
    """Render a scrolling live feedback of the data/insights"""
    st.markdown("<h3 class='glow-text'>📡 Live Intelligence Feed</h3>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        if data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", f"{len(data):,}")
            with col2:
                missing = data.isnull().sum().sum()
                st.metric("Data Quality", f"{100 - (missing/(len(data)*len(data.columns))*100):.1f}%")
            
            st.markdown("#### Sample Streams")
            st.dataframe(data.head(5), use_container_width=True)
        else:
            st.warning("No data stream active.")
            
        st.markdown("</div>", unsafe_allow_html=True)
