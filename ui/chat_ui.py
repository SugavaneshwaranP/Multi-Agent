import streamlit as st
from agents.chat_agent import ChatAgent

def render_chat_tab(data_source_obj, model_name="llama3"):
    """
    Renders the chat interface tab.
    args:
        data_source_obj: Object containing .data property (DataFrame)
        model_name: Name of the LLM model to use
    """
    st.markdown("### 💬 Chat with your Data")
    st.markdown("Ask questions to gain deeper insights from your dataset.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    # Handle different object types
                    if hasattr(data_source_obj, 'data'):
                        df = data_source_obj.data
                    elif isinstance(data_source_obj, dict): # Handle dict results if passed purely
                         # Try to find a dataframe in the dict, or fail gracefully
                         st.error("Chat currently only supports raw dataset access.")
                         return
                    else:
                        df = None
                        
                    agent = ChatAgent(df, model=model_name)
                    response = agent.query(prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
