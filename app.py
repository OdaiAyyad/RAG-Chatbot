"""
Streamlit UI for RAG Chatbot

This is the user-facing application.
It provides a clean interface for:
- Asking questions
- Viewing answers
- Inspecting retrieved sources
- Seeing confidence scores

Design Principles:
- Simple and professional
- Show confidence/sources for transparency
- Clear error messages
- Responsive feedback
"""

import streamlit as st
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from retriever import Retriever
from llm_interface import RAGPipeline
from config import PAGE_TITLE, PAGE_ICON


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",  # Use full width
    initial_sidebar_state="expanded"
)


# ============================================================================
# INITIALIZATION (with caching for performance)
# ============================================================================

@st.cache_resource
def initialize_pipeline():
    """
    Initialize RAG pipeline.
    
    @st.cache_resource ensures this only runs once, even if user interacts
    with the UI. This is crucial because:
    - Loading FAISS index takes time
    - Loading embedding model takes time
    - We don't want to reload on every button click
    """
    try:
        retriever = Retriever()
        pipeline = RAGPipeline(retriever)
        return pipeline, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# SIDEBAR (Settings & Info)
# ============================================================================

def render_sidebar(pipeline):
    """Render sidebar with settings and system info."""
    st.sidebar.title("⚙️ System Info")
    
    if pipeline is not None:
        # Show index statistics
        num_chunks = pipeline.retriever.faiss_index.index.ntotal
        st.sidebar.metric("Indexed Chunks", f"{num_chunks:,}")
        
        # Show current settings
        st.sidebar.metric("Confidence Threshold", 
                         f"{pipeline.retriever.confidence_threshold:.2f}")
        st.sidebar.metric("Top K Retrieval", 
                         pipeline.retriever.top_k)
    
    st.sidebar.markdown("---")
    
    # Instructions
    st.sidebar.title("📖 How to Use")
    st.sidebar.markdown("""
    1. **Type your question** in the text box
    2. **Click "Ask"** or press Enter
    3. **Review the answer** and sources
    
    **Confidence Score Meaning:**
    - 🟢 **0.80+**: Very confident match
    - 🟡 **0.65-0.80**: Moderate confidence
    - 🔴 **Below 0.65**: Will escalate to human
    
    **Source Attribution:**
    Each answer shows which documents were used,
    allowing you to verify the information.
    """)
    
    st.sidebar.markdown("---")
    
    # About section
    with st.sidebar.expander("ℹ️ About This System"):
        st.markdown("""
        This is a **Retrieval-Augmented Generation (RAG)** chatbot.
        
        **How it works:**
        1. Your question is converted to a vector
        2. Similar documents are retrieved
        3. Only high-confidence matches are used
        4. An LLM generates an answer from those documents
        5. If confidence is low, your query is escalated
        
        **Why this prevents hallucination:**
        - Answers come ONLY from your knowledge base
        - No external knowledge is used
        - Low-confidence queries are escalated, not guessed
        """)


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

def render_chat_interface(pipeline):
    """Render main chat interface."""
    
    # Header
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("Ask me anything about our products and services!")
    
    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Question input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Your Question:",
            placeholder="e.g., How do I reset my password?",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("🤔 Thinking..."):
            # Get answer
            result = pipeline.answer_question(question)
            
            # Add to history
            st.session_state.history.append({
                'question': question,
                'result': result
            })
    
    # Clear history button
    if st.session_state.history:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Display chat history (most recent first)
    st.markdown("---")
    
    for i, chat in enumerate(reversed(st.session_state.history)):
        render_chat_item(chat, len(st.session_state.history) - i)


def render_chat_item(chat: dict, index: int):
    """Render a single question-answer pair."""
    
    question = chat['question']
    result = chat['result']
    
    # Question
    st.markdown(f"### 💬 Question #{index}")
    st.info(question)
    
    # Answer section
    st.markdown("### 🤖 Answer")
    
    # Show confidence indicator
    confidence = result['confidence']
    should_escalate = result['should_escalate']
    
    if should_escalate:
        st.error(f"⚠️ **Confidence too low** (best match: {confidence:.2f})")
        st.warning(result['answer'])
    else:
        # Color-code confidence
        if confidence >= 0.80:
            st.success(f"🟢 **High Confidence** ({confidence:.2f})")
        elif confidence >= 0.65:
            st.warning(f"🟡 **Moderate Confidence** ({confidence:.2f})")
        
        st.write(result['answer'])
    
    # Sources section (if any)
    if result['sources']:
        with st.expander(f"📚 View {len(result['sources'])} Source Document(s)"):
            for source in result['sources']:
                st.markdown(f"""
                **Document {source['rank']}** (ID: `{source['doc_id']}`)  
                *Similarity Score: {source['score']:.3f}*
                
                > {source['text'][:300]}{'...' if len(source['text']) > 300 else ''}
                """)
                st.markdown("---")
    
    st.markdown("---")


# ============================================================================
# ERROR HANDLING
# ============================================================================

def render_error_page(error_message: str):
    """Show error page if initialization fails."""
    st.error("⚠️ System Initialization Failed")
    
    st.markdown(f"""
    **Error Details:**
    ```
    {error_message}
    ```
    
    **Possible causes:**
    1. **FAISS index not built yet**
       → Run `build_index.py` first
    
    2. **Missing API key**
       → Set `GEMINI_API_KEY` environment variable
    
    3. **Missing dependencies**
       → Run `pip install -r requirements.txt`
    
    4. **Corrupted index files**
       → Delete `embeddings/` folder and rebuild
    
    **To fix:**
    1. Check the error message above
    2. Follow the setup instructions in README.md
    3. Restart the app after fixing
    """)


# ============================================================================
# MAIN APP LOGIC
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize pipeline
    pipeline, error = initialize_pipeline()
    
    # Check for initialization errors
    if error is not None:
        render_error_page(error)
        return
    
    # Render UI
    render_sidebar(pipeline)
    render_chat_interface(pipeline)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()