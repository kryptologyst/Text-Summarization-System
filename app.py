"""
Streamlit Web Interface for Modern Text Summarization
====================================================

A beautiful, modern web interface for the text summarization system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

from modern_summarizer import ModernTextSummarizer, SummarizationConfig, MockDatabase

# Page configuration
st.set_page_config(
    page_title="🧠 Modern Text Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .method-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .summary-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    if 'database' not in st.session_state:
        st.session_state.database = MockDatabase()
    if 'summaries_history' not in st.session_state:
        st.session_state.summaries_history = []

def create_summarizer(config: SummarizationConfig) -> ModernTextSummarizer:
    """Create and cache summarizer instance"""
    if st.session_state.summarizer is None:
        with st.spinner("Initializing summarization models..."):
            st.session_state.summarizer = ModernTextSummarizer(config)
    return st.session_state.summarizer

def display_summary_results(summary: list, scores: dict, method: str):
    """Display summary results with metrics"""
    st.markdown(f"### 📝 {method.upper()} Summary")
    
    # Display summary sentences
    summary_text = ""
    for i, sentence in enumerate(summary, 1):
        st.markdown(f"**{i}.** {sentence}")
        summary_text += sentence + " "
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROUGE-1 F1", f"{scores['rouge1_fmeasure']:.3f}")
    with col2:
        st.metric("ROUGE-2 F1", f"{scores['rouge2_fmeasure']:.3f}")
    with col3:
        st.metric("ROUGE-L F1", f"{scores['rougeL_fmeasure']:.3f}")
    
    return summary_text.strip()

def create_metrics_chart(scores_data: list):
    """Create interactive metrics chart"""
    if not scores_data:
        return
    
    df = pd.DataFrame(scores_data)
    
    fig = go.Figure()
    
    methods = df['method'].unique()
    metrics = ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']
    
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=methods,
            y=df[metric],
            mode='lines+markers',
            name=metric.replace('_', '-').upper(),
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Summarization Quality Metrics Comparison",
        xaxis_title="Method",
        yaxis_title="Score",
        hovermode='x unified',
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Modern Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced extractive text summarization using multiple state-of-the-art algorithms
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Summarization Method",
        ["tfidf", "textrank", "bert", "sentence_bert"],
        help="Choose the algorithm for text summarization"
    )
    
    # Parameters
    max_sentences = st.sidebar.slider(
        "Maximum Sentences",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of sentences in the summary"
    )
    
    min_sentence_length = st.sidebar.slider(
        "Minimum Sentence Length",
        min_value=5,
        max_value=50,
        value=10,
        help="Minimum character length for sentences"
    )
    
    max_sentence_length = st.sidebar.slider(
        "Maximum Sentence Length",
        min_value=100,
        max_value=500,
        value=200,
        help="Maximum character length for sentences"
    )
    
    remove_stopwords = st.sidebar.checkbox("Remove Stopwords", value=True)
    stem_words = st.sidebar.checkbox("Stem Words", value=False)
    
    # Create configuration
    config = SummarizationConfig(
        max_sentences=max_sentences,
        min_sentence_length=min_sentence_length,
        max_sentence_length=max_sentence_length,
        method=method,
        remove_stopwords=remove_stopwords,
        stem_words=stem_words
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summarize Text", "📚 Sample Documents", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Text Summarization")
        
        # Text input options
        input_option = st.radio(
            "Choose input method:",
            ["Manual Input", "Sample Document"],
            horizontal=True
        )
        
        text = ""
        
        if input_option == "Manual Input":
            text = st.text_area(
                "Enter text to summarize:",
                height=200,
                placeholder="Paste your text here..."
            )
        else:
            # Sample document selection
            documents = st.session_state.database.get_documents()
            if documents:
                doc_options = {f"{doc['title']} ({doc['category']})": doc for doc in documents}
                selected_doc = st.selectbox("Select a sample document:", list(doc_options.keys()))
                
                if selected_doc:
                    doc = doc_options[selected_doc]
                    text = doc['content']
                    st.text_area("Document content:", value=text, height=200, disabled=True)
            else:
                st.warning("No sample documents available.")
        
        # Summarize button
        if st.button("🚀 Generate Summary", type="primary", disabled=not text.strip()):
            if text.strip():
                # Create summarizer
                summarizer = create_summarizer(config)
                
                # Generate summary
                with st.spinner("Generating summary..."):
                    try:
                        summary, scores = summarizer.summarize(text, method)
                        
                        # Display results
                        summary_text = display_summary_results(summary, scores, method)
                        
                        # Save to history
                        st.session_state.summaries_history.append({
                            'timestamp': datetime.now(),
                            'method': method,
                            'summary': summary_text,
                            'scores': scores,
                            'text_length': len(text),
                            'summary_length': len(summary_text)
                        })
                        
                        # Save to database if using sample document
                        if input_option == "Sample Document" and documents:
                            doc = doc_options[selected_doc]
                            st.session_state.database.save_summary(
                                doc['id'], method, summary_text, scores
                            )
                        
                        st.success("Summary generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
            else:
                st.warning("Please enter some text to summarize.")
    
    with tab2:
        st.header("Sample Documents")
        
        documents = st.session_state.database.get_documents()
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['title']} ({doc['category']})"):
                    st.write(doc['content'])
                    
                    # Quick summarize buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button(f"TF-IDF", key=f"tfidf_{doc['id']}"):
                            config.method = "tfidf"
                            summarizer = create_summarizer(config)
                            summary, scores = summarizer.summarize(doc['content'], "tfidf")
                            st.write("**TF-IDF Summary:**")
                            for i, sent in enumerate(summary, 1):
                                st.write(f"{i}. {sent}")
                    
                    with col2:
                        if st.button(f"TextRank", key=f"textrank_{doc['id']}"):
                            config.method = "textrank"
                            summarizer = create_summarizer(config)
                            summary, scores = summarizer.summarize(doc['content'], "textrank")
                            st.write("**TextRank Summary:**")
                            for i, sent in enumerate(summary, 1):
                                st.write(f"{i}. {sent}")
                    
                    with col3:
                        if st.button(f"BERT", key=f"bert_{doc['id']}"):
                            config.method = "bert"
                            summarizer = create_summarizer(config)
                            summary, scores = summarizer.summarize(doc['content'], "bert")
                            st.write("**BERT Summary:**")
                            for i, sent in enumerate(summary, 1):
                                st.write(f"{i}. {sent}")
                    
                    with col4:
                        if st.button(f"Sentence-BERT", key=f"sbert_{doc['id']}"):
                            config.method = "sentence_bert"
                            summarizer = create_summarizer(config)
                            summary, scores = summarizer.summarize(doc['content'], "sentence_bert")
                            st.write("**Sentence-BERT Summary:**")
                            for i, sent in enumerate(summary, 1):
                                st.write(f"{i}. {sent}")
        else:
            st.info("No sample documents available.")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if st.session_state.summaries_history:
            # Convert history to DataFrame
            df = pd.DataFrame(st.session_state.summaries_history)
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Summaries", len(df))
            
            with col2:
                avg_rouge1 = df['scores'].apply(lambda x: x['rouge1_fmeasure']).mean()
                st.metric("Avg ROUGE-1 F1", f"{avg_rouge1:.3f}")
            
            with col3:
                avg_rouge2 = df['scores'].apply(lambda x: x['rouge2_fmeasure']).mean()
                st.metric("Avg ROUGE-2 F1", f"{avg_rouge2:.3f}")
            
            with col4:
                avg_compression = (df['summary_length'] / df['text_length']).mean()
                st.metric("Avg Compression Ratio", f"{avg_compression:.2%}")
            
            # Method comparison chart
            scores_data = []
            for _, row in df.iterrows():
                scores_data.append({
                    'method': row['method'],
                    'rouge1_fmeasure': row['scores']['rouge1_fmeasure'],
                    'rouge2_fmeasure': row['scores']['rouge2_fmeasure'],
                    'rougeL_fmeasure': row['scores']['rougeL_fmeasure']
                })
            
            create_metrics_chart(scores_data)
            
            # Recent summaries table
            st.subheader("Recent Summaries")
            recent_df = df[['timestamp', 'method', 'summary_length', 'scores']].tail(10)
            recent_df['rouge1_f1'] = recent_df['scores'].apply(lambda x: x['rouge1_fmeasure'])
            recent_df['rouge2_f1'] = recent_df['scores'].apply(lambda x: x['rouge2_fmeasure'])
            
            display_df = recent_df[['timestamp', 'method', 'summary_length', 'rouge1_f1', 'rouge2_f1']]
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("No summaries generated yet. Try summarizing some text!")
    
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ## 🧠 Modern Text Summarization System
        
        This application demonstrates advanced extractive text summarization techniques using multiple state-of-the-art algorithms.
        
        ### 🔬 Methods Implemented
        
        1. **TF-IDF + Cosine Similarity**: Traditional approach using term frequency-inverse document frequency
        2. **TextRank**: Graph-based algorithm inspired by PageRank
        3. **BERT**: Bidirectional Encoder Representations from Transformers
        4. **Sentence-BERT**: Optimized sentence embeddings for semantic similarity
        
        ### 📊 Evaluation Metrics
        
        - **ROUGE-1**: Measures overlap of unigrams between summary and reference
        - **ROUGE-2**: Measures overlap of bigrams between summary and reference  
        - **ROUGE-L**: Measures longest common subsequence between summary and reference
        
        ### 🛠️ Technologies Used
        
        - **Python**: Core programming language
        - **Streamlit**: Web interface framework
        - **Transformers**: Hugging Face transformers library
        - **scikit-learn**: Machine learning utilities
        - **NLTK**: Natural language processing
        - **spaCy**: Advanced NLP processing
        - **SQLite**: Local database for sample documents
        
        ### 🚀 Features
        
        - Multiple summarization algorithms
        - Real-time evaluation metrics
        - Interactive web interface
        - Sample document database
        - Analytics dashboard
        - Configurable parameters
        
        ### 📈 Performance Notes
        
        - **TF-IDF**: Fast, good for general text
        - **TextRank**: Better for longer documents
        - **BERT**: High quality but slower
        - **Sentence-BERT**: Good balance of speed and quality
        
        ### 🔧 Installation
        
        ```bash
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
        ```
        
        ### 🎯 Usage
        
        1. Select your preferred summarization method
        2. Adjust parameters in the sidebar
        3. Enter text or choose a sample document
        4. Click "Generate Summary" to see results
        5. View analytics in the dashboard tab
        
        ### 📝 License
        
        This project is open source and available under the MIT License.
        """)

if __name__ == "__main__":
    main()
