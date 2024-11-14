import streamlit as st
from io import StringIO
from pathlib import Path
import tempfile
from app import DocumentProcessor  # Your existing code in a separate file

def create_temp_file(uploaded_file):
    """Create a temporary file from uploaded content"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'results' not in st.session_state:
        st.session_state.results = None

def display_results(results):
    """Display processing results in a structured format"""
    if results:
        with st.expander("Summary", expanded=True):
            st.markdown(results["summary"])
        
        with st.expander("Extracted Medical Entities"):
            st.markdown(results["entities"])
        
        with st.expander("Enhanced Insights"):
            st.markdown(results["enhanced_insights"])
        
        with st.expander("Final Assessment"):
            st.markdown(results["final_assessment"])

def display_step_result(step_name: str, result: str):
    """Display individual step result as it completes"""
    with st.expander(f"{step_name} Result", expanded=True):
        st.markdown(result)

def update_status(placeholder, completed_steps, current_step=None):
    """Update status display with all completed steps and current processing step"""
    status_text = "Processing Status:\n"
    for step, done in completed_steps.items():
        if done:
            status_text += f"✅ {step} Complete\n"
        elif step == current_step:
            status_text += f"⌛ {step} In Progress...\n"
        else:
            status_text += f"⏳ {step} Pending\n"
    placeholder.text(status_text)

def process_with_status(processor, input_source, status_placeholder):
    """Process input with real-time status updates and incremental results"""
    try:
        # Initialize status tracking
        completed_steps = {
            "Summarization": False,
            "Entity Extraction": False,
            "RAG Analysis": False,
            "Final Assessment": False
        }
        update_status(status_placeholder, completed_steps)
        
        # Load and extract text
        text = processor.document_loader.load(input_source)
        
        if not text.strip():
            raise ValueError("Extracted text is empty")
        
        # Initialize results container
        results = {}
        
        # Summarization
        update_status(status_placeholder, completed_steps, "Summarization")
        results["summary"] = processor.summarize_text(text)
        completed_steps["Summarization"] = True
        update_status(status_placeholder, completed_steps)
        display_step_result("Summary", results["summary"])
        
        # Entity Extraction
        update_status(status_placeholder, completed_steps, "Entity Extraction")
        results["entities"] = processor.extract_entities(text)
        completed_steps["Entity Extraction"] = True
        update_status(status_placeholder, completed_steps)
        display_step_result("Extracted Entities", results["entities"])
        
        # RAG Analysis
        update_status(status_placeholder, completed_steps, "RAG Analysis")
        results["enhanced_insights"] = processor.enhance_with_rag(
            text, 
            results["summary"], 
            results["entities"]
        )
        completed_steps["RAG Analysis"] = True
        update_status(status_placeholder, completed_steps)
        display_step_result("Enhanced Insights", results["enhanced_insights"])
        
        # Final Assessment
        update_status(status_placeholder, completed_steps, "Final Assessment")
        results["final_assessment"] = processor.judge_results(
            results["summary"],
            results["entities"],
            results["enhanced_insights"]
        )
        completed_steps["Final Assessment"] = True
        update_status(status_placeholder, completed_steps)
        display_step_result("Final Assessment", results["final_assessment"])
        
        return results
        
    except Exception as e:
        status_placeholder.text("Status: Error ❌")
        raise e

def main():
    st.set_page_config(
        page_title="Medical Document Analyzer",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical Document Analyzer")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for model information and status
    with st.sidebar:
        st.header("Model Information")
        st.write("Currently using:")
        st.write("- Summarization: LLaMA 2")
        st.write("- Entity Extraction: Mistral")
        st.write("- RAG Analysis: Neural-Chat")
        st.write("- Final Assessment: OpenChat")
        
        st.markdown("---")
        st.header("Processing Status")
        status_placeholder = st.empty()
    
    # Main content area
    st.write("Upload a medical document (PDF or TXT) or enter text directly for analysis.")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Enter Text"]
    )
    
    try:
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload a medical document",
                type=["pdf", "txt"],
                help="Upload a PDF or text file containing medical information"
            )
            
            if uploaded_file:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        # Create temporary file
                        temp_file_path = create_temp_file(uploaded_file)
                        
                        try:
                            # Process with status updates
                            results = process_with_status(
                                st.session_state.processor,
                                temp_file_path,
                                status_placeholder
                            )
                            st.session_state.results = results
                            
                        finally:
                            # Clean up temporary file
                            Path(temp_file_path).unlink()
        
        else:  # Enter Text
            text_input = st.text_area(
                "Enter medical text",
                height=200,
                help="Paste or type medical text for analysis"
            )
            
            if text_input and st.button("Process Text"):
                with st.spinner("Processing text..."):
                    # Create StringIO object with the input text
                    text_io = StringIO(text_input)
                    
                    try:
                        # Process with status updates
                        results = process_with_status(
                            st.session_state.processor,
                            text_io,
                            status_placeholder
                        )
                        st.session_state.results = results
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
        
        # Add download button if results exist
        if st.session_state.results:
            if st.button("Download Complete Results"):
                results_text = f"""Medical Document Analysis Results

SUMMARY:
{st.session_state.results['summary']}

EXTRACTED ENTITIES:
{st.session_state.results['entities']}

ENHANCED INSIGHTS:
{st.session_state.results['enhanced_insights']}

FINAL ASSESSMENT:
{st.session_state.results['final_assessment']}
"""
                st.download_button(
                    label="Download Results as Text",
                    data=results_text,
                    file_name="medical_analysis_results.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        status_placeholder.write("Status: Error ❌")
        
    # Footer
    st.markdown("---")
    st.markdown(
        "This tool uses multiple LLMs to analyze medical documents. "
        "All processing is done locally using Ollama models."
    )

if __name__ == "__main__":
    main()