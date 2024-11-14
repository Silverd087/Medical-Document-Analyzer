import streamlit as st
from io import StringIO
from pathlib import Path
import tempfile
from app import DocumentProcessor  # Your existing code in a separate file
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import datetime

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

def format_markdown_text(text: str) -> str:
    """Convert markdown text to formatted text"""
    # Replace markdown list items with proper bullet points
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Handle bullet points
        if line.strip().startswith('*') or line.strip().startswith('-'):
            line = '• ' + line.strip()[1:].strip()
        
        # Handle headers
        if line.strip().startswith('#'):
            line = line.strip().lstrip('#').strip()
            
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def create_pdf_report(results: dict) -> bytes:
    """Create a well-formatted PDF report of the results"""
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2c3e50')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#34495e')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        leading=14,
        bulletIndent=20,
        leftIndent=20
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=body_style,
        leftIndent=30,
        firstLineIndent=0,
        spaceBefore=5,
        spaceAfter=5
    )
    
    # Create the document content
    content = []
    
    # Title
    content.append(Paragraph("Medical Document Analysis Report", title_style))
    content.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    content.append(Spacer(1, 20))
    
    # Summary Section
    content.append(Paragraph("Summary", heading_style))
    formatted_summary = format_markdown_text(results["summary"])
    paragraphs = formatted_summary.split('\n')
    for p in paragraphs:
        if p.strip().startswith('•'):
            content.append(Paragraph(p, bullet_style))
        else:
            content.append(Paragraph(p, body_style))
    content.append(Spacer(1, 20))
    
    # Entities Section
    content.append(Paragraph("Extracted Medical Entities", heading_style))
    formatted_entities = format_markdown_text(results["entities"])
    paragraphs = formatted_entities.split('\n')
    for p in paragraphs:
        if p.strip().startswith('•'):
            content.append(Paragraph(p, bullet_style))
        else:
            content.append(Paragraph(p, body_style))
    content.append(Spacer(1, 20))
    
    # Enhanced Insights Section
    content.append(Paragraph("Enhanced Insights", heading_style))
    formatted_insights = format_markdown_text(results["enhanced_insights"])
    paragraphs = formatted_insights.split('\n')
    for p in paragraphs:
        if p.strip().startswith('•'):
            content.append(Paragraph(p, bullet_style))
        else:
            content.append(Paragraph(p, body_style))
    content.append(Spacer(1, 20))
    
    # Final Assessment Section
    content.append(Paragraph("Final Assessment", heading_style))
    formatted_assessment = format_markdown_text(results["final_assessment"])
    paragraphs = formatted_assessment.split('\n')
    for p in paragraphs:
        if p.strip().startswith('•'):
            content.append(Paragraph(p, bullet_style))
        else:
            content.append(Paragraph(p, body_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the value of the BytesIO buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

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
        
        # First display the results
        if st.session_state.results:
            st.header("Analysis Results")
            display_results(st.session_state.results)
            
            # Then add download buttons below the results
            st.markdown("---")
            st.subheader("Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # PDF Download
                pdf_bytes = create_pdf_report(st.session_state.results)
                st.download_button(
                    label="📄 Download as PDF",
                    data=pdf_bytes,
                    file_name="medical_analysis_report.pdf",
                    mime="application/pdf",
                    key="pdf_download"  # Add unique key
                )
            
            with col2:
                # Text Download
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
                    label="📝 Download as Text",
                    data=results_text,
                    file_name="medical_analysis_results.txt",
                    mime="text/plain",
                    key="text_download"  # Add unique key
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