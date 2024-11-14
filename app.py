from pathlib import Path
import PyPDF2
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gc
import psutil
import time
from contextlib import contextmanager
import os
from typing import Union, TextIO
import fitz  # PyMuPDF for better PDF handling
from io import StringIO
import shutil

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.base_memory = self.get_memory_usage()
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def is_memory_clear(self, threshold_mb=500):
        """Check if memory has returned close to baseline"""
        current_memory = self.get_memory_usage()
        memory_diff = current_memory - self.base_memory
        return memory_diff < threshold_mb
    
    def wait_for_memory_clear(self, max_attempts=10):
        """Wait for memory to clear before proceeding"""
        attempts = 0
        while not self.is_memory_clear() and attempts < max_attempts:
            gc.collect()
            time.sleep(1)  # Give OS time to free memory
            attempts += 1
            
        if not self.is_memory_clear():
            raise MemoryError("Failed to clear memory after unloading model")

    @contextmanager
    def load_model(self, model_name: str):
        """Safely load a model, ensuring previous model is unloaded"""
        if self.current_model is not None:
            raise RuntimeError("Another model is still loaded! Must unload first.")
        
        try:
            print(f"Loading model: {model_name}")
            print(f"Current memory usage: {self.get_memory_usage():.2f} MB")
            
            model = Ollama(model=model_name)
            self.current_model = model_name
            yield model
            
        finally:
            print(f"Unloading model: {model_name}")
            del model
            self.current_model = None
            gc.collect()
            
            self.wait_for_memory_clear()
            print(f"Memory usage after cleanup: {self.get_memory_usage():.2f} MB")

class DocumentLoader:
    """Handles document loading and text extraction"""
    
    @staticmethod
    def is_pdf(file_path: Union[str, Path]) -> bool:
        """Check if file is PDF"""
        return str(file_path).lower().endswith('.pdf')
    
    @staticmethod
    def extract_from_pdf(file_path: Union[str, Path]) -> str:
        """Extract text from PDF using PyMuPDF (more reliable than PyPDF2)"""
        try:
            text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error with PyMuPDF: {e}")
            # Fallback to PyPDF2 if PyMuPDF fails
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e:
                raise Exception(f"Failed to extract PDF text: {e}")

    @staticmethod
    def extract_from_text(file_path: Union[str, Path]) -> str:
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings if utf-8 fails
            encodings = ['latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise Exception("Failed to decode text file with multiple encodings")

    @classmethod
    def load(cls, input_source: Union[str, Path, TextIO]) -> str:
        """Main method to load text from various sources"""
        if isinstance(input_source, (str, Path)):
            path = Path(input_source)
            
            # Check if file exists
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            # Check file size
            file_size = path.stat().st_size
            max_size = 10 * 1024 * 1024  # 10MB limit
            if file_size > max_size:
                raise ValueError(f"File too large: {file_size/1024/1024:.2f}MB (max {max_size/1024/1024}MB)")
            
            # Process based on file type
            if cls.is_pdf(path):
                return cls.extract_from_pdf(path)
            else:
                return cls.extract_from_text(path)
        
        elif isinstance(input_source, StringIO):
            return input_source.read()
        
        elif isinstance(input_source, TextIO):
            return input_source.read()
        
        else:
            raise ValueError(f"Input must be a file path or text stream, got {type(input_source)}")

class DocumentProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self.model_names = {
            "summarizer": "llama2",
            "ner": "mistral",
            "rag": "neural-chat",
            "judge": "openchat",
            "embeddings": "mistral"
        }
        self.document_loader = DocumentLoader()

    def summarize_text(self, text: str) -> str:
        """Generate a medical-focused summary using Llama 2"""
        with self.model_manager.load_model(self.model_names["summarizer"]) as summarizer:
            prompt = f"""As a medical professional, provide a comprehensive summary of the following text. 
            
            Focus on these key aspects:
            1. Primary medical conditions or symptoms
            2. Key diagnostic findings
            3. Treatment plans or recommendations
            4. Critical lab results or test values
            5. Relevant patient history
            6. Important medical observations
            7. Follow-up recommendations
            
            Guidelines:
            - Maintain medical accuracy and terminology
            - Preserve numerical values and measurements
            - Highlight abnormal findings
            - Include temporal relationships of symptoms/treatments
            - Maintain patient-critical information
            - Structure the summary in clear sections
            
            Text to summarize:
            {text}
            
            Please provide the summary in the following format:
            
            CHIEF CONCERNS:
            [Key symptoms or main medical issues]
            
            SIGNIFICANT FINDINGS:
            [Important diagnostic results, observations, and measurements]
            
            ASSESSMENTS/DIAGNOSES:
            [Medical conclusions and confirmed conditions]
            
            TREATMENT/RECOMMENDATIONS:
            [Prescribed treatments, medications, and follow-up plans]
            
            ADDITIONAL NOTES:
            [Any other clinically relevant information]"""
            
            try:
                summary = summarizer.invoke(prompt)
                
                # Verify summary quality
                verification_prompt = f"""Review the generated summary and verify:
                1. Are all critical medical details preserved?
                2. Is the medical terminology accurate?
                3. Are all numerical values correctly maintained?
                
                If any issues are found, regenerate the summary addressing these issues.
                
                Original Summary:
                {summary}
                
                Verification Result:"""
                
                verification = summarizer.invoke(verification_prompt)
                
                # If verification indicates issues, regenerate
                if "issue" in verification.lower() or "error" in verification.lower():
                    summary = summarizer.invoke(prompt)
                
                return summary
                
            except Exception as e:
                print(f"Error during summarization: {str(e)}")
                # Fallback to a simpler prompt if the detailed one fails
                fallback_prompt = f"""Provide a clear medical summary of the following text, 
                focusing on key symptoms, diagnoses, and treatments:
                
                {text}"""
                return summarizer.invoke(fallback_prompt)

    def extract_entities(self, text: str) -> str:
        """Extract named entities, especially focusing on symptoms"""
        with self.model_manager.load_model(self.model_names["ner"]) as ner_model:
            prompt = f"""Extract key medical entities from the text, focusing on:
            - Symptoms
            - Conditions
            - Treatments
            - Medications
            
            Text: {text}"""
            return ner_model.invoke(prompt)

    def enhance_with_rag(self, text: str, summary: str, entities: str) -> str:
        """Use RAG to enhance the analysis"""
        import uuid
        import time
        
        # Create a unique directory for this session
        session_id = str(uuid.uuid4())
        persist_directory = f"chroma_db_{session_id}"
        
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)
            
            # First load embeddings model and create embeddings
            embeddings = OllamaEmbeddings(model=self.model_names["embeddings"])
            
            # Create vector store with persistence
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(text)[:40]  # Limit chunks
            
            # Create and use vectorstore
            vectorstore = None
            try:
                vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name=f"medical_docs_{session_id}"
                )
                
                # Then load RAG model separately
                with self.model_manager.load_model(self.model_names["rag"]) as rag_model:
                    retriever = vectorstore.as_retriever()
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=rag_model,
                        chain_type="stuff",
                        retriever=retriever
                    )
                    
                    prompt = f"""Based on the following information:
                    Summary: {summary}
                    Extracted Entities: {entities}
                    
                    Please provide additional insights and context."""
                    
                    result = qa_chain.run(prompt)
                    return result
                    
            finally:
                # Properly close and cleanup vectorstore
                if vectorstore is not None:
                    try:
                        vectorstore.delete_collection()
                        del vectorstore
                    except Exception as e:
                        print(f"Error cleaning up vectorstore: {e}")
                
                # Force garbage collection
                gc.collect()
                
                # Give the system a moment to release file handles
                time.sleep(1)
                
        finally:
            # Clean up the directory with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(persist_directory):
                        # On Windows, sometimes we need to retry due to file handles
                        shutil.rmtree(persist_directory, ignore_errors=True)
                        time.sleep(1)  # Give OS time to release handles
                    break
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"Warning: Could not remove temporary directory {persist_directory}: {e}")
                    time.sleep(1)  # Wait before retry

    def judge_results(self, summary: str, entities: str, rag_insights: str) -> str:
        """Use judge model to evaluate and combine all results"""
        with self.model_manager.load_model(self.model_names["judge"]) as judge_model:
            prompt = f"""Review and evaluate the following analysis components:
            Summary: {summary}
            Extracted Entities: {entities}
            Additional Insights: {rag_insights}
            
            Please provide:
            1. An evaluation of the consistency
            2. Any contradictions or concerns
            3. A final, consolidated assessment
            """
            return judge_model.invoke(prompt)

    def process_input(self, input_source: Union[str, Path, TextIO]) -> dict:
        """Process either a file path or text content"""
        try:
            # Load and extract text
            text = self.document_loader.load(input_source)
            
            if not text.strip():
                raise ValueError("Extracted text is empty")
            
            # Process the text through the pipeline
            results = {}
            
            print("Starting summarization...")
            results["summary"] = self.summarize_text(text)
            
            print("Starting entity extraction...")
            results["entities"] = self.extract_entities(text)
            
            print("Starting RAG analysis...")
            results["enhanced_insights"] = self.enhance_with_rag(
                text, 
                results["summary"], 
                results["entities"]
            )
            
            print("Starting final assessment...")
            results["final_assessment"] = self.judge_results(
                results["summary"],
                results["entities"],
                results["enhanced_insights"]
            )
            
            return results
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise

def main():
    processor = DocumentProcessor()
    
    # Example with different input types
    try:
        # Process a PDF file
        pdf_results = processor.process_input("path/to/document.pdf")
        print("\n=== PDF Results ===")
        print_results(pdf_results)
        
        # Process a text file
        text_results = processor.process_input("path/to/document.txt")
        print("\n=== Text File Results ===")
        print_results(text_results)
        
        # Process direct text input
        from io import StringIO
        text_content = StringIO("Your direct text content here...")
        direct_results = processor.process_input(text_content)
        print("\n=== Direct Text Results ===")
        print_results(direct_results)
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
    finally:
        gc.collect()

def print_results(results: dict):
    print("Summary:", results["summary"])
    print("\nExtracted Entities:", results["entities"])
    print("\nEnhanced Insights:", results["enhanced_insights"])
    print("\nFinal Assessment:", results["final_assessment"])
