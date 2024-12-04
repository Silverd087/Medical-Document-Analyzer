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
import hashlib
import datetime

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
        # Initialize model names first
        self.model_names = {
            "summarizer": "llama2",
            "ner": "mistral",
            "rag": "neural-chat",
            "judge": "openchat",
            "embeddings": "mistral",
            "chat": "llama3.2:3b"
        }
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize document loader
        self.document_loader = DocumentLoader()
        
        # Add persistent vector store path
        self.persist_directory = "persistent_vectorstore"
        self.collection_name = "medical_documents"
        
        # Initialize vector store
        self._initialize_vectorstore()
        
        # Add system prompt for medical chat
        self.system_prompt = """
        You are MedAssist-GPT, an AI medical assistant focused on providing general medical information and guidance. You excel in:

        Specializations:
        - General medical knowledge and terminology
        - Basic symptom assessment
        - Health education and prevention
        - Medical document analysis
        - Understanding medical research

        Guidelines:
        - Always provide clear, accurate medical information
        - Include appropriate disclaimers when necessary
        - Emphasize the importance of consulting healthcare professionals
        - Be clear about limitations and uncertainties
        - Use medical terminology appropriately, with explanations
        - For document analysis, provide detailed, structured analysis

        Important Disclaimers:
        - You cannot provide definitive diagnoses
        - You cannot prescribe medications
        - You cannot replace professional medical advice
        - You must emphasize seeking professional medical care for serious concerns

        Remember: Your role is to provide information and guidance while emphasizing the importance of professional medical care.
        """

    def _initialize_vectorstore(self):
        """Initialize or load the persistent vector store"""
        print("\n=== Initializing Vector Store ===")
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            print(f"Created persistent directory: {self.persist_directory}")
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=self.model_names["embeddings"])
        
        # Create or load vector store
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"Vector store loaded with {self.vectorstore._collection.count()} documents")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vectorstore = None

    def _get_document_hash(self, text: str) -> str:
        """Create a hash of the document text to check for duplicates"""
        return hashlib.md5(text.encode()).hexdigest()

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
        
        return summarizer.invoke(prompt)


    def extract_entities(self, text: str) -> str:
        """Extract named entities, especially focusing on symptoms"""
        with self.model_manager.load_model(self.model_names["ner"]) as ner_model:
            prompt = f"""
            You are a medical report analyzer specialized in extracting structured medical information. 
            From the provided text, extract and list key entities under the following categories:
            - Symptoms: Briefly describe any symptoms mentioned.
            - Conditions: List specific medical conditions or diagnoses.
            - Treatments: Include any mentioned procedures, therapies, or interventions.
            - Medications: Extract names of drugs, dosages, or prescriptions.
            Ensure your response is concise, clearly categorized, and directly answers the query.
            
            Text to analyze:
            {text}
            
            Output format:
            Symptoms: <list of symptoms>/n
            Conditions: <list of conditions>/n
            Treatments: <list of treatments>/n
            Medications: <list of medications>/n
            """
            return ner_model.invoke(prompt)

    def enhance_with_rag(self, text: str, summary: str, entities: str, is_query: bool = False) -> str:
        """Use RAG to enhance the analysis or answer queries"""
        print("\n=== Starting RAG Process ===")
        
        try:
            if not is_query:
                # Regular document processing
                doc_hash = self._get_document_hash(text)
                metadata = {
                    "hash": doc_hash,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "summary": summary,
                    "entities": entities
                }
                
                # Check if document already exists
                existing_docs = self.vectorstore._collection.get(
                    where={"hash": doc_hash}
                )
                
                if not existing_docs['ids']:
                    print("\nNew document detected, adding to vector store...")
                    # Modified text splitter configuration
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,  # Increased chunk size
                        chunk_overlap=200,  # Increased overlap
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""],
                        is_separator_regex=False
                    )
                    
                    # Clean and preprocess text
                    cleaned_text = self._preprocess_text(text)
                    
                    # Split text and validate chunks
                    chunks = text_splitter.split_text(cleaned_text)
                    valid_chunks = self._validate_chunks(chunks)
                    
                    if valid_chunks:
                        print(f"Split text into {len(valid_chunks)} valid chunks")
                        # Add chunks with individual metadata
                        self.vectorstore.add_texts(
                            texts=valid_chunks,
                            metadatas=[{
                                **metadata,
                                "chunk_index": i,
                                "total_chunks": len(valid_chunks)
                            } for i in range(len(valid_chunks))]
                        )
                        print("Document added to vector store")
                    else:
                        print("Warning: No valid chunks generated")
                else:
                    print("\nDocument already exists in vector store")
            
            # Get relevant chunks for either document analysis or query
            print(f"\nSearching vector store with {self.vectorstore._collection.count()} documents")
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )
            
            query = text if is_query else f"""Based on the following information:
            Summary: {summary}
            Extracted Entities: {entities}"""
            
            relevant_docs = retriever.get_relevant_documents(query)
            
            if len(relevant_docs) > 0:
                print("\nFound relevant information in database:")
                for i, doc in enumerate(relevant_docs, 1):
                    print(f"\nChunk {i}:")
                    print("-" * 50)
                    print(doc.page_content.strip())
                    print("-" * 50)
            else:
                print("\nNo relevant information found in database")
            
            # Perform RAG analysis
            print("\nStarting RAG analysis...")
            with self.model_manager.load_model(self.model_names["rag"]) as rag_model:
                print("RAG model loaded")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=rag_model,
                    chain_type="stuff",
                    retriever=retriever
                )
                
                if is_query:
                    prompt = f"""Using the retrieved medical documents as context, please answer the following question:
                    {text}
                    
                    If the database doesn't contain relevant information, please indicate that and provide a general response based on your knowledge."""
                else:
                    prompt = f"""Based on the following information:
                    Summary: {summary}
                    Extracted Entities: {entities}
                    
                    Please provide additional insights and context."""
                
                print("\nExecuting RAG query...")
                result = qa_chain.run(prompt)
                print("RAG analysis completed successfully")
                return result
                
        except Exception as e:
            print(f"Error in RAG process: {e}")
            raise e

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
        """Process document using full pipeline"""
        try:
            # Load and extract text
            text = self.document_loader.load(input_source)
            
            if not text.strip():
                raise ValueError("Extracted text is empty")
            
            # Process the text through the pipeline
            results = {}
            
            # Step 1: Summarization
            print("Starting summarization...")
            results["summary"] = self.summarize_text(text)
            
            # Step 2: Entity extraction
            print("Starting entity extraction...")
            results["entities"] = self.extract_entities(text)
            
            # Step 3: RAG analysis
            print("Starting RAG analysis...")
            results["enhanced_insights"] = self.enhance_with_rag(
                text, 
                results["summary"], 
                results["entities"]
            )
            
            # Step 4: Final assessment (should be last)
            print("Starting final assessment...")
            results["final_assessment"] = self.judge_results(
                results["summary"],
                results["entities"],
                results["enhanced_insights"]
            )
            
            # Return final results
            return {
                "type": "document",
                "content": results
            }
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise

    def clear_vectorstore(self):
        """Clear the entire vector store - use with caution"""
        try:
            if self.vectorstore is not None:
                self.vectorstore._collection.delete(where={})
                print("Vector store cleared successfully")
        except Exception as e:
            print(f"Error clearing vector store: {e}")

    def query_database(self, query: str) -> str:
        """Query the medical document database directly"""
        print(f"\n=== Processing Query: {query} ===")
        return self.enhance_with_rag(query, "", "", is_query=True)

    def chat(self, message: str, chat_history: list = None) -> dict:
        """Handle chat interactions"""
        if chat_history is None:
            chat_history = []
            
        # Regular chat interaction - using only llama3.2:3b
        print("\n=== Processing Chat Message ===")
        with self.model_manager.load_model(self.model_names["chat"]) as chat_model:
            prompt = f"""You are MedAssist-GPT, an AI medical assistant. Please provide a helpful response to the following message, while keeping in mind:
            1. You cannot provide diagnoses
            2. You cannot prescribe medications
            3. You must emphasize consulting healthcare professionals for serious concerns
            4. Be clear about any limitations or uncertainties

            Previous conversation:
            {' '.join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]])}

            User: {message}

            Assistant:"""
            
            response = chat_model.invoke(prompt)
            return {
                "type": "chat",
                "content": response
            }

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text before chunking"""
        # Only remove non-printable characters while preserving spaces and punctuation
        cleaned_text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)
        
        # Normalize line endings (but keep the newlines)
        cleaned_text = cleaned_text.replace('\r\n', '\n')
        
        # Remove any resulting double spaces (from non-printable char replacement)
        cleaned_text = ' '.join(filter(None, cleaned_text.split(' ')))
        
        return cleaned_text.strip()

    def _validate_chunks(self, chunks: list) -> list:
        """Validate and filter chunks"""
        valid_chunks = []
        min_chunk_length = 100  # Minimum meaningful chunk size
        
        for chunk in chunks:
            # Clean the chunk
            cleaned_chunk = chunk.strip()
            
            # Apply validation criteria
            if (len(cleaned_chunk) >= min_chunk_length and  # Length check
                any(char.isalpha() for char in cleaned_chunk) and  # Contains letters
                not cleaned_chunk.isspace() and  # Not just whitespace
                len(cleaned_chunk.split()) > 5):  # Minimum word count
                
                valid_chunks.append(cleaned_chunk)
        
        return valid_chunks

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
