"""
Agentic RAG Engine for Transformer Monitoring
Integrates with Ollama (phi3, mistral, llama3) for intelligent Q&A
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# LangChain imports - optional
LANGCHAIN_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, DirectoryLoader
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create placeholder classes for when LangChain is not available
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class HuggingFaceEmbeddings:
        """Placeholder for HuggingFaceEmbeddings when LangChain is not available"""
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs=None):
            self.model_name = model_name
            self.model_kwargs = model_kwargs or {}
        
        def embed_query(self, text: str) -> list:
            return [0.0] * 384  # Return dummy embeddings
        
        def embed_documents(self, texts: list) -> list:
            return [[0.0] * 384 for _ in texts]
    
    class TextLoader:
        """Placeholder for TextLoader when LangChain is not available"""
        def __init__(self, file_path: str, encoding: str = 'utf-8'):
            self.file_path = file_path
            self.encoding = encoding
        
        def load(self) -> list:
            try:
                with open(self.file_path, 'r', encoding=self.encoding) as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": self.file_path})]
            except Exception as e:
                print(f"Error reading file {self.file_path}: {e}")
                return []
    
    class CSVLoader:
        """Placeholder for CSVLoader when LangChain is not available"""
        def __init__(self, file_path: str, encoding: str = 'utf-8'):
            self.file_path = file_path
            self.encoding = encoding
        
        def load(self) -> list:
            import pandas as pd
            try:
                df = pd.read_csv(self.file_path)
                content = f"CSV Data from {self.file_path}:\n{df.to_string()}"
                return [Document(
                    page_content=content,
                    metadata={"source": self.file_path, "type": "csv"}
                )]
            except Exception as e:
                print(f"Error reading CSV {self.file_path}: {e}")
                return []
    
    class PyPDFLoader:
        """Placeholder for PyPDFLoader when LangChain is not available"""
        def __init__(self, file_path: str):
            self.file_path = file_path
        
        def load(self) -> list:
            print(f"PDF loading not available: {self.file_path}. Install pypdf to enable PDF support.")
            return [Document(
                page_content=f"[PDF file: {self.file_path}]",
                metadata={"source": self.file_path, "type": "pdf"}
            )]
    
    class DirectoryLoader:
        """Placeholder for DirectoryLoader when LangChain is not available"""
        def __init__(self, path: str, glob: str = "**/*", loader_cls=None):
            self.path = Path(path)
            self.glob = glob
            self.loader_cls = loader_cls or TextLoader
        
        def load(self) -> list:
            documents = []
            for file_path in self.path.glob(self.glob):
                if file_path.is_file():
                    loader = self.loader_cls(str(file_path))
                    documents.extend(loader.load())
            return documents
    
    class RecursiveCharacterTextSplitter:
        """Placeholder for RecursiveCharacterTextSplitter when LangChain is not available"""
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_documents(self, documents: list) -> list:
            if not documents:
                return []
            
            result = []
            for doc in documents:
                content = doc.page_content
                # Simple text splitting
                chunks = []
                for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                    chunk = content[i:i + self.chunk_size]
                    chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()
                    ))
                result.extend(chunks)
            
            return result
    
    class Chroma:
        """Placeholder for Chroma vector store when LangChain is not available"""
        def __init__(self, documents=None, embedding=None, persist_directory=None):
            self.documents = documents or []
            self.embedding = embedding
            self.persist_directory = persist_directory
            print(f"Chroma vector store initialized (persistence disabled without LangChain)")
        
        @classmethod
        def from_documents(cls, documents: list, embedding, persist_directory: str) -> 'Chroma':
            instance = cls(documents=documents, embedding=embedding, persist_directory=persist_directory)
            return instance
        
        def similarity_search(self, query: str, k: int = 5) -> list:
            # Return documents without actual similarity search
            return self.documents[:k]

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data_source"
TRANSFORMERS_DIR = DATA_DIR / "transformers"
CHROMA_DIR = BASE_DIR / "chroma_db"


class OllamaClient:
    """Client for Ollama API"""
    
    def __init__(self, base_url: str = None):
        # Support environment variable for Ollama URL
        if base_url is None:
            base_url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.base_url = base_url
        # Support environment variable for timeout
        self.timeout = int(os.environ.get('OLLAMA_TIMEOUT', '300'))
        self.available_models = ["phi", "phi3", "mistral", "llama3"]
    
    def generate(self, prompt: str, model: str = "phi3", 
                 system_prompt: str = None, temperature: float = 0.7) -> str:
        """Generate response from Ollama"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 512,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running on " \
                   f"{self.base_url}. You can download Ollama from https://ollama.com"
        except requests.exceptions.Timeout:
            return f"Error: Ollama request timed out after {self.timeout} seconds. " \
                   "Try using a smaller model or reducing the prompt complexity."
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, messages: List[Dict], model: str = "phi3") -> str:
        """Chat with Ollama using message history"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running. " \
                   f"Expected at: {self.base_url}"
        except requests.exceptions.Timeout:
            return f"Error: Chat request timed out after {self.timeout} seconds. " \
                   "Try a simpler question or smaller model."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            # Use a shorter timeout for availability check
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "").split(":")[0] for m in models]
        except:
            pass
        return self.available_models


class GroqClient:
    """Client for Groq API - Cloud-based high-performance inference"""
    
    def __init__(self, api_key: str = None):
        # Support environment variable for API key
        if api_key is None:
            api_key = os.environ.get('GROQ_API_KEY', '')
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.timeout = int(os.environ.get('OLLAMA_TIMEOUT', '300'))
        
        # Groq available models (high-performance inference)
        self.available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        # Model categorization
        self.ollama_models = ["phi", "phi3", "mistral", "llama3"]
        self.groq_models = self.available_models
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def is_available(self) -> bool:
        """Check if Groq API is available and configured"""
        if not self.api_key:
            return False
        try:
            # Quick check by making a simple request
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available Groq models"""
        if self.is_available():
            try:
                response = requests.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    # Filter to known Groq models
                    return [m for m in models if any(gm in m for gm in ["llama", "mixtral", "gemma"])]
            except:
                pass
        return self.available_models
    
    def chat(self, messages: List[Dict], model: str = "llama-3.3-70b-versatile") -> str:
        """Chat with Groq using message history (OpenAI-compatible API)"""
        if not self.api_key:
            return "Error: Groq API key not configured. Please set GROQ_API_KEY in your .env file."
        
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Groq API. Please check your internet connection."
        except requests.exceptions.AuthenticationError:
            return "Error: Groq API authentication failed. Please check your API key."
        except requests.exceptions.Timeout:
            return f"Error: Groq request timed out after {self.timeout} seconds."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate(self, prompt: str, model: str = "llama-3.3-70b-versatile",
                 system_prompt: str = None, temperature: float = 0.7) -> str:
        """Generate response from Groq using a simple prompt"""
        # Build messages list
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return self.chat(messages, model)


def get_model_provider(model: str) -> str:
    """Determine if a model is from Ollama or Groq"""
    ollama_models = ["phi", "phi3", "mistral", "llama3"]
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    
    if model in ollama_models:
        return "ollama"
    elif model in groq_models:
        return "groq"
    else:
        # Default to groq for unknown models (Groq has more models)
        return "groq"


class TransformerRAG:
    """RAG system for transformer-specific document retrieval"""
    
    def __init__(self, transformer_id: str = None):
        self.transformer_id = transformer_id
        self.ollama = OllamaClient()
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        
        # Initialize embeddings if LangChain is available
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e:
                print(f"Warning: Could not load embeddings: {e}")
        else:
            print("Info: LangChain not available. Using placeholder embeddings (vector search disabled).")
            self.embeddings = HuggingFaceEmbeddings()
    
    def load_transformer_documents(self, transformer_id: str) -> List[Document]:
        """Load all documents for a specific transformer"""
        self.transformer_id = transformer_id
        documents = []
        
        # Transformer-specific folder
        transformer_dir = TRANSFORMERS_DIR / transformer_id.replace("-", "-").upper()
        
        if transformer_dir.exists():
            documents.extend(self._load_directory(transformer_dir))
        
        # Also check common reports
        reports_dir = DATA_DIR / "reports"
        if reports_dir.exists():
            for file in reports_dir.glob(f"*{transformer_id.lower().replace('-', '')}*"):
                documents.extend(self._load_file(file))
        
        # Load PDFs
        pdfs_dir = DATA_DIR / "pdfs"
        if pdfs_dir.exists():
            for file in pdfs_dir.glob(f"*{transformer_id.upper()}*"):
                documents.extend(self._load_file(file))
        
        # Load scenarios
        scenarios_file = DATA_DIR / "scenarios" / "transformers_scenarios.json"
        if scenarios_file.exists():
            documents.extend(self._load_scenarios(scenarios_file, transformer_id))
        
        self.documents = documents
        return documents
    
    def _load_directory(self, directory: Path) -> List[Document]:
        """Load all files from a directory"""
        documents = []
        for file_path in directory.iterdir():
            if file_path.is_file():
                documents.extend(self._load_file(file_path))
        return documents
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its type"""
        documents = []
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
            elif suffix == '.csv':
                # Check if pandas is available
                if PANDAS_AVAILABLE:
                    # Custom CSV loading for better context
                    df = pd.read_csv(file_path)
                    content = f"CSV Data from {file_path.name}:\n"
                    content += df.to_string()
                    documents = [Document(
                        page_content=content,
                        metadata={"source": str(file_path), "type": "csv"}
                    )]
                else:
                    # Use CSVLoader as fallback
                    loader = CSVLoader(str(file_path), encoding='utf-8')
                    documents = loader.load()
            elif suffix == '.pdf':
                if LANGCHAIN_AVAILABLE:
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                else:
                    # Use placeholder
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
            elif suffix == '.md':
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
            elif suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
                documents = [Document(
                    page_content=content,
                    metadata={"source": str(file_path), "type": "json"}
                )]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        # Add source metadata
        for doc in documents:
            doc.metadata["transformer_id"] = self.transformer_id
        
        return documents
    
    def _load_scenarios(self, file_path: Path, transformer_id: str) -> List[Document]:
        """Load transformer-specific scenario from JSON"""
        documents = []
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for transformer in data.get("transformers", []):
                if transformer.get("id", "").upper() == transformer_id.upper():
                    content = f"""
Transformer: {transformer.get('name')}
ID: {transformer.get('id')}
Location: {transformer.get('location')}
Status: {transformer.get('status')}
Health Score: {transformer.get('health_score')}%
Scenario: {transformer.get('scenario')}
Description: {transformer.get('description')}

Latest Readings:
{json.dumps(transformer.get('latest_readings', {}), indent=2)}

Risk Factors: {', '.join(transformer.get('risk_factors', []))}

Recommendations: {', '.join(transformer.get('recommendations', []))}
"""
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": str(file_path), "type": "scenario"}
                    ))
                    break
        except Exception as e:
            print(f"Error loading scenarios: {e}")
        
        return documents
    
    def build_vectorstore(self):
        """Build vector store from loaded documents"""
        if not self.documents:
            return False
        
        if not self.embeddings:
            return False
        
        if not LANGCHAIN_AVAILABLE:
            # Use simple in-memory storage without vector search
            print("Info: Vector store not built (LangChain not available). Using document storage only.")
            self.vectorstore = self.documents
            return True
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(self.documents)
        
        # Create vector store
        persist_dir = str(CHROMA_DIR / (self.transformer_id or "general"))
        
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        return True
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if not self.vectorstore:
            return []
        
        if isinstance(self.vectorstore, list):
            # In placeholder mode, return all documents
            return self.vectorstore[:k]
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_context(self, query: str) -> str:
        """Get context string from retrieved documents"""
        docs = self.retrieve(query)
        if not docs:
            # Return raw document content if no vectorstore
            return "\n\n".join([d.page_content[:1000] for d in self.documents[:2]])
        
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source: {Path(source).name}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)


class AgenticRAGEngine:
    """Main agentic RAG engine for transformer monitoring"""
    
    SYSTEM_PROMPT = """You are an expert power transformer analyst AI assistant. 
You have deep knowledge of:
- Dissolved Gas Analysis (DGA) interpretation
- Transformer health assessment
- Oil quality analysis
- Predictive maintenance
- Fault diagnosis
- IEEE and IEC standards for transformer monitoring

When answering questions:
1. Use the provided context from transformer documents
2. Provide specific values and trends when available
3. Reference relevant standards and thresholds
4. Give actionable recommendations
5. If generating analysis charts, describe what visualization would be helpful

Always be precise and technical while remaining understandable."""

    def __init__(self):
        self.ollama = OllamaClient()
        self.groq = GroqClient()
        self.transformer_rags: Dict[str, TransformerRAG] = {}
        self.conversation_history: List[Dict] = []
        
        # Model categorization
        self.ollama_models = ["phi", "phi3", "mistral", "llama3"]
        self.groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        self.all_models = self.ollama_models + self.groq_models
    
    def initialize_transformer(self, transformer_id: str) -> bool:
        """Initialize RAG for a specific transformer"""
        if transformer_id not in self.transformer_rags:
            rag = TransformerRAG(transformer_id)
            docs = rag.load_transformer_documents(transformer_id)
            
            if docs:
                rag.build_vectorstore()
                self.transformer_rags[transformer_id] = rag
                return True
            return False
        return True
    
    def ask(self, question: str, transformer_id: str = None, 
            model: str = "llama-3.3-70b-versatile", include_history: bool = True) -> Dict[str, Any]:
        """Ask a question with RAG context"""
        
        # Get context and transformer data
        context = ""
        transformer_data = None
        if transformer_id:
            self.initialize_transformer(transformer_id)
            if transformer_id in self.transformer_rags:
                context = self.transformer_rags[transformer_id].get_context(question)
            transformer_data = self._load_transformer_data(transformer_id)
        
        # Determine which provider to use
        provider = get_model_provider(model)
        use_groq = provider == "groq"
        
        # Check availability of the chosen provider
        if use_groq:
            provider_available = self.groq.is_available()
        else:
            provider_available = self.ollama.is_available()
        
        if provider_available:
            # Build prompt for the AI
            if context:
                prompt = f"""Based on the following transformer monitoring data and documents:

{context}

---

Question: {question}

Please provide a detailed, technical response based on the data provided."""
            else:
                prompt = f"""Question about transformer monitoring: {question}

Please provide a detailed technical response based on your knowledge of power transformer monitoring, DGA analysis, and maintenance practices."""
            
            # Add to history
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            
            # Generate response using the appropriate provider
            if include_history and len(self.conversation_history) > 1:
                messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
                messages.extend(self.conversation_history[-6:])
                
                if use_groq:
                    response = self.groq.chat(messages, model)
                else:
                    response = self.ollama.chat(messages, model)
            else:
                if use_groq:
                    response = self.groq.generate(
                        prompt=prompt,
                        model=model,
                        system_prompt=self.SYSTEM_PROMPT
                    )
                else:
                    response = self.ollama.generate(
                        prompt=prompt,
                        model=model,
                        system_prompt=self.SYSTEM_PROMPT
                    )
            
            # Check if response indicates an error
            if response.startswith("Error:"):
                # Fall back to data-based response on error
                provider_available = False
            else:
                # Add response to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response
                })
                provider = "groq" if use_groq else "ollama"
        
        if not provider_available:
            # Fallback: Generate intelligent response from data
            response = self._generate_fallback_response(question, transformer_id, transformer_data, context)
            provider = "fallback"
        
        return {
            "question": question,
            "answer": response,
            "model": model,
            "provider": provider,
            "transformer_id": transformer_id,
            "context_used": bool(context),
            "sources": self._get_sources(transformer_id)
        }
    
    def _load_transformer_data(self, transformer_id: str) -> Dict:
        """Load transformer data from JSON"""
        scenarios_file = DATA_DIR / "scenarios" / "transformers_scenarios.json"
        if scenarios_file.exists():
            with open(scenarios_file, 'r') as f:
                data = json.load(f)
            for t in data.get("transformers", []):
                if t.get("id", "").upper() == transformer_id.upper():
                    return t
        return {}
    
    def _generate_fallback_response(self, question: str, transformer_id: str, 
                                     transformer_data: Dict, context: str) -> str:
        """Generate intelligent fallback response based on transformer data"""
        q_lower = question.lower()
        
        if transformer_data:
            name = transformer_data.get('name', transformer_id)
            status = transformer_data.get('status', 'Unknown')
            health_score = transformer_data.get('health_score', 'N/A')
            scenario = transformer_data.get('scenario', '')
            description = transformer_data.get('description', '')
            readings = transformer_data.get('latest_readings', {})
            risks = transformer_data.get('risk_factors', [])
            recommendations = transformer_data.get('recommendations', [])
            
            # Health/Status questions
            if any(kw in q_lower for kw in ['health', 'status', 'condition', 'concern']):
                response = f"**{name} ({transformer_id}) Health Analysis**\n\n"
                response += f"**Current Status:** {status}\n"
                response += f"**Health Score:** {health_score}%\n"
                response += f"**Scenario:** {scenario}\n\n"
                response += f"**Assessment:** {description}\n\n"
                
                if risks:
                    response += "**Risk Factors:**\n"
                    for r in risks:
                        response += f"• ⚠️ {r}\n"
                else:
                    response += "**Risk Factors:** None identified - operating normally.\n"
                
                if recommendations:
                    response += "\n**Recommendations:**\n"
                    for rec in recommendations:
                        response += f"• {rec}\n"
                return response
            
            # DGA Analysis questions
            if any(kw in q_lower for kw in ['dga', 'gas', 'hydrogen', 'acetylene', 'h2', 'c2h2']):
                response = f"**DGA Analysis for {transformer_id}**\n\n"
                response += "**Latest Gas Readings (ppm):**\n"
                response += f"• Hydrogen (H₂): {readings.get('h2_ppm', 'N/A')} ppm"
                h2 = readings.get('h2_ppm', 0)
                if h2 > 100:
                    response += " ⚠️ ELEVATED\n"
                else:
                    response += " ✓\n"
                response += f"• Methane (CH₄): {readings.get('ch4_ppm', 'N/A')} ppm\n"
                response += f"• Acetylene (C₂H₂): {readings.get('c2h2_ppm', 'N/A')} ppm"
                c2h2 = readings.get('c2h2_ppm', 0)
                if c2h2 > 2:
                    response += " ⚠️ INDICATES ARCING\n"
                else:
                    response += " ✓\n"
                response += f"• Carbon Monoxide (CO): {readings.get('co_ppm', 'N/A')} ppm\n"
                response += f"• Carbon Dioxide (CO₂): {readings.get('co2_ppm', 'N/A')} ppm\n\n"
                
                response += "**Interpretation:**\n"
                if c2h2 > 5:
                    response += "• High acetylene indicates active arcing or electrical discharge\n"
                if h2 > 200:
                    response += "• Elevated hydrogen suggests thermal or electrical fault\n"
                if c2h2 <= 2 and h2 < 100:
                    response += "• Gas levels within normal operating limits\n"
                
                return response
            
            # Maintenance/Recommendations
            if any(kw in q_lower for kw in ['maintenance', 'recommend', 'action', 'what should']):
                response = f"**Maintenance Recommendations for {transformer_id}**\n\n"
                response += f"**Current Status:** {status} (Health: {health_score}%)\n\n"
                
                if recommendations:
                    response += "**Recommended Actions:**\n"
                    for i, rec in enumerate(recommendations, 1):
                        priority = "🔴 HIGH" if status == "Critical" else "🟡 MEDIUM" if status == "Warning" else "🟢 ROUTINE"
                        response += f"{i}. [{priority}] {rec}\n"
                else:
                    response += "No immediate maintenance actions required. Continue routine monitoring.\n"
                
                response += f"\n**Scenario Notes:** {description}"
                return response
            
            # Prediction questions
            if any(kw in q_lower for kw in ['predict', 'forecast', 'future', 'next', 'trend']):
                response = f"**Health Prediction for {transformer_id}**\n\n"
                response += f"**Current Health Index:** {health_score}%\n"
                response += f"**Status:** {status}\n\n"
                
                if status == "Critical":
                    response += "**6-Month Forecast:** Without intervention, health index may decline to 15-20%. Immediate action required.\n"
                    response += "**Risk of Failure:** HIGH (35-45% probability within 12 months)\n"
                elif status == "Warning":
                    response += "**6-Month Forecast:** Health index expected to decline 5-10% without maintenance intervention.\n"
                    response += "**Risk Assessment:** MODERATE - Schedule maintenance within 30-60 days\n"
                else:
                    response += "**6-Month Forecast:** Health index expected to remain stable (±2%).\n"
                    response += "**Risk Assessment:** LOW - Continue routine monitoring\n"
                
                return response
            
            # Oil quality questions
            if any(kw in q_lower for kw in ['oil', 'moisture', 'bdv', 'breakdown', 'tan delta']):
                response = f"**Oil Quality Analysis for {transformer_id}**\n\n"
                response += f"• Moisture Content: {readings.get('moisture_ppm', 'N/A')} ppm"
                moisture = readings.get('moisture_ppm', 0)
                if moisture > 35:
                    response += " ⚠️ HIGH\n"
                elif moisture > 25:
                    response += " ⚡ ELEVATED\n"
                else:
                    response += " ✓ GOOD\n"
                
                response += f"• Breakdown Voltage (BDV): {readings.get('breakdown_voltage_kv', 'N/A')} kV"
                bdv = readings.get('breakdown_voltage_kv', 60)
                if bdv < 40:
                    response += " ⚠️ LOW - Oil treatment needed\n"
                elif bdv < 50:
                    response += " ⚡ BELOW OPTIMAL\n"
                else:
                    response += " ✓ GOOD\n"
                
                response += f"• Tan Delta: {readings.get('tan_delta_percent', 'N/A')}%"
                td = readings.get('tan_delta_percent', 0)
                if td > 1.5:
                    response += " ⚠️ HIGH - Insulation degradation\n"
                else:
                    response += " ✓ ACCEPTABLE\n"
                
                return response
            
            # Default response with overview
            response = f"**{name} ({transformer_id}) Overview**\n\n"
            response += f"• **Status:** {status}\n"
            response += f"• **Health Score:** {health_score}%\n"
            response += f"• **Scenario:** {scenario}\n"
            response += f"• **Oil Temperature:** {readings.get('top_oil_temp_c', 'N/A')}°C\n"
            response += f"• **Load:** {readings.get('load_percent', 'N/A')}%\n\n"
            response += f"**Summary:** {description}\n\n"
            response += "Ask me about: health status, DGA analysis, maintenance recommendations, or predictions."
            return response
        
        # Generic response when no transformer selected
        return """**Transformer Monitoring AI Assistant**

I can help you with:
• **Health Analysis** - Current status, health scores, risk factors
• **DGA Interpretation** - Gas analysis and fault detection
• **Maintenance Planning** - Recommendations and scheduling
• **Trend Prediction** - Health forecasting and failure risk
• **Oil Quality** - BDV, moisture, tan delta analysis

Please select a specific transformer from the sidebar for detailed analysis, or ask a general question about transformer monitoring practices."""
    
    def _get_sources(self, transformer_id: str) -> List[str]:
        """Get list of source documents used"""
        if transformer_id and transformer_id in self.transformer_rags:
            rag = self.transformer_rags[transformer_id]
            return list(set(
                Path(d.metadata.get("source", "")).name 
                for d in rag.documents
            ))
        return []
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def generate_chart_description(self, transformer_id: str, 
                                   chart_type: str = "health_trend") -> str:
        """Generate data for chart visualization"""
        if transformer_id not in self.transformer_rags:
            self.initialize_transformer(transformer_id)
        
        # Load CSV data for charting
        transformer_dir = TRANSFORMERS_DIR / transformer_id.upper()
        dga_file = transformer_dir / "dga_history.csv"
        
        if dga_file.exists():
            df = pd.read_csv(dga_file)
            return df.to_json(orient="records")
        
        return "[]"


# Singleton instance
_engine = None

def get_engine() -> AgenticRAGEngine:
    """Get or create the RAG engine singleton"""
    global _engine
    if _engine is None:
        _engine = AgenticRAGEngine()
    return _engine


# Sample questions for the UI
SAMPLE_QUESTIONS = {
    "conceptual": [
        "What are the main factors affecting the health and lifespan of a power transformer?",
        "How can load variations and overloading impact transformer performance and longevity?",
        "What safety or protection systems are used in a transformer for real-time fault detection?",
    ],
    "data_based": [
        "What insights can be derived from trends in Hydrogen (H2) and Acetylene (C2H2) gas levels in DGA reports?",
        "How can correlation between oil temperature and load current indicate cooling system inefficiency?",
        "Which parameters are used for computing a Transformer Health Index (THI)?",
        "What is the ideal range of moisture content (ppm) in transformer oil for healthy operation?",
        "How can temperature-compensated Tan Delta values help assess insulation degradation?",
    ],
    "scenario": [
        "A DGA report shows an increase in acetylene (C2H2) and hydrogen (H2). What could be the probable fault?",
        "The BDV test results dropped from 60 kV to 42 kV within six months. What actions would you recommend?",
        "Transformer oil moisture content increases after heavy rain even though the unit is sealed. What could be the cause?",
        "DGA indicates elevated CO2 with stable H2 and CH4 — what does this suggest about insulation condition?",
    ],
    "forecasting": [
        "What is the predicted health index for the next 6 months based on current trends?",
        "Based on current data, when is this transformer likely to require maintenance or replacement?",
        "How would the health score change if the oil moisture level continues to increase at the current rate?",
    ],
    "diagnostic": [
        "What is the most probable cause of deteriorating transformer health?",
        "Can you detect any early signs of partial discharge activity that could lead to insulation failure?",
        "Based on gas composition trends, can you identify the type of fault occurring (thermal, electrical, arcing)?",
    ]
}
