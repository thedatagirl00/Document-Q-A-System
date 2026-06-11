## Project Summary: Local RAG Pipeline with Ollama and LangChain

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline for document question-answering, leveraging a local Large Language Model (LLM) hosted by Ollama and integrated with LangChain.

### Key Components & Technologies:

*   **LangChain:** Used for orchestrating the RAG pipeline, including document loading, text splitting, prompt engineering, and LLM integration.
*   **Ollama:** Employed to run the `llama3` LLM locally, providing a cost-effective and private solution for generative AI tasks.
*   **ChromaDB:** Utilized as the vector database to store and retrieve document embeddings efficiently.
*   **HuggingFace Embeddings (`all-MiniLM-L6-v2`):** Used to generate vector representations of document chunks for semantic search.
*   **PyPDFLoader:** For loading and processing PDF documents.

### Project Flow:

1.  **Dependency Installation:** Installed necessary Python libraries for LangChain, Ollama integration, vector store, and PDF processing.
2.  **Ollama Setup:** Installed and configured the Ollama server, ensuring it was running in the background and that the `llama3` model was successfully pulled. This included implementing robust retry mechanisms to handle server startup and model download.
3.  **Document Processing:** Loaded a local PDF document (`apple-privacy-policy-en-ww.pdf`), which was then split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
4.  **Vector Database Creation:** Embeddings for the document chunks were generated using `HuggingFaceEmbeddings` and stored in a persistent ChromaDB instance.
5.  **LLM Initialization:** The `llama3` model was initialized via the `OllamaLLM` LangChain integration.
6.  **RAG Pipeline Construction:** A RAG chain was built using LangChain Expression Language (LCEL), combining a custom prompt template, a vector store retriever (fetching the top 3 relevant chunks), and the local `llama3` LLM.
7.  **Question Answering:** Demonstrated the pipeline's functionality by asking a question about the PDF content, with the LLM generating an answer based *only* on the provided context retrieved from the vector database, thus mitigating hallucination.


```python
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-ollama chromadb sentence-transformers pypdf
```

```bash
# Install zstd, a dependency for Ollama
!apt-get update && apt-get install -y zstd

# Install Ollama server
!curl -fsSL https://ollama.com/install.sh | sh
```

```python
import time
!nohup /usr/local/bin/ollama serve &
time.sleep(10)
```

```python
# Pull the llama3 model using its full path
!/usr/local/bin/ollama pull llama3
```

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("/content/apple-privacy-policy-en-ww.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Split document into {len(chunks)} chunks.")
```

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created successfully!")
```

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question ONLY using the provided context.

<context>
{context}
</context>

Question: {question}
""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)
```

```python
import time
import subprocess

question = "According to the document, why user's personal data is used by Apple?"

ollama_path = "/usr/local/bin/ollama"
model_name = "llama3"

print(f"Ensuring Ollama server is running...")
subprocess.run(["nohup", ollama_path, "serve", "&"], check=False)
time.sleep(5)

server_ready = False
for i in range(10):
    try:
        result = subprocess.run([ollama_path, "list"], capture_output=True, text=True, check=True, timeout=10)
        server_ready = True
        if model_name in result.stdout:
            print(f"Ollama server is responsive and '{model_name}' model is available.")
        else:
            print(f"Ollama server is responsive but '{model_name}' model not found. Attempting to pull...")
            pull_result = subprocess.run([ollama_path, "pull", model_name], capture_output=True, text=True, check=True, timeout=300)
            print(pull_result.stdout)
            print(f"'{model_name}' model pulled successfully.")
        break
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Attempt {i+1}/10: Ollama server not ready or pull failed. Retrying in 10 seconds. Error: {e}")
        time.sleep(10)
    except Exception as e:
        print(f"Attempt {i+1}/10: An unexpected error occurred: {e}. Retrying in 10 seconds.")
        time.sleep(10)

if not server_ready:
    print("\n--- Fatal Error ---")
    print("Ollama server did not become responsive. Please ensure Ollama is correctly installed and started.")
else:
    response = rag_chain.invoke(question)

    print("\n--- Answer ---")
    print(response)
```
