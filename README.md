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
