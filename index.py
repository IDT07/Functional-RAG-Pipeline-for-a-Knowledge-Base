import os
import pickle
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_documents, chunk_documents


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def build_vector_index(chunks: List[Document], embeddings):
    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✓ Index built successfully")
    return vectorstore


def save_index(vectorstore, index_dir: str = "vectorstore"):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    index_path = Path(index_dir) / "faiss_index"
    vectorstore.save_local(str(index_path))
    print(f"✓ Index saved to {index_path}")


def load_index(embeddings, index_dir: str = "vectorstore"):
    index_path = Path(index_dir) / "faiss_index"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Index loaded from {index_path}")
    return vectorstore


def test_retrieval(vectorstore, test_queries: List[str], k: int = 3):
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Distance: {score:.4f}")
            print(f"   Content: {doc.page_content[:200]}...")


def main():
    print("=" * 80)
    print("EMBEDDING & INDEXING")
    print("=" * 80)
    print()
    
    print("Loading documents...")
    documents = load_documents("data")
    
    if not documents:
        print("No documents found. Run ingest.py first.")
        return
    
    print(f"Loaded {len(documents)} document(s)")
    
    print("\nChunking documents...")
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    print("\nInitializing embeddings model...")
    embeddings = create_embeddings()
    print("✓ Embeddings model ready")
    
    vectorstore = build_vector_index(chunks, embeddings)
    
    save_index(vectorstore)
    
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the applications of NLP?"
    ]
    
    test_retrieval(vectorstore, test_queries, k=3)
    
    print("\n" + "=" * 80)
    print("✓ Indexing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
