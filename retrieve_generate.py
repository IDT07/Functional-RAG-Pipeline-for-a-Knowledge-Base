import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List, Dict
from index import load_index, create_embeddings

load_dotenv()


class RAGPipeline:
    def __init__(self, index_dir: str = "vectorstore"):
        print("Initializing RAG Pipeline...")
        
        self.embeddings = create_embeddings()
        self.vectorstore = load_index(self.embeddings, index_dir)
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=api_key
        )
        
        self.prompt_template = PromptTemplate(
            template="""You are an AI assistant that answers questions based on provided context. Your task is to provide accurate, well-cited answers.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the question using ONLY the information from the context above
2. Cite sources using [Source: filename] format after each claim
3. If the context doesn't contain enough information, say so clearly
4. Provide a confidence statement at the end (High/Medium/Low)
5. Be concise but comprehensive

QUESTION: {question}

ANSWER:""",
            input_variables=["context", "question"]
        )
        
        print("✓ RAG Pipeline ready")
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            context_parts.append(f"[{i}] Source: {source}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate(self, query: str, k: int = 4) -> Dict:
        retrieved_docs = self.retrieve(query, k)
        
        context = self.format_context(retrieved_docs)
        
        prompt = self.prompt_template.format(context=context, question=query)
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        sources = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]
        
        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
    
    def interactive_mode(self):
        print("\n" + "=" * 80)
        print("RAG PIPELINE - INTERACTIVE MODE")
        print("=" * 80)
        print("Commands: 'quit' to exit, or ask any question")
        print()
        
        while True:
            query = input("\nYour Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n" + "-" * 80)
            print("Retrieving and generating answer...")
            print("-" * 80)
            
            result = self.generate(query)
            
            print(f"\n📝 ANSWER:\n")
            print(result['answer'])
            
            print(f"\n📚 SOURCES USED ({result['num_sources']}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source['source']}")
                print(f"   {source['content']}")


def main():
    print("=" * 80)
    print("RETRIEVAL-AUGMENTED GENERATION")
    print("=" * 80)
    print()
    
    try:
        rag = RAGPipeline()
        
        test_queries = [
            "What is machine learning and what are its types?",
            "How do neural networks work?",
            "What are the applications of Natural Language Processing?"
        ]
        
        print("\n" + "=" * 80)
        print("TESTING RAG PIPELINE")
        print("=" * 80)
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print("=" * 80)
            
            result = rag.generate(query, k=3)
            
            print(f"\n📝 ANSWER:\n")
            print(result['answer'])
            
            print(f"\n📚 SOURCES ({result['num_sources']}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source['source']}")
                print(f"   {source['content']}")
        
        rag.interactive_mode()
        
    except FileNotFoundError:
        print("Error: Vector index not found. Run index.py first to build the index.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
