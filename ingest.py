import os
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(data_dir: str = "data") -> List[Document]:
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Creating {data_dir} directory...")
        data_path.mkdir(parents=True)
        return documents
    
    for file_path in data_path.glob("**/*"):
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                elif file_path.suffix.lower() in [".txt", ".md"]:
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    docs = loader.load()
                else:
                    continue
                
                for doc in docs:
                    doc.metadata["source"] = str(file_path.name)
                    doc.metadata["file_path"] = str(file_path)
                
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
                
            except Exception as e:
                print(f"✗ Failed to load {file_path.name}: {e}")
    
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks


def create_sample_documents(data_dir: str = "data"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    sample_docs = {
        "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence (AI) is the simulation of human intelligence by machines. AI systems can perform tasks such as learning, reasoning, problem-solving, and understanding natural language.

Key Areas of AI:
1. Machine Learning: Algorithms that improve through experience
2. Natural Language Processing: Understanding and generating human language
3. Computer Vision: Interpreting visual information
4. Robotics: Physical AI systems that interact with the world

Applications:
- Healthcare: Diagnosis, drug discovery, personalized medicine
- Finance: Fraud detection, algorithmic trading, risk assessment
- Transportation: Autonomous vehicles, route optimization
- Education: Personalized learning, automated grading

Challenges:
- Ethics and bias in AI systems
- Data privacy concerns
- Job displacement
- AI safety and control
""",
        "machine_learning.txt": """Machine Learning Fundamentals

Machine Learning (ML) is a subset of AI that focuses on building systems that learn from data. Instead of being explicitly programmed, ML models improve their performance through experience.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled training data
   - Examples: Classification, Regression
   - Applications: Image recognition, spam detection

2. Unsupervised Learning
   - Finds patterns in unlabeled data
   - Examples: Clustering, Dimensionality reduction
   - Applications: Customer segmentation, anomaly detection

3. Reinforcement Learning
   - Learns through trial and error
   - Uses rewards and penalties
   - Applications: Game playing, robotics

Popular ML Algorithms:
- Linear Regression: Predicting continuous values
- Decision Trees: Classification and regression
- Neural Networks: Complex pattern recognition
- Support Vector Machines: Classification tasks
- K-Means: Clustering similar data points

Best Practices:
- Split data into training, validation, and test sets
- Use cross-validation to prevent overfitting
- Normalize and standardize features
- Monitor for bias in training data
""",
        "deep_learning.txt": """Deep Learning and Neural Networks

Deep Learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers. These deep neural networks can automatically learn hierarchical representations of data.

Neural Network Basics:
- Input Layer: Receives raw data
- Hidden Layers: Process and transform data
- Output Layer: Produces predictions
- Activation Functions: Introduce non-linearity (ReLU, Sigmoid, Tanh)
- Backpropagation: Training algorithm that adjusts weights

Popular Deep Learning Architectures:

1. Convolutional Neural Networks (CNNs)
   - Specialized for image processing
   - Use convolutional layers to detect features
   - Applications: Image classification, object detection

2. Recurrent Neural Networks (RNNs)
   - Process sequential data
   - Maintain memory of previous inputs
   - Applications: Time series, natural language

3. Transformers
   - Use attention mechanisms
   - Highly parallelizable
   - Applications: Language models, machine translation

4. Generative Adversarial Networks (GANs)
   - Generate synthetic data
   - Consist of generator and discriminator
   - Applications: Image generation, data augmentation

Training Deep Learning Models:
- Requires large datasets
- GPU acceleration recommended
- Use techniques like dropout and batch normalization
- Monitor training and validation loss
- Fine-tune hyperparameters

Frameworks:
- TensorFlow: Google's deep learning framework
- PyTorch: Facebook's flexible deep learning library
- Keras: High-level API for building neural networks
""",
        "nlp_fundamentals.txt": """Natural Language Processing

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. It combines linguistics, computer science, and machine learning.

Core NLP Tasks:

1. Text Classification
   - Sentiment analysis
   - Spam detection
   - Topic categorization

2. Named Entity Recognition (NER)
   - Identifying people, organizations, locations
   - Extracting structured information

3. Machine Translation
   - Translating between languages
   - Neural machine translation

4. Question Answering
   - Understanding questions
   - Retrieving relevant answers

5. Text Generation
   - Writing articles, stories
   - Chatbots and dialogue systems

NLP Preprocessing:
- Tokenization: Breaking text into words or subwords
- Lemmatization: Reducing words to base form
- Stop word removal: Filtering common words
- Part-of-speech tagging: Identifying grammatical roles

Modern NLP Techniques:
- Word Embeddings: Word2Vec, GloVe
- Contextualized Embeddings: BERT, GPT
- Transfer Learning: Fine-tuning pre-trained models
- Attention Mechanisms: Focusing on relevant parts

Large Language Models (LLMs):
- GPT (Generative Pre-trained Transformer)
- BERT (Bidirectional Encoder Representations)
- T5 (Text-to-Text Transfer Transformer)
- LLaMA: Open-source language models

Applications:
- Virtual assistants (Siri, Alexa)
- Search engines
- Content moderation
- Document summarization
- Language translation
"""
    }
    
    for filename, content in sample_docs.items():
        file_path = data_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created sample document: {filename}")


def main():
    print("=" * 80)
    print("DOCUMENT INGESTION & PREPROCESSING")
    print("=" * 80)
    print()
    
    data_dir = "data"
    
    if not Path(data_dir).exists() or not list(Path(data_dir).glob("*")):
        print("No documents found. Creating sample documents...")
        create_sample_documents(data_dir)
        print()
    
    print("Loading documents...")
    documents = load_documents(data_dir)
    print(f"\nLoaded {len(documents)} document(s)")
    
    if not documents:
        print("No documents to process. Add files to the 'data' directory.")
        return
    
    print(f"\nChunking documents...")
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i}:")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")
    
    print("\n" + "=" * 80)
    print("✓ Ingestion complete!")
    print("=" * 80)
    
    return chunks


if __name__ == "__main__":
    main()
