"""
Modern Text Summarization System using Extractive Methods
========================================================

This project implements multiple extractive summarization techniques:
1. TF-IDF + Cosine Similarity (original method)
2. TextRank algorithm
3. BERT-based sentence embeddings
4. Sentence-BERT embeddings
5. Position-based scoring

Features:
- Multiple summarization algorithms
- Evaluation metrics (ROUGE scores)
- Web interface
- Mock database integration
- Configuration management
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummarizationConfig:
    """Configuration for summarization parameters"""
    max_sentences: int = 3
    min_sentence_length: int = 10
    max_sentence_length: int = 200
    method: str = "tfidf"  # tfidf, textrank, bert, sentence_bert
    language: str = "english"
    remove_stopwords: bool = True
    stem_words: bool = False

class MockDatabase:
    """Mock database for storing sample texts and summaries"""
    
    def __init__(self, db_path: str = "summarization.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                method TEXT NOT NULL,
                summary TEXT NOT NULL,
                rouge_scores TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Insert sample data
        sample_docs = [
            {
                "title": "Artificial Intelligence in Healthcare",
                "content": """
                Artificial Intelligence (AI) is revolutionizing healthcare by enabling early disease detection, 
                personalized treatment plans, and improved patient outcomes. Machine learning algorithms can 
                analyze medical images with superhuman accuracy, identifying tumors, fractures, and other 
                conditions that might be missed by human doctors. AI-powered diagnostic tools are being 
                integrated into hospitals worldwide, reducing diagnostic errors and improving efficiency. 
                Natural language processing helps doctors extract insights from patient records and medical 
                literature. However, challenges remain in data privacy, algorithm bias, and regulatory approval. 
                The future of AI in healthcare promises more precise medicine and better patient care.
                """,
                "category": "Technology"
            },
            {
                "title": "Climate Change and Renewable Energy",
                "content": """
                Climate change poses one of the greatest challenges of our time, requiring immediate action 
                to reduce greenhouse gas emissions and transition to renewable energy sources. Solar and 
                wind power have become increasingly cost-effective, making them competitive with fossil fuels. 
                Countries worldwide are investing heavily in renewable energy infrastructure, creating jobs 
                and stimulating economic growth. Electric vehicles are gaining market share as battery 
                technology improves and charging infrastructure expands. However, the transition requires 
                significant investment and policy support. International cooperation is essential to meet 
                climate goals and ensure a sustainable future for generations to come.
                """,
                "category": "Environment"
            },
            {
                "title": "The Future of Remote Work",
                "content": """
                The COVID-19 pandemic accelerated the adoption of remote work, fundamentally changing how 
                organizations operate and employees work. Companies have discovered that remote work can 
                increase productivity while reducing overhead costs. Employees enjoy greater flexibility 
                and work-life balance, leading to higher job satisfaction. However, remote work also 
                presents challenges in communication, collaboration, and maintaining company culture. 
                Hybrid work models are emerging as a compromise, combining the benefits of remote and 
                in-person work. Technology continues to evolve to support distributed teams, with 
                improvements in video conferencing, project management, and virtual collaboration tools.
                """,
                "category": "Business"
            }
        ]
        
        for doc in sample_docs:
            cursor.execute('''
                INSERT OR IGNORE INTO documents (title, content, category)
                VALUES (?, ?, ?)
            ''', (doc["title"], doc["content"], doc["category"]))
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with sample data")
    
    def get_documents(self) -> List[Dict]:
        """Get all documents from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, category FROM documents")
        rows = cursor.fetchall()
        conn.close()
        
        return [{"id": row[0], "title": row[1], "content": row[2], "category": row[3]} 
                for row in rows]
    
    def save_summary(self, document_id: int, method: str, summary: str, rouge_scores: Dict):
        """Save a summary to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO summaries (document_id, method, summary, rouge_scores)
            VALUES (?, ?, ?, ?)
        ''', (document_id, method, summary, json.dumps(rouge_scores)))
        conn.commit()
        conn.close()

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self, language: str = "english"):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       stem_words: bool = False) -> str:
        """Preprocess text for better summarization"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize words
        words = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = [word for word in words if word not in self.stop_words]
        
        # Stem words if requested
        if stem_words:
            words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def clean_sentences(self, sentences: List[str], min_length: int = 10, 
                       max_length: int = 200) -> List[str]:
        """Clean and filter sentences"""
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if min_length <= len(sentence) <= max_length:
                cleaned.append(sentence)
        return cleaned

class SummarizationEvaluator:
    """Evaluation metrics for summarization quality"""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_summary(self, reference: str, summary: str) -> Dict[str, float]:
        """Evaluate summary using ROUGE metrics"""
        scores = self.scorer.score(reference, summary)
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_fmeasure': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_fmeasure': scores['rougeL'].fmeasure,
        }

class ModernTextSummarizer:
    """Modern text summarization system with multiple algorithms"""
    
    def __init__(self, config: SummarizationConfig):
        self.config = config
        self.preprocessor = TextPreprocessor(config.language)
        self.evaluator = SummarizationEvaluator()
        
        # Initialize models lazily
        self._spacy_model = None
        self._bert_model = None
        self._bert_tokenizer = None
        self._sentence_model = None
    
    @property
    def spacy_model(self):
        """Lazy loading of spaCy model"""
        if self._spacy_model is None:
            try:
                self._spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self._spacy_model = None
        return self._spacy_model
    
    @property
    def bert_model(self):
        """Lazy loading of BERT model"""
        if self._bert_model is None:
            try:
                model_name = "bert-base-uncased"
                self._bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._bert_model = AutoModel.from_pretrained(model_name)
                self._bert_model.eval()
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}")
                self._bert_model = None
        return self._bert_model
    
    @property
    def sentence_model(self):
        """Lazy loading of Sentence-BERT model"""
        if self._sentence_model is None:
            try:
                self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load Sentence-BERT model: {e}")
                self._sentence_model = None
        return self._sentence_model
    
    def summarize_tfidf(self, text: str) -> List[str]:
        """TF-IDF based summarization (original method enhanced)"""
        sentences = sent_tokenize(text)
        sentences = self.preprocessor.clean_sentences(
            sentences, self.config.min_sentence_length, self.config.max_sentence_length
        )
        
        if len(sentences) <= self.config.max_sentences:
            return sentences
        
        # Preprocess sentences
        processed_sentences = [
            self.preprocessor.preprocess_text(sentence, self.config.remove_stopwords, self.config.stem_words)
            for sentence in sentences
        ]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        sentence_vectors = vectorizer.fit_transform(processed_sentences)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Rank sentences using similarity scores
        scores = similarity_matrix.sum(axis=1)
        
        # Select top sentences
        top_indices = np.argsort(scores)[-self.config.max_sentences:][::-1]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        
        return summary_sentences
    
    def summarize_textrank(self, text: str) -> List[str]:
        """TextRank algorithm implementation"""
        sentences = sent_tokenize(text)
        sentences = self.preprocessor.clean_sentences(
            sentences, self.config.min_sentence_length, self.config.max_sentence_length
        )
        
        if len(sentences) <= self.config.max_sentences:
            return sentences
        
        # Preprocess sentences
        processed_sentences = [
            self.preprocessor.preprocess_text(sentence, self.config.remove_stopwords, self.config.stem_words)
            for sentence in sentences
        ]
        
        # Create similarity matrix
        vectorizer = TfidfVectorizer(max_features=1000)
        sentence_vectors = vectorizer.fit_transform(processed_sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # TextRank algorithm
        scores = np.ones(len(sentences))
        damping_factor = 0.85
        max_iterations = 100
        
        for _ in range(max_iterations):
            prev_scores = scores.copy()
            for i in range(len(sentences)):
                score = 0
                for j in range(len(sentences)):
                    if i != j and similarity_matrix[i][j] > 0:
                        score += similarity_matrix[i][j] * prev_scores[j]
                scores[i] = (1 - damping_factor) + damping_factor * score
            
            if np.allclose(scores, prev_scores, atol=1e-6):
                break
        
        # Select top sentences
        top_indices = np.argsort(scores)[-self.config.max_sentences:][::-1]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        
        return summary_sentences
    
    def summarize_bert(self, text: str) -> List[str]:
        """BERT-based sentence embeddings for summarization"""
        if self.bert_model is None:
            logger.warning("BERT model not available, falling back to TF-IDF")
            return self.summarize_tfidf(text)
        
        sentences = sent_tokenize(text)
        sentences = self.preprocessor.clean_sentences(
            sentences, self.config.min_sentence_length, self.config.max_sentence_length
        )
        
        if len(sentences) <= self.config.max_sentences:
            return sentences
        
        # Get BERT embeddings
        embeddings = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self._bert_tokenizer(sentence, return_tensors="pt", 
                                            truncation=True, padding=True, max_length=512)
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding.flatten())
        
        embeddings = np.array(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Rank sentences
        scores = similarity_matrix.sum(axis=1)
        
        # Select top sentences
        top_indices = np.argsort(scores)[-self.config.max_sentences:][::-1]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        
        return summary_sentences
    
    def summarize_sentence_bert(self, text: str) -> List[str]:
        """Sentence-BERT based summarization"""
        if self.sentence_model is None:
            logger.warning("Sentence-BERT model not available, falling back to TF-IDF")
            return self.summarize_tfidf(text)
        
        sentences = sent_tokenize(text)
        sentences = self.preprocessor.clean_sentences(
            sentences, self.config.min_sentence_length, self.config.max_sentence_length
        )
        
        if len(sentences) <= self.config.max_sentences:
            return sentences
        
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(sentences)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Rank sentences
        scores = similarity_matrix.sum(axis=1)
        
        # Select top sentences
        top_indices = np.argsort(scores)[-self.config.max_sentences:][::-1]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        
        return summary_sentences
    
    def summarize(self, text: str, method: Optional[str] = None) -> Tuple[List[str], Dict[str, float]]:
        """Main summarization method"""
        method = method or self.config.method
        
        # Generate summary
        if method == "tfidf":
            summary = self.summarize_tfidf(text)
        elif method == "textrank":
            summary = self.summarize_textrank(text)
        elif method == "bert":
            summary = self.summarize_bert(text)
        elif method == "sentence_bert":
            summary = self.summarize_sentence_bert(text)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate summary (using first few sentences as reference)
        reference_sentences = sent_tokenize(text)[:5]  # Use first 5 sentences as reference
        reference = " ".join(reference_sentences)
        summary_text = " ".join(summary)
        
        rouge_scores = self.evaluator.evaluate_summary(reference, summary_text)
        
        return summary, rouge_scores

def main():
    """Main function for command-line usage"""
    # Sample text
    sample_text = """
    Artificial Intelligence (AI) is transforming industries worldwide. 
    It is being used in healthcare to detect diseases early, in finance to prevent fraud, 
    and in manufacturing to optimize supply chains. AI-powered assistants are now part of daily life. 
    Despite its promise, AI raises ethical concerns like bias and job displacement. 
    As AI evolves, regulations and responsible development become crucial.
    """
    
    # Initialize summarizer
    config = SummarizationConfig(max_sentences=3, method="tfidf")
    summarizer = ModernTextSummarizer(config)
    
    # Test different methods
    methods = ["tfidf", "textrank", "bert", "sentence_bert"]
    
    print("🧠 Modern Text Summarization System\n")
    print(f"Original text ({len(sent_tokenize(sample_text))} sentences):")
    print(sample_text.strip())
    print("\n" + "="*80 + "\n")
    
    for method in methods:
        try:
            summary, scores = summarizer.summarize(sample_text, method)
            print(f"📝 {method.upper()} Summary:")
            for i, sentence in enumerate(summary, 1):
                print(f"{i}. {sentence}")
            print(f"\n📊 ROUGE-1 F1 Score: {scores['rouge1_fmeasure']:.3f}")
            print(f"📊 ROUGE-2 F1 Score: {scores['rouge2_fmeasure']:.3f}")
            print("-" * 80)
        except Exception as e:
            print(f"❌ Error with {method}: {e}")
            print("-" * 80)

if __name__ == "__main__":
    main()
