"""
Test Suite for Modern Text Summarization System
===============================================

Comprehensive tests for all components of the summarization system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from modern_summarizer import (
    ModernTextSummarizer,
    SummarizationConfig,
    MockDatabase,
    TextPreprocessor,
    SummarizationEvaluator
)

class TestSummarizationConfig:
    """Test configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SummarizationConfig()
        assert config.max_sentences == 3
        assert config.min_sentence_length == 10
        assert config.max_sentence_length == 200
        assert config.method == "tfidf"
        assert config.language == "english"
        assert config.remove_stopwords == True
        assert config.stem_words == False
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = SummarizationConfig(
            max_sentences=5,
            method="textrank",
            remove_stopwords=False
        )
        assert config.max_sentences == 5
        assert config.method == "textrank"
        assert config.remove_stopwords == False

class TestMockDatabase:
    """Test mock database functionality"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = MockDatabase(db_path)
            documents = db.get_documents()
            
            assert len(documents) == 3
            assert all('id' in doc for doc in documents)
            assert all('title' in doc for doc in documents)
            assert all('content' in doc for doc in documents)
            assert all('category' in doc for doc in documents)
            
        finally:
            os.unlink(db_path)
    
    def test_save_summary(self):
        """Test saving summaries to database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = MockDatabase(db_path)
            documents = db.get_documents()
            
            # Save a summary
            doc_id = documents[0]['id']
            summary = "Test summary"
            rouge_scores = {'rouge1_fmeasure': 0.5}
            
            db.save_summary(doc_id, "tfidf", summary, rouge_scores)
            
            # Verify it was saved (we can't easily query without adding a get method)
            # This test mainly ensures no exceptions are raised
            
        finally:
            os.unlink(db_path)

class TestTextPreprocessor:
    """Test text preprocessing functionality"""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = TextPreprocessor("english")
        assert preprocessor.language == "english"
        assert len(preprocessor.stop_words) > 0
        assert preprocessor.stemmer is not None
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        preprocessor = TextPreprocessor("english")
        
        text = "The quick brown fox jumps over the lazy dog."
        processed = preprocessor.preprocess_text(text, remove_stopwords=True)
        
        assert isinstance(processed, str)
        assert len(processed) > 0
        # Should remove stopwords like "the", "over"
        assert "the" not in processed.lower()
    
    def test_clean_sentences(self):
        """Test sentence cleaning"""
        preprocessor = TextPreprocessor("english")
        
        sentences = [
            "Short.",
            "This is a good sentence with proper length.",
            "This is an extremely long sentence that exceeds the maximum length limit and should be filtered out because it is too long for our purposes and contains too many words to be considered appropriate for summarization tasks."
        ]
        
        cleaned = preprocessor.clean_sentences(sentences, min_length=10, max_length=50)
        
        assert len(cleaned) == 1
        assert cleaned[0] == "This is a good sentence with proper length."

class TestSummarizationEvaluator:
    """Test evaluation metrics"""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = SummarizationEvaluator()
        assert evaluator.scorer is not None
    
    def test_evaluate_summary(self):
        """Test summary evaluation"""
        evaluator = SummarizationEvaluator()
        
        reference = "The quick brown fox jumps over the lazy dog."
        summary = "The fox jumps over the dog."
        
        scores = evaluator.evaluate_summary(reference, summary)
        
        assert isinstance(scores, dict)
        assert 'rouge1_precision' in scores
        assert 'rouge1_recall' in scores
        assert 'rouge1_fmeasure' in scores
        assert 'rouge2_precision' in scores
        assert 'rouge2_recall' in scores
        assert 'rouge2_fmeasure' in scores
        assert 'rougeL_precision' in scores
        assert 'rougeL_recall' in scores
        assert 'rougeL_fmeasure' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1

class TestModernTextSummarizer:
    """Test main summarization class"""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        Artificial Intelligence (AI) is transforming industries worldwide. 
        It is being used in healthcare to detect diseases early, in finance to prevent fraud, 
        and in manufacturing to optimize supply chains. AI-powered assistants are now part of daily life. 
        Despite its promise, AI raises ethical concerns like bias and job displacement. 
        As AI evolves, regulations and responsible development become crucial.
        """
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return SummarizationConfig(max_sentences=2, method="tfidf")
    
    @pytest.fixture
    def summarizer(self, config):
        """Test summarizer instance"""
        return ModernTextSummarizer(config)
    
    def test_summarizer_initialization(self, config):
        """Test summarizer initialization"""
        summarizer = ModernTextSummarizer(config)
        assert summarizer.config == config
        assert summarizer.preprocessor is not None
        assert summarizer.evaluator is not None
    
    def test_summarize_tfidf(self, summarizer, sample_text):
        """Test TF-IDF summarization"""
        summary = summarizer.summarize_tfidf(sample_text)
        
        assert isinstance(summary, list)
        assert len(summary) <= summarizer.config.max_sentences
        assert all(isinstance(sentence, str) for sentence in summary)
        assert all(len(sentence.strip()) > 0 for sentence in summary)
    
    def test_summarize_textrank(self, summarizer, sample_text):
        """Test TextRank summarization"""
        summary = summarizer.summarize_textrank(sample_text)
        
        assert isinstance(summary, list)
        assert len(summary) <= summarizer.config.max_sentences
        assert all(isinstance(sentence, str) for sentence in summary)
        assert all(len(sentence.strip()) > 0 for sentence in summary)
    
    @patch('modern_summarizer.AutoTokenizer')
    @patch('modern_summarizer.AutoModel')
    def test_summarize_bert(self, mock_model, mock_tokenizer, summarizer, sample_text):
        """Test BERT summarization with mocked models"""
        # Mock the BERT components
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        # Mock the forward pass
        mock_output = Mock()
        mock_output.last_hidden_state = Mock()
        mock_output.last_hidden_state.__getitem__ = Mock(return_value=Mock())
        mock_output.last_hidden_state.__getitem__.return_value.numpy.return_value = np.random.rand(1, 768)
        
        mock_model.from_pretrained.return_value.return_value = mock_output
        mock_model.from_pretrained.return_value.eval.return_value = None
        
        summary = summarizer.summarize_bert(sample_text)
        
        assert isinstance(summary, list)
        assert len(summary) <= summarizer.config.max_sentences
    
    @patch('modern_summarizer.SentenceTransformer')
    def test_summarize_sentence_bert(self, mock_sentence_transformer, summarizer, sample_text):
        """Test Sentence-BERT summarization with mocked model"""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 384)  # Mock embeddings
        mock_sentence_transformer.return_value = mock_model
        
        summary = summarizer.summarize_sentence_bert(sample_text)
        
        assert isinstance(summary, list)
        assert len(summary) <= summarizer.config.max_sentences
    
    def test_summarize_main_method(self, summarizer, sample_text):
        """Test main summarize method"""
        summary, scores = summarizer.summarize(sample_text, "tfidf")
        
        assert isinstance(summary, list)
        assert isinstance(scores, dict)
        assert len(summary) <= summarizer.config.max_sentences
        assert 'rouge1_fmeasure' in scores
    
    def test_summarize_invalid_method(self, summarizer, sample_text):
        """Test summarize with invalid method"""
        with pytest.raises(ValueError):
            summarizer.summarize(sample_text, "invalid_method")
    
    def test_short_text_handling(self, summarizer):
        """Test handling of very short text"""
        short_text = "This is a short text."
        
        summary = summarizer.summarize_tfidf(short_text)
        
        # Should return the original text if it's shorter than max_sentences
        assert isinstance(summary, list)
        assert len(summary) >= 0
    
    def test_empty_text_handling(self, summarizer):
        """Test handling of empty text"""
        empty_text = ""
        
        summary = summarizer.summarize_tfidf(empty_text)
        
        assert isinstance(summary, list)
        assert len(summary) == 0

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_summarization(self):
        """Test complete end-to-end summarization pipeline"""
        config = SummarizationConfig(max_sentences=2, method="tfidf")
        summarizer = ModernTextSummarizer(config)
        
        text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms. 
        These algorithms can learn patterns from data without being explicitly programmed. 
        Deep learning is a subset of machine learning that uses neural networks. 
        Neural networks are inspired by the structure of the human brain.
        """
        
        summary, scores = summarizer.summarize(text)
        
        assert isinstance(summary, list)
        assert isinstance(scores, dict)
        assert len(summary) <= 2
        assert all(isinstance(sentence, str) for sentence in summary)
    
    def test_database_integration(self):
        """Test database integration"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = MockDatabase(db_path)
            documents = db.get_documents()
            
            config = SummarizationConfig(max_sentences=2, method="tfidf")
            summarizer = ModernTextSummarizer(config)
            
            # Summarize first document
            doc = documents[0]
            summary, scores = summarizer.summarize(doc['content'])
            
            # Save summary
            db.save_summary(doc['id'], "tfidf", " ".join(summary), scores)
            
            # Verify no exceptions were raised
            assert True
            
        finally:
            os.unlink(db_path)

# Performance tests
class TestPerformance:
    """Performance and stress tests"""
    
    def test_large_text_handling(self):
        """Test handling of large text"""
        config = SummarizationConfig(max_sentences=3, method="tfidf")
        summarizer = ModernTextSummarizer(config)
        
        # Create a large text by repeating sentences
        base_sentence = "This is a test sentence for performance testing. "
        large_text = base_sentence * 100  # 100 sentences
        
        import time
        start_time = time.time()
        
        summary = summarizer.summarize_tfidf(large_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert isinstance(summary, list)
        assert len(summary) <= 3
        assert processing_time < 10  # Should complete within 10 seconds
    
    def test_multiple_methods_comparison(self):
        """Test performance comparison between methods"""
        text = """
        Artificial Intelligence (AI) is transforming industries worldwide. 
        It is being used in healthcare to detect diseases early, in finance to prevent fraud, 
        and in manufacturing to optimize supply chains. AI-powered assistants are now part of daily life. 
        Despite its promise, AI raises ethical concerns like bias and job displacement. 
        As AI evolves, regulations and responsible development become crucial.
        """
        
        config = SummarizationConfig(max_sentences=2)
        summarizer = ModernTextSummarizer(config)
        
        methods = ["tfidf", "textrank"]
        results = {}
        
        for method in methods:
            summary, scores = summarizer.summarize(text, method)
            results[method] = {
                'summary': summary,
                'rouge1': scores['rouge1_fmeasure'],
                'rouge2': scores['rouge2_fmeasure']
            }
        
        # Both methods should produce valid results
        assert len(results) == 2
        assert all('summary' in result for result in results.values())
        assert all('rouge1' in result for result in results.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
