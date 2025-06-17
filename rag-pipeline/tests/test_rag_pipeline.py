import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from document_processor import DocumentProcessor, VectorStore, DocumentChunk
from qwen_agent import QwenAgent, AgenticRAG

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor"""
    
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        dirty_text = "This   is  a   test    with\nexcessive\t\twhitespace."
        clean_text = self.processor.clean_text(dirty_text)
        expected = "This is a test with excessive whitespace."
        self.assertEqual(clean_text, expected)
    
    def test_chunk_text(self):
        """Test text chunking functionality"""
        text = "This is sentence one. This is sentence two. " * 50
        metadata = {'source': 'test.pdf', 'page_number': 1}
        
        chunks = self.processor.chunk_text(text, metadata)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], DocumentChunk)
        self.assertEqual(chunks[0].source, 'test.pdf')
        self.assertEqual(chunks[0].page_number, 1)

class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore"""
    
    def setUp(self):
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        self.vector_store = VectorStore(collection_name="test_collection")
    
    def tearDown(self):
        # Clean up temporary directory
        os.chdir('/')
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_and_search_documents(self):
        """Test adding documents and searching"""
        # Create test chunks
        chunks = [
            DocumentChunk(
                text="Mechanical ventilation is used for respiratory support.",
                metadata={'source': 'test.pdf', 'page_number': 1},
                chunk_id="test_chunk_1",
                source="test.pdf",
                page_number=1
            ),
            DocumentChunk(
                text="PEEP (Positive End-Expiratory Pressure) improves oxygenation.",
                metadata={'source': 'test.pdf', 'page_number': 2},
                chunk_id="test_chunk_2",
                source="test.pdf",
                page_number=2
            )
        ]
        
        # Add documents
        self.vector_store.add_documents(chunks)
        
        # Test search
        results = self.vector_store.search("mechanical ventilation", n_results=2)
        
        self.assertGreater(len(results), 0)
        self.assertIn('text', results[0])
        self.assertIn('metadata', results[0])
        
        # Test collection stats
        stats = self.vector_store.get_collection_stats()
        self.assertEqual(stats['total_documents'], 2)

class TestQwenAgent(unittest.TestCase):
    """Test cases for QwenAgent (without loading actual model)"""
    
    def test_analyze_query_intent(self):
        """Test query intent analysis without loading model"""
        # Create a mock agent for testing intent analysis only
        class MockQwenAgent:
            def analyze_query_intent(self, query):
                from qwen_agent import QwenAgent
                agent = QwenAgent.__new__(QwenAgent)  # Create without __init__
                return agent.analyze_query_intent(query)
        
        mock_agent = MockQwenAgent()
        
        # Test ventilation query
        ventilation_query = "How do I set up mechanical ventilation for COPD patients?"
        result = mock_agent.analyze_query_intent(ventilation_query)
        
        self.assertIn('concepts', result)
        self.assertTrue(result['concepts']['ventilation'])
        self.assertEqual(result['question_type'], 'procedural')
        
        # Test pathology query
        pathology_query = "What causes ARDS in patients?"
        result = mock_agent.analyze_query_intent(pathology_query)
        
        self.assertTrue(result['concepts']['pathology'])
        self.assertEqual(result['question_type'], 'explanatory')

class TestAgenticRAG(unittest.TestCase):
    """Test cases for AgenticRAG (integration tests)"""
    
    def setUp(self):
        # Create temporary directory for test
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
        
        # Create mock components
        self.vector_store = VectorStore(collection_name="test_rag")
        
        # Add some test data
        chunks = [
            DocumentChunk(
                text="Mechanical ventilation is a life-support method used in respiratory failure.",
                metadata={'source': 'textbook.pdf', 'page_number': 45},
                chunk_id="rag_test_1",
                source="textbook.pdf",
                page_number=45
            )
        ]
        self.vector_store.add_documents(chunks)
    
    def tearDown(self):
        os.chdir('/')
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_enhance_query(self):
        """Test query enhancement functionality"""
        # Create mock RAG system for testing enhancement
        class MockAgenticRAG:
            def _enhance_query(self, original_query, intent_analysis):
                from qwen_agent import AgenticRAG
                rag = AgenticRAG.__new__(AgenticRAG)
                return rag._enhance_query(original_query, intent_analysis)
        
        mock_rag = MockAgenticRAG()
        
        intent_analysis = {
            'concepts': {
                'ventilation': True,
                'pathology': False,
                'procedures': False
            }
        }
        
        original_query = "How to set PEEP?"
        enhanced_query = mock_rag._enhance_query(original_query, intent_analysis)
        
        self.assertIn("mechanical ventilation", enhanced_query)
        self.assertIn(original_query, enhanced_query)

def create_test_suite():
    """Create and return test suite"""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDocumentProcessor))
    suite.addTest(unittest.makeSuite(TestVectorStore))
    suite.addTest(unittest.makeSuite(TestQwenAgent))
    suite.addTest(unittest.makeSuite(TestAgenticRAG))
    
    return suite

if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
