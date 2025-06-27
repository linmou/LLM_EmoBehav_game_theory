import unittest
from unittest.mock import patch, MagicMock
import os

# Import the function to be tested
# Assuming model_loader_test.py is in the same directory or in PYTHONPATH
from model_loader_test import load_model_and_tokenizer

class TestModelLoader(unittest.TestCase):

    @patch('model_loader_test.os.path.isdir')
    @patch('model_loader_test.AutoTokenizer.from_pretrained')
    @patch('model_loader_test.AutoModel.from_pretrained')
    def test_load_model_and_tokenizer_success(self, mock_auto_model, mock_auto_tokenizer, mock_isdir):
        """Test successful loading of model and tokenizer."""
        mock_isdir.return_value = True
        mock_auto_tokenizer.return_value = MagicMock()  # Simulate a tokenizer object
        mock_auto_model.return_value = MagicMock()      # Simulate a model object

        model_path = "/fake/model/path"
        success, error = load_model_and_tokenizer(model_path)

        self.assertTrue(success)
        self.assertIsNone(error)
        mock_isdir.assert_called_once_with(model_path)
        mock_auto_tokenizer.assert_called_once_with(model_path, trust_remote_code=True)
        mock_auto_model.assert_called_once_with(model_path, trust_remote_code=True)

    @patch('model_loader_test.os.path.isdir')
    def test_load_model_path_not_dir(self, mock_isdir):
        """Test when the model path is not a directory or doesn't exist."""
        mock_isdir.return_value = False
        model_path = "/invalid/path"
        
        success, error = load_model_and_tokenizer(model_path)
        
        self.assertFalse(success)
        self.assertIn("Path not found or not a directory", error)
        mock_isdir.assert_called_once_with(model_path)

    @patch('model_loader_test.os.path.isdir')
    @patch('model_loader_test.AutoTokenizer.from_pretrained')
    def test_load_tokenizer_fails(self, mock_auto_tokenizer, mock_isdir):
        """Test failure when tokenizer loading raises an exception."""
        mock_isdir.return_value = True
        mock_auto_tokenizer.side_effect = Exception("Tokenizer load error")
        
        model_path = "/fake/model/path"
        success, error = load_model_and_tokenizer(model_path)
        
        self.assertFalse(success)
        self.assertIn("Tokenizer load error", error)
        mock_isdir.assert_called_once_with(model_path)
        mock_auto_tokenizer.assert_called_once_with(model_path, trust_remote_code=True)

    @patch('model_loader_test.os.path.isdir')
    @patch('model_loader_test.AutoTokenizer.from_pretrained')
    @patch('model_loader_test.AutoModel.from_pretrained')
    def test_load_model_fails(self, mock_auto_model, mock_auto_tokenizer, mock_isdir):
        """Test failure when model loading raises an exception."""
        mock_isdir.return_value = True
        mock_auto_tokenizer.return_value = MagicMock()
        mock_auto_model.side_effect = Exception("Model load error")

        model_path = "/fake/model/path"
        success, error = load_model_and_tokenizer(model_path)

        self.assertFalse(success)
        self.assertIn("Model load error", error)
        mock_isdir.assert_called_once_with(model_path)
        mock_auto_tokenizer.assert_called_once_with(model_path, trust_remote_code=True)
        mock_auto_model.assert_called_once_with(model_path, trust_remote_code=True)

if __name__ == '__main__':
    unittest.main() 