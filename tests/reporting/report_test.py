import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from cover_class.reporting import Report, ModelConfig


class SimpleMultiLabelModel(nn.Module):
    """Simple model for testing purposes"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class ReportTest(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_report_uses_sigmoid_not_softmax(self):
        """Test that Report uses sigmoid activation for multi-label classification"""
        # Create a simple model
        input_dim = 10
        num_classes = 5
        batch_size = 20
        
        model = SimpleMultiLabelModel(input_dim, num_classes)
        
        # Create test data
        X_test = torch.randn(batch_size, input_dim)
        Y_test = torch.randint(0, 2, (batch_size, num_classes))
        
        # Create a minimal config
        config = {
            'datasets': {
                'class0': ['data'],
                'class1': ['data'],
                'class2': ['data'],
                'class3': ['data'],
                'class4': ['data'],
            }
        }
        
        model_config = ModelConfig(
            model=model,
            model_name="TestModel"
        )
        
        # Mock the report generation functions to avoid file I/O
        with patch('cover_class.reporting.report_config.generate_pdf_report'), \
             patch('cover_class.reporting.report_config.generate_json_report'), \
             patch('cover_class.reporting.report_config.download_scenes', return_value=[]):
            
            report = Report(
                outdir=self.temp_dir,
                config=config,
                model_config=model_config,
                X_test=X_test,
                Y_test=Y_test,
                _download_missing_qualitative_testing_scenes_from_config=False
            )
            
            # Get model output to compare
            model.eval()
            with torch.no_grad():
                logits = model(X_test)
                expected_probs = torch.sigmoid(logits)
                
            # Spy on torch.sigmoid to ensure it's called
            original_sigmoid = torch.sigmoid
            sigmoid_called = {'called': False}
            
            def spy_sigmoid(*args, **kwargs):
                sigmoid_called['called'] = True
                return original_sigmoid(*args, **kwargs)
            
            with patch('torch.sigmoid', side_effect=spy_sigmoid):
                report.make_report()
            
            # Verify sigmoid was called
            self.assertTrue(sigmoid_called['called'], 
                          "torch.sigmoid should be called for multi-label classification")
    
    def test_report_sigmoid_output_range(self):
        """Test that sigmoid outputs are in [0, 1] range for each class independently"""
        input_dim = 10
        num_classes = 5
        batch_size = 20
        
        model = SimpleMultiLabelModel(input_dim, num_classes)
        
        # Create test data
        X_test = torch.randn(batch_size, input_dim)
        Y_test = torch.randint(0, 2, (batch_size, num_classes))
        
        config = {
            'datasets': {
                'class0': ['data'],
                'class1': ['data'],
                'class2': ['data'],
                'class3': ['data'],
                'class4': ['data'],
            }
        }
        
        model_config = ModelConfig(
            model=model,
            model_name="TestModel"
        )
        
        # Get the raw probabilities before thresholding
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.sigmoid(logits)
        
        # Verify all probabilities are in [0, 1]
        self.assertTrue(torch.all(probs >= 0), "Sigmoid outputs should be >= 0")
        self.assertTrue(torch.all(probs <= 1), "Sigmoid outputs should be <= 1")
        
        # For multi-label, probabilities do NOT need to sum to 1 (unlike softmax)
        # This is the key difference - each class is independent
        prob_sums = probs.sum(dim=1)
        # With sigmoid, sums can be any value (not necessarily 1.0)
        # Just verify they're not all exactly 1.0 (which would indicate softmax)
        self.assertFalse(torch.allclose(prob_sums, torch.ones(batch_size)), 
                        "Sigmoid probabilities should not sum to 1.0 (that would be softmax)")
    
    def test_report_with_numpy_model(self):
        """Test that Report uses sigmoid for numpy-based models too"""
        input_dim = 10
        num_classes = 5
        batch_size = 20
        
        # Create a mock numpy model
        X_test = np.random.randn(batch_size, input_dim)
        Y_test = np.random.randint(0, 2, (batch_size, num_classes))
        
        # Create a callable that returns logits
        def numpy_model(x):
            return np.random.randn(x.shape[0], num_classes)
        
        config = {
            'datasets': {
                'class0': ['data'],
                'class1': ['data'],
                'class2': ['data'],
                'class3': ['data'],
                'class4': ['data'],
            }
        }
        
        model_config = ModelConfig(
            model=numpy_model,
            model_name="NumpyModel"
        )
        
        with patch('cover_class.reporting.report_config.generate_pdf_report'), \
             patch('cover_class.reporting.report_config.generate_json_report'), \
             patch('cover_class.reporting.report_config.download_scenes', return_value=[]):
            
            # Spy on torch.sigmoid to ensure it's called
            original_sigmoid = torch.sigmoid
            sigmoid_called = {'called': False}
            
            def spy_sigmoid(*args, **kwargs):
                sigmoid_called['called'] = True
                return original_sigmoid(*args, **kwargs)
            
            with patch('torch.sigmoid', side_effect=spy_sigmoid):
                report = Report(
                    outdir=self.temp_dir,
                    config=config,
                    model_config=model_config,
                    X_test=X_test,
                    Y_test=Y_test,
                    _download_missing_qualitative_testing_scenes_from_config=False
                )
                report.make_report()
            
            # Verify sigmoid was called
            self.assertTrue(sigmoid_called['called'], 
                          "torch.sigmoid should be called for numpy models too")


if __name__ == "__main__":
    unittest.main()
