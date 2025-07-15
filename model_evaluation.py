import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config

class ModelEvaluator:
    """Comprehensive model evaluation and testing suite"""
    
    def __init__(self, model_path):
        self.config = Config()
        self.model_path = Path(model_path)
        
        # Load model and tokenizer
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Use CPU to match training device
        self.device = torch.device('cpu')
        self.model.to(self.device)
        print(f"Device set to use: {self.device}")
        
        # Create sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=-1  # Force CPU usage
        )
        
        # Load evaluation results if available
        self.eval_results = self._load_evaluation_results()
    
    def _load_evaluation_results(self):
        """Load saved evaluation results"""
        eval_path = self.model_path / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                return json.load(f)
        return None
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text with detailed output"""
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move inputs to CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': text,
            'sentiment': self.config.SENTIMENT_MAP[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            },
            'predicted_class': predicted_class
        }
    
    def batch_predict(self, texts, batch_size=32):
        """Predict sentiment for a batch of texts"""
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                pred_class = predictions[j].item()
                confidence = probabilities[j][pred_class].item()
                
                results.append({
                    'text': text,
                    'sentiment': self.config.SENTIMENT_MAP[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        'negative': probabilities[j][0].item(),
                        'neutral': probabilities[j][1].item(),
                        'positive': probabilities[j][2].item()
                    },
                    'predicted_class': pred_class
                })
        
        return results
    
    def benchmark_performance(self, test_texts, true_labels=None):
        """Benchmark model performance on speed and accuracy"""
        
        print("Benchmarking model performance...")
        
        # Speed benchmark
        start_time = time.time()
        results = self.batch_predict(test_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_sample = total_time / len(test_texts)
        
        print(f"Performance Metrics:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per sample: {avg_time_per_sample:.4f} seconds")
        print(f"Throughput: {len(test_texts) / total_time:.1f} samples/second")
        
        # Accuracy benchmark if labels provided
        if true_labels:
            predictions = [r['predicted_class'] for r in results]
            accuracy = accuracy_score(true_labels, predictions)
            print(f"Accuracy on test set: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(
                true_labels, 
                predictions,
                target_names=['Negative', 'Neutral', 'Positive']
            ))
        
        return results
    
    def test_edge_cases(self):
        """Test model on edge cases and challenging examples"""
        
        print("\n=== Testing Edge Cases ===")
        
        edge_cases = {
            "Very Short": [
                "Good",
                "Bad", 
                "OK",
                "Meh"
            ],
            "Sarcasm/Irony": [
                "Oh great, another broken product. Just what I needed.",
                "Wow, this is exactly as terrible as I expected.",
                "Perfect! It broke immediately. Money well spent.",
                "Amazing quality! It lasted a whole 5 minutes."
            ],
            "Mixed Sentiment": [
                "The product quality is great but the shipping was terrible.",
                "Love the design but hate the price.",
                "Fast shipping but poor quality materials.",
                "Good value but terrible customer service."
            ],
            "Neutral/Ambiguous": [
                "I received the product.",
                "It exists and does things.",
                "This is a product that I purchased.",
                "It arrived in a box."
            ],
            "Extreme Positive": [
                "ABSOLUTELY AMAZING! BEST PRODUCT EVER! 10/10 WOULD RECOMMEND!!!",
                "This changed my life! I can't live without it!",
                "Perfect in every single way! Exceeded all expectations!",
                "Outstanding quality! Worth every penny! LOVE IT!"
            ],
            "Extreme Negative": [
                "WORST PRODUCT EVER! COMPLETE WASTE OF MONEY!",
                "Absolutely terrible! Don't waste your time!",
                "Horrible quality! Broke immediately! AVOID!",
                "Completely useless! Worst purchase of my life!"
            ]
        }
        
        all_results = {}
        
        for category, texts in edge_cases.items():
            print(f"\n{category} Examples:")
            category_results = []
            
            for text in texts:
                result = self.predict_sentiment(text)
                category_results.append(result)
                
                print(f"Text: {text}")
                print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
                print(f"Probabilities: Neg: {result['probabilities']['negative']:.3f}, "
                      f"Neu: {result['probabilities']['neutral']:.3f}, "
                      f"Pos: {result['probabilities']['positive']:.3f}")
                print("-" * 50)
            
            all_results[category] = category_results
        
        return all_results
    
    def analyze_confidence_distribution(self, test_data):
        """Analyze confidence score distribution"""
        
        print("\n=== Confidence Distribution Analysis ===")
        
        if isinstance(test_data, list) and isinstance(test_data[0], str):
            # If test_data is list of texts, predict first
            results = self.batch_predict(test_data)
        else:
            # Assume test_data is already results
            results = test_data
        
        confidences = [r['confidence'] for r in results]
        sentiments = [r['sentiment'] for r in results]
        
        # Overall confidence statistics
        print(f"Confidence Statistics:")
        print(f"Mean confidence: {np.mean(confidences):.3f}")
        print(f"Median confidence: {np.median(confidences):.3f}")
        print(f"Min confidence: {np.min(confidences):.3f}")
        print(f"Max confidence: {np.max(confidences):.3f}")
        print(f"Std deviation: {np.std(confidences):.3f}")
        
        # Confidence by sentiment
        df = pd.DataFrame({
            'sentiment': sentiments,
            'confidence': confidences
        })
        
        print(f"\nConfidence by sentiment:")
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            sentiment_confidences = df[df['sentiment'] == sentiment]['confidence']
            if len(sentiment_confidences) > 0:
                print(f"{sentiment}: Mean={sentiment_confidences.mean():.3f}, "
                      f"Std={sentiment_confidences.std():.3f}")
        
        # Plot confidence distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Overall confidence histogram
        axes[0].hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Confidence Score Distribution')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by sentiment
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            sentiment_confidences = df[df['sentiment'] == sentiment]['confidence']
            if len(sentiment_confidences) > 0:
                axes[1].hist(sentiment_confidences, bins=15, alpha=0.6, 
                           label=sentiment, edgecolor='black')
        
        axes[1].set_title('Confidence Distribution by Sentiment')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Identify low confidence predictions
        low_confidence_threshold = 0.6
        low_confidence_predictions = [r for r in results if r['confidence'] < low_confidence_threshold]
        
        print(f"\nLow confidence predictions (< {low_confidence_threshold}):")
        print(f"Count: {len(low_confidence_predictions)} / {len(results)} ({len(low_confidence_predictions)/len(results)*100:.1f}%)")
        
        if low_confidence_predictions:
            print("\nSample low confidence predictions:")
            for i, pred in enumerate(low_confidence_predictions[:5]):
                print(f"{i+1}. Text: {pred['text'][:80]}...")
                print(f"   Prediction: {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
        
        return {
            'mean_confidence': np.mean(confidences),
            'confidence_by_sentiment': df.groupby('sentiment')['confidence'].agg(['mean', 'std']).to_dict(),
            'low_confidence_count': len(low_confidence_predictions),
            'low_confidence_rate': len(low_confidence_predictions) / len(results)
        }
    
    def compare_with_baseline(self, test_texts, true_labels=None):
        """Compare model performance with simple baselines"""
        
        print("\n=== Baseline Comparison ===")
        
        # Get model predictions
        model_results = self.batch_predict(test_texts)
        model_predictions = [r['predicted_class'] for r in model_results]
        
        if true_labels:
            model_accuracy = accuracy_score(true_labels, model_predictions)
            print(f"Model Accuracy: {model_accuracy:.4f}")
            
            # Random baseline
            random_predictions = np.random.choice([0, 1, 2], size=len(test_texts))
            random_accuracy = accuracy_score(true_labels, random_predictions)
            print(f"Random Baseline: {random_accuracy:.4f}")
            
            # Majority class baseline
            majority_class = max(set(true_labels), key=true_labels.count)
            majority_predictions = [majority_class] * len(test_texts)
            majority_accuracy = accuracy_score(true_labels, majority_predictions)
            print(f"Majority Class Baseline: {majority_accuracy:.4f}")
            
            # Simple keyword-based baseline
            keyword_predictions = self._keyword_baseline(test_texts)
            keyword_accuracy = accuracy_score(true_labels, keyword_predictions)
            print(f"Keyword Baseline: {keyword_accuracy:.4f}")
            
            # Improvement over baselines
            print(f"\nImprovement over baselines:")
            print(f"vs Random: +{(model_accuracy - random_accuracy)*100:.2f}%")
            print(f"vs Majority: +{(model_accuracy - majority_accuracy)*100:.2f}%")
            print(f"vs Keywords: +{(model_accuracy - keyword_accuracy)*100:.2f}%")
            
            return {
                'model_accuracy': model_accuracy,
                'random_accuracy': random_accuracy,
                'majority_accuracy': majority_accuracy,
                'keyword_accuracy': keyword_accuracy
            }
    
    def _keyword_baseline(self, texts):
        """Simple keyword-based sentiment classifier"""
        
        positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome', 'fantastic', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor', 'disappointing', 'useless', 'broken'}
        
        predictions = []
        
        for text in texts:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                predictions.append(2)  # Positive
            elif negative_count > positive_count:
                predictions.append(0)  # Negative
            else:
                predictions.append(1)  # Neutral
        
        return predictions
    
    def generate_model_report(self, test_texts=None, true_labels=None, save_path=None):
        """Generate comprehensive model evaluation report"""
        
        print("\n" + "="*60)
        print("           MODEL EVALUATION REPORT")
        print("="*60)
        
        # Model information
        print(f"\nModel Path: {self.model_path}")
        print(f"Model Type: {self.model.config.model_type}")
        print(f"Number of Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Saved evaluation results
        if self.eval_results:
            print(f"\n=== Saved Evaluation Results ===")
            print(f"Test Accuracy: {self.eval_results['accuracy']:.4f}")
            print(f"Macro F1-Score: {self.eval_results['macro_f1']:.4f}")
            
            print(f"\nPer-class Performance:")
            for sentiment, metrics in self.eval_results['per_class_metrics'].items():
                print(f"{sentiment.capitalize():8} - Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        report_data = {}
        
        # Performance benchmarking
        if test_texts:
            print(f"\n=== Performance Benchmarking ===")
            benchmark_results = self.benchmark_performance(test_texts, true_labels)
            report_data['benchmark'] = benchmark_results
            
            # Confidence analysis
            confidence_analysis = self.analyze_confidence_distribution(benchmark_results)
            report_data['confidence_analysis'] = confidence_analysis
            
            # Baseline comparison
            if true_labels:
                baseline_comparison = self.compare_with_baseline(test_texts, true_labels)
                report_data['baseline_comparison'] = baseline_comparison
        
        # Edge case testing
        edge_case_results = self.test_edge_cases()
        report_data['edge_cases'] = edge_case_results
        
        # Save report
        if save_path:
            report_path = Path(save_path)
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_path}")
        
        print("\n" + "="*60)
        print("           REPORT COMPLETE")
        print("="*60)
        
        return report_data
    
    def interactive_testing(self):
        """Interactive mode for testing custom inputs"""
        
        print("\n=== Interactive Testing Mode ===")
        print("Enter reviews to test sentiment analysis (type 'quit' to exit):")
        
        while True:
            user_input = input("\nEnter review: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Predict sentiment
            result = self.predict_sentiment(user_input)
            
            print(f"\nResults:")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Detailed probabilities:")
            for sentiment, prob in result['probabilities'].items():
                print(f"  {sentiment.capitalize()}: {prob:.3f}")
        
        print("Interactive testing ended.")

def main():
    """Main evaluation pipeline"""
    
    # Specify model path - update this to your trained model
    model_path = "./models/simple_sentiment_model"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        print("Please train a model first using the training script.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Sample test data for demonstration
    test_reviews = [
        "This product is absolutely fantastic! I love everything about it.",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It's okay, nothing special but works as expected.",
        "Amazing customer service and fast shipping. Highly recommend!",
        "Poor quality materials, not worth the price at all.",
        "Good value for money, satisfied with my purchase.",
        "Excellent product, exceeded my expectations completely!",
        "Disappointing experience, product doesn't match description.",
        "Average product, meets basic requirements but nothing more.",
        "Outstanding quality and design, will definitely buy again!"
    ]
    
    # Corresponding true labels for accuracy testing (0=Negative, 1=Neutral, 2=Positive)
    true_labels = [2, 0, 1, 2, 0, 2, 2, 0, 1, 2]
    
    # Generate comprehensive evaluation report
    report = evaluator.generate_model_report(
        test_texts=test_reviews,
        true_labels=true_labels,
        save_path="model_evaluation_report.json"
    )
    
    # Interactive testing
    print("\nWould you like to try interactive testing? (y/n)")
    if input().lower().startswith('y'):
        evaluator.interactive_testing()

if __name__ == "__main__":
    main()