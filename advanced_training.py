import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_loader import DataLoader

class AdvancedSentimentTrainer:
    """Advanced training class with comprehensive monitoring and evaluation"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.config = Config()
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Training metrics storage
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': [],
            'learning_rates': []
        }
        
        # Model performance tracking
        self.best_metrics = {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'epoch': 0
        }
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        
        # Store metrics for tracking
        self.training_history['eval_accuracy'].append(accuracy)
        self.training_history['eval_f1'].append(f1)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples['review_text'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors=None
        )
    
    def prepare_datasets(self, df):
        """Prepare train/validation/test datasets"""
        
        print("Preparing datasets...")
        
        # Split into train and temp (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=self.config.RANDOM_STATE,
            stratify=df['sentiment']
        )
        
        # Split temp into validation and test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,  # 15% of total data for test, 15% for validation
            random_state=self.config.RANDOM_STATE,
            stratify=temp_df['sentiment']
        )
        
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples") 
        print(f"Test: {len(test_df)} samples")
        
        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(train_df[['review_text', 'sentiment']])
        val_dataset = Dataset.from_pandas(val_df[['review_text', 'sentiment']])
        test_dataset = Dataset.from_pandas(test_df[['review_text', 'sentiment']])
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # Rename columns for HuggingFace
        train_dataset = train_dataset.rename_column('sentiment', 'labels')
        val_dataset = val_dataset.rename_column('sentiment', 'labels')
        test_dataset = test_dataset.rename_column('sentiment', 'labels')
        
        # Set format
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, train_dataset, val_dataset, 
                   num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the sentiment analysis model with advanced monitoring"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.MODEL_DIR / f"sentiment_model_{timestamp}"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.config.LOGS_DIR),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            report_to=None,
            dataloader_pin_memory=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
        )
        
        # Custom callback for tracking metrics
        class MetricsCallback:
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    if 'train_loss' in logs:
                        self.trainer_instance.training_history['train_loss'].append(logs['train_loss'])
                    if 'eval_loss' in logs:
                        self.trainer_instance.training_history['eval_loss'].append(logs['eval_loss'])
                    if 'learning_rate' in logs:
                        self.trainer_instance.training_history['learning_rates'].append(logs['learning_rate'])
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                MetricsCallback(self)
            ]
        )
        
        # Start training
        print("Starting training...")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        final_model_dir = self.config.MODEL_DIR / "sentiment_model_final"
        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))
        
        # Save training history
        history_path = final_model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            history_json = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    history_json[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                else:
                    history_json[key] = value
            json.dump(history_json, f, indent=2)
        
        print(f"Model and training history saved to: {final_model_dir}")
        
        return trainer, final_model_dir
    
    def evaluate_model(self, trainer, test_dataset, model_dir=None):
        """Comprehensive model evaluation"""
        
        print("\nEvaluating model on test set...")
        
        if model_dir:
            # Load model from directory
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            model = trainer.model
            tokenizer = self.tokenizer
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
        
        # Create detailed classification report
        print("\n=== Model Performance ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        
        print("\nPer-class metrics:")
        for i, (sentiment_name) in enumerate(['Negative', 'Neutral', 'Positive']):
            print(f"{sentiment_name:8} - Precision: {precision[i]:.4f}, "
                  f"Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}, Support: {support[i]}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm)
        
        # Plot training curves
        self._plot_training_curves()
        
        # Error analysis
        self._analyze_errors(test_dataset, y_true, y_pred)
        
        # Save evaluation results
        eval_results = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'per_class_metrics': {
                'negative': {'precision': float(precision[0]), 'recall': float(recall[0]), 'f1': float(f1[0])},
                'neutral': {'precision': float(precision[1]), 'recall': float(recall[1]), 'f1': float(f1[1])},
                'positive': {'precision': float(precision[2]), 'recall': float(recall[2]), 'f1': float(f1[2])}
            },
            'confusion_matrix': cm.tolist(),
            'support': support.tolist()
        }
        
        if model_dir:
            eval_path = Path(model_dir) / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def _plot_training_curves(self):
        """Plot training curves"""
        if not self.training_history['train_loss']:
            print("No training history available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        if self.training_history['train_loss'] and self.training_history['eval_loss']:
            axes[0, 0].plot(self.training_history['train_loss'], label='Training Loss', color='blue')
            axes[0, 0].plot(self.training_history['eval_loss'], label='Validation Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Validation accuracy
        if self.training_history['eval_accuracy']:
            axes[0, 1].plot(self.training_history['eval_accuracy'], label='Validation Accuracy', color='green')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Evaluation Steps')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Validation F1 score
        if self.training_history['eval_f1']:
            axes[1, 0].plot(self.training_history['eval_f1'], label='Validation F1', color='purple')
            axes[1, 0].set_title('Validation F1 Score')
            axes[1, 0].set_xlabel('Evaluation Steps')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if self.training_history['learning_rates']:
            axes[1, 1].plot(self.training_history['learning_rates'], label='Learning Rate', color='orange')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_errors(self, test_dataset, y_true, y_pred):
        """Analyze prediction errors"""
        
        # Find misclassified examples
        errors = []
        correct = []
        
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            example = {
                'text': test_dataset[i]['review_text'] if 'review_text' in test_dataset[i] else "N/A",
                'true_sentiment': self.config.SENTIMENT_MAP[true_label],
                'predicted_sentiment': self.config.SENTIMENT_MAP[pred_label],
                'correct': true_label == pred_label
            }
            
            if true_label == pred_label:
                correct.append(example)
            else:
                errors.append(example)
        
        print(f"\n=== Error Analysis ===")
        print(f"Correct predictions: {len(correct)}")
        print(f"Incorrect predictions: {len(errors)}")
        print(f"Error rate: {len(errors) / (len(correct) + len(errors)):.3f}")
        
        # Show sample errors
        print("\nSample misclassified reviews:")
        for i, error in enumerate(errors[:5]):  # Show first 5 errors
            print(f"\n{i+1}. True: {error['true_sentiment']}, Predicted: {error['predicted_sentiment']}")
            print(f"   Text: {error['text'][:100]}...")
    
    def predict_single(self, text, model_path=None):
        """Predict sentiment for a single text"""
        
        if model_path:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model = self.model
            tokenizer = self.tokenizer
        
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
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
            }
        }

def main():
    """Main training pipeline"""
    
    # Load data
    loader = DataLoader()
    df = loader.load_processed_data()
    
    if df is None:
        # Load and preprocess data if not available
        df = loader.load_amazon_reviews(sample_size=5000)  # Adjust sample size as needed
        df = loader.preprocess_data(df)
        loader.save_processed_data(df)
    
    print(f"\nTotal data available: {len(df)} samples")
    
    # Initialize trainer
    trainer = AdvancedSentimentTrainer()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(df)
    
    # Train model
    trained_model, model_dir = trainer.train_model(
        train_dataset, 
        val_dataset,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Evaluate model
    eval_results = trainer.evaluate_model(trained_model, test_dataset, model_dir)
    
    # Test single predictions
    print("\n=== Testing Single Predictions ===")
    test_texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality, completely useless. Don't buy this.",
        "It's okay, nothing special but does what it's supposed to do."
    ]
    
    for text in test_texts:
        result = trainer.predict_single(text, model_dir)
        print(f"\nText: {text}")
        print(f"Predicted: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: {result['probabilities']}")
    
    print(f"\nModel training completed! Final model saved at: {model_dir}")

if __name__ == "__main__":
    main()
