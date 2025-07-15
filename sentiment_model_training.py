import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset as HFDataset
import warnings
warnings.filterwarnings('ignore')

class AmazonReviewDataset:
    """Custom dataset class for Amazon reviews"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalysisModel:
    """Main model class for sentiment analysis"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_and_preprocess_data(self, data_path=None):
        """Load and preprocess Amazon review data"""
        
        # For demonstration, we'll create sample data
        # In practice, load from Amazon review dataset
        if data_path is None:
            # Sample data - replace with actual Amazon dataset
            reviews = [
                "This product is amazing! Best purchase ever!",
                "Terrible quality, waste of money",
                "It's okay, nothing special but works",
                "Love it! Highly recommend to everyone",
                "Poor quality, broke after one use",
                "Good value for money, satisfied",
                "Excellent product, fast shipping",
                "Not worth the price, disappointed",
                "Average product, meets expectations",
                "Outstanding quality, will buy again!"
            ] * 1000  # Simulate larger dataset
            
            ratings = [5, 1, 3, 5, 1, 4, 5, 2, 3, 5] * 1000
            
            df = pd.DataFrame({
                'review_text': reviews,
                'rating': ratings
            })
        else:
            df = pd.read_csv(data_path)
        
        # Convert ratings to sentiment labels
        # 1-2: Negative (0), 3: Neutral (1), 4-5: Positive (2)
        df['sentiment'] = df['rating'].apply(self._rating_to_sentiment)
        
        # Clean text
        df['review_text'] = df['review_text'].apply(self._clean_text)
        
        # Remove any rows with missing data
        df = df.dropna()
        
        return df
    
    def _rating_to_sentiment(self, rating):
        """Convert rating to sentiment label"""
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive
    
    def _clean_text(self, text):
        """Clean review text"""
        import re
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def prepare_datasets(self, df, test_size=0.2, val_size=0.1):
        """Split data into train/val/test sets"""
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['sentiment']
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=train_val_df['sentiment']
        )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Convert to HuggingFace Dataset format
        train_dataset = HFDataset.from_pandas(train_df)
        val_dataset = HFDataset.from_pandas(val_df)
        test_dataset = HFDataset.from_pandas(test_df)
        
        return train_dataset, val_dataset, test_dataset
    
    def tokenize_function(self, examples):
        """Tokenize function for HuggingFace datasets"""
        return self.tokenizer(
            examples['review_text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    
    def train_model(self, train_dataset, val_dataset, output_dir='./sentiment_model'):
        """Train the sentiment analysis model"""
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Rename label column
        train_dataset = train_dataset.rename_column('sentiment', 'labels')
        val_dataset = val_dataset.rename_column('sentiment', 'labels')
        
        # Set format for PyTorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to=None,  # Disable wandb
            dataloader_pin_memory=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def evaluate_model(self, test_dataset, trainer=None):
        """Evaluate the trained model"""
        
        if trainer is None:
            # Load model for evaluation
            model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model')
            tokenizer = AutoTokenizer.from_pretrained('./sentiment_model')
        else:
            model = trainer.model
            tokenizer = self.tokenizer
        
        # Tokenize test dataset
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.rename_column('sentiment', 'labels')
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Make predictions
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in DataLoader(test_dataset, batch_size=16):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                     target_names=['Negative', 'Neutral', 'Positive'])
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy, report, cm
    
    def predict_sentiment(self, text, model_path='./sentiment_model'):
        """Predict sentiment for a single text"""
        
        # Load model if not already loaded
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          padding=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Map to sentiment labels
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map[predicted_class]
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': predictions[0][0].item(),
                'neutral': predictions[0][1].item(),
                'positive': predictions[0][2].item()
            }
        }

def main():
    """Main function to train and evaluate the model"""
    
    # Initialize model
    model = SentimentAnalysisModel()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = model.load_and_preprocess_data()
    
    # Show data distribution
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = model.prepare_datasets(df)
    
    # Train model
    trainer = model.train_model(train_dataset, val_dataset)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, report, cm = model.evaluate_model(test_dataset, trainer)
    
    # Test single prediction
    print("\nTesting single prediction...")
    test_text = "This product is absolutely fantastic! I love it!"
    result = model.predict_sentiment(test_text)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")

if __name__ == "__main__":
    main()