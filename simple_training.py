import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_loader import DataLoader as CustomDataLoader

class SimpleDataset(Dataset):
    """Simple PyTorch dataset for sentiment analysis"""
    
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
        
        # Tokenize
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

class SimpleSentimentTrainer:
    """Simple sentiment trainer without HuggingFace Trainer"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def prepare_data(self, df):
        """Prepare train/validation/test datasets"""
        
        print("Preparing datasets...")
        
        # Split data
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df['sentiment']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment']
        )
        
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        # Create datasets
        train_dataset = SimpleDataset(
            train_df['review_text'].tolist(),
            train_df['sentiment'].tolist(),
            self.tokenizer
        )
        
        val_dataset = SimpleDataset(
            val_df['review_text'].tolist(),
            val_df['sentiment'].tolist(),
            self.tokenizer
        )
        
        test_dataset = SimpleDataset(
            test_df['review_text'].tolist(),
            test_df['sentiment'].tolist(),
            self.tokenizer
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, train_dataset, val_dataset, 
                   num_epochs=3, batch_size=8, learning_rate=2e-5):
        """Train the model"""
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print("Starting training...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch in train_pbar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss' : f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save model
        model_dir = self.config.MODEL_DIR / "simple_sentiment_model"
        model_dir.mkdir(exist_ok=True)
        
        self.model.save_pretrained(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))
        
        # Save training history
        with open(model_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nModel saved to: {model_dir}")
        return model_dir
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def test_model(self, test_dataset):
        """Test the final model"""
        
        print("\nTesting model...")
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, 
            all_predictions,
            target_names=['Negative', 'Neutral', 'Positive']
        )
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        return accuracy, all_predictions, all_labels
    
    def plot_training_curves(self):
        """Plot training curves"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.history['val_accuracy'], 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_text(self, text, model_path=None):
        """Predict sentiment for a single text"""
        
        if model_path:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.to(self.device)
        else:
            model = self.model
            tokenizer = self.tokenizer
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
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
    """Main training function"""
    
    # Load data
    loader = CustomDataLoader()
    df = loader.load_processed_data()
    
    if df is None:
        print("No processed data found. Please run data_loader.py first.")
        return
    
    print(f"Total data available: {len(df)} samples")
    
    # Initialize trainer
    trainer = SimpleSentimentTrainer()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(df)
    
    # Train model
    model_dir = trainer.train_model(
        train_dataset, 
        val_dataset, 
        num_epochs=3,
        batch_size=4,  # Smaller batch size for small dataset
        learning_rate=2e-5
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Test model
    accuracy, predictions, labels = trainer.test_model(test_dataset)
    
    # Test some predictions
    print("\n=== Sample Predictions ===")
    test_texts = [
        "This product is amazing! Love it!",
        "Terrible quality, don't buy this.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        result = trainer.predict_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()
