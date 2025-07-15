import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from config import Config

class DataLoader:
    """Load and preprocess Amazon review data"""
    
    def __init__(self):
        self.config = Config()
    
    def load_amazon_reviews(self, dataset_name="amazon_reviews_multi", language="en", 
                           sample_size=None, product_categories=None):
        """
        Load Amazon reviews from HuggingFace datasets
        
        Args:
            dataset_name: Name of the dataset
            language: Language code (en, de, es, fr, etc.)
            sample_size: Number of samples to load (None for all)
            product_categories: List of product categories to include
        """
        
        try:
            print(f"Loading {dataset_name} dataset...")
            
            # Load the dataset
            dataset = load_dataset(dataset_name, language)
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset['train'])
            
            print(f"Original dataset size: {len(df)}")
            
            # Filter by product categories if specified
            if product_categories:
                df = df[df['product_category'].isin(product_categories)]
                print(f"After category filtering: {len(df)}")
            
            # Sample data if specified
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=self.config.RANDOM_STATE)
                print(f"After sampling: {len(df)}")
            
            # Select relevant columns
            columns_to_keep = ['review_body', 'review_title', 'stars', 'product_category', 'verified_purchase']
            df = df[columns_to_keep]
            
            # Combine title and body
            df['review_text'] = df['review_title'].fillna('') + ' ' + df['review_body'].fillna('')
            
            return df
            
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            return self._load_sample_data()
    
    def _load_sample_data(self):
        """Create sample data for demonstration"""
        
        print("Creating sample Amazon review data...")
        
        sample_reviews = [
            # Positive reviews (5 stars)
            "This product exceeded my expectations! Amazing quality and fast shipping.",
            "Absolutely love this! Best purchase I've made in years. Highly recommend!",
            "Outstanding quality, exactly as described. Will definitely buy again!",
            "Perfect! Works exactly as expected. Great value for money.",
            "Excellent product! My family loves it. Five stars!",
            
            # Positive reviews (4 stars)
            "Really good product, minor issues but overall satisfied.",
            "Good quality, fast delivery. Worth the price.",
            "Nice product, works well. Small packaging issue but content is great.",
            "Solid product, meets expectations. Would recommend.",
            "Good purchase, happy with the quality.",
            
            # Neutral reviews (3 stars)
            "It's okay, nothing special but does the job.",
            "Average product, meets basic expectations.",
            "Not bad, not great either. Just okay.",
            "Decent quality, could be better for the price.",
            "It works, but there are better options available.",
            
            # Negative reviews (2 stars)
            "Disappointed with the quality. Expected better.",
            "Poor quality materials, not worth the money.",
            "Product doesn't match the description. Misleading.",
            "Cheap feeling, wouldn't buy again.",
            "Below average quality, has some issues.",
            
            # Negative reviews (1 star)
            "Terrible quality! Complete waste of money!",
            "Broke after one use. Do not buy this!",
            "Worst purchase ever. Completely useless.",
            "Defective product, doesn't work at all.",
            "Horrible quality, terrible customer service."
        ]
        
        ratings = [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        categories = ['Electronics'] * 25
        verified = [True] * 25
        
        # Replicate to create larger dataset
        multiplier = 400
        sample_reviews = sample_reviews * multiplier
        ratings = ratings * multiplier
        categories = categories * multiplier
        verified = verified * multiplier
        
        df = pd.DataFrame({
            'review_text': sample_reviews,
            'stars': ratings,
            'product_category': categories,
            'verified_purchase': verified
        })
        
        return df
    
    def clean_text(self, text):
        """Clean review text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def convert_rating_to_sentiment(self, rating):
        """Convert star rating to sentiment label"""
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive
    
    def preprocess_data(self, df):
        """Preprocess the loaded data"""
        
        print("Preprocessing data...")
        
        # Clean text
        df['review_text'] = df['review_text'].apply(self.clean_text)
        
        # Convert ratings to sentiment labels
        df['sentiment'] = df['stars'].apply(self.convert_rating_to_sentiment)
        
        # Remove empty reviews
        df = df[df['review_text'].str.len() > 0]
        
        # Remove very short reviews (less than 10 characters)
        df = df[df['review_text'].str.len() >= 10]
        
        # Remove very long reviews (over 1000 characters) to avoid memory issues
        df = df[df['review_text'].str.len() <= 1000]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['review_text'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print(f"Data after preprocessing: {len(df)} samples")
        
        return df
    
    def analyze_data(self, df):
        """Analyze the dataset and create visualizations"""
        
        print("\n=== Dataset Analysis ===")
        print(f"Total samples: {len(df)}")
        print(f"Unique reviews: {df['review_text'].nunique()}")
        
        # Sentiment distribution
        print("\nSentiment distribution:")
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        for sentiment, count in sentiment_counts.items():
            sentiment_name = self.config.SENTIMENT_MAP[sentiment]
            percentage = (count / len(df)) * 100
            print(f"{sentiment_name}: {count} ({percentage:.1f}%)")
        
        # Rating distribution
        print("\nRating distribution:")
        rating_counts = df['stars'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{rating} stars: {count} ({percentage:.1f}%)")
        
        # Text length statistics
        df['text_length'] = df['review_text'].str.len()
        print(f"\nText length statistics:")
        print(f"Mean: {df['text_length'].mean():.1f} characters")
        print(f"Median: {df['text_length'].median():.1f} characters")
        print(f"Min: {df['text_length'].min()} characters")
        print(f"Max: {df['text_length'].max()} characters")
        
        # Create visualizations
        self._create_visualizations(df)
        
        return df
    
    def _create_visualizations(self, df):
        """Create data visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        sentiment_labels = [self.config.SENTIMENT_MAP[i] for i in sentiment_counts.index]
        
        axes[0, 0].bar(sentiment_labels, sentiment_counts.values, color=['red', 'gray', 'green'])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # Rating distribution
        rating_counts = df['stars'].value_counts().sort_index()
        axes[0, 1].bar(rating_counts.index, rating_counts.values, color='skyblue')
        axes[0, 1].set_title('Rating Distribution')
        axes[0, 1].set_xlabel('Stars')
        axes[0, 1].set_ylabel('Count')
        
        # Text length distribution
        axes[1, 0].hist(df['text_length'], bins=50, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Text Length Distribution')
        axes[1, 0].set_xlabel('Character Count')
        axes[1, 0].set_ylabel('Frequency')
        
        # Sentiment vs Rating heatmap
        sentiment_rating_crosstab = pd.crosstab(df['sentiment'], df['stars'])
        sns.heatmap(sentiment_rating_crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Sentiment vs Rating Heatmap')
        axes[1, 1].set_xlabel('Stars')
        axes[1, 1].set_ylabel('Sentiment')
        
        plt.tight_layout()
        plt.show()
    
    def save_processed_data(self, df, filename="processed_amazon_reviews.csv"):
        """Save processed data to CSV"""
        
        filepath = self.config.get_data_path(filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to: {filepath}")
        
        # Save metadata (convert numpy types to native Python types)
        metadata = {
            'total_samples': int(len(df)),
            'sentiment_distribution': {str(k): int(v) for k, v in df['sentiment'].value_counts().to_dict().items()},
            'rating_distribution': {str(k): int(v) for k, v in df['stars'].value_counts().to_dict().items()},
            'text_length_stats': {
                'mean': float(df['review_text'].str.len().mean()),
                'median': float(df['review_text'].str.len().median()),
                'min': int(df['review_text'].str.len().min()),
                'max': int(df['review_text'].str.len().max())
            }
        }
        
        metadata_path = self.config.get_data_path("metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def load_processed_data(self, filename="processed_amazon_reviews.csv"):
        """Load previously processed data"""
        
        filepath = self.config.get_data_path(filename)
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"Loaded processed data from: {filepath}")
            return df
        else:
            print(f"No processed data found at: {filepath}")
            return None

def main():
    """Main function to load and preprocess data"""
    
    # Initialize data loader
    loader = DataLoader()
    
    # Try to load processed data first
    df = loader.load_processed_data()
    
    if df is None:
        # Load raw data
        df = loader.load_amazon_reviews(
            sample_size=10000,  # Adjust based on your needs
            product_categories=['Electronics', 'Books', 'Clothing', 'Home']
        )
        
        # Preprocess data
        df = loader.preprocess_data(df)
        
        # Save processed data
        loader.save_processed_data(df)
    
    # Analyze data
    df = loader.analyze_data(df)
    
    # Display sample reviews
    print("\n=== Sample Reviews ===")
    for sentiment in [0, 1, 2]:
        sentiment_name = loader.config.SENTIMENT_MAP[sentiment]
        sample = df[df['sentiment'] == sentiment].sample(n=2, random_state=42)
        print(f"\n{sentiment_name} Reviews:")
        for idx, row in sample.iterrows():
            print(f"★ {row['stars']}: {row['review_text'][:100]}...")
    
    return df

if __name__ == "__main__":
    main()