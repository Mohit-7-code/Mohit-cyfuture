import os
from scripts.data_scraper import scrape_twitter_data
from scripts.preprocess import preprocess_data
from scripts.train_model import train_lstm_model

def validate_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

if __name__ == "__main__":
    try:
        print("Step 1: Scraping Twitter Data...")
        scrape_twitter_data(keyword="#example", max_tweets=500)
        validate_file("data/raw/raw_data.csv")
        
        print("Step 2: Preprocessing Data...")
        preprocess_data("data/raw/raw_data.csv", "data/processed/processed_data.csv")
        validate_file("data/processed/processed_data.csv")
        
        print("Step 3: Training LSTM Model...")
        train_lstm_model("data/processed/processed_data.csv")
        
        print("All steps completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
