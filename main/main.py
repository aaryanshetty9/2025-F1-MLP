import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# Updated URLs for Australian GP data
race_url = "https://www.espn.com/f1/results/_/id/600052045"
qualifying_url = "https://www.espn.com/f1/results/_/id/600052045/type/qualifying"
practice_url = "https://www.espn.com/f1/results/_/id/600052045/type/practice1"

# Function to scrape ESPN race results
def scrape_race_results(url):
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Print the first part of the HTML to help debug
        print(f"Response status: {response.status_code}")
        print(f"URL being accessed: {url}")
        
        # Look for tables with different possible class names
        tables = soup.find_all('table')
        
        if not tables:
            print("No tables found on the page.")
            return None
        
        print(f"Found {len(tables)} tables on the page.")
        
        # Try to find a table with race results
        for i, table in enumerate(tables):
            print(f"Checking table {i+1}...")
            
            # Try to extract headers
            headers_row = table.find('tr')
            if not headers_row:
                continue
                
            headers = [th.get_text(strip=True) for th in headers_row.find_all(['th', 'td'])]
            print(f"Headers found: {headers}")
            
            # Check if this table has position and driver columns
            if 'POS' in headers and ('DRIVER' in headers or 'NAME' in headers):
                print(f"Found results table (table {i+1})!")
                
                # Extract rows
                rows = table.find_all('tr')
                
                # Extract data
                data = []
                for row in rows[1:]:  # Skip header row
                    cols = row.find_all(['td', 'th'])
                    if cols:
                        row_data = [col.get_text(strip=True) for col in cols]
                        data.append(row_data)
                
                # Create DataFrame
                df = pd.DataFrame(data, columns=headers)
                return df
        
        print("No suitable results table found.")
        return None
        
    except Exception as e:
        print(f"Error scraping results: {e}")
        return None

# Function to clean and process data
def process_data(race_df, qualifying_df, practice_df):
    if race_df is None or qualifying_df is None or practice_df is None:
        print("Missing data, cannot process.")
        return None
    
    print("\nRace data sample:")
    print(race_df.head())
    
    print("\nQualifying data sample:")
    print(qualifying_df.head())
    
    print("\nPractice data sample:")
    print(practice_df.head())
    
    # Clean position data - extract numbers only
    try:
        race_df['POS'] = race_df['POS'].str.extract(r'(\d+)').astype(float)
        qualifying_df['POS'] = qualifying_df['POS'].str.extract(r'(\d+)').astype(float)
        practice_df['POS'] = practice_df['POS'].str.extract(r'(\d+)').astype(float)
        
        # Standardize driver column name if needed
        if 'NAME' in race_df.columns and 'DRIVER' not in race_df.columns:
            race_df.rename(columns={'NAME': 'DRIVER'}, inplace=True)
        if 'NAME' in qualifying_df.columns and 'DRIVER' not in qualifying_df.columns:
            qualifying_df.rename(columns={'NAME': 'DRIVER'}, inplace=True)
        if 'NAME' in practice_df.columns and 'DRIVER' not in practice_df.columns:
            practice_df.rename(columns={'NAME': 'DRIVER'}, inplace=True)
            
        # Merge dataframes
        merged_df = pd.merge(race_df[['POS', 'DRIVER']], 
                            qualifying_df[['POS', 'DRIVER']], 
                            on='DRIVER', 
                            suffixes=('_race', '_qualifying'))
        
        merged_df = pd.merge(merged_df, 
                            practice_df[['POS', 'DRIVER']],
                            on='DRIVER')
        
        # Rename the practice position column
        merged_df.rename(columns={'POS': 'POS_practice'}, inplace=True)
        
        return merged_df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Function to create weighted features
def create_weighted_features(df):
    if df is None:
        return None
        
    # Weights based on your prioritization
    race_weight = 0.6
    qualifying_weight = 0.3
    practice_weight = 0.1
    
    # Create weighted score (lower is better for positions)
    df['weighted_score'] = (
        df['POS_race'] * race_weight +
        df['POS_qualifying'] * qualifying_weight +
        df['POS_practice'] * practice_weight
    )
    
    # Sort by weighted score
    df = df.sort_values('weighted_score')
    
    return df

# Function to predict China GP results
def predict_china_gp(model_df):
    if model_df is None:
        print("No model data available for prediction.")
        return
        
    print("\nPredicted China GP Results:")
    print("===========================")
    
    # For now, just returning the weighted prediction based on Australia
    for i, (_, row) in enumerate(model_df.iterrows(), 1):
        print(f"{i}. {row['DRIVER']}")

# Main execution
def main():
    print("Scraping F1 race data...")
    
    # Scrape data
    race_df = scrape_race_results(race_url)
    
    # Add small delays to avoid potential rate limiting
    time.sleep(1)
    qualifying_df = scrape_race_results(qualifying_url)
    
    time.sleep(1)
    practice_df = scrape_race_results(practice_url)
    
    # Process and merge data
    print("\nProcessing data...")
    merged_df = process_data(race_df, qualifying_df, practice_df)
    
    # Create weighted features
    model_df = create_weighted_features(merged_df)
    
    if model_df is not None:
        print("\nModel Data:")
        print(model_df[['DRIVER', 'POS_race', 'POS_qualifying', 'POS_practice', 'weighted_score']])
        
        # Make prediction for China GP
        predict_china_gp(model_df)
    
    print("\nNote: This prediction is based on Australian GP data.")

if __name__ == "__main__":
    main()