import pandas as pd
import json

# Load business data to filter restaurants (smaller file, fits in memory)
businesses = pd.read_json('yelp_academic_dataset_business.json', lines=True)
restaurants = businesses[businesses['categories'].str.contains('Restaurants', na=False)]
restaurant_ids = set(restaurants['business_id'])

# Process reviews in chunks
chunk_size = 10000  # Adjust based on memory
reviews = []
with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line.strip())
        if review['business_id'] in restaurant_ids:
            reviews.append({
                'Title': review['text'][:50],  # First 50 chars as title
                'Review': review['text'],
                'Rating': review['stars'],
                'Date': review['date']
            })
        # Process chunk when it reaches chunk_size
        if len(reviews) >= chunk_size:
            df_chunk = pd.DataFrame(reviews)
            reviews = []  # Clear memory
            if len(reviews) == 0:  # First chunk
                df_chunk.to_csv('yelp_reviews.csv', index=False, mode='w')
            else:
                df_chunk.to_csv('yelp_reviews.csv', index=False, mode='a', header=False)

# Process any remaining reviews
if reviews:
    df_chunk = pd.DataFrame(reviews)
    df_chunk.to_csv('yelp_reviews.csv', index=False, mode='a', header=False)

# Sample 10,000 reviews
df = pd.read_csv('yelp_reviews.csv')
df = df.sample(n=10000, random_state=42) if len(df) > 10000 else df
df.to_csv('yelp_reviews.csv', index=False)
print(f"Generated yelp_reviews.csv with {len(df)} reviews")