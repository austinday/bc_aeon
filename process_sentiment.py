import json
import os

def analyze_sentiment(text):
    positive_words = {'love', 'fantastic', 'great', 'good'}
    negative_words = {'terrible', 'waste', 'bad', 'poor'}
    text_lower = text.lower()
    score = 0
    for word in positive_words: if word in text_lower: score += 1
    for word in negative_words: if word in text_lower: score -= 1
    return score

def main():
    input_path = 'data/raw_data.json'
    output_path = 'data/processed_data.json'
    
    if not os.path.exists(input_path):
        print(f'Error: {input_path} not found')
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    results = []
    total_sentiment = 0
    
    for item in data:
        sentiment = analyze_sentiment(item['text'])
        results.append({
            'text': item['text'],
            'sentiment_score': sentiment,
            'original_score': item['score']
        })
        total_sentiment += sentiment

    avg_sentiment = total_sentiment / len(data) if data else 0
    
    final_output = {
        'processed_items': results,
        'average_sentiment': avg_sentiment
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f'Successfully processed {len(data)} items. Average sentiment: {avg_sentiment}')

if __name__ == '__main__':
    main()
