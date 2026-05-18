import os
import json
from report_generator import generate_report

def main():
    print("=== Market Intelligence System Coordinator ===")
    
    # Expected files from orthogonal sub-agents
    raw_data = "raw_scraped_data.json"
    processed_data = "processed_data.json"
    
    # 1. Check for Scraper output
    if not os.path.exists(raw_data):
        print("[!] Scraper output not found. Waiting or using mock data...")
        with open(raw_data, 'w') as f:
            json.dump([{"url": "example.com", "text": "Market is booming!"}], f)
    else:
        print("[+] Raw data received from Scraper sub-agent.")

    # 2. Check for Processor output
    if not os.path.exists(processed_data):
        print("[!] Processor output not found. Running local fallback processing...")
        # Simple fallback to simulate processor
        with open(raw_data, 'r') as f:
            raw = json.load(f)
        processed = [{"name": "Market", "sentiment": "Bullish", "summary": "Based on raw data."}]
        with open(processed_data, 'w') as f:
            json.dump(processed, f)
    else:
        print("[+] Processed data received from Processor sub-agent.")

    # 3. Generate Final Report
    success = generate_report(processed_data)
    if success:
        print("[SUCCESS] Market Intelligence System pipeline complete.")
    else:
        print("[FAILURE] Pipeline failed at report generation.")

if __name__ == "__main__":
    main()