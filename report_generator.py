import json
import os
from datetime import datetime

def generate_report(data_file, output_file="market_report.txt"):
    """
    Processes the cleaned data from the processor and generates a formatted report.
    """
    print(f"[{datetime.now()}] Generating final market intelligence report...")
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        report_content = "=== MARKET INTELLIGENCE REPORT ===\n"
        report_content += f"Generated on: {datetime.now()}\n"
        report_content += "----------------------------------\n\n"
        
        if not data:
            report_content += "No data available to report.\n"
        else:
            for item in data:
                report_content += f"Entity: {item.get('name', 'Unknown')}\n"
                report_content += f"Sentiment: {item.get('sentiment', 'N/A')}\n"
                report_content += f"Summary: {item.get('summary', 'N/A')}\n"
                report_content += "----------------------------------\n"
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"Report successfully written to {output_file}")
        return True
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

if __name__ == "__main__":
    # Mock data for standalone testing if processor hasn't finished
    test_data = "processed_data.json"
    if not os.path.exists(test_data):
        with open(test_data, 'w') as f:
            json.dump([{"name": "Test Corp", "sentiment": "Positive", "summary": "Growth expected."}], f)
    
    generate_report(test_data)