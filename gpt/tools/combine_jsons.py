#!/usr/bin/env python3
"""
Script to combine all final_query_results.json files from different papers
into a single comprehensive JSON file.

Usage:
    python combine_jsons.py

This script will:
1. Find all final_query_results.json files in paper directories
2. Parse each JSON file
3. Combine them into a single structured JSON
4. Save the combined result to combined_final_query_results.json
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any

def find_json_files(base_dir: str = "final_query_results") -> List[str]:
    """
    Find all final_query_results.json files in paper directories.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of paths to JSON files
    """
    pattern = os.path.join(base_dir, "paper*", "final_query_results.json")
    json_files = glob.glob(pattern)
    return sorted(json_files)

def parse_paper_id(file_path: str) -> str:
    """
    Extract paper ID from file path.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Paper ID (e.g., "paper2", "paper9")
    """
    # Extract paper directory name from path
    dir_name = os.path.basename(os.path.dirname(file_path))
    return dir_name

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def extract_topics_from_papers(papers_data: List[Dict]) -> List[str]:
    """
    Extract common topics from paper titles and content.
    
    Args:
        papers_data: List of paper data dictionaries
        
    Returns:
        List of identified topics
    """
    topics = set()
    
    # Common VCO-related terms
    vco_terms = [
        "VCO", "oscillator", "phase noise", "CMOS", "GHz", "MHz",
        "quadrature", "cascode", "cross-coupled", "flicker noise",
        "drain current", "trifilar", "transformer", "harmonic",
        "differential", "resonance", "tuning", "frequency"
    ]
    
    for paper in papers_data:
        if "paper_title" in paper:
            title = paper["paper_title"].lower()
            for term in vco_terms:
                if term.lower() in title:
                    topics.add(term)
    
    return sorted(list(topics))

def combine_json_files() -> Dict[str, Any]:
    """
    Combine all final_query_results.json files into a single structure.
    
    Returns:
        Combined JSON data structure
    """
    # Find all JSON files
    json_files = find_json_files()
    
    if not json_files:
        print("No final_query_results.json files found!")
        return {}
    
    print(f"Found {len(json_files)} JSON files to combine:")
    for file_path in json_files:
        print(f"  - {file_path}")
    
    # Load and process each file
    papers_data = []
    total_queries = 0
    timestamps = []
    
    for file_path in json_files:
        paper_id = parse_paper_id(file_path)
        data = load_json_file(file_path)
        
        if not data:
            continue
            
        # Extract paper information
        paper_info = {
            "paper_id": paper_id,
            "paper_title": data.get("results", [{}])[0].get("paper", "Unknown Paper") if data.get("results") else "Unknown Paper",
            "timestamp": data.get("timestamp", "unknown"),
            "query_count": len(data.get("results", [])),
            "results": data.get("results", [])
        }
        
        papers_data.append(paper_info)
        total_queries += paper_info["query_count"]
        timestamps.append(data.get("timestamp", ""))
    
    # Extract date range
    date_range = "unknown"
    if timestamps:
        # Extract dates from timestamps (format: YYYYMMDD_HHMMSS)
        dates = []
        for ts in timestamps:
            if ts and len(ts) >= 8:
                try:
                    date_part = ts[:8]  # YYYYMMDD
                    dates.append(date_part)
                except:
                    pass
        
        if dates:
            date_range = f"{min(dates)}-{max(dates)}"
    
    # Extract topics
    topics = extract_topics_from_papers(papers_data)
    
    # Create combined structure
    combined_data = {
        "combined_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_papers": len(papers_data),
        "total_queries": total_queries,
        "papers": papers_data,
        "summary": {
            "total_results": total_queries,
            "papers_covered": [paper["paper_title"] for paper in papers_data],
            "date_range": date_range,
            "topics": topics
        }
    }
    
    return combined_data

def save_combined_json(data: Dict[str, Any], output_file: str = "combined_final_query_results.json"):
    """
    Save the combined JSON data to a file.
    
    Args:
        data: Combined JSON data
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nCombined JSON saved to: {output_file}")
        print(f"Total papers: {data['total_papers']}")
        print(f"Total queries: {data['total_queries']}")
    except Exception as e:
        print(f"Error saving combined JSON: {e}")

def main():
    """
    Main function to execute the JSON combination process.
    """
    print("Starting JSON combination process...")
    
    # Combine all JSON files
    combined_data = combine_json_files()
    
    if not combined_data:
        print("No data to combine!")
        return
    
    # Save the combined result
    save_combined_json(combined_data)
    
    # Print summary
    print("\nCombination Summary:")
    print("=" * 50)
    for paper in combined_data["papers"]:
        print(f"{paper['paper_id']}: {paper['paper_title']}")
        print(f"  Queries: {paper['query_count']}")
        print(f"  Timestamp: {paper['timestamp']}")
        print()
    
    print(f"Topics identified: {', '.join(combined_data['summary']['topics'])}")

if __name__ == "__main__":
    main() 