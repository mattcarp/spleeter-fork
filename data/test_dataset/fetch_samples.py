#!/usr/bin/env python
# coding: utf8

"""Script to fetch audio samples for the test dataset."""

import os
import json
import argparse
from datetime import datetime
import requests
import urllib.request
from typing import Dict, List, Optional, Tuple

# Configuration - Add your Freesound API key here
FREESOUND_API_KEY = "YOUR_API_KEY"  # Get from https://freesound.org/apiv2/apply/
MUSDB_URL = "https://zenodo.org/records/3338373"  # Public MUSDB18 repository

# Valid licenses for audio files
VALID_LICENSES = [
    "Creative Commons 0",
    "Attribution",
    "Attribution Noncommercial",
    "CC BY",
    "CC BY-NC",
    "CC BY-SA",
    "CC BY-NC-SA",
    "Public Domain"
]

def validate_license(license_name: str) -> bool:
    """
    Validate if a license is acceptable for our test dataset.
    
    Args:
        license_name: The license name to validate
        
    Returns:
        True if license is acceptable, False otherwise
    """
    return any(valid_lic.lower() in license_name.lower() for valid_lic in VALID_LICENSES)


def fetch_from_freesound(
    query: str,
    target_dir: str,
    license_filter: Optional[str] = "Attribution",
    max_items: int = 5,
) -> List[Dict]:
    """
    Fetch audio samples from Freesound.org.
    
    Args:
        query: Search query string
        target_dir: Directory to save audio files
        license_filter: Filter by license type
        max_items: Maximum number of items to fetch
        
    Returns:
        List of metadata for downloaded files
    """
    if FREESOUND_API_KEY == "YOUR_API_KEY":
        print("ERROR: Please add your Freesound API key to the script.")
        return []
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Prepare API request
    url = f"https://freesound.org/apiv2/search/text/"
    params = {
        "query": query,
        "token": FREESOUND_API_KEY,
        "fields": "id,name,tags,license,download,previews,username,filesize,duration,description",
        "filter": f"license:{license_filter}",
        "page_size": max_items,
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        metadata_list = []
        for i, result in enumerate(data.get("results", [])):
            sound_id = result["id"]
            
            # Get detailed sound info
            sound_url = f"https://freesound.org/apiv2/sounds/{sound_id}/"
            sound_params = {"token": FREESOUND_API_KEY}
            sound_response = requests.get(sound_url, params=sound_params)
            sound_response.raise_for_status()
            sound_data = sound_response.json()
            
            # Validate license
            license_name = sound_data.get("license", "")
            if not validate_license(license_name):
                print(f"Skipping {sound_data['name']} - License not acceptable: {license_name}")
                continue
                
            # Prepare filename and path
            clean_name = "".join(c if c.isalnum() or c in " ._-" else "_" for c in result["name"])
            filename = f"{sound_id}_{clean_name}"
            if not filename.lower().endswith((".wav", ".mp3", ".flac")):
                filename += ".wav"
            filepath = os.path.join(target_dir, filename)
            
            # Download sound
            download_url = sound_data["download"]
            download_params = {"token": FREESOUND_API_KEY}
            print(f"Downloading {filename}...")
            download_response = requests.get(download_url, params=download_params)
            download_response.raise_for_status()
            
            with open(filepath, "wb") as f:
                f.write(download_response.content)
            
            # Store metadata
            metadata = {
                "id": sound_id,
                "filename": filename,
                "title": result["name"],
                "tags": result.get("tags", []),
                "license": license_name,
                "license_url": sound_data.get("license_url", ""),
                "duration": result.get("duration", 0),
                "author": result.get("username", ""),
                "description": result.get("description", ""),
                "source": "freesound.org",
                "source_url": f"https://freesound.org/s/{sound_id}/",
                "download_date": datetime.now().strftime("%Y-%m-%d"),
                "filesize": result.get("filesize", 0),
                "attribution": f"Sound '{result['name']}' by {result.get('username', '')} from Freesound.org, under {license_name}"
            }
            metadata_list.append(metadata)
            
            print(f"Downloaded {i+1}/{len(data.get('results', []))} - {filename}")
            print(f"License: {license_name}")
            
        return metadata_list
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Freesound: {e}")
        return []


def download_musdb_sample(
    target_dir: str,
) -> List[Dict]:
    """
    Download sample track from MUSDB18 dataset if available.
    
    Args:
        target_dir: Directory to save audio files
        
    Returns:
        List of metadata for downloaded files
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # MUSDB18 is under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
    license_name = "CC BY-NC-SA 4.0"
    if not validate_license(license_name):
        print(f"Warning: MUSDB18 license ({license_name}) may not be acceptable for all uses")
    
    # Sample URLs for publicly available MUSDB18 examples
    # These are just examples and might need to be updated
    sample_urls = [
        "https://zenodo.org/records/3338373/files/train/A%20Classic%20Education%20-%20NightOwl/mixture.wav",
        "https://zenodo.org/records/3338373/files/train/A%20Classic%20Education%20-%20NightOwl/drums.wav",
        "https://zenodo.org/records/3338373/files/train/A%20Classic%20Education%20-%20NightOwl/bass.wav",
        "https://zenodo.org/records/3338373/files/train/A%20Classic%20Education%20-%20NightOwl/vocals.wav",
        "https://zenodo.org/records/3338373/files/train/A%20Classic%20Education%20-%20NightOwl/other.wav",
    ]
    
    metadata_list = []
    for url in sample_urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(target_dir, filename)
        
        try:
            print(f"Downloading {filename} from MUSDB18...")
            urllib.request.urlretrieve(url, filepath)
            
            track_name = url.split("/")[-2]
            
            # Store metadata
            metadata = {
                "filename": filename,
                "title": track_name,
                "source": "MUSDB18",
                "source_url": MUSDB_URL,
                "license": license_name,
                "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "download_date": datetime.now().strftime("%Y-%m-%d"),
                "attribution": f"'{track_name}' from MUSDB18 dataset, available at {MUSDB_URL}, under {license_name}"
            }
            metadata_list.append(metadata)
            
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    return metadata_list


def update_metadata_file(
    metadata_file: str,
    new_metadata: List[Dict],
) -> None:
    """
    Update the metadata.json file with new entries.
    
    Args:
        metadata_file: Path to metadata.json file
        new_metadata: List of new metadata entries to add
    """
    # Read existing metadata if file exists
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {"samples": []}
    else:
        metadata = {"samples": []}
    
    # Add new metadata entries
    metadata["samples"].extend(new_metadata)
    
    # Update dataset stats
    total_files = len(metadata["samples"])
    total_duration = sum(sample.get("duration", 0) for sample in metadata["samples"])
    
    metadata["total_files"] = total_files
    metadata["total_duration_seconds"] = total_duration
    metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    
    # Count by category
    genres = ["pop", "rock", "jazz", "classical", "electronic", "other"]
    challenges = ["reverberation", "distortion", "low_volume", "similar_timbres", "dynamic_range"]
    
    for genre in genres:
        metadata["genres"][genre] = sum(1 for sample in metadata["samples"] 
                                      if sample.get("category", "").lower() == genre.lower())
    
    for challenge in challenges:
        metadata["challenges"][challenge] = sum(1 for sample in metadata["samples"] 
                                             if sample.get("category", "").lower() == challenge.lower())
    
    # Write updated metadata back to file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main function to fetch sample audio files."""
    parser = argparse.ArgumentParser(description="Fetch audio samples for test dataset")
    parser.add_argument(
        "--target", 
        choices=["pop", "rock", "jazz", "classical", "electronic", "other", 
                 "reverberation", "distortion", "low_volume", "similar_timbres", "dynamic_range"],
        required=True,
        help="Target category to fetch samples for"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Search query for Freesound API"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=5, 
        help="Number of samples to fetch"
    )
    parser.add_argument(
        "--license", 
        type=str, 
        default="Attribution", 
        help="License filter for Freesound API"
    )
    parser.add_argument(
        "--source", 
        choices=["freesound", "musdb"], 
        default="freesound", 
        help="Source to fetch samples from"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true",
        help="Only check licenses without downloading files"
    )
    
    args = parser.parse_args()
    
    # Determine target directory based on category
    if args.target in ["pop", "rock", "jazz", "classical", "electronic", "other"]:
        target_dir = os.path.join("data", "test_dataset", "genres", args.target)
    else:
        target_dir = os.path.join("data", "test_dataset", "challenges", args.target)
    
    metadata = []
    
    # Fetch samples from selected source
    if args.source == "freesound":
        if not args.query:
            parser.error("--query is required when using --source=freesound")
        metadata = fetch_from_freesound(
            query=args.query,
            target_dir=target_dir,
            license_filter=args.license,
            max_items=args.samples
        )
    elif args.source == "musdb":
        metadata = download_musdb_sample(target_dir)
    
    # Update category for each sample
    for item in metadata:
        item["category"] = args.target
    
    # Update metadata file
    if metadata:
        metadata_file = os.path.join("data", "test_dataset", "metadata.json")
        update_metadata_file(metadata_file, metadata)
        print(f"Updated metadata in {metadata_file}")
        print(f"Added {len(metadata)} samples to category '{args.target}'")


if __name__ == "__main__":
    main() 