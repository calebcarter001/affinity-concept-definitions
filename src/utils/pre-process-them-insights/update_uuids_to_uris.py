#!/usr/bin/env python3

import json
import re

def load_uuid_uri_mapping(mapping_file: str) -> dict:
    """Load the UUID to URI mapping from the mapping file."""
    uuid_to_uri = {}
    current_uuid = None
    current_uri = None
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('UUID: '):
                current_uuid = line.replace('UUID: ', '').strip()
            elif line.startswith('URI: '):
                current_uri = line.replace('URI: ', '').strip()
                if current_uuid and current_uri != 'URI not found':
                    uuid_to_uri[current_uuid] = current_uri
                current_uuid = None
                current_uri = None
    
    return uuid_to_uri

def update_uuids_to_uris(data, uuid_to_uri: dict):
    """Recursively update UUIDs to URIs in the data structure."""
    if isinstance(data, dict):
        # Create a list of items to avoid dictionary size change during iteration
        items = list(data.items())
        for key, value in items:
            if key == 'UUID' and isinstance(value, str) and value in uuid_to_uri:
                data['URI'] = uuid_to_uri[value]
            elif isinstance(value, (dict, list)):
                update_uuids_to_uris(value, uuid_to_uri)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                update_uuids_to_uris(item, uuid_to_uri)

def main():
    # File paths
    mapping_file = 'uuid_uri_mapping.txt'
    json_file = 'parsed_theme_rules_v3/all_business_themes_structured.json'
    output_file = 'parsed_theme_rules_v3/all_business_themes_structured_with_uris.json'
    
    print("Loading UUID to URI mapping...")
    uuid_to_uri = load_uuid_uri_mapping(mapping_file)
    print(f"Loaded {len(uuid_to_uri)} UUID to URI mappings")
    
    print("\nReading JSON file...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Updating UUIDs to URIs...")
    update_uuids_to_uris(data, uuid_to_uri)
    
    print(f"\nWriting updated JSON to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main() 