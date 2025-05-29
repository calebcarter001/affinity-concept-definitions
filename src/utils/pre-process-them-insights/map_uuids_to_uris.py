#!/usr/bin/env python3

import json
import re
from collections import defaultdict
from typing import Dict, Set, List, Tuple

def extract_uuids_from_themes(json_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract all UUIDs from the themes JSON file, keeping track of which theme and context they came from.
    Returns a dict mapping UUID -> list of (theme_name, context) tuples.
    """
    uuid_map = defaultdict(list)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        themes = json.load(f)
    
    def process_dict(d: dict, theme_name: str, context: str = ""):
        for k, v in d.items():
            if k == "UUID" and isinstance(v, str) and len(v) == 36:  # Standard UUID length
                uuid_map[v].append((theme_name, context))
            elif isinstance(v, dict):
                process_dict(v, theme_name, f"{context}/{k}" if context else k)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        process_dict(item, theme_name, f"{context}/{k}[{i}]" if context else f"{k}[{i}]")
    
    for theme in themes:
        theme_name = theme.get("theme_attribute_name", "Unknown Theme")
        process_dict(theme, theme_name)
    
    return uuid_map

def find_uris_in_rdf(rdf_file: str, uuids: Set[str]) -> Dict[str, str]:
    """
    Search through the RDF file to find URIs matching the given UUIDs.
    Returns a dict mapping UUID -> URI.
    """
    uri_map = {}
    
    # Pattern to match URIs containing UUIDs in the format:
    # urn:expediagroup:taxonomies:acs:#UUID
    uri_pattern = re.compile(r'(?:rdf:about|rdf:resource)="(urn:expediagroup:taxonomies:acs:#([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}))"')
    
    with open(rdf_file, 'r', encoding='utf-8') as f:
        for line in f:
            matches = uri_pattern.finditer(line)
            for match in matches:
                uri = match.group(1)
                uuid = match.group(2)
                if uuid in uuids:
                    uri_map[uuid] = uri
    
    return uri_map

def main():
    # File paths
    themes_json = "parsed_theme_rules_v3/all_business_themes_structured.json"
    rdf_file = "/Users/calebcarter/PycharmProjects/PythonProject/datasources/acs.rdf"
    output_file = "uuid_uri_mapping.txt"
    
    print("Extracting UUIDs from themes JSON...")
    uuid_map = extract_uuids_from_themes(themes_json)
    print(f"Found {len(uuid_map)} unique UUIDs")
    
    print("\nSearching for URIs in RDF file...")
    uri_map = find_uris_in_rdf(rdf_file, set(uuid_map.keys()))
    print(f"Found URIs for {len(uri_map)} UUIDs")
    
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("UUID to URI Mapping\n")
        f.write("==================\n\n")
        
        # Group by theme for better organization
        theme_groups = defaultdict(list)
        for uuid, contexts in uuid_map.items():
            for theme, context in contexts:
                theme_groups[theme].append((uuid, context))
        
        for theme in sorted(theme_groups.keys()):
            f.write(f"\n{theme}\n")
            f.write("-" * len(theme) + "\n")
            
            for uuid, context in sorted(theme_groups[theme]):
                uri = uri_map.get(uuid, "URI not found")
                f.write(f"\nContext: {context}\n")
                f.write(f"UUID: {uuid}\n")
                f.write(f"URI:  {uri}\n")
    
    print("Done!")
    
    # Print some statistics
    total_uuids = len(uuid_map)
    found_uris = len(uri_map)
    print(f"\nStatistics:")
    print(f"Total unique UUIDs found: {total_uuids}")
    print(f"URIs found: {found_uris}")
    print(f"URIs not found: {total_uuids - found_uris}")
    print(f"Coverage: {found_uris/total_uuids*100:.1f}%")

if __name__ == "__main__":
    main() 