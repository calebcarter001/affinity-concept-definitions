#!/usr/bin/env python3

import re
import json
from preprocess_theme_insights import parse_markdown_table_flexible, parse_family_theme_qualifying_attributes, get_theme_parser

def test_family_theme_parsing():
    """Test parsing of Family theme with focus on the split sections and attributes."""
    
    # Extract the Family theme section from the insights file
    with open('business-domain-insights.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Family theme section
    family_pattern = r'# Family.*?(?=^# [A-Z]|\Z)'
    family_match = re.search(family_pattern, content, re.MULTILINE | re.DOTALL)
    
    if not family_match:
        print("Could not find Family theme section")
        return
    
    family_text = family_match.group(0)
    lines = family_text.split('\n')
    
    print(f"Found Family theme section with {len(lines)} lines")
    
    # Find the start of qualifying attributes section
    qa_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('## Qualifying Attributes'):
            qa_start = i
            break
    
    if qa_start == -1:
        print("Could not find Qualifying Attributes section")
        return
    
    print(f"Qualifying Attributes section starts at line {qa_start}")
    
    # Show context around qualifying attributes section
    print("\n=== FAMILY THEME QUALIFYING ATTRIBUTES CONTEXT ===")
    start_context = max(0, qa_start - 2)
    end_context = min(len(lines), qa_start + 30)
    
    for i in range(start_context, end_context):
        marker = " >>> " if i == qa_start else "     "
        print(f"{i:3d}{marker}{lines[i]}")
    
    # Test the theme parser selection
    print("\n=== TESTING THEME PARSER SELECTION ===")
    parser = get_theme_parser("Family")
    print(f"Selected parser: {parser.__name__}")
    
    # Test the Family theme parser specifically
    print("\n=== TESTING FAMILY THEME PARSER ===")
    section_data, _ = parse_family_theme_qualifying_attributes(lines, qa_start)
    
    print(f"Parser returned structure:")
    print(json.dumps(section_data, indent=2))
    
    # Look for specific sections that should be present
    print("\n=== CHECKING FOR EXPECTED SECTIONS ===")
    
    # Look for "Conventional Lodgings" and "Vacation Rentals" sections
    conventional_found = False
    vacation_found = False
    primary_found = False
    secondary_found = False
    
    for i, line in enumerate(lines[qa_start:], qa_start):
        line_stripped = line.strip()
        if "Conventional Lodgings" in line_stripped:
            conventional_found = True
            print(f"Found 'Conventional Lodgings' at line {i}: {line_stripped}")
        elif "Vacation Rentals" in line_stripped:
            vacation_found = True
            print(f"Found 'Vacation Rentals' at line {i}: {line_stripped}")
        elif "Primary Attributes" in line_stripped:
            primary_found = True
            print(f"Found 'Primary Attributes' at line {i}: {line_stripped}")
        elif "Secondary Attributes" in line_stripped:
            secondary_found = True
            print(f"Found 'Secondary Attributes' at line {i}: {line_stripped}")
    
    print(f"\nSections found:")
    print(f"  Conventional Lodgings: {conventional_found}")
    print(f"  Vacation Rentals: {vacation_found}")
    print(f"  Primary Attributes: {primary_found}")
    print(f"  Secondary Attributes: {secondary_found}")

if __name__ == "__main__":
    test_family_theme_parsing() 