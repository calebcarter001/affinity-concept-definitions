#!/usr/bin/env python3

import re
import json
from preprocess_theme_insights import parse_markdown_table_flexible, parse_adventure_theme_qualifying_attributes

def test_ski_attributes_parsing():
    """Test parsing of Adventure theme with focus on ski attributes."""
    
    # Extract the Adventure theme section from the insights file
    with open('business-domain-insights.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Adventure theme section
    adventure_pattern = r'# Adventure Sport.*?(?=^# [A-Z]|\Z)'
    adventure_match = re.search(adventure_pattern, content, re.MULTILINE | re.DOTALL)
    
    if not adventure_match:
        print("Could not find Adventure theme section")
        return
    
    adventure_text = adventure_match.group(0)
    lines = adventure_text.split('\n')
    
    print(f"Found Adventure theme section with {len(lines)} lines")
    
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
    
    # Find ski-related section
    ski_section_line = -1
    for i in range(qa_start, len(lines)):
        if 'Ski-Related Attributes' in lines[i]:
            ski_section_line = i
            print(f"Found ski section at line {i}: {lines[i]}")
            break
    
    if ski_section_line == -1:
        print("Could not find Ski-Related Attributes section")
        return
    
    # Show context around ski section
    print("\n=== CONTEXT AROUND SKI SECTION ===")
    start_context = max(0, ski_section_line - 5)
    end_context = min(len(lines), ski_section_line + 15)
    
    for i in range(start_context, end_context):
        marker = " >>> " if i == ski_section_line else "     "
        print(f"{i:3d}{marker}{lines[i]}")
    
    # Test the Adventure theme parser specifically
    print("\n=== TESTING ADVENTURE THEME PARSER ===")
    section_data, _ = parse_adventure_theme_qualifying_attributes(lines, qa_start)
    
    print(f"Parser returned {len(section_data['attribute_groups'])} attribute groups:")
    for i, group in enumerate(section_data['attribute_groups']):
        print(f"  Group {i}: {group['name']} - {len(group['attributes'])} attributes")
        if group['name'] == 'ski_attributes':
            print(f"    Ski attributes: {group['attributes']}")
    
    # Test table parsing on ski table specifically
    print("\n=== TESTING SKI TABLE PARSING ===")
    
    # Extract just the ski table lines
    ski_table_lines = []
    in_ski_table = False
    
    for i in range(ski_section_line, len(lines)):
        line = lines[i].strip()
        if line.startswith('|'):
            ski_table_lines.append(line)
            in_ski_table = True
        elif in_ski_table and line == '':
            continue  # Allow blank lines within table
        elif in_ski_table and not line.startswith('|'):
            # End of table
            break
    
    print(f"Extracted {len(ski_table_lines)} ski table lines:")
    for i, line in enumerate(ski_table_lines):
        print(f"  {i}: {line}")
    
    if ski_table_lines:
        ski_parsed = parse_markdown_table_flexible(ski_table_lines)
        print(f"\nParsed ski table into {len(ski_parsed)} rows:")
        for i, row in enumerate(ski_parsed):
            print(f"  Row {i}: {row}")

if __name__ == "__main__":
    test_ski_attributes_parsing() 