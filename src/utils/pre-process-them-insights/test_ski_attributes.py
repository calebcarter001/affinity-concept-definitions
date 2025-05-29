import json

def parse_markdown_table_flexible(table_lines: list[str]) -> list[dict]:
    """
    Parses a simple markdown table into a list of dictionaries.
    More robust to variations in header and separator lines.
    """
    if not table_lines:
        return []

    header_line_index = -1
    data_start_index = -1

    # Find header: the first line starting and ending with '|' that is not a separator
    for i, line in enumerate(table_lines):
        line_strip = line.strip()
        if line_strip.startswith('|') and line_strip.endswith('|'):
            if '---' not in line_strip: # Potential header
                header_line_index = i
                # Check if next line is a separator
                if i + 1 < len(table_lines) and table_lines[i+1].strip().startswith('|') and '---' in table_lines[i+1]:
                    data_start_index = i + 2
                else: # No separator or separator missing, data starts next line
                    data_start_index = i + 1
                break # Found header
    
    if header_line_index == -1:
        print(f"Warning: Could not determine headers for table: {table_lines[:2]}")
        return []

    headers = [h.strip() for h in table_lines[header_line_index].strip('|').split('|')]
    
    parsed_rows = []
    if data_start_index < len(table_lines):
        for line_num in range(data_start_index, len(table_lines)):
            line = table_lines[line_num]
            line_strip = line.strip()
            if not (line_strip.startswith('|') and line_strip.endswith('|')):
                if parsed_rows: # only break if we've already parsed some data rows
                    break
                continue # Skip initial non-table lines in data part

            values = [v.strip() for v in line_strip.strip('|').split('|')]
            
            row_dict = {}
            for i, header in enumerate(headers):
                row_dict[header] = values[i] if i < len(values) else "" # Handle rows with fewer cells than headers

            # Only add if it has the same number of columns as headers or if it's not just empty separators
            if len(values) == len(headers) or any(v for v in values if v.strip()):
                 parsed_rows.append(row_dict)

    return parsed_rows

def test_ski_attributes_parsing():
    # Sample input data with both main attributes and ski-related attributes
    test_input = """# Adventure Theme Definition

## Qualifying Attributes

Properties must have 5 of the following attributes OR 4 of the following attributes AND 1 of the ski-related attributes.

| Attribute Name | AID | UUID |
|----------------|-----|------|
| Hiking | 2359 | 123e4567-e89b-12d3-a456-426614174000 |
| Kayaking | 2361 | 123e4567-e89b-12d3-a456-426614174001 |
| Rock climbing | 2363 | 123e4567-e89b-12d3-a456-426614174002 |

Ski-Related Attributes:

| Attribute Name | AID | UUID |
|----------------|-----|------|
| Ski lessons nearby | 2360 | aaf52aef-f5de-4cf2-af5d-9a126ea3ac37 |
| Ski lessons on site | 3836 | 18ef2adf-6321-40fb-bcc2-0a9a85d5c051 |
| Ski rentals nearby | 2362 | 03be3a63-876f-4664-a8b2-f9878689046d |
| Ski rentals on site | 3835 | 1e31114d-68a6-49cc-a13e-71df9ef22385 |
"""

    # Split input into lines
    lines = test_input.split('\n')

    # Find the qualifying attributes section
    start_idx = -1
    for i, line in enumerate(lines):
        if "## Qualifying Attributes" in line:
            start_idx = i
            break

    if start_idx == -1:
        print("Error: Could not find Qualifying Attributes section")
        return

    # Parse the section
    section_data = parse_adventure_section(lines[start_idx:])
    
    # Print results
    print("\nParsed Results:")
    print(json.dumps(section_data, indent=2))

def parse_adventure_section(lines: list[str]) -> dict:
    """
    Parse adventure theme section with both main and ski-related attributes.
    """
    section_data = {
        "description": "",
        "main_attributes": [],
        "ski_related_attributes": [],
        "rules": []
    }
    
    current_section = "main"  # or "ski"
    current_content = []
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for ski-related section
        if "Ski-Related Attributes" in line:
            # Process any pending table content
            if current_content and current_content[0].startswith('|'):
                table_data = parse_markdown_table_flexible(current_content)
                if current_section == "main":
                    section_data["main_attributes"].extend(table_data)
                else:
                    section_data["ski_related_attributes"].extend(table_data)
            current_section = "ski"
            current_content = []
            in_table = False
            continue
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                # Start of a new table
                current_content = []
                in_table = True
            current_content.append(line)
        else:
            # Process completed table
            if in_table and current_content:
                table_data = parse_markdown_table_flexible(current_content)
                if current_section == "main":
                    section_data["main_attributes"].extend(table_data)
                else:
                    section_data["ski_related_attributes"].extend(table_data)
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('##'):
                section_data["description"] += line + "\n"
                # Look for rules
                if "must have" in line.lower() or "of the following" in line:
                    section_data["rules"].append({
                        "type": "compound",
                        "operator": "OR",
                        "conditions": [
                            {
                                "type": "simple",
                                "required_count": 5,
                                "from": "main_attributes"
                            },
                            {
                                "type": "compound",
                                "operator": "AND",
                                "conditions": [
                                    {
                                        "type": "simple",
                                        "required_count": 4,
                                        "from": "main_attributes"
                                    },
                                    {
                                        "type": "simple",
                                        "required_count": 1,
                                        "from": "ski_related_attributes"
                                    }
                                ]
                            }
                        ]
                    })
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        if current_section == "main":
            section_data["main_attributes"].extend(table_data)
        else:
            section_data["ski_related_attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = section_data["description"].strip()
    
    return section_data

if __name__ == "__main__":
    test_ski_attributes_parsing() 