def normalize_theme_name(name: str) -> str:
    """
    Normalize theme names for exact matching.
    Each theme must have its own specific parser, so we need exact matches.
    """
    if not name:
        return ""
        
    # Remove common suffixes
    name = name.strip()
    name = name.replace(" Theme Definition", "")
    name = name.replace(" Theme", "")
    name = name.replace(" Property", "")
    name = name.replace(" property", "")
    name = name.replace(" Friendly*", "")
    name = name.replace(" Friendly", "")
    name = name.replace("-Friendly", "")
    name = name.replace(" certified", "")
    name = name.replace(" Certified", "")
    
    # Remove any trailing dash and spaces
    name = name.rstrip("-").strip()
    
    return name

def get_theme_parser(theme_name: str) -> callable:
    """
    Returns the appropriate parser function for a given theme.
    Each theme has its own specific parser due to unique structures and requirements.
    """
    print(f"DEBUG: get_theme_parser called with theme_name: '{theme_name}'")
    
    theme_parsers = {
        "Adventure Sport": parse_adventure_theme_qualifying_attributes,
        "Beach": parse_beach_theme_qualifying_attributes,
        "Boutique": parse_boutique_theme_qualifying_attributes,
        "Business": parse_business_theme_qualifying_attributes,
        "Casino": parse_casino_theme_qualifying_attributes,
        "Eco-certified": parse_eco_certified_theme_qualifying_attributes,
        "Family": parse_family_theme_qualifying_attributes,
        "Golf": parse_golf_theme_qualifying_attributes,
        "Historic": parse_historic_theme_qualifying_attributes,
        "Hot Springs": parse_hot_springs_theme_qualifying_attributes,
        "LGBTQ Friendly": parse_lgbtq_theme_qualifying_attributes,
        "Luxury": parse_luxury_theme_qualifying_attributes,
        "Pet Friendly": parse_pet_friendly_theme_qualifying_attributes,
        "Romantic": parse_romantic_theme_qualifying_attributes,
        "Shopping": parse_shopping_theme_qualifying_attributes,
        "Ski": parse_ski_theme_qualifying_attributes,
        "Spa": parse_spa_theme_qualifying_attributes,
        "Winery": parse_winery_theme_qualifying_attributes
    }
    
    if not theme_name:
        raise ValueError("No theme name provided")
    
    # Normalize theme name for matching
    normalized_name = normalize_theme_name(theme_name)
    print(f"DEBUG: Normalized theme name: '{normalized_name}'")
    
    # Try exact match first
    for key in theme_parsers:
        normalized_key = normalize_theme_name(key)
        print(f"DEBUG: Checking if '{normalized_key}' matches '{normalized_name}'")
        if normalized_key == normalized_name:
            print(f"DEBUG: Found exact match! Using parser for '{key}'")
            return theme_parsers[key]
    
    raise ValueError(f"No specific parser found for theme: {theme_name}. Each theme must have its own parser.")

def parse_theme_block_from_insights(theme_block_text: str) -> dict:
    theme_data = {"name_from_header": None}
    lines = theme_block_text.split('\n')
    
    # Extract theme name from header
    for line in lines:
        if line.startswith('# '):
            theme_data["name_from_header"] = line.lstrip('# ').split('-')[0].strip()
            break
    
    if not theme_data["name_from_header"]:
        raise ValueError("Could not find theme name in header")
    
    # Get the appropriate parser for this theme
    parser = get_theme_parser(theme_data["name_from_header"])
    
    # Parse the theme block using the theme-specific parser
    parsed_rules_details = parser(lines, 0)
    theme_data["parsed_rules_details"] = parsed_rules_details
    
    return theme_data

def parse_adventure_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Adventure theme which has:
    1. Simple numeric requirement (5 attributes)
    2. Alternative requirement (4 + 1 ski-related)
    3. Two attribute tables (main and ski-related)
    """
    print("\nStarting to parse Adventure theme section...")
    
    section_data = {
        "description": "",
        "main_attributes": [],
        "ski_related_attributes": [],
        "rules": []
    }
    
    current_idx = start_idx
    current_section = "main"  # or "ski"
    current_content = []
    in_table = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        print(f"\nProcessing line {current_idx}: {line[:50]}...")
        
        # Check for next theme section
        if line.startswith('# ') and current_idx > start_idx:
            print(f"Found next theme section, breaking: {line}")
            break
            
        # Check for ski-related section using exact separator
        if "### Ski-Related Attributes" in line:
            print(f"\n=== DEBUG: Found ski-related section ===")
            print(f"Current line: {line}")
            print(f"Current section before switch: {current_section}")
            # Process any pending table content
            if current_content and current_content[0].startswith('|'):
                print("\nProcessing pending table content before switching to ski section")
                print(f"Current content: {current_content}")
                table_data = parse_markdown_table_flexible(current_content)
                if current_section == "main":
                    print(f"Adding {len(table_data)} rows to main attributes")
                    section_data["main_attributes"].extend(table_data)
                    print(f"Main attributes now has {len(section_data['main_attributes'])} total rows")
            current_section = "ski"
            print(f"Switched to section: {current_section}")
            current_content = []
            in_table = False
            current_idx += 1
            continue
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                print(f"\n=== Starting new table in {current_section} section ===")
                current_content = []
                in_table = True
            current_content.append(line)
            print(f"Added table line to {current_section} section: {line}")
        else:
            # Process completed table
            if in_table and current_content:
                print(f"\n=== Processing completed table in {current_section} section ===")
                print(f"Table content to process: {current_content}")
                table_data = parse_markdown_table_flexible(current_content)
                if current_section == "main":
                    print(f"Adding {len(table_data)} rows to main attributes")
                    section_data["main_attributes"].extend(table_data)
                    print(f"Main attributes now has {len(section_data['main_attributes'])} total rows")
                else:
                    print(f"Adding {len(table_data)} rows to ski attributes")
                    section_data["ski_related_attributes"].extend(table_data)
                    print(f"Ski attributes now has {len(section_data['ski_related_attributes'])} total rows")
                    print("Ski attributes content:")
                    for attr in table_data:
                        print(f"  {attr}")
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Look for rules
                if "must have" in line.lower() or "of the following" in line:
                    print(f"\n=== Found rule in line ===")
                    print(f"Rule text: {line}")
                    rule = {
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
                    }
                    section_data["rules"].append(rule)
                    print("Added rule structure:")
                    print(f"  {rule}")
        
        current_idx += 1
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        print(f"\n=== Processing final table content in {current_section} section ===")
        print(f"Final table content: {current_content}")
        table_data = parse_markdown_table_flexible(current_content)
        if current_section == "main":
            print("Adding final rows to main attributes")
            section_data["main_attributes"].extend(table_data)
            print(f"Main attributes final count: {len(section_data['main_attributes'])}")
        else:
            print("Adding final rows to ski attributes")
            section_data["ski_related_attributes"].extend(table_data)
            print(f"Ski attributes final count: {len(section_data['ski_related_attributes'])}")
            print("Final ski attributes content:")
            for attr in table_data:
                print(f"  {attr}")
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    final_structure = {
        "description": section_data["description"],
        "rules": section_data["rules"],
        "attribute_groups": [
            {
                "name": "main_attributes",
                "description": "Main adventure sport attributes",
                "attributes": section_data["main_attributes"]
            },
            {
                "name": "ski_related_attributes", 
                "description": "Ski-related attributes",
                "attributes": section_data["ski_related_attributes"]
            }
        ]
    }
    
    return final_structure, current_idx

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

    # Clean and normalize headers
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

            # Clean and normalize values
            values = [v.strip() for v in line_strip.strip('|').split('|')]
            
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    value = values[i].strip()
                    if value:  # Only add non-empty values
                        row_dict[header] = value

            # Only add if it has actual data
            if row_dict and any(v for v in row_dict.values()):
                parsed_rows.append(row_dict)

    return parsed_rows

def clean_section_text(text: str) -> str:
    """Clean up section text by removing extra whitespace and empty lines."""
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)

def parse_golf_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Golf theme which has:
    1. Simple qualifying attributes (golf course on site or adjacent)
    2. No complex rules or sections
    """
    section_data = {
        "description": "",
        "attributes": [],
        "rules": []
    }
    
    current_idx = start_idx
    current_content = []
    in_table = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next theme section
        if line.startswith('# ') and current_idx > start_idx:
            break
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                current_content = []
                in_table = True
            current_content.append(line)
        else:
            # Process completed table
            if in_table and current_content:
                table_data = parse_markdown_table_flexible(current_content)
                section_data["attributes"].extend(table_data)
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('###'):
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_historic_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Historic theme which has:
    1. National historic designation requirements
    2. Program-based qualification
    """
    section_data = {
        "description": "",
        "programs": [],
        "attributes": [],
        "rules": []
    }
    
    current_idx = start_idx
    current_content = []
    in_table = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next theme section
        if line.startswith('# ') and current_idx > start_idx:
            break
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                current_content = []
                in_table = True
            current_content.append(line)
        else:
            # Process completed table
            if in_table and current_content:
                table_data = parse_markdown_table_flexible(current_content)
                if "Program" in table_data[0] if table_data else False:
                    section_data["programs"].extend(table_data)
                else:
                    section_data["attributes"].extend(table_data)
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('###'):
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        if "Program" in table_data[0] if table_data else False:
            section_data["programs"].extend(table_data)
        else:
            section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_hot_springs_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Hot Springs theme which has:
    1. Simple qualifying attributes (hot springs on-site or nearby)
    2. No complex rules or sections
    """
    section_data = {
        "description": "",
        "attributes": [],
        "rules": []
    }
    
    current_idx = start_idx
    current_content = []
    in_table = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next theme section
        if line.startswith('# ') and current_idx > start_idx:
            break
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                current_content = []
                in_table = True
            current_content.append(line)
        else:
            # Process completed table
            if in_table and current_content:
                table_data = parse_markdown_table_flexible(current_content)
                section_data["attributes"].extend(table_data)
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('###'):
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_lgbtq_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for LGBTQ theme which has:
    1. Manual assignment criteria
    2. Country exclusions
    3. Documentation requirements
    """
    section_data = {
        "description": "",
        "attributes": [],
        "rules": [],
        "country_exclusions": [],
        "documentation_requirements": []
    }
    
    current_idx = start_idx
    current_content = []
    in_table = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next theme section
        if line.startswith('# ') and current_idx > start_idx:
            break
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                current_content = []
                in_table = True
            current_content.append(line)
        else:
            # Process completed table
            if in_table and current_content:
                table_data = parse_markdown_table_flexible(current_content)
                section_data["attributes"].extend(table_data)
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Look for documentation requirements
                if "proof is required" in line.lower():
                    section_data["documentation_requirements"].append(line)
                # Look for country exclusions
                elif "countries that impose civil penalties" in line.lower():
                    section_data["country_exclusions"].append(line)
        
        current_idx += 1
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_luxury_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Luxury theme which has:
    1. Star rating requirements
    2. Awards and Affiliations requirements
    3. Structure type exclusions
    """
    section_data = {
        "description": "",
        "attributes": [],
        "rules": [],
        "excluded_structure_types": []
    }
    
    current_idx = start_idx
    current_content = []
    in_table = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next theme section
        if line.startswith('# ') and current_idx > start_idx:
            break
            
        # Handle table lines
        if line.startswith('|'):
            if not in_table:
                current_content = []
                in_table = True
            current_content.append(line)
        else:
            # Process completed table
            if in_table and current_content:
                table_data = parse_markdown_table_flexible(current_content)
                section_data["attributes"].extend(table_data)
                current_content = []
                in_table = False
            
            # Add non-table lines to description if meaningful
            if not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Look for structure type exclusions
                if "Structure Types" in line and "not considered" in line:
                    types = line.split(":")[-1].strip()
                    section_data["excluded_structure_types"].extend([t.strip() for t in types.split(",")])
        
        current_idx += 1
    
    # Process any final table content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

if __name__ == "__main__":
    try:
        # Read the business-domain-insights.txt file
        with open('src/utils/pre-process-them-insights/business-domain-insights.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Successfully read {len(lines)} lines from file")
        
        # Find the Adventure theme section
        start_idx = 0
        for i, line in enumerate(lines):
            if "# Adventure Sport" in line:
                start_idx = i
                print(f"Found Adventure Sport section at line {start_idx}")
                break
        
        # Parse the Adventure theme section
        print("\nParsing Adventure theme section...")
        result, end_idx = parse_adventure_theme_qualifying_attributes(lines, start_idx)
        
        # Print the results
        print("\nProcessing Results:")
        print("\nMain Attributes:")
        for attr in result["attribute_groups"][0]["attributes"]:
            print(f"  - {attr}")
        
        print("\nSki-Related Attributes:")
        for attr in result["attribute_groups"][1]["attributes"]:
            print(f"  - {attr}")
        
        print("\nRules:")
        for rule in result["rules"]:
            print(f"  - {rule}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}") 