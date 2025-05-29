import re
import json
import os
from collections import defaultdict

# (parse_markdown_table_flexible and parse_key_value_lines remain the same for now,
#  as the main issue seems to be higher-level block/section parsing)

def parse_markdown_table_flexible(table_lines: list[str]) -> list[dict]:
    """
    Parses a simple markdown table into a list of dictionaries.
    More robust to variations in header and separator lines.
    """
    print("\n=== DEBUG: Parsing table ===")
    print(f"Input table lines: {table_lines}")
    
    if not table_lines:
        print("No table lines provided")
        return []

    header_line_index = -1
    separator_line_index = -1 # Not strictly needed for parsing data if header is found
    data_start_index = -1

    # Find header: the first line starting and ending with '|' that is not a separator
    for i, line in enumerate(table_lines):
        line_strip = line.strip()
        print(f"Checking line {i}: {line_strip}")
        if line_strip.startswith('|') and line_strip.endswith('|'):
            if '---' not in line_strip: # Potential header
                header_line_index = i
                print(f"Found header at line {i}: {line_strip}")
                # Check if next line is a separator
                if i + 1 < len(table_lines) and table_lines[i+1].strip().startswith('|') and '---' in table_lines[i+1]:
                    data_start_index = i + 2
                    print(f"Found separator, data starts at line {data_start_index}")
                else: # No separator or separator missing, data starts next line
                    data_start_index = i + 1
                    print(f"No separator found, data starts at line {data_start_index}")
                break # Found header
    
    if header_line_index == -1:
        print("Warning: Could not determine headers for table")
        return []

    headers = [h.strip() for h in table_lines[header_line_index].strip('|').split('|')]
    print(f"Parsed headers: {headers}")
    
    parsed_rows = []
    if data_start_index < len(table_lines):
        for line_num in range(data_start_index, len(table_lines)):
            line = table_lines[line_num]
            line_strip = line.strip()
            print(f"Processing data line {line_num}: {line_strip}")
            if not (line_strip.startswith('|') and line_strip.endswith('|')):
                print(f"Found non-table line at {line_num}, stopping")
                if parsed_rows:
                    break
                continue

            values = [v.strip() for v in line_strip.strip('|').split('|')]
            print(f"Parsed values: {values}")
            
            row_dict = {}
            for i, header in enumerate(headers):
                row_dict[header] = values[i] if i < len(values) else ""

            if len(values) == len(headers) or any(v for v in values if v.strip()):
                print(f"Adding row: {row_dict}")
                parsed_rows.append(row_dict)
            elif not any(v for v in values if v.strip()):
                print("Skipping empty separator row")
            else:
                print(f"Warning: Row-header mismatch. Headers:{len(headers)}, Values:{len(values)}")

    print(f"Final parsed rows: {parsed_rows}")
    return parsed_rows


def parse_key_value_lines(lines: list[str], keys: list[str]) -> dict:
    """
    Parse key-value pairs from lines that match the format:
    - **Key**: Value
    or
    **Key**: Value
    """
    result = {}
    for line in lines:
        line = line.strip('- ')  # Remove leading dash and space if present
        for key in keys:
            pattern = fr'\*\*{key}\*\*:\s*(.+)'
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Convert key to lowercase for consistency
                normalized_key = key.lower().replace(' ', '_')
                result[normalized_key] = value
                break
    return result


def parse_attribute_logic_block(content_lines: list[str]) -> dict:
    """
    Parses a block of text and tables describing qualifying/disqualifying attributes.
    Tries to identify distinct conditions or groups of attributes.
    """
    block_data = {"overall_description": "", "conditions": []}
    current_condition = {"description": "", "attribute_tables": [], "applies_to_structure_types": []}
    text_buffer = []
    
    # Patterns to identify start of new logical conditions or sub-sections
    condition_starters = r"^(Properties must have|Minimum requirement is|Minimum requirement:|Applies to:|Primary Attributes|Secondary Attributes|\*\*\d+\s+of|\*\*AND\s+\d+\s+of|OR\s+\d+\s+of|AND\s+\d+)"
    structure_type_pattern = r"Applies to:\s*(.*)"

    for line in content_lines:
        stripped_line = line.strip()

        if re.match(condition_starters, stripped_line, re.IGNORECASE) or stripped_line.startswith("### "):
            if text_buffer or current_condition["attribute_tables"]: # Save previous condition/text
                if not current_condition["description"] and not current_condition["attribute_tables"]:
                    # This text is part of the overall description of the section
                    block_data["overall_description"] += "\n".join(text_buffer).strip() + "\n"
                else:
                    # Text belongs to the current condition's description
                    current_condition["description"] = (current_condition["description"] + "\n" + "\n".join(text_buffer)).strip()
                    if current_condition["description"] or current_condition["attribute_tables"]:
                         block_data["conditions"].append(current_condition)
                text_buffer = []
            
            current_condition = {"description": stripped_line, "attribute_tables": [], "applies_to_structure_types": []}
            
            st_match = re.search(structure_type_pattern, stripped_line, re.IGNORECASE)
            if st_match:
                current_condition["applies_to_structure_types"] = [s.strip() for s in st_match.group(1).split(',')]
                current_condition["description"] = current_condition["description"].replace(st_match.group(0),'').strip() # Remove from main desc

        elif stripped_line.startswith("|") and stripped_line.endswith("|"):
            # Start of a table
            if text_buffer: # Text before table is part of current condition's description
                current_condition["description"] = (current_condition["description"] + "\n" + "\n".join(text_buffer)).strip()
                text_buffer = []
            
            # Collect all lines for this table
            current_table_lines = [line]
            # This logic is flawed, it doesn't use line_idx. We need to pass the full list and current index
            # For simplicity here, assume tables are contiguous blocks.
            # This will be handled by passing current_section_content to helper.
            # Here, we are parsing a sub-block already.

            # The `parse_attribute_list_section` should actually handle this better
            # by finding tables within its `current_section_content`.
            # This function is more about structuring multiple "condition" blocks.
            # For now, let's assume a table found here belongs to the current condition.
            # This needs robust table detection within the loop.

            # For simplicity in this specific function, we'll assume parse_markdown_table_flexible
            # is called on pre-identified table blocks.
            # This function parse_attribute_logic_block is more about identifying the conditions around tables.
            pass # Table parsing should happen when calling this function with table lines

        elif stripped_line:
            text_buffer.append(stripped_line)

    # Finalize last condition/text
    if text_buffer or current_condition["attribute_tables"] or current_condition["description"].strip():
        current_condition["description"] = (current_condition["description"] + "\n" + "\n".join(text_buffer)).strip()
        if not block_data["conditions"] and not block_data["overall_description"] and not current_condition["attribute_tables"]:
             block_data["overall_description"] = current_condition["description"] # If it's all just one block of text
        elif current_condition["description"] or current_condition["attribute_tables"]:
             block_data["conditions"].append(current_condition)
        elif not block_data["conditions"] and not block_data["overall_description"]: # if only text_buffer remained
            block_data["overall_description"] = "\n".join(text_buffer).strip()


    block_data["overall_description"] = block_data["overall_description"].strip()
    block_data["conditions"] = [c for c in block_data["conditions"] if c["description"] or c["attribute_tables"]]
    return block_data


def clean_section_text(text: str) -> str:
    """Clean up section text by removing unwanted characters and normalizing whitespace."""
    # Remove separator lines
    text = re.sub(r'=+\s*', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def parse_rule_text(text: str) -> dict:
    """Parse rule text to extract requirements."""
    rule = {"type": "simple", "description": text}
    
    # Look for star ratings
    star_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*stars?', text, re.IGNORECASE)
    star_requirements = []
    for match in star_matches:
        star_requirements.append(float(match.group(1)))
    if star_requirements:
        rule["star_requirements"] = star_requirements
    
    # Look for basic number requirements like "5 of the following"
    count_match = re.search(r'(?:\*\*)?(\d+)(?:\s+of)(?:\*\*)?', text)
    if count_match:
        rule["required_count"] = int(count_match.group(1))
    
    # Look for compound requirements with AND/OR
    if " AND " in text.upper() or " OR " in text.upper():
        rule["type"] = "compound"
        rule["conditions"] = []
        
        # Split on OR first to get main alternatives
        or_parts = re.split(r'\s+OR\s+', text, flags=re.IGNORECASE)
        
        for or_part in or_parts:
            # Split each OR part on AND to get required components
            and_parts = re.split(r'\s+AND\s+', or_part, flags=re.IGNORECASE)
            
            if len(and_parts) > 1:
                # Multiple AND conditions
                and_condition = {
                    "type": "and",
                    "conditions": []
                }
                
                for part in and_parts:
                    condition = {}
                    # Look for star ratings
                    star_match = re.search(r'(\d+(?:\.\d+)?)\s*stars?', part, re.IGNORECASE)
                    if star_match:
                        condition["stars"] = float(star_match.group(1))
                    
                    # Look for count requirements
                    count_match = re.search(r'(?:\*\*)?(\d+)(?:\s+of)(?:\*\*)?', part)
                    if count_match:
                        condition["required_count"] = int(count_match.group(1))
                    
                    # Look for references to attribute groups
                    group_ref = re.search(r'(?:from|of)\s+(?:the\s+)?([\w-]+(?:\s+[\w-]+)*?)(?:\s+attributes?|$)', part, re.IGNORECASE)
                    if group_ref:
                        condition["group_name"] = group_ref.group(1).lower()
                    
                    if condition:
                        and_condition["conditions"].append(condition)
                
                if and_condition["conditions"]:
                    rule["conditions"].append(and_condition)
            else:
                # Single condition
                condition = {}
                # Look for star ratings
                star_match = re.search(r'(\d+(?:\.\d+)?)\s*stars?', and_parts[0], re.IGNORECASE)
                if star_match:
                    condition["stars"] = float(star_match.group(1))
                
                # Look for count requirements
                count_match = re.search(r'(?:\*\*)?(\d+)(?:\s+of)(?:\*\*)?', and_parts[0])
                if count_match:
                    condition["required_count"] = int(count_match.group(1))
                
                # Look for references to attribute groups
                group_ref = re.search(r'(?:from|of)\s+(?:the\s+)?([\w-]+(?:\s+[\w-]+)*?)(?:\s+attributes?|$)', and_parts[0], re.IGNORECASE)
                if group_ref:
                    condition["group_name"] = group_ref.group(1).lower()
                
                if condition:
                    rule["conditions"].append(condition)
        
        if len(or_parts) > 1:
            rule["operator"] = "OR"
        elif rule["conditions"] and isinstance(rule["conditions"][0], dict) and rule["conditions"][0].get("type") == "and":
            rule["operator"] = "AND"
    
    return rule

def find_rules_in_text(text: str) -> list:
    """Find and parse rules in text."""
    rules = []
    
    # Look for rule patterns
    patterns = [
        r'Properties must have.*?(?=\n|$)',  # "Properties must have X of the following"
        r'(?:\*\*.*?\*\*(?:\s*(?:OR|AND)\s*\*\*.*?\*\*)*)',  # Bold text with AND/OR
        r'Minimum requirement is.*?(?=\n|$)',  # "Minimum requirement is X"
        r'(?:Must have|Required:).*?(?=\n|$)',  # Other common rule formats
        r'(?:Property has|Properties have)\s+.*?(?=(?:\s*(?:AND|OR|<br>|\n)|$))'  # Property has X stars
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            rule_text = match.group(0)
            if any(keyword in rule_text.lower() for keyword in 
                ["must have", "of the following", "minimum requirement", "property has", "properties have"]):
                rule = parse_rule_text(rule_text)
                rules.append(rule)
    
    return rules

def parse_family_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Family theme which has:
    1. Split between Conventional Lodgings and Vacation Rentals
    2. Primary vs Secondary attributes
    3. Complex category-based requirements
    """
    section_data = {
        "conventional_lodging": {
            "description": "",
            "applicable_structure_types": [],
            "rules": [],
            "primary_attributes": [],
            "secondary_attributes": []
        },
        "vacation_rental": {
            "description": "",
            "applicable_structure_types": [],
            "rules": [],
            "primary_attributes": [],
            "secondary_attributes": []
        }
    }
    
    current_idx = start_idx
    current_section = None
    current_subsection = None
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Detect subsection headers FIRST (before checking for main section breaks)
        if "Primary Attributes" in line and line.startswith('###'):
            # Process any pending table before changing subsection
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section and current_subsection:
                    if current_section == "conventional":
                        if current_subsection == "primary":
                            section_data["conventional_lodging"]["primary_attributes"].extend(table_data)
                        elif current_subsection == "secondary":
                            section_data["conventional_lodging"]["secondary_attributes"].extend(table_data)
                    elif current_section == "vacation":
                        if current_subsection == "primary":
                            section_data["vacation_rental"]["primary_attributes"].extend(table_data)
                        elif current_subsection == "secondary":
                            section_data["vacation_rental"]["secondary_attributes"].extend(table_data)
                current_content = []
            current_subsection = "primary"
        elif "Secondary Attributes" in line and line.startswith('###'):
            # Process any pending table before changing subsection
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section and current_subsection:
                    if current_section == "conventional":
                        if current_subsection == "primary":
                            section_data["conventional_lodging"]["primary_attributes"].extend(table_data)
                        elif current_subsection == "secondary":
                            section_data["conventional_lodging"]["secondary_attributes"].extend(table_data)
                    elif current_section == "vacation":
                        if current_subsection == "primary":
                            section_data["vacation_rental"]["primary_attributes"].extend(table_data)
                        elif current_subsection == "secondary":
                            section_data["vacation_rental"]["secondary_attributes"].extend(table_data)
                current_content = []
            current_subsection = "secondary"
        # Check for next main section (only if it's not a subsection header)
        elif line.startswith('##') and not "Qualifying Attributes" in line:
            break
        # Detect section headers
        elif "Conventional Lodgings" in line:
            current_section = "conventional"
        elif "Vacation Rentals" in line:
            current_section = "vacation"
        # Handle structure type lists
        elif line.startswith("**Applicable Structure Types**:"):
            types_text = line.replace("**Applicable Structure Types**:", "").strip()
            structure_types = [t.strip() for t in types_text.split(',')]
            if current_section == "conventional":
                section_data["conventional_lodging"]["applicable_structure_types"] = structure_types
            elif current_section == "vacation":
                section_data["vacation_rental"]["applicable_structure_types"] = structure_types
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section and current_subsection:
                    if current_section == "conventional":
                        if current_subsection == "primary":
                            section_data["conventional_lodging"]["primary_attributes"].extend(table_data)
                        elif current_subsection == "secondary":
                            section_data["conventional_lodging"]["secondary_attributes"].extend(table_data)
                    elif current_section == "vacation":
                        if current_subsection == "primary":
                            section_data["vacation_rental"]["primary_attributes"].extend(table_data)
                        elif current_subsection == "secondary":
                            section_data["vacation_rental"]["secondary_attributes"].extend(table_data)
                current_content = []
            
            # Add description text and parse rules
            if line and not line.startswith('###'):
                if current_section == "conventional":
                    section_data["conventional_lodging"]["description"] += line + "\n"
                elif current_section == "vacation":
                    section_data["vacation_rental"]["description"] += line + "\n"
                
                # Parse rules
                if "must have" in line.lower():
                    rule_info = {"type": "complex", "description": line}
                    if current_section == "conventional":
                        section_data["conventional_lodging"]["rules"].append(rule_info)
                    elif current_section == "vacation":
                        section_data["vacation_rental"]["rules"].append(rule_info)
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if table_data and current_section and current_subsection:
            if current_section == "conventional":
                if current_subsection == "primary":
                    section_data["conventional_lodging"]["primary_attributes"].extend(table_data)
                elif current_subsection == "secondary":
                    section_data["conventional_lodging"]["secondary_attributes"].extend(table_data)
            elif current_section == "vacation":
                if current_subsection == "primary":
                    section_data["vacation_rental"]["primary_attributes"].extend(table_data)
                elif current_subsection == "secondary":
                    section_data["vacation_rental"]["secondary_attributes"].extend(table_data)
    
    # Clean up descriptions
    section_data["conventional_lodging"]["description"] = clean_section_text(section_data["conventional_lodging"]["description"])
    section_data["vacation_rental"]["description"] = clean_section_text(section_data["vacation_rental"]["description"])
    
    return section_data, current_idx

def parse_luxury_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Luxury theme which has:
    1. Star-based requirements (5 stars, 4.5 stars, 4 stars + affiliations)
    2. Awards and Affiliations table
    """
    section_data = {
        "description": "",
        "rules": [
            {"type": "star_rating", "description": "Property has 5 stars", "stars": 5.0},
            {"type": "star_rating", "description": "Property has 4.5 stars", "stars": 4.5},
            {"type": "star_rating_with_affiliations", "description": "Property has 4 stars AND at least 1 award/affiliation", "stars": 4.0}
        ],
        "awards_affiliations": []
    }
    
    current_idx = start_idx
    current_content = []
    in_affiliations_section = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Detect affiliations section FIRST (before checking for main section breaks)
        if "Awards and Affiliations" in line:
            # Process any pending table before changing section
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and in_affiliations_section:
                    section_data["awards_affiliations"].extend(table_data)
                current_content = []
            in_affiliations_section = True
            current_idx += 1
            continue
        # Check for next main section (only if it's not a subsection header)
        elif line.startswith('##') and not "Qualifying Attributes" in line:
            break
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and in_affiliations_section:
                    section_data["awards_affiliations"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if in_affiliations_section:
            section_data["awards_affiliations"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_spa_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Spa theme which has:
    1. Full-service spa OR complex requirements
    2. Primary attributes (2 of) AND Additional attributes (3 of)
    """
    section_data = {
        "description": "",
        "rules": [
            {"type": "alternative", "description": "Full-service spa OR 2 primary + 3 additional attributes"}
        ],
        "full_service_spa": [],
        "primary_attributes": [],
        "additional_attributes": []
    }
    
    current_idx = start_idx
    current_section = None
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Detect sections FIRST (before checking for main section breaks)
        if "Primary Attributes" in line:
            # Process any pending table before changing section
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section:
                    if current_section == "primary":
                        section_data["primary_attributes"].extend(table_data)
                    elif current_section == "additional":
                        section_data["additional_attributes"].extend(table_data)
                    elif current_section == "full_service":
                        section_data["full_service_spa"].extend(table_data)
                current_content = []
            current_section = "primary"
        elif "Additional Attributes" in line:
            # Process any pending table before changing section
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section:
                    if current_section == "primary":
                        section_data["primary_attributes"].extend(table_data)
                    elif current_section == "additional":
                        section_data["additional_attributes"].extend(table_data)
                    elif current_section == "full_service":
                        section_data["full_service_spa"].extend(table_data)
                current_content = []
            current_section = "additional"
        elif "Full-Service Spa" in line:
            # Process any pending table before changing section
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section:
                    if current_section == "primary":
                        section_data["primary_attributes"].extend(table_data)
                    elif current_section == "additional":
                        section_data["additional_attributes"].extend(table_data)
                    elif current_section == "full_service":
                        section_data["full_service_spa"].extend(table_data)
                current_content = []
            current_section = "full_service"
        # Check for next main section (only if it's not a subsection header)
        elif line.startswith('##') and not "Qualifying Attributes" in line:
            break
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data and current_section:
                    if current_section == "primary":
                        section_data["primary_attributes"].extend(table_data)
                    elif current_section == "additional":
                        section_data["additional_attributes"].extend(table_data)
                    elif current_section == "full_service":
                        section_data["full_service_spa"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if current_section == "primary":
            section_data["primary_attributes"].extend(table_data)
        elif current_section == "additional":
            section_data["additional_attributes"].extend(table_data)
        elif current_section == "full_service":
            section_data["full_service_spa"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_shopping_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Shopping theme which has:
    1. City-based qualification requirements
    2. Shopping district destinations list
    """
    section_data = {
        "description": "",
        "rules": [
            {"type": "location_based", "description": "Must be in shopping district of qualifying city"}
        ],
        "qualifying_cities": [],
        "requirements": []
    }
    
    current_idx = start_idx
    current_content = []
    in_destinations_section = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Detect destinations section
        if "Shopping Theme Destinations" in line:
            in_destinations_section = True
            current_idx += 1
            continue
        # Handle city list items
        elif line.startswith('- ') and in_destinations_section:
            city_name = line[2:].strip()  # Remove '- ' prefix
            section_data["qualifying_cities"].append(city_name)
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                section_data["requirements"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###') and not in_destinations_section:
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["requirements"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_winery_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Winery theme which has:
    1. Alternative requirements (Winery attached OR Both vineyard + tasting room)
    2. Clear OR structure
    """
    section_data = {
        "description": "",
        "rules": [
            {"type": "alternative", "description": "Winery attached OR Both vineyard + tasting room"}
        ],
        "single_requirements": [],
        "combined_requirements": []
    }
    
    current_idx = start_idx
    current_content = []
    in_combined_section = False
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Detect sections
        if "Both" in line and "following" in line:
            in_combined_section = True
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if in_combined_section:
                    section_data["combined_requirements"].extend(table_data)
                else:
                    section_data["single_requirements"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if in_combined_section:
            section_data["combined_requirements"].extend(table_data)
        else:
            section_data["single_requirements"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def get_theme_parser(theme_name: str) -> callable:
    """
    Returns the appropriate parser function for a given theme.
    Each theme may have unique parsing requirements.
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
        "Luxury": parse_luxury_theme_qualifying_attributes,
        "Shopping": parse_shopping_theme_qualifying_attributes,
        "Spa": parse_spa_theme_qualifying_attributes,
        "Winery": parse_winery_theme_qualifying_attributes,
    }
    
    if not theme_name:
        print("DEBUG: No theme name provided, using default parser")
        return parse_default_theme_qualifying_attributes
    
    # First try exact match
    if theme_name in theme_parsers:
        print(f"DEBUG: Found exact match for '{theme_name}'")
        return theme_parsers[theme_name]
    
    # Then try normalized match
    normalized_name = normalize_theme_name(theme_name)
    print(f"DEBUG: Normalized theme name: '{normalized_name}'")
    
    for key in theme_parsers:
        normalized_key = normalize_theme_name(key)
        print(f"DEBUG: Checking if '{normalized_key}' in '{normalized_name}'")
        if normalized_key in normalized_name or normalized_name in normalized_key:
            print(f"DEBUG: Found match! Using parser for '{key}'")
            return theme_parsers[key]
    
    print("DEBUG: No specific parser found, using default parser")
    return parse_default_theme_qualifying_attributes

def parse_adventure_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Adventure theme which has:
    1. Main attributes list
    2. Separate ski-related attributes list
    3. Complex requirements (5 from main OR 4 from main + 1 from ski)
    """
    section_data = {
        "description": "",
        "rules": [],
        "attribute_groups": [
            {
                "name": "main_attributes",
                "description": "Main adventure sport attributes",
                "attributes": []
            },
            {
                "name": "ski_attributes",
                "description": "Ski-related attributes",
                "attributes": []
            }
        ]
    }
    
    current_idx = start_idx
    current_content = []
    in_ski_section = False
    
    print(f"\n=== DEBUG ADVENTURE PARSER: Starting at index {start_idx} ===")
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section or end markers
        if (line.startswith('## ') and not "Qualifying Attributes" in line) or line.startswith('===='):
            print(f"DEBUG ADVENTURE: Breaking at line {current_idx}: {line[:50]}")
            break
            
        # Check for ski section header (allow additional text after "Ski-Related Attributes")
        if line.startswith('### Ski-Related Attributes'):
            print(f"DEBUG ADVENTURE: Found ski section at line {current_idx}")
            print(f"DEBUG ADVENTURE: Current content has {len(current_content)} lines before processing")
            # Process any pending table as main attributes before switching
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data:
                    print(f"DEBUG ADVENTURE: Adding {len(table_data)} rows to main_attributes")
                    section_data["attribute_groups"][0]["attributes"].extend(table_data)
                current_content = []
            in_ski_section = True
            print(f"DEBUG ADVENTURE: Set in_ski_section = True")
            current_idx += 1
            continue
            
        # Handle table lines
        if line.startswith('|'):
            current_content.append(line)
            if in_ski_section and len(current_content) <= 3:  # Show first few ski table lines
                print(f"DEBUG ADVENTURE: Ski table line {len(current_content)}: {line}")
        elif line == '':
            # Blank line: just continue
            pass
        else:
            # Process completed table when we hit non-table, non-blank line
            if current_content and any(l.startswith('|') for l in current_content):
                print(f"DEBUG ADVENTURE: Processing table with {len(current_content)} lines, in_ski_section = {in_ski_section}")
                table_data = parse_markdown_table_flexible(current_content)
                if table_data:
                    if in_ski_section:
                        print(f"DEBUG ADVENTURE: Adding {len(table_data)} rows to ski_attributes")
                        section_data["attribute_groups"][1]["attributes"].extend(table_data)
                    else:
                        print(f"DEBUG ADVENTURE: Adding {len(table_data)} rows to main_attributes")
                        section_data["attribute_groups"][0]["attributes"].extend(table_data)
                current_content = []
            
            # Add meaningful description text (but not delimiter lines)
            if line and not line.startswith('###') and not line.startswith('===='):
                section_data["description"] += line + "\n"
                # Parse rules from description
                if "must have" in line.lower() or "of the following" in line:
                    rule_info = {
                        "type": "compound",
                        "description": line,
                        "required_count": 5,
                        "conditions": [
                            {"required_count": 5, "group_name": "main_attributes"},
                            {"required_count": 4, "group_name": "main_attributes", "additional_required": 1, "additional_group": "ski_attributes"}
                        ],
                        "operator": "OR"
                    }
                    section_data["rules"].append(rule_info)
        
        current_idx += 1
    
    # Process any final table - this is crucial for the ski attributes
    print(f"DEBUG ADVENTURE: Final processing - current_content has {len(current_content)} lines, in_ski_section = {in_ski_section}")
    if current_content and any(l.startswith('|') for l in current_content):
        print(f"DEBUG ADVENTURE: Final table processing...")
        table_data = parse_markdown_table_flexible(current_content)
        if table_data:
            if in_ski_section:
                print(f"DEBUG ADVENTURE: FINAL - Adding {len(table_data)} rows to ski_attributes")
                section_data["attribute_groups"][1]["attributes"].extend(table_data)
            else:
                print(f"DEBUG ADVENTURE: FINAL - Adding {len(table_data)} rows to main_attributes")
                section_data["attribute_groups"][0]["attributes"].extend(table_data)
    
    print(f"DEBUG ADVENTURE: Final counts - main: {len(section_data['attribute_groups'][0]['attributes'])}, ski: {len(section_data['attribute_groups'][1]['attributes'])}")
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_beach_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Beach theme which has:
    1. Two-part qualification: Location (1 of) AND View/Amenities (1 of)
    2. Clear structure separation
    """
    section_data = {
        "description": "",
        "rules": [],
        "location_attributes": [],
        "view_amenity_attributes": [],
        "disqualifications": []
    }
    
    current_idx = start_idx
    current_section = None
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Detect sections based on content
        if line == "**1 of:**":
            # Process any previous table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if current_section == "location":
                    section_data["location_attributes"].extend(table_data)
                elif current_section == "view_amenity":
                    section_data["view_amenity_attributes"].extend(table_data)
                current_content = []
            current_section = "location"
            section_data["rules"].append({"type": "location", "count": 1})
        elif line == "**AND 1 of:**":
            # Process previous table as location attributes
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                section_data["location_attributes"].extend(table_data)
                current_content = []
            current_section = "view_amenity"
            section_data["rules"].append({"type": "view_amenity", "count": 1})
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if current_section == "location":
                    section_data["location_attributes"].extend(table_data)
                elif current_section == "view_amenity":
                    section_data["view_amenity_attributes"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Look for disqualifications
                if "do not qualify" in line.lower():
                    section_data["disqualifications"].append(line)
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if current_section == "location":
            section_data["location_attributes"].extend(table_data)
        elif current_section == "view_amenity":
            section_data["view_amenity_attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_default_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Default parser for themes without special requirements.
    Captures basic structure of attributes and rules.
    """
    section_data = {
        "description": "",
        "attributes": [],
        "rules": []
    }
    
    current_idx = start_idx
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Handle table lines
        if line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data:
                    section_data["attributes"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Parse rules
                if "must have" in line.lower() or "of the following" in line:
                    section_data["rules"].append(parse_rule_text(line))
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if table_data:
            section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_qualifying_attributes_section(lines: list[str], start_idx: int, theme_name: str = None) -> tuple[dict, int]:
    """
    Parse qualifying attributes section, using specialized parsers for specific themes.
    """
    if not theme_name:
        # Try to determine theme from the content
        for i in range(max(0, start_idx - 20), start_idx):
            if i < len(lines):
                line = lines[i].strip()
                if line.startswith("# "):
                    theme_name = line.lstrip("# ").split("-")[0].strip()
                    break
    
    # Get the appropriate parser for this theme
    parser = get_theme_parser(theme_name)
    return parser(lines, start_idx)

def parse_theme_block_from_insights(theme_block_text: str) -> dict:
    theme_data = {"name_from_header": None}
    lines = theme_block_text.split('\n')
    
    # Extract theme name from header
    for line in lines:
        if line.startswith('# '):
            theme_data["name_from_header"] = line.lstrip('# ').split('-')[0].strip()
            break
    
    # Initialize parsed_rules_details
    parsed_rules_details = {
        "overview_data": {},
        "overview_text": "",
        "definition": {
            "text": "",
            "uuid": None,
            "urn": None,
            "attributes": []
        },
        "qualifying_attributes": {
            "description": "",
            "rules": [],
            "attribute_groups": [],
            "attributes": []
        },
        "disqualifying_attributes": {
            "description": "",
            "attributes": []
        },
        "special_sections": {}
    }
    
    # Parse each section
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('## Overview'):
            overview_data, overview_text, new_idx = parse_overview_section(lines, i + 1)
            parsed_rules_details["overview_data"] = overview_data
            parsed_rules_details["overview_text"] = overview_text
            # Copy UUID and URN to definition section
            if "uuid" in overview_data:
                parsed_rules_details["definition"]["uuid"] = overview_data["uuid"]
            if "urn" in overview_data:
                parsed_rules_details["definition"]["urn"] = overview_data["urn"]
            i = new_idx
        elif line.startswith('## Definition'):
            definition_data, new_idx = parse_definition_section(lines, i + 1)
            parsed_rules_details["definition"].update(definition_data)
            i = new_idx
        elif line.startswith('## Qualifying Attributes'):
            qa_data, new_idx = parse_qualifying_attributes_section(lines, i, theme_data["name_from_header"])
            # Integrate specialized parser output into the theme data structure
            if isinstance(qa_data, dict):
                if "conventional_lodging" in qa_data and "vacation_rental" in qa_data:
                    parsed_rules_details["qualifying_attributes"] = {
                        "split_sections": True,
                        "conventional_lodging": qa_data["conventional_lodging"],
                        "vacation_rental": qa_data["vacation_rental"]
                    }
                    if "qualifying_brands" in qa_data:
                        parsed_rules_details["qualifying_attributes"]["qualifying_brands"] = qa_data["qualifying_brands"]
                elif "attribute_groups" in qa_data:
                    # Handle structured attribute groups (like Adventure theme)
                    print("\n=== DEBUG: Processing attribute groups in parse_theme_block_from_insights ===")
                    print(f"Raw qa_data structure: {json.dumps(qa_data, indent=2)}")
                    print(f"Number of attribute groups: {len(qa_data['attribute_groups'])}")
                    for group in qa_data["attribute_groups"]:
                        print(f"\nGroup name: {group['name']}")
                        print(f"Group description: {group.get('description', 'No description')}")
                        print(f"Number of attributes in group: {len(group['attributes'])}")
                        if group['name'] == 'ski_related_attributes':
                            print("Ski-related attributes content:")
                            for attr in group['attributes']:
                                print(f"  {attr}")
                    
                    # Preserve the attribute_groups structure
                    parsed_rules_details["qualifying_attributes"] = {
                        "description": qa_data["description"],
                        "rules": qa_data["rules"],
                        "attribute_groups": qa_data["attribute_groups"]
                    }
                    
                    print("\nVerifying structure after assignment:")
                    print(f"Final structure: {json.dumps(parsed_rules_details['qualifying_attributes'], indent=2)}")
                elif "location_attributes" in qa_data and "view_amenity_attributes" in qa_data:
                    parsed_rules_details["qualifying_attributes"] = {
                        "description": qa_data["description"],
                        "rules": qa_data["rules"],
                        "location_attributes": qa_data["location_attributes"],
                        "view_amenity_attributes": qa_data["view_amenity_attributes"],
                        "disqualifications": qa_data.get("disqualifications", [])
                    }
                elif "manual_assignment" in qa_data:
                    parsed_rules_details["qualifying_attributes"] = {
                        "description": qa_data["description"],
                        "manual_assignment": True,
                        "country_rules": qa_data["country_rules"],
                        "criteria": qa_data["criteria"]
                    }
                elif "certification_programs" in qa_data:
                    parsed_rules_details["qualifying_attributes"] = {
                        "description": qa_data["description"],
                        "certification_programs": qa_data["certification_programs"],
                        "update_frequency": qa_data["update_frequency"],
                        "documentation_requirements": qa_data["documentation_requirements"],
                        "rules": qa_data["rules"]
                    }
                elif "exceptions" in qa_data:
                    parsed_rules_details["qualifying_attributes"] = {
                        "description": qa_data["description"],
                        "attributes": qa_data["attributes"],
                        "exceptions": qa_data["exceptions"]
                    }
                else:
                    parsed_rules_details["qualifying_attributes"] = qa_data
            i = new_idx
        elif line.startswith('## Disqualifying Attributes'):
            disq_data, new_idx = parse_disqualifying_attributes_section(lines, i + 1)
            parsed_rules_details["disqualifying_attributes"] = disq_data
            i = new_idx
        else:
            i += 1
    
    theme_data["parsed_rules_details"] = parsed_rules_details
    return theme_data


def load_themes_summary(filepath: str) -> dict:
    """Loads the themes_summary.txt table into a dictionary keyed by URN."""
    themes_map = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        table_match = re.search(r"## Themes Definitions\s*\n(\|.*?\n)((?:\|--.*?--\|\s*\n)?)(.*?)(?=\n##|\n\*\*Note|$)", content, re.DOTALL | re.IGNORECASE)
        if not table_match:
            print("Error: Could not find 'Themes Definitions' table in themes_summary.txt")
            return {}
            
        table_header_line = table_match.group(1)
        # Group 2 is the separator, Group 3 is the data
        table_data_lines_str = table_match.group(3)
        
        # Construct full table text for parser
        full_table_text = table_header_line.strip() + "\n"
        if table_match.group(2): # Include separator if present
             full_table_text += table_match.group(2).strip() + "\n"
        full_table_text += table_data_lines_str.strip()

        table_lines = [line.strip() for line in full_table_text.split('\n') if line.strip().startswith('|')]

        parsed_table = parse_markdown_table_flexible(table_lines)
        
        for row in parsed_table:
            urn = row.get("URN", "").strip()
            if urn:
                themes_map[urn] = {
                    "theme_attribute_name": row.get("Theme Attribute Name", "").strip(),
                    "lcm_id": row.get("Attribute LCM ID", "").strip(),
                    "uuid": row.get("UUID", "").strip(),
                    "theme_automation": row.get("Theme Automation", "").strip().lower() == "yes",
                    "is_business_rule_theme": True, # All themes from this file are
                    "live_site_filter": row.get("Live site filter (as of early 2024)", "").strip().lower() == "yes",
                    "properties_count_early_2024": row.get("# Properties (as of early 2024)", "").strip()
                }
            else:
                print(f"Warning: Row in themes_summary.txt missing URN: {row}")
        print(f"Loaded {len(themes_map)} themes from themes_summary.txt")
    except Exception as e:
        print(f"Error loading themes_summary.txt: {e}")
        import traceback
        traceback.print_exc()
    return themes_map

def load_lgbtq_excluded_countries(filepath: str) -> list[str]:
    excluded_countries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num == 0 and "Country code" in line and "Country" in line: # Header
                    continue
                parts = re.split(r'\s{2,}|[\t]', line.strip())
                if len(parts) >= 2:
                    country_code = parts[1].strip()
                    if "ARE (or UAE)" in country_code: country_code = "ARE"
                    if country_code and len(country_code) == 3 and country_code.isalpha():
                        excluded_countries.append(country_code.upper())
        print(f"Loaded {len(excluded_countries)} LGBTQ+ excluded country codes.")
    except FileNotFoundError:
        print(f"Warning: LGBTQ countries file not found at {filepath}. Proceeding without it.")
    except Exception as e:
        print(f"Error loading LGBTQ excluded countries file '{filepath}': {e}")
    return excluded_countries


def normalize_theme_name(name: str) -> str:
    """Normalize theme names for matching by removing common variations and special characters."""
    normalized = name.lower()
    # Remove common suffixes and prefixes
    normalized = normalized.replace(" theme definition", "")
    normalized = normalized.replace(" property", "")
    normalized = normalized.replace(" friendly*", "")
    normalized = normalized.replace(" friendly", "")
    normalized = normalized.replace("-friendly", "")
    normalized = normalized.replace("certified", "")
    # Remove special characters and extra spaces
    normalized = normalized.replace("-", "")
    normalized = normalized.replace("*", "")
    normalized = normalized.strip()
    return normalized

def pre_processor(insights_filepath: str, summary_filepath: str, lgbtq_countries_filepath:str, output_dir: str):
    if not os.path.exists(insights_filepath):
        print(f"Error: Insights file not found at {insights_filepath}"); return
    if not os.path.exists(summary_filepath):
        print(f"Error: Themes summary file not found at {summary_filepath}"); return

    os.makedirs(output_dir, exist_ok=True)

    summary_themes_data = load_themes_summary(summary_filepath)
    if not summary_themes_data:
        print("Halting due to failure in loading themes_summary.txt"); return

    # Create UUID to theme data mapping from summary data
    summary_uuid_to_theme = {data["uuid"]: {"urn": urn, "data": data} 
                           for urn, data in summary_themes_data.items()}

    lgbtq_excluded_list = load_lgbtq_excluded_countries(lgbtq_countries_filepath)

    with open(insights_filepath, 'r', encoding='utf-8') as f:
        insights_content = f.read().strip()
    
    # Remove any leading delimiter line
    insights_content = re.sub(r'^=+\n', '', insights_content)
    
    # Split on delimiter lines
    theme_insight_blocks = re.split(r'\n=+\n', insights_content)
    
    parsed_insights_map = {} # Keyed by UUID
    unmatched_blocks = [] # Store blocks that couldn't be matched by UUID

    for block_num, block in enumerate(theme_insight_blocks):
        block = block.strip()
        if not block or not block.startswith("# "):
            if block: print(f"Skipping minor block/remnant: {block[:50]}...")
            continue
        print(f"\n--- Parsing Insight Block {block_num+1} ---")
        print(block[:200] + "...\n")
        detailed_theme_data = parse_theme_block_from_insights(block)
        print(f"[DEBUG] Parsed theme data for block {block_num+1}: {json.dumps(detailed_theme_data, indent=2)[:1000]}...")
        
        block_header_name = detailed_theme_data.get("name_from_header")
        block_uuid = None
        
        # Look for UUID in definition section first
        if detailed_theme_data.get("parsed_rules_details", {}).get("definition", {}).get("uuid"):
            block_uuid = detailed_theme_data["parsed_rules_details"]["definition"]["uuid"]
        
        # Look in overview data if not found in definition
        if not block_uuid:
            block_uuid = detailed_theme_data.get("parsed_rules_details", {}).get("overview_data", {}).get("uuid")
        
        # Look in any tables if still not found
        if not block_uuid:
            for section_name, section_data in detailed_theme_data.get("parsed_rules_details", {}).items():
                if isinstance(section_data, dict):
                    # Check attributes list
                    if "attributes" in section_data:
                        for attr in section_data["attributes"]:
                            if "uuid" in attr:
                                uuid_match = re.match(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', attr["uuid"])
                                if uuid_match:
                                    block_uuid = uuid_match.group(0)
                                    break
                        if block_uuid:
                            break
                    
                    # Check nested attribute groups
                    if "attribute_groups" in section_data:
                        for group in section_data["attribute_groups"]:
                            for attr in group.get("available_attributes", []):
                                if "uuid" in attr:
                                    uuid_match = re.match(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', attr["uuid"])
                                    if uuid_match:
                                        block_uuid = uuid_match.group(0)
                                        break
                            if block_uuid:
                                break
                    if block_uuid:
                        break

        # Try to match with summary themes by UUID
        if block_uuid and block_uuid in summary_uuid_to_theme:
            matched_summary = summary_uuid_to_theme[block_uuid]
            print(f"Matched by UUID: {block_uuid} -> {matched_summary['urn']}")
            
            # Update the URN in the parsed data to match summary
            if "parsed_rules_details" not in detailed_theme_data:
                detailed_theme_data["parsed_rules_details"] = {}
            if "definition" not in detailed_theme_data["parsed_rules_details"]:
                detailed_theme_data["parsed_rules_details"]["definition"] = {}
            detailed_theme_data["parsed_rules_details"]["definition"]["urn"] = matched_summary["urn"]
            
            parsed_insights_map[block_uuid] = detailed_theme_data
        else:
            if block_uuid:
                print(f"Warning: Could not match UUID '{block_uuid}' from theme block '{block_header_name}' with any summary UUID")
            elif block_header_name:
                print(f"Warning: No UUID found in theme block '{block_header_name}'")
            else:
                print(f"Warning: Could not determine UUID or Name for insight block: {block[:100]}...")
            unmatched_blocks.append((block_header_name, detailed_theme_data))

    # Generate output files
    final_themes_output = []
    for summary_uuid, summary_info in summary_uuid_to_theme.items():
        merged_theme_data = summary_info["data"].copy()
        merged_theme_data["urn"] = summary_info["urn"]

        if summary_uuid in parsed_insights_map:
            detailed_insight = parsed_insights_map[summary_uuid]
            # Merge the parsed rules details
            if "parsed_rules_details" in detailed_insight:
                merged_theme_data["parsed_rules_details"] = detailed_insight["parsed_rules_details"]
        else:
            print(f"Warning: No detailed insights found for UUID: {summary_uuid} (URN: {summary_info['urn']}, Name: {summary_info['data'].get('theme_attribute_name')})")
            merged_theme_data["parsed_rules_details"] = {"status": "No detailed rules found in insights file"}

        if summary_info["urn"] == "LGBTQIAWelcoming" and lgbtq_excluded_list:
            if "special_sections" not in merged_theme_data.get("parsed_rules_details", {}):
                if "parsed_rules_details" not in merged_theme_data: 
                    merged_theme_data["parsed_rules_details"] = {}
                merged_theme_data["parsed_rules_details"]["special_sections"] = defaultdict(lambda: {"text": "", "tables": [], "attributes": []})

            merged_theme_data["parsed_rules_details"]["special_sections"]["country_exclusions_lgbtq"] = {
                "lists": [sorted(list(set(lgbtq_excluded_list)))],
                "text": "List of countries where the LGBTQ Friendly theme should not be applied due to civil penalties.",
            }
        
        print(f"[DEBUG] Writing merged theme data for {summary_info['urn']}: {json.dumps(merged_theme_data, indent=2)[:1000]}...")
        final_themes_output.append(merged_theme_data)

        filename_theme_urn = re.sub(r'[^\w_-]', '', summary_info["urn"].lower())
        output_filename = os.path.join(output_dir, f"theme_{filename_theme_urn}.json")
        try:
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                json.dump(merged_theme_data, outfile, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing JSON for theme URN '{summary_info['urn']}': {e}")

    combined_output_filename = os.path.join(output_dir, "all_business_themes_structured.json")
    with open(combined_output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(final_themes_output, outfile, indent=2, ensure_ascii=False)
    print(f"Successfully wrote {len(final_themes_output)} merged themes to {combined_output_filename}")

    if unmatched_blocks:
        print("\nWarning: The following theme blocks could not be matched by UUID:")
        for name, _ in unmatched_blocks:
            print(f"- {name}")

def parse_boutique_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Boutique theme which has:
    1. Manual assignment criteria
    2. Country-specific rules (Greece, Turkey)
    3. Complex qualification criteria
    """
    section_data = {
        "description": "",
        "manual_assignment": True,
        "country_rules": {},
        "criteria": []
    }
    
    current_idx = start_idx
    current_country = None
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Check for country-specific rules
        if "Special Rule for" in line:
            current_country = line.split("for")[-1].strip()
            section_data["country_rules"][current_country] = {"rules": [], "urns": []}
        elif current_country and line.startswith('-'):
            if 'URN' in line:
                urn = re.search(r'`([^`]+)`', line)
                if urn:
                    section_data["country_rules"][current_country]["urns"].append(urn.group(1))
            else:
                section_data["country_rules"][current_country]["rules"].append(line.lstrip('- '))
        # Add other lines to description
        elif line and not line.startswith('###'):
            section_data["description"] += line + "\n"
            # Look for criteria
            if any(keyword in line.lower() for keyword in ["must", "typically", "feature", "furnished"]):
                section_data["criteria"].append(line)
        
        current_idx += 1
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_business_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Business theme which has:
    1. Split between Conventional Lodgings and Vacation Rentals
    2. Qualifying brands list
    3. Complex attribute requirements
    """
    section_data = {
        "split_sections": True,
        "conventional_lodging": {
            "description": "",
            "applicable_structure_types": [],
            "rules": [],
            "attributes": []
        },
        "vacation_rental": {
            "description": "",
            "applicable_structure_types": [],
            "rules": [],
            "attributes": []
        },
        "qualifying_brands": []
    }
    
    current_idx = start_idx
    current_section = None
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Detect section headers FIRST (before checking for main section breaks)
        if "Conventional Lodgings" in line:
            current_section = "conventional"
        elif "Vacation Rentals" in line:
            current_section = "vacation"
        elif "Qualifying Brands" in line or "Business Theme Qualifying Brands" in line:
            current_section = "brands"
        # Check for next main section (only if it's not a section header)
        elif line.startswith('##') and not "Qualifying Attributes" in line:
            break
        # Handle structure type lists
        elif line.startswith("Applies to:"):
            # Extract structure types from the line
            types_text = line.replace("Applies to:", "").strip()
            structure_types = [t.strip() for t in types_text.split(',')]
            if current_section == "conventional":
                section_data["conventional_lodging"]["applicable_structure_types"] = structure_types
            elif current_section == "vacation":
                section_data["vacation_rental"]["applicable_structure_types"] = structure_types
        # Handle table lines
        elif line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and any(l.startswith('|') for l in current_content):
                table_data = parse_markdown_table_flexible(current_content)
                if table_data:
                    if current_section == "brands":
                        section_data["qualifying_brands"].extend(table_data)
                    elif current_section == "conventional":
                        section_data["conventional_lodging"]["attributes"].extend(table_data)
                    elif current_section == "vacation":
                        section_data["vacation_rental"]["attributes"].extend(table_data)
                current_content = []
            
            # Add description text
            if line and not line.startswith('###'):
                if current_section == "conventional":
                    section_data["conventional_lodging"]["description"] += line + "\n"
                elif current_section == "vacation":
                    section_data["vacation_rental"]["description"] += line + "\n"
                
                # Parse rules
                if "minimum requirement" in line.lower():
                    rule_info = {"type": "simple", "description": line}
                    if current_section == "conventional":
                        section_data["conventional_lodging"]["rules"].append(rule_info)
                    elif current_section == "vacation":
                        section_data["vacation_rental"]["rules"].append(rule_info)
        
        current_idx += 1
    
    # Process any final content
    if current_content and any(l.startswith('|') for l in current_content):
        table_data = parse_markdown_table_flexible(current_content)
        if current_section == "brands":
            section_data["qualifying_brands"].extend(table_data)
        elif current_section:
            section_data[current_section]["attributes"].extend(table_data)
    
    # Clean up descriptions
    section_data["conventional_lodging"]["description"] = clean_section_text(section_data["conventional_lodging"]["description"])
    section_data["vacation_rental"]["description"] = clean_section_text(section_data["vacation_rental"]["description"])
    
    return section_data, current_idx

def parse_casino_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Casino theme which has:
    1. Simple single attribute requirement
    2. Adjacent casino exception
    """
    section_data = {
        "description": "",
        "attributes": [],
        "exceptions": []
    }
    
    current_idx = start_idx
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Handle table lines
        if line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and current_content[0].startswith('|'):
                table_data = parse_markdown_table_flexible(current_content)
                section_data["attributes"].extend(table_data)
                current_content = []
            
            # Add non-table lines to description if meaningful
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Look for exceptions
                if "adjacent to" in line.lower():
                    section_data["exceptions"].append(line)
        
        current_idx += 1
    
    # Process any final content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["attributes"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_eco_certified_theme_qualifying_attributes(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Specialized parser for Eco-certified theme which has:
    1. Program-based certification requirements
    2. Quarterly updates
    3. Documentation requirements
    """
    section_data = {
        "description": "",
        "certification_programs": [],
        "update_frequency": "quarterly",
        "documentation_requirements": [],
        "rules": []
    }
    
    current_idx = start_idx
    current_content = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Qualifying Attributes" in line:
            break
            
        # Handle table lines
        if line.startswith('|'):
            current_content.append(line)
        else:
            # Process completed table
            if current_content and current_content[0].startswith('|'):
                table_data = parse_markdown_table_flexible(current_content)
                section_data["certification_programs"].extend(table_data)
                current_content = []
            
            # Add non-table lines to description if meaningful
            if line and not line.startswith('###'):
                section_data["description"] += line + "\n"
                # Look for documentation requirements
                if "documentation" in line.lower() or "certificate" in line.lower():
                    section_data["documentation_requirements"].append(line)
                # Look for rules
                elif "must" in line.lower():
                    section_data["rules"].append(parse_rule_text(line))
        
        current_idx += 1
    
    # Process any final content
    if current_content and current_content[0].startswith('|'):
        table_data = parse_markdown_table_flexible(current_content)
        section_data["certification_programs"].extend(table_data)
    
    # Clean up description
    section_data["description"] = clean_section_text(section_data["description"])
    
    return section_data, current_idx

def parse_overview_section(lines: list[str], start_idx: int) -> tuple[dict, str, int]:
    """
    Parse the overview section of a theme block.
    Returns (overview_data, overview_text, new_index).
    """
    overview_data = {}
    overview_text = ""
    current_idx = start_idx
    section_lines = []
    table_lines = []
    description_lines = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Overview" in line:
            break
            
        # Handle table lines
        if line.startswith('|'):
            table_lines.append(line)
        else:
            # Process completed table
            if table_lines:
                table_data = parse_markdown_table_flexible(table_lines)
                if table_data:
                    if "tables" not in overview_data:
                        overview_data["tables"] = []
                    overview_data["tables"].append(table_data)
                    
                    # Look for rules in table descriptions
                    for row in table_data:
                        if "Description" in row:
                            rules = find_rules_in_text(row["Description"])
                            if rules:
                                if "rules" not in overview_data:
                                    overview_data["rules"] = []
                                overview_data["rules"].extend(rules)
                table_lines = []
            
            # Add non-table lines to appropriate collection
            if line:
                if line.startswith('-') or '**' in line:
                    section_lines.append(line)
                else:
                    description_lines.append(line)
        
        current_idx += 1
    
    # Process any final table
    if table_lines:
        table_data = parse_markdown_table_flexible(table_lines)
        if table_data:
            if "tables" not in overview_data:
                overview_data["tables"] = []
            overview_data["tables"].append(table_data)
            
            # Look for rules in table descriptions
            for row in table_data:
                if "Description" in row:
                    rules = find_rules_in_text(row["Description"])
                    if rules:
                        if "rules" not in overview_data:
                            overview_data["rules"] = []
                        overview_data["rules"].extend(rules)
    
    # Parse key-value pairs from section lines
    overview_data.update(parse_key_value_lines(section_lines, 
        ["UUID", "URN", "Owned by", "Last Updated", "Automation", "Attribute ID"]))
    
    # Clean up and set overview text
    overview_text = clean_section_text("\n".join(description_lines))
    
    return overview_data, overview_text, current_idx

def parse_definition_section(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Parse the definition section of a theme block.
    Returns (definition_data, new_index).
    """
    definition_data = {
        "text": "",
        "uuid": None,
        "urn": None,
        "attributes": []
    }
    
    current_idx = start_idx
    table_lines = []
    description_lines = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Definition" in line:
            break
            
        # Handle table lines
        if line.startswith('|'):
            table_lines.append(line)
        else:
            # Process completed table
            if table_lines:
                table_data = parse_markdown_table_flexible(table_lines)
                if table_data:
                    definition_data["attributes"].extend(table_data)
                table_lines = []
            
            # Add non-table lines to description
            if line and not line.startswith('###'):
                description_lines.append(line)
        
        current_idx += 1
    
    # Process any final table
    if table_lines:
        table_data = parse_markdown_table_flexible(table_lines)
        if table_data:
            definition_data["attributes"].extend(table_data)
    
    # Clean up and set definition text
    definition_data["text"] = clean_section_text("\n".join(description_lines))
    
    return definition_data, current_idx

def parse_disqualifying_attributes_section(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Parse the disqualifying attributes section of a theme block.
    Returns (disqualifying_data, new_index).
    """
    section_data = {
        "description": "",
        "attributes": []
    }
    
    current_idx = start_idx
    table_lines = []
    description_lines = []
    
    while current_idx < len(lines):
        line = lines[current_idx].strip()
        
        # Check for next main section
        if line.startswith('##') and not "Disqualifying Attributes" in line:
            break
            
        # Handle table lines
        if line.startswith('|'):
            table_lines.append(line)
        else:
            # Process completed table
            if table_lines:
                table_data = parse_markdown_table_flexible(table_lines)
                if table_data:
                    section_data["attributes"].extend(table_data)
                table_lines = []
            
            # Add non-table lines to description
            if line and not line.startswith('###'):
                description_lines.append(line)
        
        current_idx += 1
    
    # Process any final table
    if table_lines:
        table_data = parse_markdown_table_flexible(table_lines)
        if table_data:
            section_data["attributes"].extend(table_data)
    
    # Clean up and set description
    section_data["description"] = clean_section_text("\n".join(description_lines))
    
    return section_data, current_idx

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python script.py <insights_file> <summary_file> <lgbtq_countries_file> <output_dir>")
        print(f"Received {len(sys.argv)} arguments: {sys.argv}")
        sys.exit(1)
    
    INSIGHTS_FILE = sys.argv[1]
    SUMMARY_FILE = sys.argv[2]
    LGBTQ_COUNTRIES_FILE = sys.argv[3]
    OUTPUT_DIR = sys.argv[4]
    
    print(f"Using files:")
    print(f"  Insights: {INSIGHTS_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")  
    print(f"  LGBTQ Countries: {LGBTQ_COUNTRIES_FILE}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    
    pre_processor(INSIGHTS_FILE, SUMMARY_FILE, LGBTQ_COUNTRIES_FILE, OUTPUT_DIR)

    print(f"\n--- Pre-processing complete. Check the '{OUTPUT_DIR}' directory. ---")
    # print("\n--- Sample of first theme from combined output: ---")
    # try:
    #     with open(os.path.join(OUTPUT_DIR, "all_business_themes_structured.json"), 'r', encoding='utf-8') as f:
    #         sample_data_all = json.load(f)
    #         if sample_data_all:
    #             print(json.dumps(sample_data_all[0], indent=2, ensure_ascii=False))
    # except Exception as e:
    #     print(f"Could not print sample from combined file: {e}")