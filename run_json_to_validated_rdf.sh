#!/bin/bash

# Default values
BATCH_SIZE=100
TAXONOMY_DIR="../datasources"

# Help function
show_help() {
    echo "Usage: $0 <input_json_file> <output_turtle_file> [-b batch_size] [-t taxonomy_dir]"
    echo
    echo "Options:"
    echo "  -b <size>    Batch size for processing (default: 100)"
    echo "  -t <dir>     Directory containing taxonomy files (default: ../datasources)"
    echo "  -h           Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--taxonomy-dir)
            TAXONOMY_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$INPUT_JSON" ]]; then
                INPUT_JSON="$1"
            elif [[ -z "$OUTPUT_TTL" ]]; then
                OUTPUT_TTL="$1"
            else
                echo "Error: Unexpected argument '$1'"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_JSON" || -z "$OUTPUT_TTL" ]]; then
    echo "Error: Input JSON file and output Turtle file are required"
    show_help
    exit 1
fi

# Validate input file exists
if [[ ! -f "$INPUT_JSON" ]]; then
    echo "Error: Input JSON file '$INPUT_JSON' not found"
    exit 1
fi

# Validate taxonomy directory exists
if [[ ! -d "$TAXONOMY_DIR" ]]; then
    echo "Error: Taxonomy directory '$TAXONOMY_DIR' not found"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_TTL")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "Converting JSON to RDF with validation..."
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_TTL"
echo "Batch size: $BATCH_SIZE"
echo "Taxonomy directory: $TAXONOMY_DIR"

# Run the converter
python3 json_to_validated_rdf.py "$INPUT_JSON" "$OUTPUT_TTL" "$TAXONOMY_DIR" "$BATCH_SIZE"

# Check exit code
if [[ $? -ne 0 ]]; then
    echo "Conversion or validation failed with errors"
    exit 1
fi 