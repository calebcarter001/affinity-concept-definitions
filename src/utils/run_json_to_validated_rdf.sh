#!/bin/bash

# Default values
BATCH_SIZE=100
TAXONOMY_DIR="../datasources"

# Help function
show_help() {
    echo "Usage: $0 <input_json_file> <output_rdf_file> [options]"
    echo
    echo "Options:"
    echo "  -b, --batch-size <size>     Batch size for processing (default: 100)"
    echo "  -t, --taxonomy-dir <dir>    Directory containing taxonomy files (default: ../datasources)"
    echo "  -h, --help                  Show this help message"
    echo
    echo "Example:"
    echo "  $0 ./output_v34.14/affinity_definitions.json ./rdf_outputs/affinity_definitions.ttl -b 200"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--taxonomy-dir)
            TAXONOMY_DIR="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT_JSON" ]; then
                INPUT_JSON="$1"
            elif [ -z "$OUTPUT_RDF" ]; then
                OUTPUT_RDF="$1"
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
if [ -z "$INPUT_JSON" ] || [ -z "$OUTPUT_RDF" ]; then
    echo "Error: Input JSON and output RDF files are required"
    show_help
    exit 1
fi

# Validate input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input JSON file does not exist: $INPUT_JSON"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_RDF")
mkdir -p "$OUTPUT_DIR"

# Run the converter with validation
echo "Converting JSON to RDF with validation..."
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_RDF"
echo "Batch size: $BATCH_SIZE"
echo "Taxonomy directory: $TAXONOMY_DIR"

python json_to_validated_rdf.py "$INPUT_JSON" "$OUTPUT_RDF" "$TAXONOMY_DIR" "$BATCH_SIZE"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Conversion and validation completed successfully"
    exit 0
else
    echo "Conversion or validation failed with errors"
    exit 1
fi 