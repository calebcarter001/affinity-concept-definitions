#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Load Environment Variables ---
ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
else
    echo "Warning: $ENV_FILE not found. Environment variables will not be loaded."
fi

# --- Configuration ---
SCRIPT_NAME="affinity_generator_v34.0.14.py" # UPDATED Script name
PYTHON_EXECUTABLE="./venv/bin/python"

# --- Define default paths ---
DEFAULT_CONCEPTS_FILE="./datasources/verified_raw_travel_concepts.txt"
DEFAULT_TAXONOMY_DIR="./datasources/"
DEFAULT_CONFIG_FILE="./affinity_config_v34.12.json" # UPDATED Config file
DEFAULT_OUTPUT_DIR="./output_v34.14" # UPDATED Output dir
DEFAULT_CACHE_DIR="./cache_v34.14"  # UPDATED Cache dir

# --- Initialize variables ---
CONCEPTS_FILE="$DEFAULT_CONCEPTS_FILE"
TAXONOMY_DIR="$DEFAULT_TAXONOMY_DIR"
CONFIG_FILE_ARG="$DEFAULT_CONFIG_FILE"
OUTPUT_DIR_ARG="$DEFAULT_OUTPUT_DIR"
CACHE_DIR_ARG="$DEFAULT_CACHE_DIR"
DEBUG_FLAG=""
REBUILD_FLAG=""
LLM_PROVIDER_ARG=""
LLM_MODEL_ARG=""

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_FLAG="true" ;;
        --rebuild-cache|--rebuild_cache) REBUILD_FLAG="true" ;;
        --config) CONFIG_FILE_ARG="$2"; shift ;;
        --output-dir|--output_dir) OUTPUT_DIR_ARG="$2"; shift ;;
        --cache-dir|--cache_dir) CACHE_DIR_ARG="$2"; shift ;;
        --concepts|--input-concepts-file|--input_concepts_file) CONCEPTS_FILE="$2"; shift ;;
        --taxonomy-dir|--taxonomy_dir) TAXONOMY_DIR="$2"; shift ;;
        --llm-provider|--llm_provider) LLM_PROVIDER_ARG="$2"; shift ;;
        --llm-model|--llm_model) LLM_MODEL_ARG="$2"; shift ;;
        *) echo "Unknown shell script option: $1"; exit 1 ;;
    esac
    shift
done

# --- Validation ---
if [[ ! -f "$CONCEPTS_FILE" ]]; then echo "Error: Input concepts file not found at '$CONCEPTS_FILE'."; exit 1; fi
if [[ ! -d "$TAXONOMY_DIR" ]]; then echo "Error: Taxonomy directory not found at '$TAXONOMY_DIR'."; exit 1; fi
if [[ ! -f "$CONFIG_FILE_ARG" ]]; then echo "Error: Configuration file not found at '$CONFIG_FILE_ARG'."; exit 1; fi
if [[ ! -f "$SCRIPT_NAME" ]]; then echo "Error: Python script '$SCRIPT_NAME' not found."; exit 1; fi

# --- Setup ---
mkdir -p "$OUTPUT_DIR_ARG"
mkdir -p "$CACHE_DIR_ARG"
export TOKENIZERS_PARALLELISM=false

# --- Construct Python Command Arguments ---
CMD_ARGS=(
    "--input-concepts-file"  "$CONCEPTS_FILE"
    "--taxonomy-dir"         "$TAXONOMY_DIR"
    "--output-dir"           "$OUTPUT_DIR_ARG"
    "--cache-dir"            "$CACHE_DIR_ARG"
    "--config"               "$CONFIG_FILE_ARG"
)
if [[ -n "$DEBUG_FLAG" ]]; then CMD_ARGS+=("--debug"); fi
if [[ -n "$REBUILD_FLAG" ]]; then CMD_ARGS+=("--rebuild-cache"); fi
if [[ -n "$LLM_PROVIDER_ARG" ]]; then CMD_ARGS+=("--llm-provider" "$LLM_PROVIDER_ARG"); fi
if [[ -n "$LLM_MODEL_ARG" ]]; then CMD_ARGS+=("--llm-model" "$LLM_MODEL_ARG"); fi

# --- Execution ---
echo "Executing Script: $SCRIPT_NAME"
printf "Command: %s %s " "$PYTHON_EXECUTABLE" "$SCRIPT_NAME"
printf "'%s' " "${CMD_ARGS[@]}"
echo # Newline

"$PYTHON_EXECUTABLE" "$SCRIPT_NAME" "${CMD_ARGS[@]}"

EXECUTION_STATUS=$?
if [[ $EXECUTION_STATUS -ne 0 ]]; then
    echo "------------------------------------"
    echo "Error: Failed to run $SCRIPT_NAME (Exit Code: $EXECUTION_STATUS)."
    echo "Check the log file in $OUTPUT_DIR_ARG for details."
    echo "------------------------------------"
    exit 1
else
    echo "------------------------------------"
    echo "Success: Script $SCRIPT_NAME finished."
    echo "Output directory: $OUTPUT_DIR_ARG"
    echo "(Log file name depends on CACHE_VERSION set in config file '$CONFIG_FILE_ARG')"
    echo "------------------------------------"
fi

exit 0