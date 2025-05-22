#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Script Directory Detection ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Load Environment Variables ---
# Priority: 
# 1. Command line argument (--env-file)
# 2. Environment variable (AFFINITY_ENV_FILE)
# 3. Default locations (.env in script directory)
DEFAULT_ENV_FILE="${SCRIPT_DIR}/.env"
ENV_FILE="${AFFINITY_ENV_FILE:-$DEFAULT_ENV_FILE}"

# --- Configuration ---
SCRIPT_NAME="affinity_generator_v34.0.14.py"
PYTHON_EXECUTABLE="./venv/bin/python"

# --- Define default paths ---
DEFAULT_CONCEPTS_FILE="./datasources/verified_raw_travel_concepts.txt"
DEFAULT_TAXONOMY_DIR="./datasources/"
DEFAULT_CONFIG_FILE="./affinity_config_v34.12.json"
DEFAULT_OUTPUT_DIR="./output_v34.14"
DEFAULT_CACHE_DIR="./cache_v34.14"

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
        --env-file) ENV_FILE="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug                     Enable debug logging"
            echo "  --rebuild-cache             Force rebuild of caches"
            echo "  --config FILE               Custom config file path"
            echo "  --output-dir DIR            Custom output directory"
            echo "  --cache-dir DIR             Custom cache directory"
            echo "  --concepts FILE             Custom input concepts file"
            echo "  --taxonomy-dir DIR          Custom taxonomy directory"
            echo "  --llm-provider PROVIDER     Specify LLM provider"
            echo "  --llm-model MODEL           Specify LLM model"
            echo "  --env-file FILE             Custom environment file (default: .env)"
            echo "  -h, --help                  Show this help message"
            echo
            echo "Environment Variables:"
            echo "  AFFINITY_ENV_FILE           Override default environment file location"
            echo "  GOOGLE_API_KEY              Required for Google AI LLM provider"
            echo "  OPENAI_API_KEY              Required for OpenAI LLM provider"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# --- Environment File Handling ---
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Warning: Environment file not found at '$ENV_FILE'"
    echo "Required environment variables:"
    echo "  - GOOGLE_API_KEY    (for Google AI)"
    echo "  - OPENAI_API_KEY    (for OpenAI)"
    echo
    echo "You can:"
    echo "1. Create an .env file in the project root"
    echo "2. Set AFFINITY_ENV_FILE environment variable"
    echo "3. Use --env-file argument"
    echo
    read -p "Continue without environment file? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Loading environment variables from: $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

# --- Validate Environment Variables ---
MISSING_KEYS=()
if [[ -z "${GOOGLE_API_KEY}" ]]; then MISSING_KEYS+=("GOOGLE_API_KEY"); fi
if [[ -z "${OPENAI_API_KEY}" ]]; then MISSING_KEYS+=("OPENAI_API_KEY"); fi

if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
    echo "Error: Missing required API keys in environment:"
    for key in "${MISSING_KEYS[@]}"; do
        echo "  - $key"
    done
    echo
    echo "Please ensure your .env file contains these variables:"
    echo "GOOGLE_API_KEY=your_google_api_key_here"
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    echo
    read -p "Continue without required API keys? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if LLM provider is specified but missing corresponding API key
if [[ -n "$LLM_PROVIDER_ARG" ]]; then
    case "$LLM_PROVIDER_ARG" in
        "google")
            if [[ -z "${GOOGLE_API_KEY}" ]]; then
                echo "Error: Google AI provider specified but GOOGLE_API_KEY is missing" >&2
                exit 1
            fi
            ;;
        "openai")
            if [[ -z "${OPENAI_API_KEY}" ]]; then
                echo "Error: OpenAI provider specified but OPENAI_API_KEY is missing" >&2
                exit 1
            fi
            ;;
    esac
fi

# --- Validation ---
if [[ ! -f "$CONCEPTS_FILE" ]]; then echo "Error: Input concepts file not found at '$CONCEPTS_FILE'" >&2; exit 1; fi
if [[ ! -d "$TAXONOMY_DIR" ]]; then echo "Error: Taxonomy directory not found at '$TAXONOMY_DIR'" >&2; exit 1; fi
if [[ ! -f "$CONFIG_FILE_ARG" ]]; then echo "Error: Configuration file not found at '$CONFIG_FILE_ARG'" >&2; exit 1; fi
if [[ ! -f "$SCRIPT_NAME" ]]; then echo "Error: Python script '$SCRIPT_NAME' not found" >&2; exit 1; fi

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
echo

"$PYTHON_EXECUTABLE" "$SCRIPT_NAME" "${CMD_ARGS[@]}"

EXECUTION_STATUS=$?
if [[ $EXECUTION_STATUS -ne 0 ]]; then
    echo "------------------------------------"
    echo "Error: Failed to run $SCRIPT_NAME (Exit Code: $EXECUTION_STATUS)"
    echo "Check the log file in $OUTPUT_DIR_ARG for details"
    echo "------------------------------------"
    exit 1
else
    echo "------------------------------------"
    echo "Success: Script $SCRIPT_NAME finished"
    echo "Output directory: $OUTPUT_DIR_ARG"
    echo "(Log file name depends on CACHE_VERSION set in config file '$CONFIG_FILE_ARG')"
    echo "------------------------------------"
fi

exit 0