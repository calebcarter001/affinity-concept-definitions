#!/bin/bash
# Run the affinity generator with debug, cache rebuild, and OpenAI GPT-4.1
cd "$(dirname "$0")"

echo "==== Starting script debug at $(date) ====" > run_debug.log

# Check if required files exist
echo "Checking for required files..." | tee -a run_debug.log
for file in affinity_generator_v34.0.14.py affinity_config_v34.12.json utils.py; do
  if [ -f "$file" ]; then
    echo "✓ Found $file (size: $(wc -c < "$file") bytes)" | tee -a run_debug.log
  else
    echo "✗ Missing $file!" | tee -a run_debug.log
    exit 1
  fi
done

# Check if the input file exists
INPUT_FILE="./datasources/verified_raw_travel_concepts.txt"
if [ -f "$INPUT_FILE" ]; then
  echo "✓ Found input concepts file" | tee -a run_debug.log
  echo "   First 5 lines of input file:" | tee -a run_debug.log
  head -n 5 "$INPUT_FILE" | tee -a run_debug.log
else
  echo "✗ Missing input concepts file: $INPUT_FILE" | tee -a run_debug.log
  exit 1
fi

# Create the output directory
mkdir -p ./output_v34.14
echo "Creating output directory: ./output_v34.14" | tee -a run_debug.log

# Try running the script directly first to see any Python errors
echo "Testing Python script execution directly..." | tee -a run_debug.log
./venv/bin/python -c "import sys; sys.path.append('.'); import affinity_generator_v34.0.14; print('Import successful')" 2>&1 | tee -a run_debug.log

# Run the actual script
echo "Running full script with arguments..." | tee -a run_debug.log
./run_affinity_generator.sh --debug --rebuild-cache --llm-provider openai --llm-model gpt-4.1 2>&1 | tee -a run_debug.log

# Check output directory after execution
echo "Checking output directory after execution:" | tee -a run_debug.log
ls -la ./output_v34.14/ | tee -a run_debug.log

echo "==== Debug complete at $(date) ====" | tee -a run_debug.log
echo "Press Enter to close this window..."
read 