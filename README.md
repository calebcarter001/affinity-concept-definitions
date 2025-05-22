# Travel Concept Affinity Generator

This project generates and validates travel concept affinities for use in TopBraid. It consists of a pipeline that processes travel concepts from taxonomies, generates affinity definitions, and converts them to RDF format.

## Project Structure

```
PythonProject/
├── affinity_generator_v34.0.14.py  # Main generator script
├── json_to_validated_rdf.py        # JSON to RDF converter with validation
├── affinity_config_v34.12.json     # Configuration file
├── run_affinity_generator.sh       # Shell script to run the generator
├── run_json_to_validated_rdf.sh    # Shell script to run the converter
├── .env                           # Environment variables (API keys)
├── datasources/                    # Input data directory
│   ├── verified_raw_travel_concepts.txt  # Input concepts to process
│   ├── transformed_acs_tracker.csv      # ACS enrichment data
│   ├── activities.rdf             # Taxonomy files
│   ├── places.rdf
│   ├── trip_preferences.rdf
│   └── ...
├── src/
│   └── utils/
│       ├── utils.py               # Utility functions (v34.0.2+)
│       └── AffinityDefinitionModel.ttl  # RDF ontology
└── output_v34.14/                 # Latest output directory
    ├── affinity_definitions_v[DATE].affinity.34.0.14.json  # Main output
    ├── affinity_generation_v[DATE].affinity.34.0.14.log   # Main log
    ├── nan_embeddings_v[DATE].affinity.34.0.14.log        # NaN detection log
    ├── llm_interactions_v[DATE].affinity.34.0.14.log      # LLM calls log
    └── stage1_candidates_v[DATE].affinity.34.0.14.log     # Stage 1 processing log
```

## Latest Features (v34.0.14)

### Affinity Generator
- Weights attributes within themes based on combined score (BM25+SBERT+Bias)
- Dynamic lodging type determination ('CL', 'VR', 'Both')
- Detailed LLM negation candidate selection logging
- Inherited features from v34.0.13:
  - Reverted LLM slotting prompt
  - Unthemed high-scoring concept capture
  - Dynamic requires_geo_check flag
  - LLM Negation Identification & Filtering

### JSON to RDF Converter
- Batch processing for large datasets
- Parallel taxonomy loading with multi-threading
- Real-time validation during conversion:
  - Format validation
  - Taxonomy concept verification
  - Namespace validation
  - Data type checking
  - Missing data detection
- Detailed progress tracking with ETA
- Intermediate result saving for large datasets
- Comprehensive validation report including:
  - Conversion statistics
  - RDF graph metrics
  - Taxonomy validation details
  - Issue categorization by severity and type
- Automatic unique filename generation
- Safe URI handling and encoding

## Input Requirements

1. **Travel Concepts File**
   - Location: `datasources/verified_raw_travel_concepts.txt`
   - Format: One concept per line
   - Contains the travel concepts to generate affinities for

2. **Taxonomy Files**
   - Location: `datasources/*.rdf`
   - Format: RDF/XML
   - Contains all concept definitions and relationships

3. **Configuration**
   - Main config: `affinity_config_v34.12.json`
   - Environment variables: `.env` (for API keys)
   ```
   GOOGLE_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

## Configuration Details

The `affinity_config_v34.12.json` contains several key sections:

### 1. Model Settings
```json
{
    "SBERT_MODEL_NAME": "sentence-transformers/all-mpnet-base-v2",
    "CACHE_VERSION": "v20250420.affinity.34.0.14",
    "LLM_MODEL": "gemini-2.0-flash",
    "LLM_PROVIDER": "google"
}
```

### 2. Processing Stages
- **Stage 1**: Initial candidate selection
  - Keyword expansion
  - Semantic similarity scoring
  - BM25 scoring
  - Candidate filtering

- **Stage 2**: Theme-based refinement
  - LLM-based attribute classification
  - Weight calculation
  - Lodging type determination
  - Unthemed concept capture

### 3. Themes
The configuration defines 11 base themes:
- Location
- Privacy
- Indoor Amenities
- Outdoor Amenities
- Activities
- Imagery
- Spaces
- Seasonality
- Group Relevance
- Technology
- Sentiment

Each theme has:
- Description
- Type (decision/comfort/structural/etc.)
- Weight
- Subscores
- Rules

### 4. Namespace Configuration
- Preferred namespaces for concept lookup
- Namespace biasing rules
- Domain-specific namespace groups (lodging, activity, location, etc.)

### 5. Scoring Configuration
- Minimum similarity thresholds
- Keyword scoring settings
- Namespace boost/penalty factors
- Subscore weights

### 6. Special Handling
- Concept overrides for specific terms
- ACS data enrichment settings
- LLM negation detection
- Lodging type inference
- Dynamic lodging type determination based on:
  - Anchor/attribute analysis
  - Configurable hints
  - Theme-based rules

## Utility Functions (utils.py)

Key functionality provided by `utils.py`:

1. **Data Loading**
   - Taxonomy concept loading
   - Cache management
   - Configuration loading
   - ACS data integration

2. **Text Processing**
   - Concept normalization
   - Label extraction
   - URI parsing
   - Keyword indexing

3. **Embedding Operations**
   - SBERT model management
   - Embedding computation
   - Similarity calculations
   - NaN detection

4. **Logging**
   - Multi-file logging setup
   - Detailed diagnostics
   - Process tracking
   - Error handling

5. **Knowledge Graph**
   - RDF data extraction
   - Type label processing
   - Namespace management
   - URI normalization

## Dependencies

Required Python packages:
- rdflib>=6.0.0
- sentence-transformers>=2.2.0
- numpy>=1.25.0
- pandas>=2.0.0
- bm25s>=0.2.12
- tqdm>=4.65.0
- scikit-learn>=1.3.0
- openai>=1.0.0 (optional for OpenAI LLM)
- google-generativeai>=0.3.0 (optional for Google AI)

## Environment Setup

1. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. **API Keys**
   Create `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

3. **Cache Directories**
   ```bash
   mkdir -p cache_v34.14
   mkdir -p output_v34.14
   ```

## Running the Pipeline

### 1. Generate Affinity Definitions

The main script can be run in two ways:

#### Using the Shell Script (Recommended)
```bash
./run_affinity_generator.sh
```

The shell script provides several options:
```bash
./run_affinity_generator.sh [options]
  --debug                     Enable debug logging
  --rebuild-cache            Force rebuild of caches
  --config FILE              Custom config file path
  --output-dir DIR           Custom output directory
  --cache-dir DIR           Custom cache directory
  --concepts FILE            Custom input concepts file
  --taxonomy-dir DIR         Custom taxonomy directory
  --llm-provider PROVIDER    Specify LLM provider
  --llm-model MODEL          Specify LLM model
```

#### Direct Python Execution
```bash
python affinity_generator_v34.0.14.py \
  --input-concepts-file datasources/verified_raw_travel_concepts.txt \
  --taxonomy-dir datasources/ \
  --output-dir output_v34.14 \
  --cache-dir cache_v34.14 \
  --config affinity_config_v34.12.json
```

This will:
- Process concepts from the input file
- Use taxonomies from datasources/
- Generate JSON output in output_v34.14/
- Use configuration from affinity_config_v34.12.json

### 2. Convert to RDF Format

The JSON to RDF converter can be run using the provided shell script:

```bash
./run_json_to_validated_rdf.sh [options]
  --input FILE               Input JSON file path
  --output FILE             Output Turtle file path
  --batch-size N            Number of definitions per batch (default: 100)
  --taxonomy-dir DIR        Directory containing taxonomy files
  --debug                   Enable debug logging
```

The converter provides:

1. **Validation Categories**:
   - FORMAT: JSON structure and RDF syntax
   - TAXONOMY: Concept existence and relationships
   - NAMESPACE: URI and prefix validation
   - DATA_MISSING: Required field checks
   - DATA_TYPE: Value type verification

2. **Statistics Tracking**:
   - Conversion metrics:
     - Total definitions processed
     - Triples generated
     - Batches completed
   - RDF graph analysis:
     - Unique subjects/predicates/objects
     - Namespace usage
     - URI/literal distribution
   - Taxonomy validation:
     - Files processed
     - Valid concepts found
     - Source file mapping

3. **Error Handling**:
   - Severity levels (ERROR, WARNING, INFO)
   - Detailed error messages
   - Location tracking
   - Suggested fixes
   - Batch recovery

4. **Output Files**:
   - RDF in Turtle format
   - Validation report (JSON)
   - Processing statistics
   - Error logs

Example usage:
```bash
./run_json_to_validated_rdf.sh \\
  --input output_v34.14/affinity_definitions_v20250502.affinity.34.0.14.json \\
  --output rdf_outputs/AffinityDefinitionModel_v34.0.14.ttl \\
  --batch-size 200 \\
  --taxonomy-dir datasources
```

The converter will:
1. Load and validate taxonomy files in parallel
2. Process JSON input in configurable batches
3. Perform real-time validation
4. Generate detailed statistics
5. Save intermediate results
6. Create a comprehensive validation report

## Output Files

The pipeline produces several files in output_v34.14/:

1. **Main Output**
   - `affinity_definitions_v[DATE].affinity.34.0.14.json`
   - Contains affinity definitions with themes and weights

2. **Main Log**
   - `affinity_generation_v[DATE].affinity.34.0.14.log`
   - Overall process logging

3. **Diagnostic Logs**
   - `nan_embeddings_v[DATE].affinity.34.0.14.log`
     - NaN detection in embeddings
   - `llm_interactions_v[DATE].affinity.34.0.14.log`
     - LLM API calls and responses
   - `stage1_candidates_v[DATE].affinity.34.0.14.log`
     - Initial candidate selection details

## Version History

Latest versions:
- Generator: v34.0.14
- Utils: v34.0.2+
- Config: v34.12
- Output: v34.14

## Notes

- Always use the latest versions of components
- Ensure API keys are properly set in .env
- Check all log files for issues
- Validate RDF before importing to TopBraid
- Monitor NaN detection logs for embedding issues
- Keep cache versions aligned with config CACHE_VERSION

## Input Data

The `/datasources` directory contains:
- Core taxonomy files (*.rdf)
- Travel concept definitions
- Property and amenity taxonomies
- Activity and location taxonomies

## Output

The pipeline produces:
1. JSON affinity definitions
2. RDF representation
3. Validation reports
4. Processing logs

## Dependencies

- Python 3.7+
- rdflib
- Required Python packages in requirements.txt

## Version History

Latest versions:
- Generator: v34.0.14
- Config: v34.12
- Output: v34.14

## Notes

- Always use the latest versions of components
- Validate RDF before importing to TopBraid
- Check logs for any processing issues 