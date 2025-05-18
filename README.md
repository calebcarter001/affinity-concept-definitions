# Affinity Concept Definitions

A Python-based tool for generating travel concept affinities using semantic matching and AI assistance.

## Overview

This project analyzes travel concepts and generates affinity definitions using multiple methods:
- Semantic embedding similarity
- BM25 text similarity
- LLM-based concept classification

## Key Files

- `affinity_generator_v34.0.14.py` - Main generator script
- `affinity_config_v34.12.json` - Configuration settings
- `utils.py` - Utility functions
- `run_affinity_generator.sh` - Shell script to run the generator

## Usage

```bash
./run_affinity_generator.sh
```

Or with custom options:

```bash
python affinity_generator_v34.0.14.py --input-concepts-file /path/to/concepts.txt --taxonomy-dir /path/to/taxonomy/ --output-dir ./output --cache-dir ./cache --config ./config.json --debug
```

## Requirements

- Python 3.10+
- sentence-transformers
- scikit-learn
- rdflib
- bm25s
- tqdm
- pandas
- openai (optional for LLM support) 