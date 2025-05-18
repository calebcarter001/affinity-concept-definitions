#!/usr/bin/env python3
"""Main script for generating travel affinity definitions with enhanced debugging."""

import os
import argparse
import json
import logging
import time
import gc
import traceback
import signal
import sys
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from rdflib import Graph
from sentence_transformers import SentenceTransformer

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - [%(process)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("affinity_debug.log")
    ]
)

logger = logging.getLogger("affinity_generator")

try:
    from affinity_taxonomy import TaxonomyParser, load_rdf_files
except ImportError as e:
    logger.error(f"Failed to import TaxonomyParser or load_rdf_files: {e}")
    raise

from affinity_scoring import (
    process_concept,
    normalize_concept,
    load_configs,
    BASE_THEMES,
    SPECIALIZED_SUBSCORES,
    assign_required_theme_attributes,
    KnowledgeBase,
    get_sbert_model
)
from utils import load_corpus_terms


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Define this at the module level for pickling compatibility
def process_concept_wrapper(args):
    """Wrapper for process_concept to handle exceptions and add timeout."""
    try:
        concept, parser, themes_config, input_graph, sbert_model, corpus_terms, worker_id, timeout = args

        logger.info(f"Worker {worker_id} starting to process concept: {concept}")
        start_time = time.time()

        # Force garbage collection before processing
        gc.collect()

        # Log memory usage of key objects
        logger.debug(f"Worker {worker_id} graph size: {len(input_graph)}")
        logger.debug(f"Worker {worker_id} corpus terms: {len(corpus_terms)}")

        # Process the concept
        result = process_concept((concept, parser, themes_config, input_graph, sbert_model, corpus_terms))

        elapsed = time.time() - start_time
        logger.info(f"Worker {worker_id} completed concept {concept} in {elapsed:.2f} seconds")

        # Force garbage collection after processing
        gc.collect()

        return result
    except Exception as e:
        logger.error(f"Error processing concept {args[0]}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def batch_process_concepts(concepts: List[str], batch_size: int = 5) -> List[List[str]]:
    """Split concepts into manageable batches."""
    return [concepts[i:i + batch_size] for i in range(0, len(concepts), batch_size)]


def generate_affinity_definitions(
        concepts_file: str,
        taxonomy_dir: str,
        output_dir: str = ".",
        metadata_file: str = "taxonomy_metadata.json",
        themes_config_file: str = "themes_config.json",
        travel_corpus: Optional[str] = None,
        batch_size: int = 5,
        max_workers: int = 4,
        timeout: int = 300,
        max_concepts: int = 0  # 0 means process all concepts
) -> None:
    """Generate affinity definitions from RDF taxonomies with enhanced debugging."""
    try:
        logger.info("Starting affinity definitions generation")
        overall_start = time.time()

        # Initialize Sentence-BERT model once in main process
        model_start = time.time()
        sbert_model = get_sbert_model()
        logger.info(f"Initialized Sentence-BERT model in {time.time() - model_start:.2f} seconds")

        # Precompute corpus embeddings if provided
        corpus_start = time.time()
        corpus_terms = load_corpus_terms(travel_corpus, sbert_model) if travel_corpus else {}
        if corpus_terms:
            logger.info(
                f"Precomputed embeddings for {len(corpus_terms)} corpus terms in {time.time() - corpus_start:.2f} seconds")

        # Load configs
        config_start = time.time()
        metadata, themes_config = load_configs(metadata_file, themes_config_file)
        logger.info(f"Loaded config files in {time.time() - config_start:.2f} seconds")

        # Load concepts
        concepts_start = time.time()
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                concepts = [line.strip() for line in f if line.strip()]

            # Limit concepts for testing if needed
            if max_concepts > 0 and max_concepts < len(concepts):
                logger.info(f"Limiting to {max_concepts} concepts for testing")
                concepts = concepts[:max_concepts]

        except FileNotFoundError:
            logger.error(f"Concepts file not found: {concepts_file}")
            raise
        except PermissionError:
            logger.error(f"Permission denied reading concepts file: {concepts_file}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading concepts file: {e}")
            raise
        if not concepts:
            logger.error("Concepts file is empty")
            raise ValueError("Concepts file is empty")
        logger.info(f"Loaded {len(concepts)} concepts in {time.time() - concepts_start:.2f} seconds")

        # Load taxonomy
        taxonomy_start = time.time()
        input_graph = load_rdf_files(taxonomy_dir)
        if not input_graph:
            logger.error("Taxonomy graph is empty")
            raise ValueError("No taxonomy data loaded")
        parser = TaxonomyParser(input_graph)
        logger.info(f"Loaded taxonomy in {time.time() - taxonomy_start:.2f} seconds")

        output_graph = Graph()
        affinity_definitions = {"travel_concepts": [], "themes": []}
        seen_uris = set()

        # Process in smaller batches to avoid memory issues
        batches = batch_process_concepts(concepts, batch_size)
        logger.info(f"Split {len(concepts)} concepts into {len(batches)} batches of size {batch_size}")

        # Use fewer processes to avoid deadlocks - cap at max_workers
        num_processes = min(cpu_count(), max_workers)
        logger.info(f"Using {num_processes} processes for processing concept batches")

        # For single worker case, just process sequentially without multiprocessing
        if num_processes == 1:
            logger.info("Using sequential processing (single worker)")
            batch_start = time.time()
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} concepts")
                batch_process_start = time.time()

                results = []
                for i, concept in enumerate(tqdm(batch, desc=f"Batch {batch_idx + 1}/{len(batches)}")):
                    worker_args = (concept, parser, themes_config, input_graph, sbert_model, corpus_terms, 0, timeout)
                    results.append(process_concept_wrapper(worker_args))

                # Process results for this batch
                result_count = 0
                error_count = 0

                for result in results:
                    if "error" in result:
                        error_count += 1
                        continue

                    if result and result.get("uri"):
                        uri = result["uri"]
                        if uri in seen_uris:
                            logger.debug(f"Skipped duplicate URI: {uri}")
                            continue
                        seen_uris.add(uri)
                        affinity_definitions["travel_concepts"].append(
                            {k: v for k, v in result.items() if k != "graph"})
                        if "graph" in result:
                            output_graph += result["graph"]
                        result_count += 1

                logger.info(
                    f"Batch {batch_idx + 1} completed: {result_count} successes, {error_count} errors, in {time.time() - batch_process_start:.2f} seconds")

                # Explicit cleanup
                gc.collect()

        else:
            # Process batches using Pool
            batch_start = time.time()
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} concepts")
                batch_process_start = time.time()

                # Prepare worker arguments
                worker_args = [
                    (concept, parser, themes_config, input_graph, sbert_model, corpus_terms, i % num_processes, timeout)
                    for i, concept in enumerate(batch)
                ]

                # Create a separate pool for each batch to avoid memory issues
                try:
                    with Pool(processes=num_processes, maxtasksperchild=1) as pool:
                        results = list(tqdm(
                            pool.map(process_concept_wrapper, worker_args),
                            total=len(batch),
                            desc=f"Batch {batch_idx + 1}/{len(batches)}",
                            leave=True
                        ))
                except Exception as e:
                    logger.error(f"Error during batch processing: {e}")
                    logger.error(traceback.format_exc())
                    # Continue to next batch
                    continue

                # Process results for this batch
                result_count = 0
                error_count = 0

                for result in results:
                    if "error" in result:
                        error_count += 1
                        continue

                    if result and result.get("uri"):
                        uri = result["uri"]
                        if uri in seen_uris:
                            logger.debug(f"Skipped duplicate URI: {uri}")
                            continue
                        seen_uris.add(uri)
                        affinity_definitions["travel_concepts"].append(
                            {k: v for k, v in result.items() if k != "graph"})
                        if "graph" in result:
                            output_graph += result["graph"]
                        result_count += 1

                logger.info(
                    f"Batch {batch_idx + 1} completed: {result_count} successes, {error_count} errors, in {time.time() - batch_process_start:.2f} seconds")

                # Explicit cleanup
                gc.collect()

        logger.info(f"All batches processed in {time.time() - batch_start:.2f} seconds")

        # Handle required themes
        themes_start = time.time()
        required_themes = [
            theme for theme_name, theme in themes_config.items()
            if isinstance(theme, dict) and theme.get("required", False)
        ]
        logger.info(f"Processing {len(required_themes)} required themes")

        # Create a new KB instance for theme processing
        kb = KnowledgeBase(input_graph, themes_config, corpus_terms=corpus_terms, sbert_model=sbert_model)
        required_assignments = assign_required_theme_attributes(required_themes, concepts, kb)

        # Build theme data
        for theme_name, theme_config in themes_config.items():
            if not isinstance(theme_config, dict) or "name" not in theme_config:
                logger.warning(f"Skipping invalid theme config: {theme_name}")
                continue
            theme_attrs = set()
            theme_data = {
                "name": theme_name,
                "type": theme_config.get("type", "structural"),
                "rule": theme_config.get("rule", "Optional"),
                "theme_weight": theme_config.get("theme_weight", 0.05),
                "subScore": theme_config.get("subScore", f"{theme_name}Affinity"),
                "attributes": []
            }

            if theme_name in required_assignments and required_assignments[theme_name]["status"] == "assigned":
                for attr in required_assignments[theme_name]["attributes"]:
                    uri = attr.get(
                        "uri") or f"urn:expediagroup:taxonomies:core:#{attr['skos:prefLabel'].replace(' ', '').replace(':', '')}"
                    theme_data["attributes"].append({
                        "skos:prefLabel": attr["skos:prefLabel"],
                        "uri": uri,
                        "concept_weight": attr["score"],
                        "matched_reason": f"match: {attr['reason']}: {attr['score']}",
                        "scoring_guidance": {
                            "min_weight": theme_config.get("min_weight", 0.75),
                            "max_weight": 1.0,
                            "required": theme_config.get("required", False)
                        },
                        "metadata": {}
                    })
                    theme_attrs.add(attr["skos:prefLabel"].lower().replace(" ", "").replace(":", ""))
            else:
                for concept in affinity_definitions["travel_concepts"]:
                    for theme in concept.get("themes", []):
                        if theme["name"] == theme_name:
                            theme_data["attributes"].append({
                                "skos:prefLabel": concept["skos:prefLabel"],
                                "uri": concept["uri"],
                                "concept_weight": theme["concept_weight"],
                                "matched_reason": theme["matched_reason"],
                                "scoring_guidance": theme["scoring_guidance"],
                                "metadata": concept.get("metadata", {})
                            })
                            theme_attrs.add(concept["skos:prefLabel"].lower().replace(" ", "").replace(":", ""))

            # Skip themes with no attributes for optional themes
            if not theme_data["attributes"] and not theme_config.get("required", False):
                logger.debug(f"Skipping empty optional theme: {theme_name}")
                continue

            theme_data["dynamic_rule"] = {
                "rule_type": theme_config.get("rule", "optional").lower(),
                "condition": "attribute_presence",
                "target": theme_name,
                "criteria": {"attributes": sorted(theme_attrs), "minimum_count": 1},
                "effect": "boost" if theme_config.get("rule", "").lower() == "optional" else "pass"
            }
            affinity_definitions["themes"].append(theme_data)

        logger.info(f"Theme processing completed in {time.time() - themes_start:.2f} seconds")

        # Log themes needing review
        for theme_name, assignment in required_assignments.items():
            if assignment["status"] == "manual_review_needed":
                logger.warning(f"Required theme {theme_name} has no valid attributes; flagged for review")

        # Write output files
        output_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        ttl_output_file = os.path.join(output_dir, "affinity_definitions.ttl")
        json_output_file = os.path.join(output_dir, "affinity_definitions.json")

        try:
            output_graph.serialize(destination=ttl_output_file, format="turtle")
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(affinity_definitions, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            logger.info(f"Generated affinity definitions at {json_output_file} and {ttl_output_file}")
        except (PermissionError, FileNotFoundError) as e:
            logger.error(f"Failed to write output files: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error writing output files: {e}")
            raise

        logger.info(f"Output files written in {time.time() - output_start:.2f} seconds")
        logger.info(f"Total processing time: {time.time() - overall_start:.2f} seconds")

    except Exception as e:
        logger.error(f"Failed to generate affinity definitions: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Parse arguments and run affinity generation."""
    parser = argparse.ArgumentParser(description="Generate travel affinity definitions from RDF taxonomies.")
    parser.add_argument("--concepts", required=True, help="Path to concepts file")
    parser.add_argument("--taxonomy-dir", required=True, help="Directory containing RDF taxonomies")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--metadata-file", default="taxonomy_metadata.json", help="Metadata JSON file")
    parser.add_argument("--themes-config-file", default="themes_config.json", help="Themes configuration JSON file")
    parser.add_argument("--travel-corpus", help="Path to external travel corpus file")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing concepts")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for processing a concept in seconds")
    parser.add_argument("--max-concepts", type=int, default=0, help="Max concepts to process (0=all, for testing)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    generate_affinity_definitions(
        args.concepts,
        args.taxonomy_dir,
        args.output_dir,
        args.metadata_file,
        args.themes_config_file,
        args.travel_corpus,
        args.batch_size,
        args.max_workers,
        args.timeout,
        args.max_concepts
    )


if __name__ == "__main__":
    main()