#!/usr/bin/env python3
"""
Suggest Candidate Opposites Script (Helper for Affinity Definition)

Purpose:
Analyzes a taxonomy to suggest potential 'opposite' concept URIs for a given
list of core input concepts (e.g., luxury -> budget, adults only -> family friendly).
It uses a combination of semantic dissimilarity and keyword spotting to find
candidates.

Output:
Generates a JSON file mapping each core input concept (normalized) to a list
of candidate opposite concept URIs and their labels/scores. This output is
intended for HUMAN REVIEW AND SELECTION to populate the final
`INPUT_TO_OPPOSITE_URI_SET_MAP` in the main affinity generation script.

Method:
1. Loads taxonomy concepts and primary embeddings (using functions imported
   from the main generator script).
2. For each specified core input concept:
   - Gets its embedding (using TRAVEL_CONTEXT).
   - Identifies relevant 'negative keywords' associated with its opposite.
   - Iterates through all taxonomy concepts with embeddings:
     - Calculates semantic distance (1 - similarity) to the input concept.
     - Checks if the concept's text contains any negative keywords.
     - Ranks concepts based on a combination of high semantic distance and
       presence of negative keywords.
   - Outputs the top N ranked candidates.

Limitations:
- This is a HEURISTIC approach for candidate generation, not a perfect oppositeness detector.
- Relies on the quality of embeddings and the definition of negative keywords.
- The generated list REQUIRES HUMAN VALIDATION AND SELECTION before use.
- **Crucially depends on the function signatures of `load_taxonomy_concepts` and
  `precompute_taxonomy_embeddings` matching those in the main generator script.**
- **NOTE:** This version assumes the imported `precompute_taxonomy_embeddings` does NOT
  accept a `context_prefix` argument. Embeddings will be generated/loaded based
  on the logic within that imported function. Ensure consistency if context is important.

Example Usage:
python suggest_opposite_candidates.py --core-concepts-file core_concepts.txt --taxonomy-dir path/to/taxonomy/files --cache-dir ./cache --output-file suggested_opposites.json

Version: 2025-04-14-opposite-suggester-v1.3-debug (Added detailed logging for core concepts list)
"""

import os
import json
import logging
import argparse
import re
import pickle
import numpy as np
import sys
from collections import defaultdict
import time
import traceback
from typing import Dict, List, Set, Tuple, Optional, Any

# --- Assume Core Utilities are available (Copy or Import) ---

# --- START COPIED/IMPORTED UTILITIES ---
try: from sentence_transformers import SentenceTransformer
except ImportError: print("CRITICAL ERROR: sentence-transformers missing. `pip install sentence-transformers`", file=sys.stderr); sys.exit(1)
try: from sklearn.metrics.pairwise import cosine_similarity
except ImportError: print("CRITICAL ERROR: scikit-learn missing. `pip install scikit-learn`", file=sys.stderr); sys.exit(1)
try: from rdflib import Graph, Namespace, URIRef, Literal, RDF, util # Needed for concept loading if cache empty
except ImportError: print("Warning: rdflib not found, concept cache rebuild will fail. `pip install rdflib`", file=sys.stderr)

# NLTK (Optional)
NLTK_AVAILABLE = False
STOP_WORDS = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'to', 'and', 'or', 'is', 'are', 'was', 'were'])
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    print("Info: NLTK stopwords loaded.")
except (ImportError, LookupError):
    print("Warning: NLTK stopwords not found. Using basic list.")

# TQDM (Optional)
def tqdm_dummy(iterable, *args, **kwargs): return iterable
tqdm = tqdm_dummy
try: from tqdm import tqdm as real_tqdm; tqdm = real_tqdm; print("Info: tqdm loaded.")
except ImportError: print("Warning: tqdm not found, progress bars disabled. `pip install tqdm`")

# Basic Logging Setup
log_filename_suggester = "opposite_suggester.log"
# Ensure logging is configured BEFORE any logger calls
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", handlers=[logging.FileHandler(log_filename_suggester, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger_suggester = logging.getLogger("opposite_suggester")

# Namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

# Global Caches (Specific to this script's context if run standalone)
_model_instance_suggester: Optional[SentenceTransformer] = None
_taxonomy_concepts_cache_suggester: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache_suggester: Optional[Tuple[Dict, Dict, List]] = None

# --- Configuration - Ensure these match the main script if sharing caches ---
CACHE_VERSION = os.environ.get("AFFINITY_CACHE_VERSION", "v27.4") # Ensure this matches!
TRAVEL_CONTEXT = os.environ.get("AFFINITY_TRAVEL_CONTEXT", "travel and accommodation: ") # Used for embedding *input* concepts

# Log config *after* basicConfig is set
logger_suggester.info(f"Using Cache Version: {CACHE_VERSION}")
logger_suggester.info(f"Using Travel Context for input concepts: '{TRAVEL_CONTEXT}'")


def get_sbert_model_suggester() -> SentenceTransformer:
    """Loads or returns the cached Sentence-BERT model instance."""
    global _model_instance_suggester
    if _model_instance_suggester is None:
        model_name = 'all-MiniLM-L6-v2'
        logger_suggester.info(f"Loading Sentence-BERT model ('{model_name}')...")
        start_time = time.time()
        try: _model_instance_suggester = SentenceTransformer(model_name)
        except Exception as e: logger_suggester.error(f"Fatal: Failed to load SBERT model: {e}", exc_info=True); raise RuntimeError("SBERT Model loading failed") from e
        logger_suggester.info("Loaded SBERT model in %.2f seconds", time.time() - start_time)
    return _model_instance_suggester

def normalize_concept_suggester(concept: Optional[str]) -> str:
    """Normalizes a concept string."""
    if not isinstance(concept, str) or not concept: return ""
    try:
        # CamelCase to space, handle hyphens/underscores
        normalized = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept)
        normalized = normalized.replace("-", " ").replace("_", " ")
        # Remove punctuation (except internal spaces), possessives, convert to lower, strip extra spaces
        normalized = re.sub(r'[^\w\s]|(\'s\b)', '', normalized)
        normalized = ' '.join(normalized.lower().split())
        return normalized
    except Exception:
        # Fallback for safety
        return concept.lower().strip()

def calculate_semantic_similarity_suggester(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> float:
    """Calculates cosine similarity, handling potential issues."""
    if embedding1 is None or embedding2 is None: return 0.0
    try:
        # Ensure they are numpy arrays
        if not isinstance(embedding1, np.ndarray): return 0.0
        if not isinstance(embedding2, np.ndarray): return 0.0

        # Handle cases where they might be empty arrays
        if embedding1.size == 0 or embedding2.size == 0: return 0.0

        # Reshape if they are 1D
        if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)

        # Check shape compatibility after reshape
        if embedding1.shape[1] == 0 or embedding2.shape[1] == 0: return 0.0
        if embedding1.shape[1] != embedding2.shape[1]:
             logger_suggester.warning(f"Embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape}. Returning 0 similarity.")
             return 0.0

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        # Clamp value between 0 and 1
        return max(0.0, min(1.0, float(sim)))
    except Exception as e:
        logger_suggester.error(f"Error calculating similarity: {e}", exc_info=True)
        return 0.0

def get_pref_label_suggester(uri: str, taxonomy_concepts: Dict[str, Dict], fallback: str = "Unknown Label") -> str:
    """Safely retrieves the preferred label for a URI."""
    if not uri or uri not in taxonomy_concepts: return fallback
    concept_data = taxonomy_concepts.get(uri, {})
    # Prioritize skos:prefLabel
    pref_labels = concept_data.get("prefLabel", [])
    if pref_labels and isinstance(pref_labels, list) and pref_labels[0] and isinstance(pref_labels[0], (str, Literal)):
        return str(pref_labels[0]).strip()
    # Fallback to altLabel
    alt_labels = concept_data.get("altLabel", [])
    if alt_labels and isinstance(alt_labels, list) and alt_labels[0] and isinstance(alt_labels[0], (str, Literal)):
        return str(alt_labels[0]).strip()
    # Fallback to rdfs:label
    rdfs_labels = concept_data.get("rdfsLabel", [])
    if rdfs_labels and isinstance(rdfs_labels, list) and rdfs_labels[0] and isinstance(rdfs_labels[0], (str, Literal)):
        return str(rdfs_labels[0]).strip()
    # Final fallback: derive from URI
    try:
        uri_fallback = uri.split('/')[-1].split('#')[-1].replace('_', ' ').replace('-', ' ').title()
        return uri_fallback if uri_fallback else fallback
    except Exception:
        pass
    return fallback

def get_concept_text_suggester(uri: str, taxonomy_concepts: Dict) -> str:
    """Concatenates relevant textual properties (labels) for keyword checking."""
    if not uri or uri not in taxonomy_concepts: return ""
    concept_data = taxonomy_concepts.get(uri, {})
    texts: List[str] = []
    # Focus on labels and hidden labels for keyword checking
    label_props = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel"]
    for prop in label_props:
        values = concept_data.get(prop, [])
        if isinstance(values, list):
            texts.extend([str(v).strip() for v in values if v and isinstance(v, (str, Literal))])
        elif values and isinstance(values, (str, Literal)):
            texts.append(str(values).strip())

    full_text = " ".join(filter(None, texts))
    # Normalize for consistent keyword matching
    return normalize_concept_suggester(full_text).lower()

# --- Import functions from the main script ---
# ** Replace 'generate_affinity_definitions' if your main script has a different filename **
try:
    # Check if the main script exists before trying to import
    main_script_name = "generate_affinity_definitions.py" # Adjust if your script name is different
    if not os.path.exists(main_script_name):
         logger_suggester.critical(f"Main generator script '{main_script_name}' not found in the current directory.")
         logger_suggester.critical("Please ensure it's present or in the PYTHONPATH.")
         sys.exit(1)

    from generate_affinity_definitions_v30 import load_taxonomy_concepts, precompute_taxonomy_embeddings
    logger_suggester.info(f"Successfully imported functions from {main_script_name}.")
except ImportError as e:
    logger_suggester.critical(f"Failed to import required functions from {main_script_name}: {e}")
    logger_suggester.critical("Ensure the script exists and all its dependencies are installed.")
    sys.exit(1)
except AttributeError as e:
    logger_suggester.critical(f"An attribute required by the imported functions might be missing or named differently in {main_script_name}: {e}")
    logger_suggester.critical("Ensure the main script is up-to-date and compatible.")
    sys.exit(1)
# --- END COPIED/IMPORTED UTILITIES ---


# --- Suggester Script Configuration ---
# (OPPOSITE_KEYWORD_SETS definition remains the same as before)
OPPOSITE_KEYWORD_SETS = {
    "luxury":           {"budget", "economy", "basic", "cheap", "value", "affordable", "hostel", "shared", "dormitory", "low cost", "inexpensive"},
    "budget":           {"luxury", "premium", "suite", "exclusive", "boutique", "deluxe", "gourmet", "opulent", "high end", "upscale", "lavish"},
    "adults only":      {"child", "children", "kid", "kids", "family", "families", "infant", "infants", "baby", "babies", "teen", "teens", "underage", "minor", "minors", "playground", "crib", "all ages", "welcome children"},
    "family friendly":  {"adults only", "no children", "no kids", "age restriction", "18+", "21+", "casino", "nightclub", "bar hopping", "couples only"},
    "quiet":            {"noisy", "loud", "vibrant", "lively", "party", "nightclub", "music venue", "bar hopping", "entertainment", "active", "bustling", "energetic"},
    "lively":           {"quiet", "peaceful", "serene", "tranquil", "calm", "relaxing", "secluded", "isolated", "restful"},
    "historic":         {"modern", "contemporary", "newly built", "new build", "futuristic", "state of the art", "updated", "renovated"},
    "modern":           {"historic", "ancient", "old", "traditional", "heritage", "antique", "classic", "period"},
    "business travel":  {"leisure", "vacation", "holiday", "pleasure", "recreation", "tourism", "relaxation", "family trip"},
    "leisure travel":   {"business", "corporate", "work", "conference", "meeting", "convention", "professional"},
    "beach destination": {"city break", "urban", "metropolitan", "mountain", "ski", "countryside", "forest", "inland", "landlocked"},
    "city break":       {"beach", "coastal", "seaside", "rural", "countryside", "nature", "remote", "wilderness", "island"},
}
logger_suggester.info(f"Defined opposite keyword sets for {len(OPPOSITE_KEYWORD_SETS)} normalized concepts.")

# (Candidate Generation Parameters - using relaxed values from previous debugging)
NUM_CANDIDATES_TO_SHOW = 15
OPPOSITE_SIMILARITY_UPPER_BOUND = 0.4 # Kept high for debug
MIN_NEGATIVE_KEYWORD_SCORE = 0.06    # Kept at 0 for debug
SEMANTIC_DISTANCE_WEIGHT = 0.5
KEYWORD_SCORE_WEIGHT = 0.5
logger_suggester.warning(f"DEBUG: Using relaxed parameters: OPPOSITE_SIMILARITY_UPPER_BOUND={OPPOSITE_SIMILARITY_UPPER_BOUND}, MIN_NEGATIVE_KEYWORD_SCORE={MIN_NEGATIVE_KEYWORD_SCORE}")


def suggest_opposite_candidates(
    core_concepts_list: List[str],
    taxonomy_concepts: Dict[str, Dict],
    primary_embeddings: Dict[str, np.ndarray],
    sbert_model: SentenceTransformer,
    output_file: str,
    args: argparse.Namespace):
    """
    Generates candidate opposite concepts for a list of core input concepts.
    """
    # --->>> DEBUG STEP 3: Check list passed to the function <<<---
    logger_suggester.info(f"DEBUG: Entered suggest_opposite_candidates function. Received core_concepts_list with length: {len(core_concepts_list)}")
    if args.debug and core_concepts_list:
        logger_suggester.debug(f"DEBUG: First 5 concepts received by function: {core_concepts_list[:5]}")
    if not core_concepts_list:
         logger_suggester.error("DEBUG: suggest_opposite_candidates function received an EMPTY list. No processing possible.")
         # Decide if we should return early or let it run (it will produce empty output)
         return # Return early if the list is empty
    # --->>> END DEBUG STEP 3 <<<---

    logger_suggester.info(f"Starting opposite candidate suggestion for {len(core_concepts_list)} core concepts.")
    suggestions_map = defaultdict(list)
    processed_concepts = set()

    if not primary_embeddings:
        logger_suggester.critical("Primary embeddings map is empty inside suggest_opposite_candidates. Cannot proceed.")
        return # Return early

    uris_with_embeddings = set(primary_embeddings.keys())
    all_concept_uris = set(taxonomy_concepts.keys())
    missing_embeddings_uris = all_concept_uris - uris_with_embeddings
    if missing_embeddings_uris:
        logger_suggester.warning(f"{len(missing_embeddings_uris)} concepts lack primary embeddings (out of {len(all_concept_uris)} total). They won't be considered as candidates.")

    disable_tqdm = not logger_suggester.isEnabledFor(logging.INFO) or args.debug
    # Main loop processing each input concept
    for input_concept_raw in tqdm(core_concepts_list, desc="Processing Core Concepts", disable=disable_tqdm):
        norm_input = normalize_concept_suggester(input_concept_raw)
        if not norm_input: continue
        if norm_input in processed_concepts: continue
        processed_concepts.add(norm_input)

        logger_suggester.info(f"--- Finding candidates for: '{input_concept_raw}' (Normalized: '{norm_input}') ---")

        try:
            input_embedding = sbert_model.encode([TRAVEL_CONTEXT + norm_input])[0]
            if not isinstance(input_embedding, np.ndarray) or input_embedding.size == 0:
                 logger_suggester.error(f"Failed to get valid embedding for input '{norm_input}'. Skipping.")
                 continue
            # Debug input embedding
            if args.debug:
                logger_suggester.debug(f"DEBUG: Successfully embedded input '{norm_input}'. Shape: {input_embedding.shape}, Non-zero elements: {np.count_nonzero(input_embedding)}")

        except Exception as e:
            logger_suggester.error(f"Failed to embed input concept '{norm_input}': {e}", exc_info=args.debug)
            continue

        negative_keywords = OPPOSITE_KEYWORD_SETS.get(norm_input, set())
        if not negative_keywords:
            # logger_suggester.warning(f"No opposite keyword set defined for '{norm_input}'. Keyword scoring skipped.") # Reduced verbosity
            effective_keyword_weight = 0.0
            effective_distance_weight = 1.0
        else:
            effective_keyword_weight = KEYWORD_SCORE_WEIGHT
            effective_distance_weight = SEMANTIC_DISTANCE_WEIGHT

        candidate_scores: List[Tuple[float, Dict]] = []
        # Debug loop start
        logger_suggester.debug(f"DEBUG: Starting iteration over {len(primary_embeddings)} primary embeddings for input '{norm_input}'.")
        processed_candidate_count = 0

        # Loop comparing input concept to all taxonomy concepts with embeddings
        iter_uris = tqdm(primary_embeddings.items(), desc=f"  Scanning candidates for '{norm_input}'", leave=False, disable=disable_tqdm)
        for uri, embedding in iter_uris:
            # Debug first few comparisons
            if args.debug and processed_candidate_count < 5:
                 logger_suggester.debug(f"DEBUG: Comparing '{norm_input}' against candidate URI: {uri}")
            processed_candidate_count += 1

            if uri not in taxonomy_concepts: continue

            similarity = calculate_semantic_similarity_suggester(input_embedding, embedding)
            semantic_distance = 1.0 - similarity

            # Similarity Filter (Currently very relaxed: <= 0.8)
            if similarity >= OPPOSITE_SIMILARITY_UPPER_BOUND:
                if args.debug: logger_suggester.debug(f"DEBUG: Skip {uri} due to similarity ({similarity:.3f} >= {OPPOSITE_SIMILARITY_UPPER_BOUND})")
                continue

            keyword_score = 0.0
            matched_keywords = []
            if negative_keywords and effective_keyword_weight > 0:
                concept_text_lower = get_concept_text_suggester(uri, taxonomy_concepts)
                if concept_text_lower:
                    found_negative_keywords = sum(1 for neg_keyword in negative_keywords if re.search(r'\b' + re.escape(neg_keyword) + r'\b', concept_text_lower))
                    matched_keywords = [kw for kw in negative_keywords if re.search(r'\b' + re.escape(kw) + r'\b', concept_text_lower)]
                    keyword_score = (found_negative_keywords / len(negative_keywords)) if negative_keywords else 0.0

            # Keyword Filter (Currently disabled: MIN_NEGATIVE_KEYWORD_SCORE = 0.0)
            if negative_keywords and MIN_NEGATIVE_KEYWORD_SCORE > 0 and keyword_score < MIN_NEGATIVE_KEYWORD_SCORE:
                if args.debug: logger_suggester.debug(f"DEBUG: Skip {uri} due to keyword score ({keyword_score:.3f} < {MIN_NEGATIVE_KEYWORD_SCORE})")
                continue

            ranking_score = (effective_distance_weight * semantic_distance) + (effective_keyword_weight * keyword_score)

            if ranking_score > 0:
                pref_label = get_pref_label_suggester(uri, taxonomy_concepts, fallback=f"Label missing for {uri}")
                candidate_info = {
                    "uri": uri,
                    "skos:prefLabel": pref_label,
                    "ranking_score": round(ranking_score, 4),
                    "semantic_distance": round(semantic_distance, 4),
                    "cosine_similarity": round(similarity, 4),
                    "keyword_score": round(keyword_score, 4),
                    "matched_keywords": sorted(list(set(matched_keywords)))
                }
                candidate_scores.append((ranking_score, candidate_info))
                if args.debug and len(candidate_scores) <= 5: # Log first few found candidates
                     logger_suggester.debug(f"DEBUG: Added candidate {uri} (Score: {ranking_score:.3f})")

        # Debug loop end
        logger_suggester.debug(f"DEBUG: Finished iteration. Processed {processed_candidate_count} candidates for '{norm_input}'. Found {len(candidate_scores)} potential matches before sorting.")

        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [info for score, info in candidate_scores[:NUM_CANDIDATES_TO_SHOW]]
        suggestions_map[norm_input] = top_candidates

        logger_suggester.info(f"Found {len(top_candidates)} potential opposite candidates for '{norm_input}'.")
        # (Debug logging for top candidates remains the same)

    # --- Write Suggestions to JSON ---
    logger_suggester.info(f"Writing suggestions for {len(suggestions_map)} concepts to {output_file}")
    try:
        output_data = {
             "_metadata": {
                "description": "Candidate opposite URIs suggested by suggest_opposite_candidates.py.",
                "instructions": "REQUIRES HUMAN REVIEW AND SELECTION before use in INPUT_TO_OPPOSITE_URI_SET_MAP.",
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "cache_version": CACHE_VERSION,
                "travel_context_used_for_input": TRAVEL_CONTEXT,
                "num_candidates_per_concept": NUM_CANDIDATES_TO_SHOW,
                "similarity_upper_bound": OPPOSITE_SIMILARITY_UPPER_BOUND,
                "min_keyword_score_filter": MIN_NEGATIVE_KEYWORD_SCORE,
                "semantic_distance_weight": SEMANTIC_DISTANCE_WEIGHT,
                "keyword_score_weight": KEYWORD_SCORE_WEIGHT,
                "source_core_concepts_file": getattr(args, 'core_concepts_file', 'N/A'),
            },
            "candidate_opposites": dict(suggestions_map)
        }
        output_dir = os.path.dirname(output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger_suggester.info(f"Successfully wrote suggestions JSON to '{output_file}'")
    except Exception as e:
        logger_suggester.error(f"Failed to write suggestions JSON to '{output_file}': {e}", exc_info=True)


# --- Main Execution Guard ---
if __name__ == "__main__":
    start_time_script = time.time()
    parser = argparse.ArgumentParser(
        description=f"Suggest candidate opposite concepts... Uses cache version '{CACHE_VERSION}'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--core-concepts-file", required=True, help="Input text file with core concepts...")
    parser.add_argument("--taxonomy-dir", required=True, help="Directory containing RDF taxonomy files...")
    parser.add_argument("--cache-dir", required=True, help=f"Directory for storing/reading cache files...")
    parser.add_argument("--output-file", default="./candidate_opposites.json", help="Output JSON file path...")
    parser.add_argument("--rebuild-cache", action="store_true", default=False, help="Force rebuild of caches...")
    parser.add_argument("--skip-embeddings", action="store_true", default=False, help="Assume embeddings cache is up-to-date...")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable detailed debug logging...")
    args = parser.parse_args()

    # --- Setup Logging Level ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Ensure logger level is set correctly
    logger_suggester.setLevel(log_level)
    # Ensure handlers also respect the level
    for handler in logger_suggester.handlers:
        handler.setLevel(log_level)
    logger_suggester.info(f"Logging level set to: {logging.getLevelName(log_level)}")


    print(f"--- Running Opposite Candidate Suggester (Log File: {log_filename_suggester}) ---")
    print(f" Config:")
    print(f"  Core Concepts File: {args.core_concepts_file}")
    print(f"  Taxonomy Dir:       {args.taxonomy_dir}")
    print(f"  Cache Dir:          {args.cache_dir}")
    print(f"  Output File:        {args.output_file}")
    print(f" Parameters:")
    print(f"  Cache Version:      {CACHE_VERSION}")
    print(f"  Travel Context:     '{TRAVEL_CONTEXT}'")
    print(f"  Rebuild Cache:      {args.rebuild_cache}")
    print(f"  Skip Embeddings:    {args.skip_embeddings}")
    print(f"  Debug Logging:      {args.debug}")
    print(f" Suggestion Tuning:")
    print(f"  Num Candidates:     {NUM_CANDIDATES_TO_SHOW}")
    print(f"  Sim. Upper Bound:   {OPPOSITE_SIMILARITY_UPPER_BOUND}")
    print(f"  Min Keyword Score:  {MIN_NEGATIVE_KEYWORD_SCORE}")
    print(f"  Distance Weight:    {SEMANTIC_DISTANCE_WEIGHT}")
    print(f"  Keyword Weight:     {KEYWORD_SCORE_WEIGHT}")
    print("-" * 30)


    # --- Create Cache Directory if it doesn't exist ---
    try:
        os.makedirs(args.cache_dir, exist_ok=True)
        logger_suggester.info(f"Cache directory checked/created: {args.cache_dir}")
    except OSError as e:
        logger_suggester.critical(f"Failed to create cache directory '{args.cache_dir}': {e}")
        sys.exit(1)


    # --- Load core concepts from file ---
    core_concepts_to_process: List[str] = [] # Initialize empty list
    try:
        logger_suggester.info(f"Attempting to open core concepts file: {args.core_concepts_file}")
        line_count = 0
        accepted_count = 0
        temp_loaded_concepts = [] # Use temporary list inside 'with'
        with open(args.core_concepts_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                stripped_line = line.strip()
                is_comment = line.startswith('#')

                if args.debug: # Optional line-by-line debug
                    logger_suggester.debug(f"DEBUG Line {line_num}: Raw='{line.rstrip()}', Stripped='{stripped_line}', StartsWith#: {is_comment}, Keep: {bool(stripped_line and not is_comment)}")

                if stripped_line and not is_comment:
                    temp_loaded_concepts.append(stripped_line)
                    accepted_count += 1

        # --->>> CRITICAL DEBUG STEP 1: Check immediately after closing the file <<<---
        logger_suggester.info(f"DEBUG: File closed. Read {line_count} total lines. Accepted {accepted_count} concepts into temporary list.")
        logger_suggester.info(f"DEBUG: Length of 'temp_loaded_concepts' list: {len(temp_loaded_concepts)}")
        if args.debug and temp_loaded_concepts:
             logger_suggester.debug(f"DEBUG: First 5 concepts in temp list: {temp_loaded_concepts[:5]}")
        # --->>> END DEBUG STEP 1 <<<---

        # Assign to the main variable AFTER closing the file
        core_concepts_to_process = temp_loaded_concepts

        # Check the main variable immediately after assignment
        logger_suggester.info(f"DEBUG: Assigned to 'core_concepts_to_process'. Current length: {len(core_concepts_to_process)}")

        # Check if the list is empty and provide feedback
        if not core_concepts_to_process and line_count > 0:
             logger_suggester.warning("Core concepts file was read, but no valid (non-empty, non-comment) lines were found/assigned.")
        elif not core_concepts_to_process and line_count == 0:
             logger_suggester.warning("Core concepts file appears to be completely empty.")

        # Final confirmation log message
        logger_suggester.info(f"Loaded {len(core_concepts_to_process)} core concepts from '{args.core_concepts_file}'.")

    except FileNotFoundError:
        logger_suggester.critical(f"Core concepts file not found: '{args.core_concepts_file}'")
        sys.exit(1)
    except Exception as e:
        logger_suggester.critical(f"Failed to read core concepts file '{args.core_concepts_file}': {e}", exc_info=True)
        sys.exit(1)

    # --->>> CRITICAL DEBUG STEP 2: Check right before the next major block <<<---
    logger_suggester.info(f"DEBUG: Checkpoint before loading taxonomy/embeddings. Length of 'core_concepts_to_process': {len(core_concepts_to_process)}")
    if not core_concepts_to_process:
         logger_suggester.error("DEBUG: core_concepts_to_process is EMPTY before loading data! No concepts to process.")
         # Optionally exit if no concepts loaded
         print("Error: No core concepts loaded to process. Exiting.", file=sys.stderr)
         sys.exit(1) # Exit if the list is empty, as nothing can be done
    # --->>> END DEBUG STEP 2 <<<---


    # --- Load necessary data (taxonomy, embeddings) using functions from main script ---
    try:
        # 1. Load SBERT Model
        sbert_model_instance = get_sbert_model_suggester()

        # 2. Load Taxonomy Concepts (uses caching)
        concepts_cache_path = os.path.join(args.cache_dir, f"concepts_{CACHE_VERSION}.json")
        logger_suggester.info(f"Attempting to load concepts from: {concepts_cache_path}")
        taxonomy_concepts_instance = load_taxonomy_concepts(args.taxonomy_dir, concepts_cache_path, args)
        if not taxonomy_concepts_instance:
            raise RuntimeError("Failed to load taxonomy concepts.")
        logger_suggester.info(f"Loaded {len(taxonomy_concepts_instance)} concepts.")

        # 3. Load/Compute Taxonomy Embeddings (uses caching)
        embeddings_cache_path = os.path.join(args.cache_dir, f"embeddings_{CACHE_VERSION}.pkl")
        logger_suggester.info(f"Attempting to load embeddings from: {embeddings_cache_path}")
        embeddings_data = precompute_taxonomy_embeddings(
            taxonomy_concepts=taxonomy_concepts_instance,
            model=sbert_model_instance,
            cache_file=embeddings_cache_path, # Use correct keyword
            args=args
        )
        if not embeddings_data: raise RuntimeError("Failed to load or compute taxonomy embeddings.")

        # Extract primary embeddings (using simple strategy)
        all_uri_embeddings_data, _, _ = embeddings_data
        primary_embeddings_instance = {}
        missing_primary = 0
        if isinstance(all_uri_embeddings_data, dict):
            for uri, embedding_list in all_uri_embeddings_data.items():
                if embedding_list and isinstance(embedding_list, list):
                    if len(embedding_list[0]) >= 3 and isinstance(embedding_list[0][2], np.ndarray):
                        primary_embeddings_instance[uri] = embedding_list[0][2]
                    else: missing_primary += 1 # Track unexpected structure
                else: missing_primary += 1 # Track URIs with empty embedding lists
        else:
             logger_suggester.error("Loaded embedding data structure is not a dictionary as expected.")

        if missing_primary > 0:
             logger_suggester.warning(f"Could not determine a primary embedding for {missing_primary} URIs from the loaded embedding data.")

        if not primary_embeddings_instance:
             raise RuntimeError("Extracted primary embeddings map is empty. Cannot proceed.")
        logger_suggester.info(f"Extracted {len(primary_embeddings_instance)} primary embeddings for comparison.")


        # Check the list one last time before passing it to the function
        logger_suggester.info(f"DEBUG: Checkpoint before calling suggest_opposite_candidates function. Length of 'core_concepts_to_process': {len(core_concepts_to_process)}")
        if not core_concepts_to_process:
            logger_suggester.error("DEBUG: Core concepts list became empty before calling main function!")
            sys.exit(1) # Exit if empty


        # 4. Run the suggestion process
        suggest_opposite_candidates(
            core_concepts_list=core_concepts_to_process, # Pass the confirmed list
            taxonomy_concepts=taxonomy_concepts_instance,
            primary_embeddings=primary_embeddings_instance,
            sbert_model=sbert_model_instance,
            output_file=args.output_file,
            args=args
        )

        end_time_script = time.time()
        print("-" * 30)
        print(f"Candidate suggestion process complete.")
        print(f"Output written to: '{args.output_file}'")
        print(f"Total execution time: {end_time_script - start_time_script:.2f} seconds.")
        print("Next Step: Manually review the JSON file and select URIs...")
        print("-" * 30)

    except FileNotFoundError as e:
         logger_suggester.critical(f"File/Dir not found: {e}"); print(f"\nERROR: Not found: {e.filename}", file=sys.stderr); sys.exit(1)
    except TypeError as e:
        logger_suggester.critical(f"TypeError during function call (likely signature mismatch): {e}", exc_info=args.debug)
        logger_suggester.critical("Ensure function signatures match between this script and the main generator script.")
        print(f"\nTYPE ERROR: {e}. Check function definitions/calls. See log '{log_filename_suggester}'.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        logger_suggester.critical(f"Runtime error: {e}", exc_info=args.debug); print(f"\nRUNTIME ERROR: {e}. See log.", file=sys.stderr); sys.exit(1)
    except Exception as e:
        logger_suggester.critical(f"Unexpected error: {e}", exc_info=True); print(f"\nCRITICAL ERROR. See log.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)