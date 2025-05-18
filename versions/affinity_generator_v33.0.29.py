# -*- coding: utf-8 -*-
"""
Generate affinity definitions for travel concepts using combined scoring (v33.0.29 - BM25 Keyword Scoring).

Implements:
- Okapi BM25 for keyword relevance scoring instead of TF-IDF.
- Concept-specific overrides (alpha, seeding, filtering, expansion, query split) driven by config file.
- TF-IDF Field Weighting (used for constructing BM25 documents).
- Multi-key sorting with namespace priority tie-breaking.
- Combined BM25 Keyword + SBERT Similarity score for candidate selection.
- Absolute SBERT Filter (0.15).
- Conditional Keyword Score Dampening (based on SBERT, Thresh=0.35, Factor=0.15).
- Optional LLM keyword expansion for weak concepts.
- LLM-assisted theme slotting.
- Relies on shared utility functions in utils.py.

Changes from v33.0.28:
- Replaced TF-IDF indexing/search with Okapi BM25 using rank-bm25 library.
- Updated build_keyword_index and get_candidate_concepts_keyword.
- Renamed TFIDF_CONFIG section in config checks/logs to KEYWORD_SCORING_CONFIG.
- Updated SCRIPT_VERSION and related logs.
- Requires installing rank-bm25 (pip install rank-bm25).
- NOTE: Requires keyword index cache rebuild (--rebuild-cache).
"""

import argparse
import json
import logging
import math
import os
import pickle
import re
import sys
import time
import traceback
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any, Set

# --- Third-Party Imports ---
try:
    import numpy as np
except ImportError:
    print("CRITICAL ERROR: numpy not found.", file=sys.stderr); sys.exit(1)

try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF
    from rdflib import util as rdflib_util
except ImportError:
    pass # Checked via utils import

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass # Checked via utils import

# Remove Scikit-learn TFIDF/Cosine - No longer needed for keyword scoring
# try:
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity
# except ImportError:
#     pass # Checked via utils import

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found.", file=sys.stderr)
    def tqdm(iterable, *args, **kwargs):
        return iterable # Define dummy

# --- NEW: BM25 Import ---
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    print("ERROR: rank-bm25 library not found. Please install: pip install rank-bm25", file=sys.stderr)
    RANK_BM25_AVAILABLE = False
    class BM25Okapi: pass # Dummy for type hinting

# --- LLM Imports & Placeholders ---
class APITimeoutError(Exception): pass
class APIConnectionError(Exception): pass
class RateLimitError(Exception): pass
class DummyLLMClient: pass

OpenAI_Type = DummyLLMClient
GoogleAI_Type = DummyLLMClient
OpenAI = None
OPENAI_AVAILABLE = False
genai = None
GOOGLE_AI_AVAILABLE = False

try:
    from openai import OpenAI as RealOpenAI, APITimeoutError as RealAPITimeoutError, \
        APIConnectionError as RealAPIConnectionError, RateLimitError as RealRateLimitError
    OpenAI = RealOpenAI
    OpenAI_Type = RealOpenAI
    APITimeoutError = RealAPITimeoutError
    APIConnectionError = RealAPIConnectionError
    RateLimitError = RealRateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    logging.debug("OpenAI library not found (optional).")
except Exception as e:
    logging.error(f"OpenAI import error: {e}")

try:
    import google.generativeai as genai_import
    genai = genai_import
    GoogleAI_Type = genai_import.GenerativeModel
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    logging.debug("google.generativeai library not found (optional).")
except Exception as e:
    GOOGLE_AI_AVAILABLE = False; genai = None; logging.error(f"Google AI import error: {e}")

# --- Utility Function Imports ---
try:
    # Ensure utils.py includes all required functions from previous versions
    # Note: SKLEARN_AVAILABLE check might be removed from utils if only used for TFIDF here
    from utils import (
        setup_logging, normalize_concept, get_primary_label, get_concept_type_labels,
        get_sbert_model, load_affinity_config, get_cache_filename, load_cache, save_cache,
        load_taxonomy_concepts, precompute_taxonomy_embeddings, get_concept_embedding,
        get_batch_embedding_similarity, get_kg_data,
        build_keyword_label_index, # Still needed for simple lookups
        save_results_json,
        RDFLIB_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE as UTILS_ST_AVAILABLE
        # SKLEARN_AVAILABLE as UTILS_SK_AVAILABLE # Removed if only for TFIDF
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Failed import from 'utils.py': {e}", file=sys.stderr); sys.exit(1)
except Exception as e:
     print(f"CRITICAL ERROR: Unexpected error importing from 'utils.py': {e}", file=sys.stderr); sys.exit(1)

# --- Config Defaults & Constants ---
SCRIPT_VERSION = "affinity-rule-engine-v33.0.29 (BM25 Keyword Scoring)" # Updated version
DEFAULT_CACHE_VERSION = "v20250420.affinity.33.0.29" # Update cache version due to BM25 change
DEFAULT_TAXONOMY_DIR = "./datasources/"
DEFAULT_CACHE_DIR = "./cache_v33/"
DEFAULT_CONFIG_FILE = "./affinity_config_v33.2.json"
OUTPUT_FILE_TEMPLATE = "affinity_definitions_{cache_version}.json"
LOG_FILE_TEMPLATE = "affinity_generation_{cache_version}.log"

# --- Core Parameters (Defaults - Config will override) ---
COMBINED_SCORE_ALPHA = 0.6
ABSOLUTE_MIN_SBERT_SCORE = 0.15
DAMPENING_SBERT_THRESHOLD = 0.35
DAMPENING_FACTOR = 0.15
ENABLE_KW_EXPANSION = True

# --- Other Tunable Parameters (Load overrides from config file) ---
MAX_CANDIDATES_FOR_LLM = 75
EVIDENCE_MIN_SIMILARITY = 0.30
KEYWORD_MIN_SCORE = 0.01 # Default for BM25 - likely needs tuning based on observed scores
KEYWORD_TOP_N_SEARCH = 500 # How many candidates to consider from BM25 ranking
THEME_ATTRIBUTE_MIN_WEIGHT = 0.001
TRAVEL_CONTEXT = "travel "
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 5
LLM_TEMPERATURE = 0.2
LLM_RETRY_DELAY_SECONDS = 5
MIN_KEYWORD_CANDIDATES_FOR_EXPANSION = 5
KW_EXPANSION_TEMPERATURE = 0.5
TOP_N_DEFINING_ATTRIBUTES_DEFAULT = 25

# --- List of known abstract/package concepts ---
ABSTRACT_CONCEPTS_LIST = ["luxury", "budget", "value"] # Config override handles others

# --- Globals ---
_config_data: Optional[Dict] = None
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict[str, np.ndarray], List[str]]] = None
_keyword_label_index: Optional[Dict[str, Set[str]]] = None
_bm25_model: Optional[BM25Okapi] = None # Replaced TF-IDF globals
_keyword_corpus_uris: Optional[List[str]] = None # Renamed for clarity
_openai_client: Optional[OpenAI_Type] = None
_google_client: Optional[Any] = None
logger = logging.getLogger(__name__)
# Global reference to args for build_keyword_index (simpler than passing down)
args: Optional[argparse.Namespace] = None


# --- Helper Functions Retained in Main Script ---
# (get_theme_definition_for_prompt, get_dynamic_theme_config, normalize_weights, deduplicate_attributes, validate_llm_assignments unchanged)
def get_theme_definition_for_prompt(theme_name: str, theme_data: Dict) -> str:
    """Gets a theme description suitable for an LLM prompt."""
    desc = theme_data.get("description")
    if isinstance(desc, str) and desc.strip(): return desc.strip()
    logger.warning(f"Theme '{theme_name}' missing description. Using fallback.")
    return f"Theme related to {theme_name} ({theme_data.get('type', 'general')} aspects)."

def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: Dict) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    """Gets theme config considering base and concept-specific overrides."""
    base_themes = config.get("base_themes", {})
    concept_overrides = config.get("concept_overrides", {}) # Read top-level overrides
    base_data = base_themes.get(theme_name)
    if not base_data: logger.error(f"Base theme '{theme_name}' not found!"); return "Optional", 0.0, None, None
    # Get concept-specific overrides, including theme overrides nested within it
    concept_override_data = concept_overrides.get(normalized_concept, {})
    theme_override = concept_override_data.get("themes", {}).get(theme_name, {}) # Check nested 'themes' key
    merged = {**base_data, **theme_override}
    rule = merged.get("rule_applied", merged.get("rule", "Optional"))
    rule = "Optional" if rule not in ["Must have 1", "Optional"] else rule
    weight = merged.get("weight", base_data.get("weight", 0.0)) # Ensure weight defaults to base if not overridden
    weight = 0.0 if not isinstance(weight, (int, float)) or weight < 0 else float(weight)
    return rule, weight, merged.get("subScore"), merged.get("fallback_logic")

def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a dictionary of weights to sum to 1.0."""
    if not weights_dict: return {}
    total = sum(weights_dict.values())
    if total <= 0: return {k: 0.0 for k in weights_dict}
    return {k: v / total for k, v in weights_dict.items()}

def deduplicate_attributes(attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates attribute list, keeping entry with highest weight for each URI."""
    best: Dict[str, Dict[str, Any]] = {}
    for attr in attributes:
        uri = attr.get("uri")
        if not uri or not isinstance(uri, str): continue
        try: weight = float(attr.get("concept_weight", 0.0))
        except (ValueError, TypeError): weight = 0.0
        current = best.get(uri)
        try: current_weight = float(current.get("concept_weight", -1.0)) if current else -1.0
        except (ValueError, TypeError): current_weight = -1.0
        if current is None or weight > current_weight: best[uri] = attr
    return list(best.values())

def validate_llm_assignments(llm_response_data: Optional[Dict[str, Any]], uris_sent: Set[str], valid_themes: Set[str], concept_label: str, diag_llm: Dict) -> Optional[Dict[str, List[str]]]:
     """Validates the structure and content of the LLM theme assignments."""
     if not llm_response_data or "theme_assignments" not in llm_response_data: logger.error(f"[{concept_label}] LLM response missing key 'theme_assignments'."); diag_llm["error"] = "Missing 'theme_assignments'."; return None
     assigns = llm_response_data["theme_assignments"]
     if not isinstance(assigns, dict): logger.error(f"[{concept_label}] LLM assignments not dict."); diag_llm["error"] = "Assignments structure not dict."; return None
     validated: Dict[str, List[str]] = {}
     uris_resp = set(assigns.keys()); extra = uris_resp - uris_sent; missing = uris_sent - uris_resp
     if extra: logger.warning(f"[{concept_label}] LLM returned {len(extra)} extra URIs. Ignoring.")
     if missing: logger.warning(f"[{concept_label}] LLM response missing {len(missing)} URIs. Adding empty."); [assigns.setdefault(uri, []) for uri in missing]
     parsed_count = 0; uris_proc = 0
     for uri, themes in assigns.items():
         if uri not in uris_sent: continue
         uris_proc += 1
         if not isinstance(themes, list): logger.warning(f"[{concept_label}] Invalid themes format for {uri}. Setting to empty list."); validated[uri] = []; continue
         valid = [t for t in themes if isinstance(t, str) and t in valid_themes]
         invalid = set(themes) - set(valid)
         if invalid: logger.warning(f"[{concept_label}] Invalid themes ignored for {uri}: {invalid}")
         validated[uri] = valid; parsed_count += len(valid)
     diag_llm["assignments_parsed_count"] = parsed_count; diag_llm["uris_in_response_count"] = uris_proc
     return validated

# --- START: Keyword Indexing Function (v33.0.29 - BM25) ---
def build_keyword_index(
    config: Dict, taxonomy_concepts_cache: Dict[str, Dict], cache_dir: str,
    cache_version: str, rebuild_cache: bool
) -> Tuple[Optional[BM25Okapi], Optional[List[str]], Optional[Dict[str, Set[str]]]]:
    """
    Builds or loads a BM25 keyword index and simple label index.
    v33.0.29: Replaced TF-IDF with BM25. Uses weighted field concatenation for documents.
              Requires rank-bm25 library and cache rebuild.
    """
    global _bm25_model, _keyword_corpus_uris, _keyword_label_index, args # Use new globals
    bm25_model, corpus_uris, label_index = None, None, None
    kw_scoring_cfg = config.get("KEYWORD_SCORING_CONFIG", config.get("TFIDF_CONFIG", {})) # Support old name for now
    use_bm25 = kw_scoring_cfg.get("enabled", False) and kw_scoring_cfg.get("algorithm", "tfidf").lower() == "bm25" and RANK_BM25_AVAILABLE

    # Define repetition factors for weighted doc construction
    pref_label_weight = 5
    alt_label_weight = 3
    definition_weight = 1

    uris_to_log_doc = { # Renamed for clarity
        "urn:expediagroup:taxonomies:acs:#1f3da634-0df6-4498-a8d9-603f895c8f3f", # AirConditioning:Unavailable
        "urn:expediagroup:taxonomies:acsPCS:#AirConditioning",
        "urn:expediagroup:taxonomies:lcm:#34528832-c2d8-312a-b52f-b27c483e5ec1", # All-inclusive resort
        "urn:expediagroup:taxonomies:lcm:#b2fa2cb3-9226-305e-8943-02d82ba90975", # Resort
        "urn:expediagroup:taxonomies:acsEnumerations:#dcecb8c7-bacf-4d04-b90f-95e4654ffa9f", # Allinclusive
    }

    if use_bm25:
        logger.info("BM25 Keyword Indexing enabled.")
        bm25_params_cfg = kw_scoring_cfg.get("bm25_params", {})
        k1 = float(bm25_params_cfg.get("k1", 1.5)) # Default BM25 k1
        b = float(bm25_params_cfg.get("b", 0.75))  # Default BM25 b
        logger.info(f"Using BM25 Params: k1={k1}, b={b}")
        logger.info(f"Applying Field Weighting: prefLabel x{pref_label_weight}, altLabel x{alt_label_weight}, definition x{definition_weight}")

        # Add BM25 params and weighting factors to cache filename
        cache_params_for_filename = {
            "alg": "bm25",
            "k1": k1, "b": b,
            "weights": f"p{pref_label_weight}a{alt_label_weight}d{definition_weight}"
        }

        model_cache_file = get_cache_filename("bm25_model", cache_version, cache_dir, cache_params_for_filename, ".pkl")
        uris_cache_file = get_cache_filename("bm25_corpus_uris", cache_version, cache_dir, cache_params_for_filename, ".pkl") # Separate URIs cache

        cache_valid = False
        if not rebuild_cache:
            # Attempt to load cache specific to BM25 and its params
            cached_model = load_cache(model_cache_file, 'pickle')
            cached_uris = load_cache(uris_cache_file, 'pickle')
            if cached_model is not None and isinstance(cached_model, BM25Okapi) and isinstance(cached_uris, list):
                # Basic check: Do URIs seem consistent with model? (More robust checks difficult)
                if hasattr(cached_model, 'doc_len_') and len(cached_model.doc_len_) == len(cached_uris):
                    bm25_model, corpus_uris, cache_valid = cached_model, cached_uris, True
                    logger.info(f"BM25 Model (Weighted Fields) loaded from cache ({len(corpus_uris)} URIs).")
                else:
                    logger.warning("BM25 cache model/URI count mismatch. Rebuilding.")
            else:
                 logger.info("BM25 cache (Weighted Fields) incomplete/invalid. Rebuilding.")

        if not cache_valid:
            logger.info("Rebuilding BM25 index with weighted fields...")
            try:
                docs, doc_uris = [], []
                debug_mode = args.debug if args else False
                disable_tqdm = not logger.isEnabledFor(logging.INFO) or debug_mode

                logger.info("Preparing documents for BM25...")
                for uri, data in tqdm(sorted(taxonomy_concepts_cache.items()), desc="Prepare BM25 Docs", disable=disable_tqdm):
                    texts_to_join = []
                    if not isinstance(data, dict): continue

                    pref_labels_raw = data.get("skos:prefLabel", [])
                    alt_labels_raw = data.get("skos:altLabel", [])
                    definitions_raw = data.get("skos:definition", [])

                    pref_labels = [str(lbl) for lbl in (pref_labels_raw if isinstance(pref_labels_raw, list) else [pref_labels_raw]) if lbl and isinstance(lbl, str)]
                    alt_labels = [str(lbl) for lbl in (alt_labels_raw if isinstance(alt_labels_raw, list) else [alt_labels_raw]) if lbl and isinstance(lbl, str)]
                    definition = None
                    if isinstance(definitions_raw, list):
                        for defin in definitions_raw:
                            if defin and isinstance(defin, str) and defin.strip():
                                definition = str(defin); break
                    elif isinstance(definitions_raw, str) and definitions_raw.strip():
                        definition = definitions_raw

                    for _ in range(pref_label_weight): texts_to_join.extend(pref_labels)
                    for _ in range(alt_label_weight): texts_to_join.extend(alt_labels)
                    if definition and definition_weight > 0: texts_to_join.append(definition)

                    raw_doc_text = " ".join(filter(None, texts_to_join))
                    norm_doc = normalize_concept(raw_doc_text) # Normalize before tokenizing

                    if uri in uris_to_log_doc and args.debug:
                        label = get_primary_label(uri, taxonomy_concepts_cache, fallback=uri)
                        raw_snippet = raw_doc_text[:200] + ('...' if len(raw_doc_text) > 200 else '')
                        norm_snippet = norm_doc[:200] + ('...' if len(norm_doc) > 200 else '')
                        logger.debug(f"[BM25 DOC LOG - Weighted] URI: {uri} Label: '{label}' Raw Snippet: '{raw_snippet}' Norm Snippet: '{norm_snippet}'")

                    if norm_doc:
                        # Tokenize for BM25 (simple space split after normalization)
                        tokenized_doc = norm_doc.split()
                        docs.append(tokenized_doc) # Store tokenized list
                        doc_uris.append(uri)

                if not docs:
                    logger.warning("No documents generated for BM25 build after weighting.")
                else:
                    logger.info(f"Tokenization complete. Building BM25 index for {len(docs)} documents...")
                    # Instantiate BM25 model
                    bm25_model = BM25Okapi(docs, k1=k1, b=b)
                    corpus_uris = doc_uris # Assign the final list of URIs
                    # Save the BM25 model and the corresponding URIs
                    save_cache(bm25_model, model_cache_file, 'pickle')
                    save_cache(corpus_uris, uris_cache_file, 'pickle')
                    logger.info("BM25 model and corpus URIs (Weighted Fields) rebuilt and saved.")
            except Exception as e:
                logger.error(f"BM25 build error with weighted fields: {e}", exc_info=True)
                bm25_model, corpus_uris = None, None

        logger.debug("--- BM25 Index Build/Load Diagnostics ---")
        logger.debug(f"BM25 Model Type: {type(bm25_model)}")
        if bm25_model:
            logger.debug(f"BM25 Params Used: k1={getattr(bm25_model, 'k1', 'N/A')}, b={getattr(bm25_model, 'b', 'N/A')}")
            logger.debug(f"Indexed Docs: {getattr(bm25_model, 'corpus_size', 'N/A')}")
        else:
             logger.debug("BM25 Model: Not available.")
        logger.debug(f"Corpus URIs Count: {len(corpus_uris) if corpus_uris is not None else 'None'}")
        logger.debug("--- End BM25 Diagnostics ---")
    else:
        logger.info("BM25 Keyword Indexing disabled or library unavailable.")

    # Build simple label index (remains the same, uses original labels)
    try:
        label_index = build_keyword_label_index(taxonomy_concepts_cache)
        if label_index is None:
            logger.warning("Failed simple label index build (returned None).")
        else:
            logger.info(f"Simple label index ready ({len(label_index)} keywords).")
    except Exception as e:
        logger.error(f"Error calling build_keyword_label_index: {e}", exc_info=True)
        label_index = None

    # Update globals
    _bm25_model = bm25_model
    _keyword_corpus_uris = corpus_uris
    _keyword_label_index = label_index

    return bm25_model, corpus_uris, label_index
# --- END: Keyword Indexing Function (v33.0.29 - BM25) ---


# --- START: Keyword Search Function (v33.0.29 - BM25) ---
def get_candidate_concepts_keyword(
    query_texts: List[str], bm25_model: BM25Okapi,
    corpus_uris: List[str], top_n: int, min_score: float = 0.01
) -> List[Dict[str, Any]]:
    """ Finds candidate concepts using BM25 similarity. """
    if not query_texts or bm25_model is None or not corpus_uris: return []
    if not hasattr(bm25_model, 'get_scores'):
         logger.error("Invalid BM25 model object passed.")
         return []
    if bm25_model.corpus_size != len(corpus_uris):
        logger.error(f"BM25 model size ({bm25_model.corpus_size}) != corpus URIs ({len(corpus_uris)}).")
        return []

    # Tokenize query similar to how documents were tokenized
    tokenized_query = [term for text in query_texts for term in normalize_concept(text).split()]
    if not tokenized_query:
        logger.warning(f"Query texts resulted in empty token list: {query_texts}")
        return []

    logger.debug(f"BM25 search query tokens: {tokenized_query[:20]}{'...' if len(tokenized_query) > 20 else ''}")

    try:
        # Get scores for all documents in the corpus
        all_scores = bm25_model.get_scores(tokenized_query)

        if args and args.debug: # Log score range only in debug mode
             min_s, max_s = np.min(all_scores), np.max(all_scores)
             logger.debug(f"BM25 Scores - Min: {min_s:.4f}, Max: {max_s:.4f}, Mean: {np.mean(all_scores):.4f}")
             if max_s < min_score:
                 logger.debug(f"Max BM25 score ({max_s:.4f}) < min_score ({min_score}). No candidates pass.")
                 return []

        # Find indices with scores >= min_score (BM25 scores can be <= 0)
        # It's generally better to filter *after* sorting for top N, unless min_score is high
        num_cands = len(all_scores)
        actual_top_n = min(top_n, num_cands)
        if actual_top_n <= 0: return []

        # Get top N indices based on score
        if actual_top_n < num_cands * 0.5: # Use argpartition for efficiency
            # Note: argpartition finds the k *smallest* elements, so we partition for the complement
            # Or, simpler, just get top N overall and then filter
            indices = np.argsort(all_scores)[::-1][:actual_top_n]
        else: # Sort all if top_n is large
            indices = np.argsort(all_scores)[::-1][:actual_top_n]

        # Filter the top N based on min_score
        candidates = []
        for i in indices:
            score = float(all_scores[i])
            # BM25 scores can be negative if query terms don't match doc terms well
            # Typically, only non-negative scores are considered relevant matches.
            # Adjust min_score check accordingly. A value > 0 is usually a good starting point.
            if score >= min_score:
                 # Check if index is valid for corpus_uris (should be, given checks above)
                 if 0 <= i < len(corpus_uris):
                      candidates.append({"uri": corpus_uris[i], "score": score, "method": "keyword_bm25"})
                 else:
                      logger.warning(f"BM25 returned invalid index {i} for corpus size {len(corpus_uris)}")
            # else: # Optional: Log how many were filtered by min_score among top N
            #     logger.debug(f"BM25 candidate {corpus_uris[i]} score {score:.4f} < min_score {min_score}. Filtered.")

        # logger.debug(f"Found {len(candidates)} BM25 candidates >= {min_score} (Top {top_n} considered)")
        return candidates

    except Exception as e:
        logger.error(f"BM25 search error: {e}", exc_info=True)
        return []
# --- END: Keyword Search Function (v33.0.29 - BM25) ---


# --- LLM Client Initialization & Call ---
# (get_openai_client, get_google_client, call_llm unchanged from v33.0.22)
try:
    from openai import OpenAI as RealOpenAI, APITimeoutError as RealAPITimeoutError, \
        APIConnectionError as RealAPIConnectionError, RateLimitError as RealRateLimitError
    OpenAI = RealOpenAI
    OpenAI_Type = RealOpenAI
    APITimeoutError = RealAPITimeoutError
    APIConnectionError = RealAPIConnectionError
    RateLimitError = RealRateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    logging.debug("OpenAI library not found (optional).")
except Exception as e:
    logging.error(f"OpenAI import error: {e}")

try:
    import google.generativeai as genai_import
    genai = genai_import
    GoogleAI_Type = genai_import.GenerativeModel
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    logging.debug("google.generativeai library not found (optional).")
except Exception as e:
    GOOGLE_AI_AVAILABLE = False; genai = None; logging.error(f"Google AI import error: {e}")

# --- LLM Client Initialization & Call ---
# (get_openai_client, get_google_client, call_llm unchanged from v33.0.22)
def get_openai_client()  -> Optional[OpenAI_Type]:
    """Initializes and returns the OpenAI client. Reduced logging."""
    global _openai_client, _config_data
    if not OPENAI_AVAILABLE: return None
    if _openai_client is None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: logger.warning("OPENAI_API_KEY env var not set."); return None
            timeout = _config_data.get("LLM_API_CONFIG", {}).get("REQUEST_TIMEOUT", 60) if _config_data else 60
            _openai_client = OpenAI(api_key=api_key, timeout=timeout)
        except Exception as e: logger.error(f"OpenAI client init failed: {e}"); return None
    return _openai_client

def get_google_client() -> Optional[Any]:
    """Configures and returns the Google AI client (genai module). Reduced logging."""
    global _google_client
    if not GOOGLE_AI_AVAILABLE: return None
    if _google_client is None:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key: logger.warning("GOOGLE_API_KEY env var not set."); return None
            genai.configure(api_key=api_key)
            _google_client = genai
        except Exception as e: logger.error(f"Google AI config failed: {e}"); return None
    return _google_client


def call_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float, provider: str) -> Dict[str, Any]:
    """Handles API calls to the specified LLM provider with retries. Reduced logging."""
    if not model_name: logger.error("LLM model name missing."); return {"success": False, "error": "LLM model missing"}
    result = {"success": False, "response": None, "error": None, "attempts": 0}
    client = get_openai_client() if provider == "openai" else get_google_client()
    if not client: result["error"] = f"{provider} client unavailable."; logger.error(result["error"]); return result
    delay = _config_data.get("LLM_API_CONFIG", {}).get("RETRY_DELAY_SECONDS", LLM_RETRY_DELAY_SECONDS) if _config_data else LLM_RETRY_DELAY_SECONDS

    for attempt in range(max_retries + 1):
        result["attempts"] = attempt + 1
        # logger.info(f"{provider} Attempt {attempt + 1}/{max_retries + 1}") # Reduced noise
        start = time.time()
        try:
            content = None
            if provider == "openai":
                if not isinstance(client, OpenAI_Type): raise RuntimeError("OpenAI client invalid.")
                resp = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=temperature, timeout=timeout)
                if resp.choices: content = resp.choices[0].message.content
            elif provider == "google":
                model_id = model_name if model_name.startswith("models/") else f"models/{model_name}"
                gen_cfg = client.types.GenerationConfig(candidate_count=1, temperature=temperature, response_mime_type="application/json")
                safety = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                model = client.GenerativeModel(model_id, system_instruction=system_prompt, safety_settings=safety)
                resp = model.generate_content([user_prompt], generation_config=gen_cfg, request_options={'timeout': timeout})
                if resp.prompt_feedback and resp.prompt_feedback.block_reason:
                    block_reason = getattr(resp.prompt_feedback, 'block_reason', 'N/A')
                    logger.warning(f"Gemini blocked prompt/response. Reason: {block_reason}. System: '{system_prompt[:100]}...', User: '{user_prompt[:100]}...'")
                    result["error"] = f"Blocked: {block_reason}"; return result
                # Check for candidates before accessing text
                if hasattr(resp, 'candidates') and resp.candidates:
                    content = resp.text
                else:
                     logger.warning(f"Gemini response received but has no candidates or valid text. Response: {resp}")
                     content = None
            else: result["error"] = f"Unsupported provider: {provider}"; logger.error(result["error"]); return result

            if content:
                try:
                    # Improved JSON extraction
                    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
                    if json_match:
                        cleaned = json_match.group(1)
                    elif '{' in content and '}' in content:
                         # Be more robust: find the first '{' and last '}'
                         first_brace = content.find('{')
                         last_brace = content.rfind('}')
                         if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                             cleaned = content[first_brace:last_brace+1]
                         else: # Fallback if braces are mismatched or content is not dict-like
                              cleaned = content.strip().strip('`')
                    else:
                         cleaned = content.strip().strip('`')

                    output = json.loads(cleaned)
                    if isinstance(output, dict):
                        result["success"] = True; result["response"] = output
                        return result
                    else: logger.error(f"{provider} response parsed but is not a dict. Type: {type(output)}. Raw: {content[:200]}...")
                except json.JSONDecodeError as e: logger.error(f"{provider} JSON parse error: {e}. Raw content start: {content[:500]}..."); result["error"] = f"JSON Parse Error: {e}"
                except Exception as e: logger.error(f"{provider} response processing error: {e}. Raw content start: {content[:500]}..."); result["error"] = f"Response Processing Error: {e}"
            else: logger.warning(f"{provider} response content was empty."); result["error"] = "Empty response from LLM"

        except (APITimeoutError, APIConnectionError, RateLimitError) as e: logger.warning(f"{provider} API Error during call: {type(e).__name__}"); result["error"] = f"{type(e).__name__}"
        except Exception as e: logger.error(f"{provider} Call Error: {e}", exc_info=True); result["error"] = str(e); return result

        should_retry = attempt < max_retries and ("Error" in str(result.get("error")) or "Empty response" in str(result.get("error")) or "Blocked" in str(result.get("error")))
        if not should_retry: logger.error(f"LLM call failed permanently for {provider} after {attempt + 1} attempts. Last error: {result.get('error')}")
        if not result.get("error"): result["error"] = "Failed after retries."
        if should_retry:
            wait = delay * (2**attempt) + np.random.uniform(0, delay*0.5); logger.info(f"Retrying in {wait:.2f}s... (Error: {result.get('error')})"); time.sleep(wait)
        else:
             return result
    return result

# --- Prompt Building Functions ---
# (construct_keyword_expansion_prompt, construct_llm_slotting_prompt, build_reprompt_prompt unchanged)
# (construct_keyword_expansion_prompt, construct_llm_slotting_prompt, build_reprompt_prompt unchanged)
def construct_keyword_expansion_prompt(input_concept: str) -> Tuple[str, str]:
    """Creates prompts for Stage 1 LLM keyword expansion (v33.0.16 - Enhanced)."""
    system_prompt = """You are a helpful assistant specializing in travel concepts and keyword analysis for search retrieval. You understand nuances like compound words and relationships between concepts. Your goal is to generate relevant keywords that will improve search results within a travel taxonomy. Output ONLY a valid JSON object containing a single key "keywords" with a list of strings as its value. Do not include any explanations or introductory text outside the JSON structure.
    """
    user_prompt = f"""
    Given the input travel concept: '{input_concept}'

    Your task is to generate a list of related keywords specifically useful for *improving keyword search retrieval* within a large travel taxonomy. Consider the following:
    1.  **Synonyms:** Include direct synonyms of the input concept.
    2.  **Constituent Parts:** If the input concept is a compound word (e.g., 'americanbreakfast'), include its meaningful parts (e.g., 'american', 'breakfast').
    3.  **Related Activities/Concepts:** List highly relevant activities, items, or concepts directly associated with the input concept (e.g., for 'americanbreakfast', include 'eggs', 'bacon', 'pancakes').
    4.  **Hypernyms/Hyponyms (Optional but helpful):** If applicable, include broader categories (hypernyms) or more specific examples (hyponyms).
    5.  **Relevance:** Focus strictly on terms highly relevant to the *core meaning* of '{input_concept}' in a travel context. Avoid overly broad or tangentially related terms.
    6.  **Simplicity:** Prefer single words or very common short phrases.

    Example Input: 'budget hotel'
    Example Output:
    ```json
    {{
      "keywords": ["budget", "hotel", "cheap", "affordable", "economy", "value", "low cost", "inn", "motel"]
    }}
    ```

    Example Input: 'scuba diving'
    Example Output:
    ```json
    {{
      "keywords": ["scuba", "diving", "scuba diving", "underwater", "dive", "reef", "snorkel", "padi", "water sports"]
    }}
    ```

    Now, generate the keywords for the input concept: '{input_concept}'

    Output ONLY the JSON object:
    """
    return system_prompt.strip(), user_prompt.strip()

def construct_llm_slotting_prompt(input_concept: str, theme_definitions: List[Dict[str, Any]], candidate_details: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[str, str]:
    """Constructs the main LLM prompt for theme slotting (v33.0.12 version)."""
    system_prompt = """You are an expert travel taxonomist creating structured definitions for travel concepts. You will be given a core concept, a list of semantic themes with descriptions, and a list of candidate concepts from a knowledge graph. Your task is to analyze EACH candidate concept and determine which of the provided themes it is semantically relevant to. Focus on the meaning conveyed by the candidate's labels, definitions, and types. Output ONLY a valid JSON object containing a single key "theme_assignments". The value of this key should be a dictionary where each key is a candidate URI you were given, and the value is a list of theme names (strings) that the candidate is relevant to. If a candidate is not relevant to any theme, its value should be an empty list []. Ensure EVERY candidate URI provided in the input is present as a key in your output JSON. Pay close attention also to candidates whose labels or definitions directly match or are highly synonymous with the input concept '{input_concept}' itself; these should be carefully slotted into appropriate themes. Focus solely on semantic relevance based on the provided information."""
    theme_defs_str = "\n".join([f"- **{t.get('name', 'N/A')}{' (Mandatory)' if t.get('is_must_have') else ''}**: {t.get('description','N/A')}" for t in theme_definitions])
    must_haves = ", ".join([t['name'] for t in theme_definitions if t.get('is_must_have')]) or "None"
    cand_details_str = "".join(f"\n{i+1}. URI: {c.get('uri', 'MISSING')}\n   PrefLabel: \"{c.get('prefLabel', 'N/A')}\"\n   AltLabels: {', '.join(c.get('skos:altLabel', []))}\n   Definition: \"{(c.get('skos:definition', '') or '')[:200].replace(chr(10), ' ')}\"\n   Types: {', '.join(c.get('type_labels', []))}" for i, c in enumerate(candidate_details)) or "\nNo candidates."
    schema_example = """```json
{
  "theme_assignments": {
    "URI_1": ["ThemeA", "ThemeB"],
    "URI_2": [],
    "URI_3": ["ThemeC"]
    // ... include ALL candidate URIs provided ...
  }
}
```"""
    task = f"""Task:
For EACH candidate URI listed below, determine which of the defined themes it semantically belongs to, based on its labels, definition, and types.
Pay close attention also to candidates whose labels or definitions directly match or are highly synonymous with the input concept '{input_concept}' itself; ensure these are appropriately slotted.
Ensure EVERY candidate URI provided in the input is present as a key in the output JSON, even if the theme list is empty. Focus solely on semantic relevance."""
    user_prompt = f"""Analyze the input concept: '{input_concept}'

Available Semantic Themes:
{theme_defs_str}
Mandatory Themes (must have at least one candidate assigned): [{must_haves}]

Candidate Concepts from Knowledge Graph:
{cand_details_str}

{task}

Output Format Requirements:
Output ONLY the JSON object adhering strictly to the schema shown below. Do not include any text before or after the JSON object.
{schema_example}
"""
    return system_prompt.strip(), user_prompt.strip()

def build_reprompt_prompt(input_concept: str, theme_name: str, theme_config: Dict, original_candidates_details_map: Dict[str, Dict]) -> Tuple[str, str]:
    """Constructs the fallback prompt for a single mandatory theme."""
    system_prompt = """You are assisting in refining travel concept definitions by assigning concepts to themes. A mandatory theme failed to get any assignments in the previous step. Review the original candidate concepts ONLY for relevance to the specific mandatory theme provided. Output ONLY a valid JSON object with the key "theme_assignments", mapping relevant candidate URIs to a list containing ONLY the target theme name. If NO candidates are plausibly relevant, output `{"theme_assignments": {}}`."""
    desc = theme_config.get('description', 'N/A')
    hints = theme_config.get("hints", {})
    hints_str = ""
    if hints.get("keywords"): hints_str += f"\n    - Keywords: {', '.join(hints['keywords'])}"
    if hints.get("uris"): hints_str += f"\n    - URIs: {', '.join(hints['uris'])}"
    if hints_str: hints_str = "  Hints:" + hints_str
    cand_list = "".join(f"\n{i+1}. URI: {uri}\n   Label: \"{cand.get('prefLabel', 'N/A')}\"\n   Types: {', '.join(cand.get('type_labels', []))}" for i, (uri, cand) in enumerate(original_candidates_details_map.items())) or "\nNo candidates."
    schema = f"```json\n{{\n  \"theme_assignments\": {{\n    \"URI_Relevant_1\": [\"{theme_name}\"],\n    \"URI_Relevant_2\": [\"{theme_name}\"]\n    // ... include ONLY relevant URIs\n  }}\n}}\n```\nIf none relevant: {{\"theme_assignments\": {{}} }}"
    user_prompt = f"""Re-evaluating '{input_concept}' for the MANDATORY theme '{theme_name}'.

Theme Definition:
- Name: {theme_name}
- Description: {desc}
{hints_str}

Original Candidate Concepts:
{cand_list}

Instructions:
Review the list of original candidates. Identify ONLY the candidates that are semantically relevant to the theme '{theme_name}' based on the theme's description and hints. Assign at least one candidate if plausible, even if the relevance is weak. Output ONLY a valid JSON object containing the URIs of the relevant candidates mapped to a list containing only the theme name '{theme_name}'. If no candidates are relevant, output an empty mapping.

Example Output Schema:
{schema}

Your Output:"""
    return system_prompt.strip(), user_prompt.strip()

# --- Keyword Expansion Helper ---
# (expand_keywords_with_llm unchanged from v33.0.28a)
def expand_keywords_with_llm(concept_label: str, config: Dict, args: argparse.Namespace) -> Tuple[List[str], bool, Optional[str]]:
    """
    Uses LLM to generate keywords. Reduced logging.
    Returns tuple: (unique_keyword_list, success_flag, error_message_or_none).
    Original normalized words from input concept are always included in the list.
    """
    llm_api_cfg = config.get("LLM_API_CONFIG", {})
    llm_stage1_cfg = config.get("STAGE1_CONFIG", {})
    temp = float(llm_stage1_cfg.get("KW_EXPANSION_TEMPERATURE", config.get("STAGE2_CONFIG", {}).get("LLM_TEMPERATURE", KW_EXPANSION_TEMPERATURE)))
    timeout = int(llm_api_cfg.get("REQUEST_TIMEOUT", LLM_TIMEOUT))
    retries = int(llm_api_cfg.get("MAX_RETRIES", LLM_MAX_RETRIES))

    # Initialize return values
    success = False
    error_message = None
    final_keyword_terms = set()

    # Always include original terms
    original_normalized_words = set(w for w in normalize_concept(concept_label).split() if len(w) > 2)
    final_keyword_terms.update(original_normalized_words)

    try:
        sys_prompt, user_prompt = construct_keyword_expansion_prompt(concept_label)
        result = call_llm(sys_prompt, user_prompt, args.llm_model, timeout, retries, temp, args.llm_provider)

        if result and result.get("success"):
            raw_phrases = result.get("response", {}).get("keywords", [])
            if isinstance(raw_phrases, list):
                added_terms = set()
                for kw_phrase in raw_phrases:
                    if isinstance(kw_phrase, str) and kw_phrase.strip():
                        normalized_phrase = normalize_concept(kw_phrase)
                        added_terms.update(term for term in normalized_phrase.split() if len(term) > 2)

                # Check if any *new* terms were added beyond the originals
                newly_expanded_kws = added_terms - original_normalized_words
                if newly_expanded_kws:
                    final_keyword_terms.update(newly_expanded_kws)
                    success = True # Mark as successful only if new terms were added
                    logger.debug(f"[{concept_label}] LLM generated {len(final_keyword_terms)} unique keyword terms ({len(newly_expanded_kws)} new) from {len(raw_phrases)} phrases.")
                else:
                    success = False # Not successful if only original terms or nothing useful came back
                    error_message = "LLM returned no new useful terms"
                    logger.info(f"[{concept_label}] LLM expansion returned no new useful keyword terms.")

            else:
                logger.error(f"[{concept_label}] LLM expansion response 'keywords' field was not a list.")
                success = False
                error_message = "LLM response format invalid (keywords not a list)"
        else:
            err_msg = result.get('error', 'Unknown Error') if result else 'No result object'
            logger.warning(f"[{concept_label}] LLM keyword expansion call failed: {err_msg}")
            success = False
            error_message = f"LLM API Call Failed: {err_msg}"

    except Exception as e:
        logger.error(f"[{concept_label}] Exception during keyword expansion: {e}", exc_info=True)
        success = False
        error_message = f"Exception: {e}"

    return list(final_keyword_terms), success, error_message

# --- Stage 1: Evidence Preparation ---
# (prepare_evidence updated for v33.0.28 to read overrides)
# --- v33.0.28a - Config-Driven Overrides, Fixed Expansion Diag ---
# ... (keep prepare_evidence function from v33.0.28a, ensuring it uses the new keyword search function correctly) ...
# *** IMPORTANT: Ensure the call inside prepare_evidence now uses get_candidate_concepts_keyword(..., bm25_model=_bm25_model, corpus_uris=_keyword_corpus_uris, ...) ***
# This change needs to be made within the prepare_evidence function where kw_cands is calculated.
# --- START: Stage 1: Evidence Preparation (v33.0.29 - BM25) ---
def prepare_evidence(
        input_concept: str, concept_embedding: Optional[np.ndarray], primary_embeddings: Dict[str, np.ndarray],
        config: Dict, args: argparse.Namespace,
        # --- Replace TF-IDF with BM25 ---
        bm25_model: Optional[BM25Okapi],
        keyword_corpus_uris: Optional[List[str]],
        # --- End Replace ---
        keyword_label_index: Optional[Dict[str, Set[str]]], taxonomy_concepts_cache: Dict[str, Dict]
) -> Tuple[List[Dict], Dict[str, Dict], Optional[Dict], Dict[str, float], Dict[str, Any], int, int, int]:
    """
    Prepares evidence candidates using combined score (BM25 + SBERT) and config-driven overrides.
    v33.0.29: Uses BM25 for keyword scoring instead of TF-IDF. Reads concept-specific
              overrides from config. Retains tie-breaking sort, dampening, abs SBERT filter.
    Returns candidate details for LLM, original candidates map, anchor candidate,
            final ORIGINAL SBERT scores, expansion diagnostics, initial SBERT count,
            initial Keyword (BM25) count, and unique candidates count before ranking.
    """
    normalized_concept = normalize_concept(input_concept)

    # --- Helper for Tie-Breaking (Unchanged) ---
    def get_sort_priority(item_uri: str) -> int:
        if not isinstance(item_uri, str): return 99
        if "urn:expediagroup:taxonomies:acsPCS:" in item_uri: return 0
        if "urn:expediagroup:taxonomies:acs:" in item_uri:
            # Example of URI-specific demotion (if needed, managed via config would be better)
            # if item_uri == "urn:expediagroup:taxonomies:acs:#1f3da634-0df6-4498-a8d9-603f895c8f3f": return 11
            return 1
        if "urn:expediagroup:taxonomies:core:" in item_uri: return 2
        if "urn:expediagroup:taxonomies:" in item_uri: return 3
        # Be more specific with expe:taxo if possible
        if "urn:expe:taxo:cars:search:filters:" in item_uri: return 4
        if "urn:expe:taxo:media-descriptors:" in item_uri: return 5
        if "urn:expe:taxo:trip-preferences:" in item_uri: return 6
        if "urn:expe:taxo:amenity-view-property-features:" in item_uri: return 7
        if "urn:expe:taxo:" in item_uri: return 8 # General expe:taxo lower
        if "urn:expe:taxo:places:" in item_uri: return 9
        if "urn:expe:taxo:events:" in item_uri: return 10
        return 15 # Default lower priority
    # --- End Helper ---

    # --- Get Base Parameters from Config or use constants ---
    stage1_cfg = config.get('STAGE1_CONFIG', {})
    dampening_sbert_threshold = float(stage1_cfg.get("DAMPENING_SBERT_THRESHOLD", DAMPENING_SBERT_THRESHOLD))
    dampening_factor = float(stage1_cfg.get("DAMPENING_FACTOR", DAMPENING_FACTOR))
    max_cands = int(stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', MAX_CANDIDATES_FOR_LLM))
    min_sim = float(stage1_cfg.get('EVIDENCE_MIN_SIMILARITY', EVIDENCE_MIN_SIMILARITY))
    # Use the specific min_score intended for BM25 (might need tuning)
    min_kw = float(stage1_cfg.get('KEYWORD_MIN_SCORE', KEYWORD_MIN_SCORE)) # Default is now 0.01
    kw_trigger = int(stage1_cfg.get('MIN_KEYWORD_CANDIDATES_FOR_EXPANSION', MIN_KEYWORD_CANDIDATES_FOR_EXPANSION))
    kw_top_n = int(stage1_cfg.get('KEYWORD_TOP_N', KEYWORD_TOP_N_SEARCH))
    abs_min_sbert = float(stage1_cfg.get("ABSOLUTE_MIN_SBERT_SCORE", ABSOLUTE_MIN_SBERT_SCORE))
    global_alpha = float(config.get("COMBINED_SCORE_ALPHA", COMBINED_SCORE_ALPHA)) # Default global alpha
    global_kw_exp_enabled = config.get("ENABLE_KW_EXPANSION", ENABLE_KW_EXPANSION) # Global expansion flag
    # --- END: Get Base Parameters ---

    # --- Get Concept-Specific Overrides from Config ---
    concept_overrides_config = config.get("concept_overrides", {})
    overrides = concept_overrides_config.get(normalized_concept, {}) # Get overrides for this concept, or empty dict if none

    # Determine effective settings using overrides or defaults
    skip_expansion_override = overrides.get("skip_expansion", False)
    kw_exp_enabled = global_kw_exp_enabled and args.llm_provider != "none" and not skip_expansion_override
    seed_uris_for_concept = overrides.get("seed_uris", [])
    boost_config = overrides.get("boost_seeds_config", {"enabled": False})
    uris_to_filter_for_concept = overrides.get("filter_uris", [])
    effective_alpha = float(overrides.get("effective_alpha", global_alpha)) # Use override or global default
    manual_query_terms = overrides.get("manual_query_split", None)
    # --- END: Get Concept-Specific Overrides ---

    # Log if overrides are active
    if overrides:
         logger.info(f"[{normalized_concept}] Applying concept-specific overrides from configuration.")
         if args.debug: logger.debug(f"Overrides for '{normalized_concept}': {overrides}")

    # --- Parameter Logging Block (Reflects Effective Settings) ---
    logger.info(f"[{normalized_concept}] Using Effective Combined Score Alpha: {effective_alpha:.2f} {'(Overridden)' if abs(effective_alpha - global_alpha) > 1e-9 else '(Default)'}")
    logger.info(f"[{normalized_concept}] Using Absolute Min SBERT Threshold: {abs_min_sbert}")
    logger.info(f"[{normalized_concept}] Using Conditional Keyword Score Dampening: Threshold={dampening_sbert_threshold}, Factor={dampening_factor}") # Updated log
    if boost_config.get("enabled"): logger.info(f"[{normalized_concept}] Seed Boost enabled via config (Threshold: {boost_config.get('threshold', 'N/A')}) for URIs: {seed_uris_for_concept}")
    logger.info(f"[{normalized_concept}] Effective LLM Keyword Expansion Enabled: {kw_exp_enabled}")
    logger.info(f"[{normalized_concept}] Applying Tie-Breaking Sort (Combined Score > Namespace Priority > SBERT Score)")
    # --- End Parameter Logging Block ---

    sim_scores, kw_scores = {}, {}
    # Initialize diagnostics dictionary
    exp_diag = {"attempted": False, "successful": False, "count": 0, "terms": [], "keyword_count": 0, "error": None} # Changed tfidf_count key

    # --- Keyword Preparation ---
    base_split_kws = set(kw for kw in normalized_concept.split() if len(kw) > 2)
    query_texts_set = set(base_split_kws)
    final_query_texts = list(query_texts_set) # Initialize final list (may be expanded)

    initial_kw_count = 0
    if keyword_label_index:
        initial_kw_count = len(set().union(*[keyword_label_index.get(kw, set()) for kw in base_split_kws]))
    else:
        logger.warning(f"[{normalized_concept}] Simple keyword index unavailable.")

    needs_exp = kw_exp_enabled and \
                (initial_kw_count < kw_trigger or normalized_concept in ABSTRACT_CONCEPTS_LIST)

    expansion_info = "Not attempted or disabled."
    if kw_exp_enabled:
        if needs_exp:
            exp_diag["attempted"] = True
            logger.info(f"[{normalized_concept}] Attempting LLM keyword expansion (Trigger: initial matches {initial_kw_count} < {kw_trigger} or abstract concept).")
            expanded_terms_list, exp_success, exp_error = expand_keywords_with_llm(input_concept, config, args)
            exp_diag["successful"] = exp_success
            exp_diag["error"] = exp_error
            final_query_texts = expanded_terms_list # Use the potentially expanded list
            if exp_success: expansion_info = f"Successful, final query terms count: {len(final_query_texts)}"
            else: expansion_info = f"Attempted, but failed or no new terms. Error: {exp_error}"
        else:
             expansion_info = f"Skipped (Initial matches {initial_kw_count} >= {kw_trigger} and not abstract)."
             logger.info(f"[{normalized_concept}] Skipping LLM KW expansion ({expansion_info}).")
    else:
         if skip_expansion_override:
             expansion_info = "Disabled via config override"
             exp_diag["attempted"] = True
             exp_diag["successful"] = False
             exp_diag["error"] = expansion_info
         else: expansion_info = "Disabled globally or provider=none."
         logger.info(f"[{normalized_concept}] LLM KW expansion disabled for this concept ({expansion_info}).")

    # --- Manual Query Split (from Config) ---
    if manual_query_terms is not None:
        if isinstance(manual_query_terms, list) and all(isinstance(t, str) for t in manual_query_terms):
            if set(final_query_texts) != set(manual_query_terms):
                logger.info(f"Applying manual query split from config, OVERRIDING previous/expanded terms: {manual_query_terms}")
                final_query_texts = manual_query_terms
                exp_diag["notes"] = "Manual query split applied."
        else:
             logger.warning(f"Invalid 'manual_query_split' format in config for {normalized_concept}. Expected list of strings. Ignoring.")
             if "notes" not in exp_diag: exp_diag["notes"] = ""
             exp_diag["notes"] += " Invalid manual_query_split in config."

    exp_diag["terms"] = final_query_texts
    exp_diag["count"] = len(final_query_texts)

    # Log Keyword Query Terms (Used for BM25)
    if args.debug:
         logger.debug(f"[BM25 QUERY LOG] Concept: '{normalized_concept}', Final Query Terms ({len(final_query_texts)}): {final_query_texts}")

    # --- Keyword Search (Using BM25) ---
    keyword_candidate_count_initial = 0
    if bm25_model and keyword_corpus_uris: # Check for BM25 model now
        if final_query_texts:
            # Call the BM25 keyword search function
            kw_cands = get_candidate_concepts_keyword(
                final_query_texts,
                bm25_model,
                keyword_corpus_uris,
                kw_top_n,
                min_kw
            )
            kw_scores = {c['uri']: c['score'] for c in kw_cands}
            keyword_candidate_count_initial = len(kw_scores)
            exp_diag["keyword_count"] = keyword_candidate_count_initial # Updated key
            logger.debug(f"[{normalized_concept}] Found {keyword_candidate_count_initial} BM25 candidates >= {min_kw}.")
        else:
            logger.warning(f"[{normalized_concept}] No query keywords derived for BM25 search.")
            kw_scores = {}
            exp_diag["keyword_count"] = 0
    else:
        logger.warning(f"[{normalized_concept}] BM25 components missing/unavailable. Skipping keyword search.")
        kw_scores = {}
        exp_diag["keyword_count"] = 0
    # --- END Keyword Search ---

    # --- SBERT Similarity Calculation ---
    sbert_candidate_count_initial = 0
    if concept_embedding is None:
        logger.error(f"[{normalized_concept}] No anchor embedding. Skipping SBERT similarity.")
        sim_scores = {}
    else:
        sim_scores = get_batch_embedding_similarity(concept_embedding, primary_embeddings)
        sbert_initial_candidates = {uri: s for uri, s in sim_scores.items() if s >= min_sim}
        sbert_candidate_count_initial = len(sbert_initial_candidates)
        logger.debug(f"[{normalized_concept}] Found {sbert_candidate_count_initial} SBERT candidates >= {min_sim}.")

    # --- Candidate Combination, Seeding, Filtering, Dampening, and Ranking ---
    all_uris_set = set(kw_scores.keys()) | set(sim_scores.keys())
    initial_unique_count = len(all_uris_set)

    # --- Apply Seeding (from Config) ---
    if seed_uris_for_concept:
        added_seeds = set(seed_uris_for_concept) - all_uris_set
        if added_seeds:
            logger.info(f"SEEDING '{normalized_concept}' with {len(added_seeds)} missing relevant URIs from config: {added_seeds}")
            all_uris_set.update(added_seeds)
            uris_needing_sbert_calc = added_seeds - set(sim_scores.keys())
            if uris_needing_sbert_calc and concept_embedding is not None:
                 seed_embeddings = {uri: primary_embeddings[uri] for uri in uris_needing_sbert_calc if uri in primary_embeddings}
                 if seed_embeddings:
                     new_sim_scores = get_batch_embedding_similarity(concept_embedding, seed_embeddings)
                     sim_scores.update(new_sim_scores)
                     logger.debug(f"Calculated SBERT for {len(new_sim_scores)} newly added seeds.")
                 else:
                     logger.warning(f"Could not find embeddings for newly added seeds: {uris_needing_sbert_calc}")

    # --- Apply Concept-Specific Filters (from Config) ---
    if uris_to_filter_for_concept:
        uris_to_remove = set()
        for uri_to_filter in uris_to_filter_for_concept:
            if uri_to_filter in all_uris_set:
                label = get_primary_label(uri_to_filter, taxonomy_concepts_cache, fallback=uri_to_filter)
                logger.info(f"Applying specific filter for '{normalized_concept}' from config: Removing {uri_to_filter} ('{label}').")
                uris_to_remove.add(uri_to_filter)
        if uris_to_remove:
             all_uris_set.difference_update(uris_to_remove)
             logger.info(f"Removed {len(uris_to_remove)} URIs via concept-specific filter for '{normalized_concept}'. New count: {len(all_uris_set)}")
    # --- END: Apply Concept-Specific Filters ---

    all_uris_list = list(all_uris_set)
    unique_candidates_before_ranking = len(all_uris_list)
    # logger.debug(f"Unique candidates before ranking: {unique_candidates_before_ranking}") # Reduced noise

    if not all_uris_list:
        logger.warning(f"[{normalized_concept}] No candidates found from Keyword/SBERT/Seeds after filtering.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, keyword_candidate_count_initial, 0

    all_details = get_kg_data(all_uris_list, taxonomy_concepts_cache)

    scored_list: List[Dict] = []
    abs_sbert_filtered_count = 0
    dampened_count = 0
    alpha_override_applied_this_concept = abs(effective_alpha - global_alpha) > 1e-9

    for i, uri in enumerate(all_uris_list):
        if uri not in all_details:
            logger.warning(f"[{normalized_concept}] Details not found for candidate URI: {uri}. Skipping.")
            continue

        s_score = sim_scores.get(uri, 0.0)
        k_score = kw_scores.get(uri, 0.0) # Now from BM25
        pref_label_for_log = all_details[uri].get('prefLabel', 'N/A')

        # --- Seed Score Boost (from Config) ---
        is_seed_uri = uri in seed_uris_for_concept
        original_s_score_before_boost = s_score
        boosted_s_score_for_calc = s_score
        boost_applied = False
        if is_seed_uri and boost_config.get("enabled", False):
            boost_threshold = float(boost_config.get("threshold", 0.80))
            if s_score < boost_threshold:
                boosted_s_score_for_calc = boost_threshold
                boost_applied = True
                logger.info(f"[Seed BOOST from Config] Boosting SBERT for {uri} ('{pref_label_for_log}') from {original_s_score_before_boost:.4f} to {boosted_s_score_for_calc:.4f}")

        # ** Absolute SBERT Filter **
        if boosted_s_score_for_calc < abs_min_sbert:
            # if args.debug: logger.debug(...) # Keep debug if needed
            abs_sbert_filtered_count += 1
            continue

        # *** Conditional Keyword Score Dampening *** (Applied to BM25 score)
        dampened_k_score = k_score
        was_dampened = False
        if boosted_s_score_for_calc < dampening_sbert_threshold:
            dampened_k_score *= dampening_factor
            if k_score > 0 and abs(k_score - dampened_k_score) > 1e-9: # Check if dampening actually happened
                 was_dampened = True
                 if args.debug: logger.debug(f"Dampening Keyword Score for {uri} ('{pref_label_for_log}'): SBERT={boosted_s_score_for_calc:.4f} < {dampening_sbert_threshold:.4f}. Keyword {k_score:.4f} -> {dampened_k_score:.4f} (Factor={dampening_factor}).")
                 dampened_count += 1

        # *** Effective Alpha (already determined from config or global) ***

        # Normalize scores
        # NOTE: BM25 scores are NOT inherently 0-1. We still clip them here, but this might lose info.
        # A better approach might be MinMax scaling *across the batch* if performance suffers,
        # but simple clipping is often sufficient for combining with SBERT.
        norm_s = max(0.0, min(1.0, boosted_s_score_for_calc))
        norm_k = max(0.0, min(1.0, dampened_k_score)) # Clip BM25 score after dampening

        # Calculate combined score using EFFECTIVE alpha
        combined = (effective_alpha * norm_k) + ((1.0 - effective_alpha) * norm_s)

        scored_list.append({
            "uri": uri,
            "details": all_details[uri],
            "sim_score": original_s_score_before_boost,
            "boosted_sim_score": boosted_s_score_for_calc if boost_applied else None,
            "kw_score": k_score, # Raw BM25 score
            "dampened_kw_score": dampened_k_score if was_dampened else None,
            "alpha_overridden": alpha_override_applied_this_concept,
            "effective_alpha": effective_alpha,
            "combined_score": combined
        })

    logger.info(f"[{normalized_concept}] Excluded {abs_sbert_filtered_count} candidates (SBERT < {abs_min_sbert} after potential boost).")
    logger.info(f"[{normalized_concept}] Conditionally dampened Keyword Score (Factor={dampening_factor}) for {dampened_count} candidates (SBERT < {dampening_sbert_threshold}).") # Updated log
    if alpha_override_applied_this_concept:
         logger.info(f"[{normalized_concept}] Applied concept-specific alpha override from config (Value: {effective_alpha:.2f}).")

    if not scored_list:
        logger.warning(f"[{normalized_concept}] No candidates remaining after filtering/dampening.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, keyword_candidate_count_initial, 0

    # --- Sorting with Tie-breaking (Unchanged) ---
    logger.debug(f"[{normalized_concept}] Sorting {len(scored_list)} candidates with tie-breaking (Priority: Combined Score > Namespace > SBERT)...")
    scored_list.sort(key=lambda x: x.get('sim_score', 0.0), reverse=True)
    scored_list.sort(key=lambda x: get_sort_priority(x.get('uri', '')), reverse=False)
    scored_list.sort(key=lambda x: x['combined_score'], reverse=True)
    # --- End Sorting ---


    # Debug logging for Top 5 (Update kw_score label)
    if args.debug:
        logger.debug(f"--- Top 5 Candidates for '{normalized_concept}' AFTER Sorting (Tie-Breaking Enabled: Score>Prio>SBERT) ---")
        logger.debug(f"    (DefaultAlpha={global_alpha}, EffectiveAlpha={effective_alpha:.2f}, DampThresh={dampening_sbert_threshold}, DampFactor={dampening_factor})")
        for i, c in enumerate(scored_list[:5]):
            damp_info = f"(Damp KW: {c.get('dampened_kw_score'):.4f})" if c.get('dampened_kw_score') is not None else "" # Updated label
            boost_info = f"(Boost SBERT: {c.get('boosted_sim_score'):.4f})" if c.get('boosted_sim_score') is not None else ""
            alpha_info = f"(EffAlpha: {c.get('effective_alpha', global_alpha):.2f})"
            prio_info = f"(Prio: {get_sort_priority(c.get('uri',''))})"
            label = c.get('details', {}).get('prefLabel', 'N/A')
            logger.debug(f"{i+1}. URI: {c.get('uri', 'MISSING')} {prio_info} Label: '{label}' "
                         f"Combined: {c.get('combined_score', 0.0):.6f} {alpha_info} "
                         f"(Orig SBERT: {c.get('sim_score', 0.0):.4f} {boost_info}, Raw KW (BM25): {c.get('kw_score', 0.0):.4f} {damp_info})") # Updated label
        logger.debug("--- End Top 5 ---")

    selected = scored_list[:max_cands]
    logger.info(f"[{normalized_concept}] Selected top {len(selected)}/{len(scored_list)} candidates based on combined score and tie-breaking.")

    if not selected:
        logger.warning(f"[{normalized_concept}] No candidates selected after scoring and filtering/dampening.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, keyword_candidate_count_initial, 0

    # Prepare outputs
    llm_details = [c["details"] for c in selected]
    orig_map = {}
    for c in selected:
         uri = c.get('uri')
         details = c.get('details')
         if uri and details:
              orig_map_entry = {
                  **details,
                  "sbert_score": c.get("sim_score", 0.0),
                  "keyword_score": c.get("kw_score", 0.0), # Now BM25 score
                  "combined_score": c.get("combined_score", 0.0),
                  "effective_alpha": c.get("effective_alpha", global_alpha)
              }
              if c.get("boosted_sim_score") is not None:
                   orig_map_entry["boosted_sbert_score"] = c.get("boosted_sim_score")
              if c.get("dampened_kw_score") is not None:
                   orig_map_entry["dampened_keyword_score"] = c.get("dampened_kw_score")
              orig_map[uri] = orig_map_entry
         else:
             logger.warning(f"[{normalized_concept}] Skipping candidate in orig_map due to missing URI or details: {c}")


    # Anchor Selection Logic (remains the same - selects first item after sort)
    anchor_data = selected[0]
    anchor_uri_from_data = anchor_data.get('uri')
    anchor = None
    if anchor_uri_from_data:
        anchor = orig_map.get(anchor_uri_from_data)
        if anchor is None:
             logger.error(f"[{normalized_concept}] Top candidate URI '{anchor_uri_from_data}' not found in orig_map! This should not happen.")
             # Fallback construction... (remains same)
             if anchor_data.get('details'):
                 anchor = {
                     **anchor_data.get('details'),
                     "sbert_score": anchor_data.get("sim_score", 0.0),
                     "keyword_score": anchor_data.get("kw_score", 0.0),
                     "combined_score": anchor_data.get("combined_score", 0.0),
                     "effective_alpha": anchor_data.get("effective_alpha", global_alpha)
                 }
                 if anchor_data.get("boosted_sim_score") is not None: anchor["boosted_sbert_score"] = anchor_data.get("boosted_sim_score")
                 if anchor_data.get("dampened_kw_score") is not None: anchor["dampened_keyword_score"] = anchor_data.get("dampened_kw_score")
                 logger.warning(f"[{normalized_concept}] Using fallback anchor constructed from scored_list data.")
             else:
                 logger.error(f"[{normalized_concept}] Fallback anchor construction failed - missing details in scored_list top item.")
    else:
        logger.error(f"[{normalized_concept}] Top candidate in 'selected' list has no URI! Data: {anchor_data}")

    # Debug Sanity Check (remains the same)
    # ... (keep this block) ...

    # Anchor Logging (Update kw_score label)
    if anchor:
        damp_info_log = f"(Damp KW: {anchor_data.get('dampened_kw_score'):.4f})" if anchor_data.get('dampened_kw_score') is not None else "" # Updated label
        boost_info_log = f"(Boost SBERT: {anchor_data.get('boosted_sim_score'):.4f})" if anchor_data.get('boosted_sim_score') is not None else ""
        alpha_log = f"(EffAlpha: {anchor_data.get('effective_alpha', global_alpha):.2f})"
        logger.info(f"[{normalized_concept}] Anchor selected: {anchor.get('prefLabel', anchor.get('uri', 'MISSING URI'))} "
                    f"(Combined: {anchor_data.get('combined_score', 0.0):.6f} {alpha_log}, "
                    f"Orig SBERT: {anchor_data.get('sim_score', 0.0):.4f} {boost_info_log}, "
                    f"Raw KW (BM25): {anchor_data.get('kw_score', 0.0):.4f} {damp_info_log})") # Updated label
    else:
        logger.warning(f"[{normalized_concept}] Could not determine anchor details. Anchor will be null.")
        if anchor_uri_from_data:
             logger.warning(f"    Reason: URI '{anchor_uri_from_data}' (from top candidate) was not found in orig_map and fallback failed.")
        else:
             logger.warning(f"    Reason: Top candidate dictionary lacked a URI.")

    # Return values (Update key name in tuple description)
    sbert_scores_final = {uri: d["sbert_score"] for uri, d in orig_map.items()}
    return llm_details, orig_map, anchor, sbert_scores_final, exp_diag, sbert_candidate_count_initial, keyword_candidate_count_initial, unique_candidates_before_ranking
# --- END: Stage 1: Evidence Preparation (v33.0.29 - BM25) ---


# --- Stage 2: Finalization ---
# (apply_rules_and_finalize updated for v33.0.28 to read overrides from config)
# ... (keep apply_rules_and_finalize function from v33.0.28) ...
def apply_rules_and_finalize(
    input_concept: str, llm_call_result: Optional[Dict[str, Any]], config: Dict,
    travel_category: Optional[Dict], anchor_candidate: Optional[Dict],
    original_candidates_map_for_reprompt: Dict[str, Dict],
    candidate_evidence_scores: Dict[str, float], # Note: sbert_scores_final from prepare_evidence passed here now
    args: argparse.Namespace, taxonomy_concepts_cache: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Applies rules, handles fallbacks, weights attributes, structures final output.
    Reduced logging. Uses combined_score from original_candidates_map_for_reprompt.
    v33.0.28: Reads overrides (lodging_type, category_type, must_not_have, etc.) from config['concept_overrides'].
    """
    start = time.time(); norm_concept = normalize_concept(input_concept)
    # Read concept-specific overrides from config
    concept_overrides = config.get("concept_overrides", {}).get(norm_concept, {})

    output: Dict[str, Any] = {
        "applicable_lodging_types": concept_overrides.get("applicable_lodging_types", "Both"), # Apply override or default
        "travel_category": travel_category or {"uri": None, "name": input_concept, "type": "Unknown"},
        "top_defining_attributes": [],
        "themes": [],
        "additional_relevant_subscores": concept_overrides.get("additional_relevant_subscores", []), # Apply override or default (empty list)
        "must_not_have": [], # Will be populated below based on overrides
        "failed_fallback_themes": {},
        "diagnostics": {"theme_processing": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "final_output": {}, "stage2": {"status": "Started", "duration_seconds": 0.0, "error": None}}
    }
    theme_diag = output["diagnostics"]["theme_processing"]; reprompt_diag = output["diagnostics"]["reprompting_fallback"]
    base_themes = config.get("base_themes", {});
    final_cfg = config.get('STAGE2_CONFIG', {}); min_weight = float(final_cfg.get('THEME_ATTRIBUTE_MIN_WEIGHT', THEME_ATTRIBUTE_MIN_WEIGHT)); top_n_attrs = int(final_cfg.get('TOP_N_DEFINING_ATTRIBUTES', TOP_N_DEFINING_ATTRIBUTES_DEFAULT))

    llm_assigns: Dict[str, List[str]] = {}; diag_val = {}
    if llm_call_result and llm_call_result.get("success"):
        validated = validate_llm_assignments(llm_call_result.get("response"), set(original_candidates_map_for_reprompt.keys()), set(base_themes.keys()), norm_concept, diag_val)
        if validated is not None: llm_assigns = validated; # logger.debug(f"[{norm_concept}] Validated LLM assignments.") # Reduced noise
        else: logger.warning(f"[{norm_concept}] LLM validation failed."); output["diagnostics"]["stage2"]["error"] = diag_val.get("error", "LLM Validation Failed")
    elif llm_call_result: logger.warning(f"[{norm_concept}] LLM slotting call unsuccessful: {llm_call_result.get('error')}"); output["diagnostics"]["stage2"]["error"] = f"LLM Call Failed: {llm_call_result.get('error')}"

    theme_map = defaultdict(list)
    for uri, themes in llm_assigns.items():
        if uri in original_candidates_map_for_reprompt:
            for t in themes:
                if t in base_themes: theme_map[t].append(uri)

    failed_rules: Dict[str, Dict] = {}
    for name in base_themes.keys():
        diag = theme_diag[name] = {"llm_assigned_count": len(theme_map.get(name, [])), "attributes_after_weighting": 0, "status": "Pending", "rule_failed": False}
        rule, _, _, _ = get_dynamic_theme_config(norm_concept, name, config) # This function already handles overrides for theme rules/weights
        if rule == "Must have 1" and not theme_map.get(name):
             logger.warning(f"[{norm_concept}] Rule FAILED: Mandatory theme '{name}' has no assigned concepts.")
             failed_rules[name] = {"reason": "No assigns."}; diag.update({"status": "Failed Rule", "rule_failed": True})

    fixed = set(); fallback_adds = []
    if failed_rules and original_candidates_map_for_reprompt and args.llm_provider != "none":
        logger.info(f"[{norm_concept}] Attempting LLM fallback for {len(failed_rules)} mandatory themes: {list(failed_rules.keys())}")
        for name in list(failed_rules.keys()):
            reprompt_diag["attempts"] += 1; base_cfg = base_themes.get(name)
            if not base_cfg: logger.error(f"[{norm_concept}] No config found for fallback theme '{name}'."); reprompt_diag["failures"] += 1; continue
            sys_p, user_p = build_reprompt_prompt(input_concept, name, base_cfg, original_candidates_map_for_reprompt)
            fb_result = call_llm(sys_p, user_p, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)
            if fb_result and fb_result.get("success"):
                fb_assigns = fb_result.get("response", {}).get("theme_assignments", {}); new_uris = set(uri for uri, ts in fb_assigns.items() if isinstance(ts, list) and name in ts and uri in original_candidates_map_for_reprompt) if isinstance(fb_assigns, dict) else set()
                if new_uris:
                    logger.info(f"[{norm_concept}] Fallback SUCCESS for '{name}': Assigned {len(new_uris)} URIs.")
                    reprompt_diag["successes"] += 1; fixed.add(name); added = 0
                    for uri in new_uris:
                        if uri not in theme_map.get(name, []): theme_map[name].append(uri); fallback_adds.append({"uri": uri, "assigned_theme": name}); added += 1
                    # logger.debug(f"[{norm_concept}] Added {added} unique URIs via fallback for '{name}'.") # Reduced noise
                    if name in theme_diag: theme_diag[name].update({"status": "Passed (Fallback)", "rule_failed": False})
                else: logger.warning(f"[{norm_concept}] Fallback LLM for '{name}' returned successfully but assigned 0 candidates."); reprompt_diag["failures"] += 1; theme_diag[name]["status"] = "Failed (Fallback - No Assigns)"
            else: err = fb_result.get("error", "Unknown") if fb_result else "None"; logger.error(f"[{norm_concept}] Fallback LLM call failed for theme '{name}'. Error: {err}"); reprompt_diag["failures"] += 1; theme_diag[name]["status"] = "Failed (Fallback - API Error)"
    elif failed_rules: logger.warning(f"[{norm_concept}] Cannot attempt fallback for failed rules ({list(failed_rules.keys())}), LLM provider is 'none'.")

    final_themes_out = []; all_final_attrs = []
    theme_w_cfg = {n: get_dynamic_theme_config(norm_concept, n, config)[1] for n in base_themes.keys()}; norm_w = normalize_weights(theme_w_cfg)

    for name, base_data in base_themes.items():
        diag = theme_diag[name]; rule, _, subscore, _ = get_dynamic_theme_config(norm_concept, name, config)
        theme_w = norm_w.get(name, 0.0); uris = theme_map.get(name, [])
        diag["llm_assigned_count"] = len(uris)

        # Use the combined_score from the detailed candidate map passed for reprompting
        scores = {u: original_candidates_map_for_reprompt.get(u, {}).get("combined_score", 0.0)
                  for u in uris if u in original_candidates_map_for_reprompt}
        total_score = sum(scores.values())
        # logger.debug(f"[{norm_concept}][{name}] URIs:{len(uris)}, Combined Score Sum:{total_score:.4f}, Theme Weight:{theme_w:.4f}") # Reduced noise

        theme_attrs = []
        if uris and theme_w > 0:
            n_uris = len(uris)
            if total_score < 1e-9 and n_uris > 0:
                # logger.warning(f"[{norm_concept}][{name}] Zero combined scores for assigned candidates. Using equal weight distribution for {n_uris} URIs.") # Reduced noise
                eq_w = (theme_w / n_uris)
                if eq_w >= min_weight:
                    for u in uris:
                        if u not in original_candidates_map_for_reprompt: continue
                        d = original_candidates_map_for_reprompt[u]; is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                        attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(eq_w, 6), "type": d.get('type_labels', [])}
                        if is_fb: attr["comment"] = "Fallback Assignment"
                        theme_attrs.append(attr); all_final_attrs.append(attr)
                # else: logger.warning(f"[{norm_concept}][{name}] Equal weight {eq_w:.6f} < {min_weight}, skipping attributes for this theme.") # Reduced noise
            elif total_score > 1e-9:
                 for u in uris:
                      if u not in original_candidates_map_for_reprompt: continue
                      d = original_candidates_map_for_reprompt[u]
                      s = scores.get(u, 0.0)
                      prop = s / total_score if total_score > 0 else 0
                      attr_w = theme_w * prop
                      if attr_w >= min_weight:
                           is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                           attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(attr_w, 6), "type": d.get('type_labels', [])}
                           if is_fb: attr["comment"] = "Fallback Assignment"
                           theme_attrs.append(attr); all_final_attrs.append(attr)

        theme_attrs.sort(key=lambda x: x['concept_weight'], reverse=True); diag["attributes_after_weighting"] = len(theme_attrs)
        if diag["status"] == "Pending": diag["status"] = "Processed (Initial)" if not diag.get("rule_failed") else diag["status"]
        final_themes_out.append({"theme_name": name, "theme_type": base_data.get("type", "unknown"), "rule_applied": rule, "normalized_theme_weight": round(theme_w, 6), "subScore": subscore or f"{name}Affinity", "llm_summary": None, "attributes": theme_attrs })
    output["themes"] = final_themes_out

    # --- Calculate Top Defining Attributes ---
    unique_attrs: Dict[str, Dict[str, Any]] = {}
    for attr in all_final_attrs:
        uri = attr.get("uri")
        if not uri: continue
        try: current_weight = float(attr.get("concept_weight", 0.0))
        except (ValueError, TypeError): current_weight = 0.0
        stored_weight = unique_attrs.get(uri, {}).get("concept_weight", -1.0)
        if uri not in unique_attrs or current_weight > stored_weight:
             # Use the full attribute dict when storing/updating
             unique_attrs[uri] = {k: v for k, v in attr.items() if k != 'comment'}

    sorted_top = sorted(unique_attrs.values(), key=lambda x: x.get('concept_weight', 0.0), reverse=True)
    output['top_defining_attributes'] = sorted_top[:top_n_attrs]

    # --- Apply Final Concept Overrides (from Config) ---
    # lodging_type and additional_subscores were handled at output initialization
    tc = output["travel_category"]
    if isinstance(tc, dict):
        # Note: anchor_candidate is preferred, but fallback to concept_overrides if anchor is missing? Careful logic needed.
        # Assuming travel_category was populated correctly by stage 1 anchor details
        tc["type"] = concept_overrides.get("travel_category_type", tc.get("type", "Unknown"))
        tc["exclusionary_concepts"] = concept_overrides.get("exclusionary_concepts", [])
    else:
        logger.error(f"[{norm_concept}] Travel category structure invalid during override application?")
        output["travel_category"] = { # Recreate if invalid
            "uri": None,
            "name": norm_concept,
            "type": concept_overrides.get("travel_category_type", "Uncategorized"),
            "exclusionary_concepts": concept_overrides.get("exclusionary_concepts", [])
        }

    mnh_config = concept_overrides.get("must_not_have", [])
    mnh_uris = set(i["uri"] for i in mnh_config if isinstance(i, dict) and "uri" in i) if isinstance(mnh_config, list) else set()
    output["must_not_have"] = [{"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "scope": None} for u in sorted(list(mnh_uris))]
    # --- End Final Overrides ---


    # --- Final Diagnostic Counts ---
    final_diag = output["diagnostics"]["final_output"]
    final_diag["must_not_have_count"] = len(output["must_not_have"]); final_diag["additional_subscores_count"] = len(output["additional_relevant_subscores"])
    final_diag["themes_count"] = len(output["themes"]); output["failed_fallback_themes"] = { n: r for n, r in failed_rules.items() if n not in fixed }
    final_diag["failed_fallback_themes_count"] = len(output["failed_fallback_themes"]); final_diag["top_defining_attributes_count"] = len(output['top_defining_attributes'])
    output["diagnostics"]["stage2"]["status"] = "Completed"; output["diagnostics"]["stage2"]["duration_seconds"] = round(time.time() - start, 2)
    return output


# --- Main Processing Loop ---
# (generate_affinity_definitions_loop updated for v33.0.29 to pass bm25 model)
def generate_affinity_definitions_loop(
    concepts_to_process: List[str], config: Dict, args: argparse.Namespace,
    sbert_model: SentenceTransformer, primary_embeddings_map: Dict[str, np.ndarray],
    taxonomy_concepts_cache: Dict[str, Dict], keyword_label_index: Optional[Dict[str, Set[str]]],
    # --- Pass BM25 model and URIs instead of TFIDF ---
    bm25_model: Optional[BM25Okapi],
    keyword_corpus_uris: Optional[List[str]]
    # --- End change ---
) -> List[Dict]:
    """ Main loop processing each concept. v33.0.29 uses BM25."""
    all_definitions = []
    cache_ver = config.get("CACHE_VERSION", DEFAULT_CACHE_VERSION)
    if not taxonomy_concepts_cache: logger.critical("FATAL: Taxonomy concepts cache is empty or None."); return []
    limit = args.limit if args.limit and args.limit > 0 else len(concepts_to_process)
    concepts_subset = concepts_to_process[:limit]; logger.info(f"Processing {len(concepts_subset)}/{len(concepts_to_process)} concepts.")

    if not concepts_subset: logger.warning("Concept subset to process is empty!"); return []

    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug

    for concept in tqdm(concepts_subset, desc="Processing Concepts", disable=disable_tqdm):
        start_time = time.time(); norm_concept = normalize_concept(concept)
        logger.info(f"=== Processing Concept: '{concept}' ('{norm_concept}') ===")

        affinity_def = {
            "input_concept": concept, "normalized_concept": norm_concept,
            "applicable_lodging_types": "Both", "travel_category": {},
            "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [], "must_not_have": [],
            "failed_fallback_themes": {},
            "processing_metadata": { "status": "Started", "version": SCRIPT_VERSION, "timestamp": None, "total_duration_seconds": 0.0, "cache_version": cache_ver, "llm_provider": args.llm_provider, "llm_model": args.llm_model if args.llm_provider != "none" else None },
            "diagnostics": {
                "stage1": { "status": "Not Started", "error": None, "selection_method": "Combined BM25+SBERT", "expansion": {}, "sbert_candidate_count_initial": 0, "tfidf_candidate_count_initial": 0, "unique_candidates_before_ranking": 0, "llm_candidate_count": 0 }, # Updated method name
                "llm_slotting": {"status": "Not Started", "error": None},
                "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0},
                "stage2": {"status": "Not Started", "error": None},
                "theme_processing": {}, "final_output": {}, "error_details": None
            }
        }
        diag1 = affinity_def["diagnostics"]["stage1"]; diag_llm = affinity_def["diagnostics"]["llm_slotting"]

        try:
            concept_emb = get_concept_embedding(norm_concept, sbert_model)
            if concept_emb is None:
                logger.error(f"[{norm_concept}] Embedding failed."); diag1["error"] = "Embedding failed"; diag1["status"] = "Failed"

            stage1_start = time.time()
            # --- Pass BM25 model and URIs instead of TFIDF ---
            cand_details, orig_map, anchor, sbert_scores, exp_diag, sbert_init_count, kw_init_count, unique_count = prepare_evidence(
                concept, concept_emb, primary_embeddings_map, config, args,
                bm25_model, # Pass BM25 model
                keyword_corpus_uris, # Pass corpus URIs
                keyword_label_index,
                taxonomy_concepts_cache
            )
            # --- End change ---
            stage1_dur = time.time() - stage1_start

            diag1["expansion"] = exp_diag
            # Update diagnostic key name for clarity
            diag1["keyword_candidate_count_initial"] = kw_init_count
            diag1.update({
                "status": "Completed" if diag1.get("error") is None and concept_emb is not None else "Failed",
                "duration_seconds": round(stage1_dur, 2),
                "sbert_candidate_count_initial": sbert_init_count,
                # "tfidf_candidate_count_initial": kw_init_count, # Removed old key
                "unique_candidates_before_ranking": unique_count,
                "llm_candidate_count": len(cand_details)
                })
            logger.info(f"[{norm_concept}] Stage 1 done ({stage1_dur:.2f}s). Status: {diag1['status']}. LLM Cands: {len(cand_details)}. Initial SBERT: {sbert_init_count}, Keyword (BM25): {kw_init_count}.") # Updated log

            affinity_def["travel_category"] = anchor if anchor and anchor.get('uri') else {"uri": None, "name": concept, "type": "Unknown"}
            if not anchor and diag1["status"] == "Completed": logger.warning(f"[{norm_concept}] Stage 1 completed but no anchor candidate found/selected.")

            # --- Stage 1.5: LLM Theme Slotting ---
            llm_result = None
            llm_start = time.time()
            diag_llm.update({"llm_provider": args.llm_provider, "llm_model": args.llm_model if args.llm_provider != "none" else None})

            if diag1["status"] == "Failed":
                logger.warning(f"[{norm_concept}] Skipping LLM slotting due to Stage 1 failure.")
                diag_llm["status"] = "Skipped (Stage 1 Failed)"
            elif not cand_details:
                logger.warning(f"[{norm_concept}] Skipping LLM slotting (No candidates generated).")
                if affinity_def["processing_metadata"]["status"] == "Started": affinity_def["processing_metadata"]["status"] = "Warning - No LLM Candidates"
                diag_llm["status"] = "Skipped (No Candidates)"
            elif args.llm_provider == "none":
                logger.info(f"[{norm_concept}] Skipping LLM slotting (Provider is 'none').")
                diag_llm["status"] = "Skipped (Provider None)"
            else:
                diag_llm["status"] = "Started"; diag_llm["llm_call_attempted"] = True
                themes_for_prompt = [
                    {"name": name, "description": get_theme_definition_for_prompt(name, data),
                     "is_must_have": get_dynamic_theme_config(norm_concept, name, config)[0] == "Must have 1"}
                    for name, data in config.get("base_themes", {}).items()
                ]
                sys_p, user_p = construct_llm_slotting_prompt(concept, themes_for_prompt, cand_details, args)
                llm_result = call_llm(sys_p, user_p, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)

                diag_llm["attempts_made"] = llm_result.get("attempts", 0) if llm_result else 0
                if llm_result and llm_result.get("success"):
                    diag_llm["llm_call_success"] = True; diag_llm["status"] = "Completed"
                else:
                    diag_llm["llm_call_success"] = False; diag_llm["status"] = "Failed"
                    diag_llm["error"] = llm_result.get("error", "Unknown Error") if llm_result else "LLM Call resulted in None"
                    logger.warning(f"[{norm_concept}] LLM slotting call failed. Error: {diag_llm['error']}")

            diag_llm["duration_seconds"] = round(time.time() - llm_start, 2)
            logger.info(f"[{norm_concept}] LLM Slotting took {diag_llm['duration_seconds']:.2f}s. Status: {diag_llm['status']}")

            # --- Stage 2: Finalization ---
            stage2_start = time.time()
            stage2_out = apply_rules_and_finalize(
                concept, llm_result, config, affinity_def["travel_category"], anchor,
                orig_map, sbert_scores, args, _taxonomy_concepts_cache # Pass sbert_scores here
            )
            stage2_dur = time.time() - stage2_start

            affinity_def.update({k: v for k, v in stage2_out.items() if k != 'diagnostics'})
            if "diagnostics" in stage2_out:
                s2d = stage2_out["diagnostics"]
                affinity_def["diagnostics"]["theme_processing"] = s2d.get("theme_processing", {})
                affinity_def["diagnostics"]["reprompting_fallback"].update(s2d.get("reprompting_fallback", {}))
                affinity_def["diagnostics"]["final_output"] = s2d.get("final_output", {})
                affinity_def["diagnostics"]["stage2"] = s2d.get("stage2", {"status": "Unknown"})
            else: affinity_def["diagnostics"]["stage2"] = {"status": "Unknown", "error": "Stage 2 diagnostics missing"}
            affinity_def["diagnostics"]["stage2"]["duration_seconds"] = round(stage2_dur, 2)

            # Determine final processing status
            final_status = affinity_def["processing_metadata"]["status"]
            if final_status == "Started":
                if diag1["status"] == "Failed": final_status = "Failed - Stage 1 Error"
                elif affinity_def.get("failed_fallback_themes"): final_status = "Success with Failed Rules"
                elif diag_llm["status"] == "Failed": final_status = "Warning - LLM Slotting Failed"
                elif diag_llm["status"] == "Skipped (No Candidates)": final_status = "Warning - No LLM Candidates"
                elif affinity_def["diagnostics"]["stage2"].get("error"): final_status = f"Warning - Finalization Error ({affinity_def['diagnostics']['stage2']['error']})"
                elif diag_llm["status"] == "Skipped (Provider None)": final_status = "Success (LLM Skipped)"
                else: final_status = "Success"
            affinity_def["processing_metadata"]["status"] = final_status


        except Exception as e:
            logger.error(f"Core processing loop failed unexpectedly for concept '{concept}': {e}", exc_info=True)
            affinity_def["processing_metadata"]["status"] = f"FATAL ERROR"
            affinity_def["diagnostics"]["error_details"] = traceback.format_exc()
            if diag1["status"] not in ["Completed", "Failed"]: diag1["status"] = "Failed (Exception)"
            if diag_llm["status"] not in ["Completed", "Failed", "Skipped (No Candidates)", "Skipped (Provider None)", "Skipped (Stage 1 Failed)"]: diag_llm["status"] = "Failed (Exception)"
            if affinity_def["diagnostics"]["stage2"]["status"] not in ["Completed", "Failed"]: affinity_def["diagnostics"]["stage2"]["status"] = "Failed (Exception)"

        finally:
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            affinity_def["processing_metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            affinity_def["processing_metadata"]["total_duration_seconds"] = duration
            all_definitions.append(affinity_def)
            log_func = logger.warning if "Warning" in affinity_def["processing_metadata"]["status"] or "Failed" in affinity_def["processing_metadata"]["status"] else logger.info
            log_func(f"--- Finished '{norm_concept}' ({duration:.2f}s). Status: {affinity_def['processing_metadata']['status']} ---")

    return all_definitions


# --- Main Execution ---
def main():
    # Make args global so build_keyword_index can access debug flag
    global _config_data, _taxonomy_concepts_cache, _taxonomy_embeddings_cache, \
           _keyword_label_index, _bm25_model, _keyword_corpus_uris, args

    parser = argparse.ArgumentParser(description=f"Generate Travel Concept Affinity Definitions ({SCRIPT_VERSION})")
    # ... (keep all argument definitions as before) ...
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_FILE, help="Config JSON path.")
    parser.add_argument("-t", "--taxonomy-dir", dest="taxonomy_dir", type=str, default=DEFAULT_TAXONOMY_DIR, help="Taxonomy RDF directory.")
    parser.add_argument("-i", "--input-concepts-file", type=str, required=True, help="Input concepts file path.")
    parser.add_argument("-o", "--output-dir", type=str, default="./output_v33", help="Output directory.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache directory.")
    parser.add_argument("--rebuild-cache", action='store_true', help="Force rebuild caches.")
    parser.add_argument("--limit", type=int, default=None, help="Limit concepts processed.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
    parser.add_argument("--llm-provider", type=str, choices=['openai', 'google', 'none'], default=None, help="Override LLM provider.")
    parser.add_argument("--llm-model", type=str, default=None, help="Override LLM model.")
    args = parser.parse_args() # Assign parsed args to the global variable 'args'

    config = load_affinity_config(args.config)
    if config is None: sys.exit(1)
    _config_data = config

    # Ensure necessary params are loaded or defaulted
    config["COMBINED_SCORE_ALPHA"] = config.get("COMBINED_SCORE_ALPHA", COMBINED_SCORE_ALPHA)
    config.setdefault("STAGE1_CONFIG", {})
    config["STAGE1_CONFIG"]["ABSOLUTE_MIN_SBERT_SCORE"] = config["STAGE1_CONFIG"].get("ABSOLUTE_MIN_SBERT_SCORE", ABSOLUTE_MIN_SBERT_SCORE)
    config["STAGE1_CONFIG"]["DAMPENING_SBERT_THRESHOLD"] = config["STAGE1_CONFIG"].get("DAMPENING_SBERT_THRESHOLD", DAMPENING_SBERT_THRESHOLD)
    config["STAGE1_CONFIG"]["DAMPENING_FACTOR"] = config["STAGE1_CONFIG"].get("DAMPENING_FACTOR", DAMPENING_FACTOR)
    # Ensure KEYWORD_MIN_SCORE uses Stage 1 config or global default
    config["STAGE1_CONFIG"]["KEYWORD_MIN_SCORE"] = config["STAGE1_CONFIG"].get("KEYWORD_MIN_SCORE", KEYWORD_MIN_SCORE)

    cache_ver = config.get("CACHE_VERSION", DEFAULT_CACHE_VERSION)
    os.makedirs(args.output_dir, exist_ok=True); os.makedirs(args.cache_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, LOG_FILE_TEMPLATE.format(cache_version=cache_ver))
    out_file = os.path.join(args.output_dir, OUTPUT_FILE_TEMPLATE.format(cache_version=cache_ver))
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level, log_file, args.debug)

    logger.info(f"Starting {SCRIPT_VERSION}"); logger.info(f"Cmd: {' '.join(sys.argv)}")
    logger.info(f"Config: {args.config}, Taxo: {args.taxonomy_dir}, Input: {args.input_concepts_file}")
    logger.info(f"Output: {args.output_dir}, Cache: {args.cache_dir}, Log: {log_file}, Outfile: {out_file}")
    logger.info(f"Rebuild: {args.rebuild_cache}, Limit: {args.limit or 'None'}, Debug: {args.debug}")
    logger.info(f"--- Key Parameters for Run (as loaded/defaulted) ---")
    logger.info(f"COMBINED_SCORE_ALPHA (Default): {config['COMBINED_SCORE_ALPHA']}")
    logger.info(f"ALPHA MECHANISM: Handled via config['concept_overrides']['<concept>']['effective_alpha']")
    logger.info(f"SORTING: Tie-Breaking Enabled (Score > Namespace Prio > SBERT)")
    logger.info(f"KEYWORD SCORING: BM25 (from rank-bm25 library)") # Updated log
    logger.info(f"TF-IDF/BM25 DOC STRATEGY: Weighted Field Concatenation (prefLabel x5, altLabel x3, definition x1)")
    logger.info(f"ABSOLUTE_MIN_SBERT_SCORE: {config['STAGE1_CONFIG']['ABSOLUTE_MIN_SBERT_SCORE']}")
    logger.info(f"DAMPENING_SBERT_THRESHOLD: {config['STAGE1_CONFIG']['DAMPENING_SBERT_THRESHOLD']}")
    logger.info(f"DAMPENING_FACTOR: {config['STAGE1_CONFIG']['DAMPENING_FACTOR']}")
    logger.info(f"KEYWORD_MIN_SCORE (BM25): {config['STAGE1_CONFIG']['KEYWORD_MIN_SCORE']} (NOTE: Scale differs from TF-IDF, may need tuning)") # Added note
    logger.info(f"ENABLE_KW_EXPANSION (Default): {config.get('ENABLE_KW_EXPANSION', ENABLE_KW_EXPANSION)}")
    kw_scoring_enabled = config.get("KEYWORD_SCORING_CONFIG", config.get("TFIDF_CONFIG", {})).get("enabled", False)
    logger.info(f"Keyword Scoring Enabled: {kw_scoring_enabled}")
    if kw_scoring_enabled:
         bm25_params_log = config.get("KEYWORD_SCORING_CONFIG", config.get("TFIDF_CONFIG", {})).get("bm25_params", {})
         logger.info(f"BM25 Params: {bm25_params_log}")
    logger.info(f"Specific Handling (Seeding/Filtering/Boosting/Splitting/Expansion): Driven by config['concept_overrides']")
    logger.info(f"--- End Key Parameters ---")

    if args.llm_provider is None: args.llm_provider = config.get('LLM_PROVIDER', 'none')
    else: logger.warning(f"Overriding LLM provider from command line: '{args.llm_provider}'")
    if args.llm_model is None: args.llm_model = config.get('LLM_MODEL', None)
    else: logger.warning(f"Overriding LLM model from command line: '{args.llm_model}'")
    logger.info(f"Effective LLM Provider: {args.llm_provider}, Model: {args.llm_model or 'N/A'}")

    if not RANK_BM25_AVAILABLE and kw_scoring_enabled:
        logger.critical("FATAL: BM25 keyword scoring is enabled in config, but 'rank-bm25' library is not installed. Please run: pip install rank-bm25")
        sys.exit(1)

    input_concepts = []
    try:
        with open(args.input_concepts_file, 'r', encoding='utf-8') as f: input_concepts = [l.strip() for l in f if l.strip()]
        if not input_concepts: raise ValueError("Input concepts file is empty or contains only whitespace.")
        logger.info(f"Loaded {len(input_concepts)} concepts from '{args.input_concepts_file}'.")
    except FileNotFoundError: logger.critical(f"Input concepts file not found: '{args.input_concepts_file}'"); sys.exit(1)
    except Exception as e: logger.critical(f"Failed to read input concepts file '{args.input_concepts_file}': {e}"); sys.exit(1)

    concepts_cache_f = get_cache_filename('concepts', cache_ver, args.cache_dir, extension=".json")
    concepts_data = load_taxonomy_concepts(args.taxonomy_dir, concepts_cache_f, args.rebuild_cache, cache_ver, args.debug)
    if concepts_data is None: logger.critical("Taxonomy concepts loading failed."); sys.exit(1)
    _taxonomy_concepts_cache = concepts_data; logger.info(f"Taxonomy concepts ready ({len(_taxonomy_concepts_cache)} concepts).");

    try:
        # Build BM25 index or load from cache
        build_keyword_index(config, _taxonomy_concepts_cache, args.cache_dir, cache_ver, args.rebuild_cache)
        if _bm25_model is not None: logger.info(f"BM25 Keyword Index ready.")
        elif kw_scoring_enabled: logger.warning("BM25 keyword scoring is enabled in config but index failed to load/build.")
        if _keyword_label_index is not None: logger.info(f"Simple Label Index ready.")
        else: logger.warning("Simple Label Index failed to build.")
    except Exception as e: logger.critical(f"Index building process failed: {e}", exc_info=True); sys.exit(1)

    try:
        sbert_name = config.get("SBERT_MODEL_NAME"); sbert_model = get_sbert_model(sbert_name)
        if sbert_model is None: raise RuntimeError("SBERT model could not be loaded (check name/path in config and utils.py)")
        logger.info(f"SBERT model '{sbert_name or 'default'}' loaded. Dimension: {sbert_model.get_sentence_embedding_dimension()}")
    except Exception as e: logger.critical(f"SBERT model loading failed: {e}", exc_info=True); sys.exit(1)

    embed_cache_params = {"model": sbert_name or "default"}
    embed_cache_f = get_cache_filename('embeddings', cache_ver, args.cache_dir, embed_cache_params, ".pkl")
    embed_data = precompute_taxonomy_embeddings(_taxonomy_concepts_cache, sbert_model, embed_cache_f, cache_ver, args.rebuild_cache, args.debug)
    if embed_data is None: logger.critical("Taxonomy embeddings loading/precomputation failed."); sys.exit(1)
    _taxonomy_embeddings_cache = embed_data; primary_embs, uris_w_embeds = _taxonomy_embeddings_cache
    logger.info(f"Taxonomy embeddings ready ({len(uris_w_embeds)} concepts embedded).");

    logger.info("Starting affinity definition generation loop...")
    start_loop = time.time()
    # --- Pass BM25 model and URIs to main loop ---
    results = generate_affinity_definitions_loop(
        input_concepts, config, args, sbert_model, primary_embs,
        _taxonomy_concepts_cache, _keyword_label_index,
        _bm25_model, _keyword_corpus_uris # Pass BM25 related globals
    )
    # --- End change ---
    end_loop = time.time(); logger.info(f"Generation loop finished ({end_loop - start_loop:.2f}s). Generated {len(results)} definitions.")

    if results: save_results_json(results, out_file)
    else: logger.warning("No results were generated to save.")
    logger.info(f"Script finished. Final log messages in: {log_file}");

# --- Standard Python Entry Point ---
if __name__ == "__main__":
    # Removed SKLEARN check as TFIDFVectorizer/cosine_similarity are no longer used
    if not all([UTILS_ST_AVAILABLE, RDFLIB_AVAILABLE]):
         print("CRITICAL: Required libraries (SentenceTransformers, RDFlib) are unavailable. Check imports in utils.py and main script.", file=sys.stderr); sys.exit(1)
    # Add check for rank-bm25 if keyword scoring is intended
    # (Actual check happens in main() after config load)

    llm_keys_found = False
    if os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API Key found.")
        llm_keys_found = True
    if os.environ.get("GOOGLE_API_KEY"):
        print("Google API Key found.")
        llm_keys_found = True
    if not llm_keys_found:
         print("Warning: No LLM API keys (OPENAI_API_KEY or GOOGLE_API_KEY) found in environment variables. LLM features will be disabled if provider is not 'none'.", file=sys.stderr)
    main()