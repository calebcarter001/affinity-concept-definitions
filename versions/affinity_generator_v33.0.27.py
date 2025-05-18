# -*- coding: utf-8 -*-
"""
Generate affinity definitions for travel concepts using combined scoring (v33.0.27 - TF-IDF Field Weighting).

Implements:
- TF-IDF Field Weighting (Weighted Field Concatenation: prefLabel x5, altLabel x3, definition x1).
- Concept-Specific Alpha Override (e.g., airconditioning=0.05).
- Multi-key sorting with namespace priority tie-breaking.
- Reduced logging verbosity for smaller log files.
- Retains critical warnings, errors, and essential diagnostic debug logs.
- Combined TF-IDF Keyword + SBERT Similarity score for candidate selection.
- Absolute SBERT Filter (0.15).
- Conditional TF-IDF Dampening (Thresh=0.35, Factor=0.15).
- TF-IDF indexing and search functionality.
- Optional LLM keyword expansion for weak concepts (except specific list).
- LLM-assisted theme slotting.
- Relies on shared utility functions in utils.py.

Changes from v33.0.26:
- Modified build_keyword_index to implement weighted field concatenation for TF-IDF documents.
- Updated SCRIPT_VERSION and relevant logs.
- NOTE: Requires TF-IDF cache rebuild (--rebuild-cache).
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

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    pass # Checked via utils import

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found.", file=sys.stderr)
    def tqdm(iterable, *args, **kwargs):
        return iterable # Define dummy

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
    from utils import (
        setup_logging, normalize_concept, get_primary_label, get_concept_type_labels,
        get_sbert_model, load_affinity_config, get_cache_filename, load_cache, save_cache,
        load_taxonomy_concepts, precompute_taxonomy_embeddings, get_concept_embedding,
        get_batch_embedding_similarity, get_kg_data,
        build_keyword_label_index, # Ensure this is still suitable for simple index
        save_results_json,
        RDFLIB_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE as UTILS_ST_AVAILABLE,
        SKLEARN_AVAILABLE as UTILS_SK_AVAILABLE
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Failed import from 'utils.py': {e}", file=sys.stderr); sys.exit(1)
except Exception as e:
     print(f"CRITICAL ERROR: Unexpected error importing from 'utils.py': {e}", file=sys.stderr); sys.exit(1)

# --- Config Defaults & Constants ---
SCRIPT_VERSION = "affinity-rule-engine-v33.0.27 (TF-IDF Field Weighting)" # Updated version
DEFAULT_CACHE_VERSION = "v20250420.affinity.33.0.27" # Update cache version due to TF-IDF change
DEFAULT_TAXONOMY_DIR = "./datasources/"
DEFAULT_CACHE_DIR = "./cache_v33/"
DEFAULT_CONFIG_FILE = "./affinity_config_v33.0.json"
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
KEYWORD_MIN_SCORE = 0.05
THEME_ATTRIBUTE_MIN_WEIGHT = 0.001
TRAVEL_CONTEXT = "travel "
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 5
LLM_TEMPERATURE = 0.2
LLM_RETRY_DELAY_SECONDS = 5
MIN_KEYWORD_CANDIDATES_FOR_EXPANSION = 5
KEYWORD_TOP_N_SEARCH = 500
KW_EXPANSION_TEMPERATURE = 0.5
TOP_N_DEFINING_ATTRIBUTES_DEFAULT = 25

# --- List of known abstract/package concepts ---
ABSTRACT_CONCEPTS_LIST = ["allinclusive", "allinclusivemeals", "allinclusiveresort", "luxury", "budget", "value"]

# --- Globals ---
_config_data: Optional[Dict] = None
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict[str, np.ndarray], List[str]]] = None
_keyword_label_index: Optional[Dict[str, Set[str]]] = None
_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix: Optional[Any] = None
_tfidf_corpus_uris: Optional[List[str]] = None
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
    overrides = config.get("concept_overrides", {})
    base_data = base_themes.get(theme_name)
    if not base_data: logger.error(f"Base theme '{theme_name}' not found!"); return "Optional", 0.0, None, None
    concept_override = overrides.get(normalized_concept, {})
    theme_override = concept_override.get("theme_overrides", {}).get(theme_name, {})
    merged = {**base_data, **theme_override}
    rule = merged.get("rule_applied", merged.get("rule", "Optional"))
    rule = "Optional" if rule not in ["Must have 1", "Optional"] else rule
    weight = merged.get("weight", 0.0)
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

# --- START: TF-IDF Indexing Function (v33.0.27) ---
def build_keyword_index(
    config: Dict, taxonomy_concepts_cache: Dict[str, Dict], cache_dir: str,
    cache_version: str, rebuild_cache: bool
) -> Tuple[Optional[TfidfVectorizer], Optional[Any], Optional[List[str]], Optional[Dict[str, Set[str]]]]:
    """
    Builds or loads TF-IDF index and simple label index.
    v33.0.27: Implements Weighted Field Concatenation for TF-IDF documents
              (prefLabel x5, altLabel x3, definition x1) to boost label terms.
              Requires cache rebuild if TF-IDF index is regenerated.
    """
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris, _keyword_label_index, args # Access global args
    vectorizer, tfidf_matrix, corpus_uris, label_index = None, None, None, None
    tfidf_cfg = config.get("TFIDF_CONFIG", {})
    tfidf_params_cfg = tfidf_cfg.get("TFIDF_VECTORIZER_PARAMS", config.get("TFIDF_VECTORIZER_PARAMS", {}))
    use_tfidf = tfidf_cfg.get("enabled", False) and UTILS_SK_AVAILABLE

    # Define repetition factors
    pref_label_weight = 5
    alt_label_weight = 3
    definition_weight = 1

    uris_to_log_tfidf_doc = {
        "urn:expediagroup:taxonomies:acs:#1f3da634-0df6-4498-a8d9-603f895c8f3f", # AirConditioning:Unavailable
        "urn:expediagroup:taxonomies:acsPCS:#AirConditioning",
        "urn:expediagroup:taxonomies:lcm:#34528832-c2d8-312a-b52f-b27c483e5ec1", # All-inclusive resort
        "urn:expediagroup:taxonomies:lcm:#b2fa2cb3-9226-305e-8943-02d82ba90975", # Resort
        "urn:expediagroup:taxonomies:acsEnumerations:#dcecb8c7-bacf-4d04-b90f-95e4654ffa9f", # Allinclusive
    }

    if use_tfidf:
        logger.info("TF-IDF Indexing enabled.")
        logger.info(f"Applying Field Weighting: prefLabel x{pref_label_weight}, altLabel x{alt_label_weight}, definition x{definition_weight}")
        tfidf_params = {"max_df": 0.95, "min_df": 2, "stop_words": "english", **tfidf_params_cfg}
        if "ngram_range" in tfidf_params and tfidf_params["ngram_range"] != (1,1):
             logger.warning(f"Unigrams required. Found ngram_range={tfidf_params['ngram_range']} in config. Removing it.")
             del tfidf_params["ngram_range"]
        logger.info(f"Using TF-IDF Vectorizer Params: {tfidf_params}")

        # Add weighting factors to cache filename parameters to ensure unique cache for this config
        cache_params_for_filename = {
            **tfidf_params,
            "weights": f"p{pref_label_weight}a{alt_label_weight}d{definition_weight}"
        }

        matrix_cache_file = get_cache_filename("tfidf_matrix", cache_version, cache_dir, cache_params_for_filename, ".pkl")
        vec_cache_file = get_cache_filename("tfidf_vectorizer", cache_version, cache_dir, cache_params_for_filename, ".pkl")
        uris_cache_file = get_cache_filename("tfidf_corpus_uris", cache_version, cache_dir, cache_params_for_filename, ".pkl")

        cache_valid = False
        if not rebuild_cache:
            # Attempt to load cache specific to these weights
            cached_matrix = load_cache(matrix_cache_file, 'pickle')
            cached_vec = load_cache(vec_cache_file, 'pickle')
            cached_uris = load_cache(uris_cache_file, 'pickle')
            if cached_matrix is not None and cached_vec is not None and isinstance(cached_uris, list):
                 loaded_params_ok = True
                 if hasattr(cached_vec, 'ngram_range') and cached_vec.ngram_range != (1, 1):
                     logger.warning(f"Loaded TF-IDF vectorizer from cache has ngram_range={cached_vec.ngram_range}, but unigrams are required. Rebuilding.")
                     loaded_params_ok = False

                 if loaded_params_ok and hasattr(cached_matrix, 'shape') and cached_matrix.shape[0] == len(cached_uris):
                     tfidf_matrix, vectorizer, corpus_uris, cache_valid = cached_matrix, cached_vec, cached_uris, True
                     logger.info(f"TF-IDF (Weighted Fields) loaded from cache ({len(corpus_uris)} URIs).")
                 elif loaded_params_ok:
                     logger.warning("TF-IDF cache dim/URI mismatch. Rebuilding.")
                 else:
                     pass # Handled below
            else:
                 logger.info("TF-IDF cache (Weighted Fields) incomplete/invalid. Rebuilding.")


        if not cache_valid:
            logger.info("Rebuilding TF-IDF index with weighted fields...")
            try:
                docs, doc_uris = [], []
                # Get debug flag from global args (set in main)
                debug_mode = args.debug if args else False
                disable_tqdm = not logger.isEnabledFor(logging.INFO) or debug_mode

                for uri, data in tqdm(sorted(taxonomy_concepts_cache.items()), desc="Prepare TF-IDF Docs", disable=disable_tqdm):
                    texts_to_join = []
                    # Ensure data is a dict
                    if not isinstance(data, dict): continue

                    # Extract text fields safely, handling missing keys and non-list/string values
                    pref_labels_raw = data.get("skos:prefLabel", [])
                    alt_labels_raw = data.get("skos:altLabel", [])
                    definitions_raw = data.get("skos:definition", []) # Keep as list or str

                    # --- Process Labels (Handle list or string) ---
                    pref_labels = []
                    if isinstance(pref_labels_raw, list):
                        pref_labels = [str(lbl) for lbl in pref_labels_raw if lbl and isinstance(lbl, str)]
                    elif isinstance(pref_labels_raw, str) and pref_labels_raw.strip():
                        pref_labels = [pref_labels_raw]

                    alt_labels = []
                    if isinstance(alt_labels_raw, list):
                        alt_labels = [str(lbl) for lbl in alt_labels_raw if lbl and isinstance(lbl, str)]
                    elif isinstance(alt_labels_raw, str) and alt_labels_raw.strip():
                        alt_labels = [alt_labels_raw]

                    # --- Process Definition (Handle list or string, take first valid) ---
                    definition = None
                    if isinstance(definitions_raw, list):
                        for defin in definitions_raw:
                            if defin and isinstance(defin, str) and defin.strip():
                                definition = str(defin)
                                break # Take the first valid one
                    elif isinstance(definitions_raw, str) and definitions_raw.strip():
                        definition = definitions_raw
                    # --- End Field Processing ---

                    # Repeat labels according to weights
                    for _ in range(pref_label_weight):
                        texts_to_join.extend(pref_labels)
                    for _ in range(alt_label_weight):
                        texts_to_join.extend(alt_labels)

                    # Add definition once if valid
                    if definition and definition_weight > 0:
                        texts_to_join.append(definition)

                    # Normalize the combined text
                    raw_doc_text = " ".join(filter(None, texts_to_join)) # Filter out potential None values
                    norm_doc = normalize_concept(raw_doc_text)

                    # Debug logging for specific URIs
                    if uri in uris_to_log_tfidf_doc:
                        label = get_primary_label(uri, taxonomy_concepts_cache, fallback=uri)
                        # Log only a snippet if the raw text is very long due to repetition
                        raw_snippet = raw_doc_text[:200] + ('...' if len(raw_doc_text) > 200 else '')
                        norm_snippet = norm_doc[:200] + ('...' if len(norm_doc) > 200 else '')
                        logger.debug(f"[TF-IDF DOC LOG - Weighted] URI: {uri} Label: '{label}' Raw Snippet: '{raw_snippet}' Norm Snippet: '{norm_snippet}'")

                    if norm_doc:
                        docs.append(norm_doc)
                        doc_uris.append(uri)

                if not docs:
                    logger.warning("No documents generated for TF-IDF build after weighting.")
                else:
                    vectorizer = TfidfVectorizer(**tfidf_params)
                    tfidf_matrix = vectorizer.fit_transform(docs)
                    corpus_uris = doc_uris
                    save_cache(tfidf_matrix, matrix_cache_file, 'pickle')
                    save_cache(vectorizer, vec_cache_file, 'pickle')
                    save_cache(corpus_uris, uris_cache_file, 'pickle')
                    logger.info("TF-IDF components (Weighted Fields) rebuilt and saved.")
            except Exception as e:
                logger.error(f"TF-IDF build error with weighted fields: {e}", exc_info=True)
                tfidf_matrix, vectorizer, corpus_uris = None, None, None

        logger.debug("--- TF-IDF Index Build/Load Diagnostics (Weighted Fields) ---")
        logger.debug(f"Vectorizer Type: {type(vectorizer)}")
        if vectorizer:
             logger.debug(f"Vectorizer Params Used: {vectorizer.get_params()}")
             if hasattr(vectorizer, 'vocabulary_'): logger.debug(f"Vocab Size: {len(vectorizer.vocabulary_)}")
             else: logger.debug("Vocab: Not available.")
        else: logger.debug("Vectorizer: Not available.")
        logger.debug(f"Matrix Shape: {tfidf_matrix.shape if tfidf_matrix is not None else 'None'}")
        logger.debug(f"Corpus URIs Count: {len(corpus_uris) if corpus_uris is not None else 'None'}")
        logger.debug("--- End TF-IDF Diagnostics ---")
    else:
        logger.info("TF-IDF Indexing disabled or sklearn unavailable.")

    # Build simple label index (remains the same, uses original labels)
    # Assuming build_keyword_label_index is correctly defined in utils.py
    try:
        label_index = build_keyword_label_index(taxonomy_concepts_cache)
        if label_index is None:
            logger.warning("Failed simple label index build (returned None).")
        else:
            logger.info(f"Simple label index ready ({len(label_index)} keywords).")
    except Exception as e:
        logger.error(f"Error calling build_keyword_label_index: {e}", exc_info=True)
        label_index = None

    _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris, _keyword_label_index = vectorizer, tfidf_matrix, corpus_uris, label_index
    return vectorizer, tfidf_matrix, corpus_uris, label_index
# --- END: TF-IDF Indexing Function (v33.0.27) ---


# (get_candidate_concepts_keyword function remains unchanged)
def get_candidate_concepts_keyword(
    query_texts: List[str], vectorizer: TfidfVectorizer, tfidf_matrix: Any,
    corpus_uris: List[str], top_n: int, min_score: float = 0.05
) -> List[Dict[str, Any]]:
    """ Finds candidate concepts using TF-IDF similarity. Reduced logging."""
    if not query_texts or tfidf_matrix is None or vectorizer is None or not corpus_uris: return []
    if tfidf_matrix.shape[0] != len(corpus_uris): logger.error(f"TF-IDF shape/URI mismatch ({tfidf_matrix.shape[0]} vs {len(corpus_uris)})."); return []
    logger.debug(f"TF-IDF search query terms (first 10): {query_texts[:10]}{'...' if len(query_texts) > 10 else ''}")
    try:
        query_matrix = vectorizer.transform(query_texts)
        logger.debug(f"Query Matrix shape: {query_matrix.shape}, Non-zero elements: {query_matrix.nnz}")
        if query_matrix.nnz == 0:
            logger.warning(f"TF-IDF Query resulted in ZERO non-zero elements for query: {query_texts}")
            if hasattr(vectorizer, 'vocabulary_'):
                missing_terms = [term for term in query_texts if term not in vectorizer.vocabulary_]
                if missing_terms:
                    logger.debug(f"Terms not in TF-IDF vocabulary: {missing_terms}")
                else:
                    logger.debug("Query terms exist in vocab, but TF-IDF vector is zero. Check TF-IDF weighting/params.")
        if query_matrix.shape[1] != tfidf_matrix.shape[1]: logger.error(f"TF-IDF vocab mismatch! Query:{query_matrix.shape[1]} vs Corpus:{tfidf_matrix.shape[1]}"); return []
        similarities = cosine_similarity(query_matrix, tfidf_matrix)
        agg_scores = np.max(similarities, axis=0) if similarities.shape[0] > 1 else similarities[0] if similarities.shape[0] == 1 else np.array([])
        if agg_scores.size == 0: logger.debug("Aggregated TF-IDF scores empty."); return []
        logger.debug(f"Agg Scores Shape: {agg_scores.shape}, Min: {np.min(agg_scores):.4f}, Max: {np.max(agg_scores):.4f}")
        if np.max(agg_scores) < min_score: logger.debug(f"Max TF-IDF score ({np.max(agg_scores):.4f}) < min_score ({min_score}). No candidates pass."); return []

        num_cands = len(agg_scores); top_n = min(top_n, num_cands)
        if top_n <= 0: return []
        # Using argpartition for efficiency if top_n is much smaller than num_cands
        if top_n < num_cands * 0.5 : # Heuristic threshold
            idx_unsorted = np.argpartition(agg_scores, -top_n)[-top_n:]
            indices = idx_unsorted[np.argsort(agg_scores[idx_unsorted])[::-1]]
        else: indices = np.argsort(agg_scores)[::-1][:top_n]

        candidates = []
        filtered = 0
        for i in indices:
            score = float(agg_scores[i])
            if score >= min_score: candidates.append({"uri": corpus_uris[i], "score": score, "method": "keyword_tfidf"})
            else: filtered += 1
        if filtered > 0: logger.debug(f"Filtered {filtered} TF-IDF candidates with score < {min_score}.")
        return candidates
    except Exception as e: logger.error(f"TF-IDF search error: {e}", exc_info=True); return []


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
        logger.info(f"{provider} Attempt {attempt + 1}/{max_retries + 1}")
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
                if resp.candidates: content = resp.text
                else: logger.warning(f"Gemini response received but has no candidates."); content = None
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
# (expand_keywords_with_llm unchanged from v33.0.22)
def expand_keywords_with_llm(concept_label: str, config: Dict, args: argparse.Namespace) -> List[str]:
    """Uses LLM to generate keywords. Reduced logging. Returns unique list including original normalized words."""
    llm_api_cfg = config.get("LLM_API_CONFIG", {})
    temp = llm_api_cfg.get("KW_EXPANSION_TEMPERATURE", KW_EXPANSION_TEMPERATURE)
    timeout = llm_api_cfg.get("REQUEST_TIMEOUT", LLM_TIMEOUT)
    retries = llm_api_cfg.get("MAX_RETRIES", LLM_MAX_RETRIES)
    sys_prompt, user_prompt = construct_keyword_expansion_prompt(concept_label)
    result = call_llm(sys_prompt, user_prompt, args.llm_model, timeout, retries, temp, args.llm_provider)
    final_keyword_terms = set()
    raw_phrases = []
    exp_diag = {}

    if result and result.get("success"):
        raw_phrases = result.get("response", {}).get("keywords", [])
        if isinstance(raw_phrases, list):
            for kw_phrase in raw_phrases:
                if isinstance(kw_phrase, str) and kw_phrase.strip():
                    normalized_phrase = normalize_concept(kw_phrase)
                    final_keyword_terms.update(term for term in normalized_phrase.split() if len(term) > 2)
            logger.info(f"[{concept_label}] LLM generated {len(final_keyword_terms)} unique keyword terms from {len(raw_phrases)} phrases.")
        else:
            logger.error(f"[{concept_label}] LLM expansion response 'keywords' field was not a list.")
            exp_diag["error"] = "LLM response format invalid"
    else:
        err_msg = result.get('error', 'N/A') if result else 'No result object'
        logger.warning(f"[{concept_label}] LLM keyword expansion failed: {err_msg}")
        exp_diag["error"] = err_msg

    original_normalized_words = set(w for w in normalize_concept(concept_label).split() if len(w) > 2)
    final_keyword_terms.update(original_normalized_words)

    return list(final_keyword_terms)


# --- START: Stage 1: Evidence Preparation (v33.0.26) ---
# --- v33.0.26 - Run 18: Tie-Breaking Sort, Concept-Specific Alpha (Aircon=0.05), Specific Filter, No Expansion, Manual Split, DampThresh=0.35, DampFactor=0.15 ---
def prepare_evidence(
        input_concept: str, concept_embedding: Optional[np.ndarray], primary_embeddings: Dict[str, np.ndarray],
        config: Dict, args: argparse.Namespace, tfidf_vectorizer: Optional[TfidfVectorizer],
        tfidf_matrix: Optional[Any], tfidf_corpus_uris: Optional[List[str]],
        keyword_label_index: Optional[Dict[str, Set[str]]], taxonomy_concepts_cache: Dict[str, Dict]
) -> Tuple[List[Dict], Dict[str, Dict], Optional[Dict], Dict[str, float], Dict[str, Any], int, int, int]:
    """
    Prepares evidence candidates using combined score with CONCEPT-SPECIFIC ALPHA OVERRIDE,
    and TIE-BREAKING based on namespace priority. Also includes:
    absolute SBERT filtering, conditional TF-IDF dampening, specific seeding/boost for 'aikido',
    disables LLM expansion for 'aikido', 'airconditioning', 'allinclusive',
    applies manual query split for 'airconditioning', and applies specific concept filters.
    v33.0.26 - Added namespace priority tie-breaking.
    Retains Concept-Specific Alpha Override (Aircon=0.05), Specific Aircon Filter, NoExp, Manual Split, DampThresh=0.35, DampFactor=0.15.
    Returns candidate details for LLM, original candidates map, anchor candidate,
            final ORIGINAL SBERT scores, expansion diagnostics, initial SBERT count,
            initial TFIDF count, and unique candidates count before ranking.
    """
    normalized_concept = normalize_concept(input_concept)
    concepts_to_log_tfidf_query = {'airconditioning', 'allinclusive'}
    concepts_to_skip_expansion = {'aikido', 'airconditioning', 'allinclusive'}

    # --- Helper for Tie-Breaking ---
    def get_sort_priority(item_uri: str) -> int:
        """Assigns a sort priority based on URI namespace. Lower is better."""
        if not isinstance(item_uri, str):
            return 99 # Lowest priority for invalid URIs

        # Highest priority for preferred namespaces/types
        if "urn:expediagroup:taxonomies:acsPCS:" in item_uri:
            return 0
        if "urn:expediagroup:taxonomies:acs:" in item_uri: # ACS attributes next
            # Penalize specific unwanted ACS concepts slightly more if needed
            if item_uri == "urn:expediagroup:taxonomies:acs:#1f3da634-0df6-4498-a8d9-603f895c8f3f": # Aircon:Unavailable
                 return 11 # Make it lower priority than default
            return 1
        if "urn:expediagroup:taxonomies:core:" in item_uri: # Core concepts
             return 2
        # General EG taxonomies catch-all
        if "urn:expediagroup:taxonomies:" in item_uri:
             return 3
        # Expedia specific taxonomies
        if "urn:expe:taxo:" in item_uri:
            # Penalize specific URIs if they cause issues across concepts
            # Example: if 'urn:expe:taxo:cars:search:filters:air-conditioning' wasn't desired
            # if item_uri == 'urn:expe:taxo:cars:search:filters:air-conditioning':
            #    return 12
            return 4
        # Add more rules if needed for other namespaces (e.g., place:, media:, events:)
        if "urn:expe:taxo:places:" in item_uri: return 5
        if "urn:expe:taxo:media-descriptors:" in item_uri: return 6
        if "urn:expe:taxo:events:" in item_uri: return 7

        return 10 # Default lower priority
    # --- End Helper ---

    # --- Get parameters from Config or use constants ---
    stage1_cfg = config.get('STAGE1_CONFIG', {})
    dampening_sbert_threshold = float(stage1_cfg.get("DAMPENING_SBERT_THRESHOLD", 0.35))
    dampening_factor = float(stage1_cfg.get("DAMPENING_FACTOR", 0.15))
    max_cands = int(stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', 75))
    min_sim = float(stage1_cfg.get('EVIDENCE_MIN_SIMILARITY', 0.30))
    min_kw = float(stage1_cfg.get('KEYWORD_MIN_SCORE', 0.05))
    kw_trigger = int(stage1_cfg.get('MIN_KEYWORD_CANDIDATES_FOR_EXPANSION', 5))
    kw_top_n = int(stage1_cfg.get('KEYWORD_TOP_N', 500))
    abs_min_sbert = float(stage1_cfg.get("ABSOLUTE_MIN_SBERT_SCORE", 0.15))
    alpha = float(config.get("COMBINED_SCORE_ALPHA", 0.6)) # Default alpha
    kw_exp_enabled = config.get("ENABLE_KW_EXPANSION", True) and args.llm_provider != "none"
    # --- END: Get parameters ---

    # --- Parameter Logging Block ---
    logger.info(f"[{normalized_concept}] Using Default Combined Score Alpha: {alpha}")
    logger.info(f"[{normalized_concept}] Applying Concept-Specific Alpha Overrides (e.g., Aircon=0.05)")
    logger.info(f"[{normalized_concept}] Using Absolute Min SBERT Threshold: {abs_min_sbert}")
    logger.info(f"[{normalized_concept}] Using Conditional Dampening: Threshold={dampening_sbert_threshold}, Factor={dampening_factor}")
    logger.info(f"[{normalized_concept}] Aikido Seed Boost enabled (Threshold: 0.80)")
    logger.info(f"[{normalized_concept}] LLM Keyword Expansion DISABLED for: {concepts_to_skip_expansion}")
    logger.info(f"[{normalized_concept}] Applying Tie-Breaking Sort (Combined Score > Namespace Priority > SBERT Score)") # Added log for tie-breaking
    # --- End Parameter Logging Block ---

    sim_scores, kw_scores = {}, {}
    exp_diag = {"attempted": False, "successful": False, "count": 0, "terms": [], "tfidf_count": 0, "error": None}

    # --- Keyword Preparation ---
    base_split_kws = set(kw for kw in normalized_concept.split() if len(kw) > 2)
    logger.debug(f"[{normalized_concept}] Base split keywords: {list(base_split_kws)}")
    query_texts_set = set(base_split_kws)
    initial_kw_count = 0
    if keyword_label_index:
        initial_kw_count = len(set().union(*[keyword_label_index.get(kw, set()) for kw in base_split_kws]))
        logger.debug(f"[{normalized_concept}] Initial simple matches (based on split words): {initial_kw_count}.")
    else:
        logger.warning(f"[{normalized_concept}] Simple keyword index unavailable.")

    needs_exp = kw_exp_enabled and \
                (initial_kw_count < kw_trigger or normalized_concept in ABSTRACT_CONCEPTS_LIST) and \
                normalized_concept not in concepts_to_skip_expansion

    expanded_kws = set()
    expansion_was_skipped_specifically = False
    if needs_exp:
        exp_diag["attempted"] = True
        logger.info(f"[{normalized_concept}] Attempting LLM keyword expansion (Trigger: initial matches {initial_kw_count} < {kw_trigger} or abstract concept, and NOT in skip list).")
        try:
            expanded_terms_list = expand_keywords_with_llm(input_concept, config, args)
            for phrase in expanded_terms_list:
                 if isinstance(phrase, str) and phrase.strip():
                     normalized_phrase = normalize_concept(phrase)
                     expanded_kws.update(term for term in normalized_phrase.split() if len(term) > 2)
            original_words_from_expansion_input = set(w for w in normalize_concept(input_concept).split() if len(w) > 2)
            newly_expanded_kws = expanded_kws - original_words_from_expansion_input
            exp_diag["successful"] = bool(newly_expanded_kws)
            if exp_diag["successful"]:
                 query_texts_set.update(newly_expanded_kws)
                 logger.info(f"[{normalized_concept}] LLM expansion added terms: {newly_expanded_kws}")
            else:
                 logger.info(f"[{normalized_concept}] LLM expansion did not add new useful keyword terms.")
                 exp_diag["error"] = exp_diag.get("error", "LLM returned no new terms")
        except Exception as llm_exp_err:
             logger.error(f"[{normalized_concept}] LLM Keyword Expansion failed: {llm_exp_err}", exc_info=True)
             exp_diag["successful"] = False
             exp_diag["error"] = str(llm_exp_err)
    else:
        if kw_exp_enabled and normalized_concept in concepts_to_skip_expansion:
            logger.info(f"[{normalized_concept}] Skipping LLM KW expansion (In skip list: {normalized_concept}).")
            exp_diag["attempted"] = True
            exp_diag["successful"] = False
            exp_diag["error"] = f"Skipped for '{normalized_concept}' (in skip list)"
            expansion_was_skipped_specifically = True
        elif not kw_exp_enabled:
             logger.info(f"[{normalized_concept}] Skipping LLM KW expansion (Globally disabled or provider='none').")
        else:
             logger.info(f"[{normalized_concept}] Skipping LLM KW expansion (Initial matches {initial_kw_count} >= {kw_trigger} and not abstract).")

    final_query_texts = list(query_texts_set)

    # --- Manual Query Split for Airconditioning ---
    if normalized_concept == 'airconditioning' and expansion_was_skipped_specifically:
        forced_query = ['air', 'conditioning']
        if set(final_query_texts) != set(forced_query):
            logger.info(f"Manually overriding query for 'airconditioning' to: {forced_query}")
            final_query_texts = forced_query
            query_texts_set = set(forced_query)
    # --- END: Manual Query Split ---

    exp_diag["count"] = len(final_query_texts)
    exp_diag["terms"] = final_query_texts

    # Log TF-IDF Query Terms
    if args.debug and normalized_concept in concepts_to_log_tfidf_query:
         logger.debug(f"[TF-IDF QUERY LOG] Concept: '{normalized_concept}', Final Query Terms ({len(final_query_texts)}): {final_query_texts}")
    elif args.debug:
         logger.debug(f"[{normalized_concept}] Using {len(final_query_texts)} keywords for TF-IDF: {final_query_texts[:10]}{'...' if len(final_query_texts)>10 else ''}")

    # --- TF-IDF Search ---
    tfidf_candidate_count_initial = 0
    if tfidf_vectorizer and tfidf_matrix is not None and tfidf_corpus_uris:
        if final_query_texts:
            kw_cands = get_candidate_concepts_keyword(final_query_texts, tfidf_vectorizer, tfidf_matrix, tfidf_corpus_uris, kw_top_n, min_kw)
            kw_scores = {c['uri']: c['score'] for c in kw_cands}
            tfidf_candidate_count_initial = len(kw_scores)
            exp_diag["tfidf_count"] = tfidf_candidate_count_initial
            logger.debug(f"[{normalized_concept}] Found {tfidf_candidate_count_initial} TF-IDF candidates >= {min_kw}.")
        else:
            logger.warning(f"[{normalized_concept}] No query keywords derived for TF-IDF search.")
            kw_scores = {}
            exp_diag["tfidf_count"] = 0
    else:
        logger.warning(f"[{normalized_concept}] TF-IDF components missing/unavailable. Skipping TF-IDF search.")
        kw_scores = {}
        exp_diag["tfidf_count"] = 0

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

    seed_uris = {
        "urn:expediagroup:taxonomies:activities:#c7b903d8-14be-4bcd-bcb3-d6d35724e7cc",
        "urn:expediagroup:taxonomies:core:#f02b51fe-eabe-4dab-961a-456bed9664e8"
    }

    if normalized_concept == 'aikido':
        added_seeds = seed_uris - all_uris_set
        if added_seeds:
            logger.info(f"SEEDING 'aikido' with {len(added_seeds)} missing relevant URIs: {added_seeds}")
            all_uris_set.update(added_seeds)
            # Calculate SBERT for newly added seeds if not already done
            uris_needing_sbert_calc = added_seeds - set(sim_scores.keys())
            if uris_needing_sbert_calc and concept_embedding is not None:
                 seed_embeddings = {uri: primary_embeddings[uri] for uri in uris_needing_sbert_calc if uri in primary_embeddings}
                 if seed_embeddings:
                     new_sim_scores = get_batch_embedding_similarity(concept_embedding, seed_embeddings)
                     sim_scores.update(new_sim_scores)
                     logger.debug(f"Calculated SBERT for {len(new_sim_scores)} newly added seeds.")
                 else:
                     logger.warning(f"Could not find embeddings for newly added seeds: {uris_needing_sbert_calc}")


    # --- Apply Concept-Specific Filters ---
    concept_specific_filters = {
        'airconditioning': [
            "urn:expe:taxo:data-element-values:flights:frequent-flyer-program-names:AP", # Air One Qualiflyer
            "urn:expediagroup:taxonomies:acs:#1f3da634-0df6-4498-a8d9-603f895c8f3f"      # AirConditioning:Unavailable
        ]
    }
    uris_to_filter_for_concept = concept_specific_filters.get(normalized_concept, [])
    if uris_to_filter_for_concept:
        original_count = len(all_uris_set)
        uris_to_remove = set()
        for uri_to_filter in uris_to_filter_for_concept:
            if uri_to_filter in all_uris_set:
                label = get_primary_label(uri_to_filter, taxonomy_concepts_cache, fallback=uri_to_filter)
                logger.info(f"Applying specific filter for '{normalized_concept}': Removing {uri_to_filter} ('{label}').")
                uris_to_remove.add(uri_to_filter)
            elif args.debug:
                 logger.debug(f"Specific filter URI {uri_to_filter} not found in candidate set for '{normalized_concept}'.")

        if uris_to_remove:
             all_uris_set.difference_update(uris_to_remove)
             logger.info(f"Removed {len(uris_to_remove)} URIs via concept-specific filter for '{normalized_concept}'. New count: {len(all_uris_set)}")
    # --- END: Apply Concept-Specific Filters ---

    all_uris_list = list(all_uris_set)
    unique_candidates_before_ranking = len(all_uris_list)
    if initial_unique_count != unique_candidates_before_ranking and len(uris_to_filter_for_concept) == 0:
        logger.debug(f"[{normalized_concept}] Unique candidates changed from {initial_unique_count} to {unique_candidates_before_ranking} after seeding.")

    if not all_uris_list:
        logger.warning(f"[{normalized_concept}] No candidates found from TF-IDF/SBERT/Seeds after filtering.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, 0

    all_details = get_kg_data(all_uris_list, taxonomy_concepts_cache)

    scored_list: List[Dict] = []
    abs_sbert_filtered_count = 0
    dampened_count = 0
    alpha_override_applied = False # Flag if override applied for this concept

    # --- Concept-Specific Alpha Overrides ---
    concept_alpha_overrides = {
        #'airconditioning': 0.05,
    }
    # --- END: Concept-Specific Alpha Overrides ---


    for i, uri in enumerate(all_uris_list):
        if uri not in all_details:
            logger.warning(f"[{normalized_concept}] Details not found for candidate URI: {uri}. Skipping.")
            continue

        s_score = sim_scores.get(uri, 0.0)
        k_score = kw_scores.get(uri, 0.0)
        pref_label_for_log = all_details[uri].get('prefLabel', 'N/A')

        is_aikido_seed_uri = (normalized_concept == 'aikido' and uri in seed_uris)

        # --- Aikido Seed Score Boost ---
        original_s_score_before_boost = s_score
        boosted_s_score_for_calc = s_score
        boost_applied = False
        if is_aikido_seed_uri:
            boost_threshold = 0.80
            if s_score < boost_threshold:
                boosted_s_score_for_calc = boost_threshold
                boost_applied = True
                logger.info(f"[Aikido Seed BOOST] Boosting SBERT for {uri} ('{pref_label_for_log}') from {original_s_score_before_boost:.4f} to {boosted_s_score_for_calc:.4f}")

        # ** Absolute SBERT Filter **
        if boosted_s_score_for_calc < abs_min_sbert:
            if args.debug:
                 filter_reason = f"SBERT {boosted_s_score_for_calc:.4f} < {abs_min_sbert:.4f}"
                 prefix = "[Aikido Seed] FILTERED seed " if is_aikido_seed_uri else "Excluding "
                 logger.debug(f"{prefix}{uri} ('{pref_label_for_log}') due to {filter_reason}. (TFIDF: {k_score:.4f})")
            if is_aikido_seed_uri:
                 logger.warning(f"[Aikido Seed] FILTERED seed URI {uri} ('{pref_label_for_log}') due to low SBERT {boosted_s_score_for_calc:.4f} (after potential boost).")
            abs_sbert_filtered_count += 1
            continue

        # *** Conditional TF-IDF Dampening ***
        dampened_k_score = k_score
        was_dampened = False
        if boosted_s_score_for_calc < dampening_sbert_threshold:
            dampened_k_score *= dampening_factor
            if k_score > 0 and abs(k_score - dampened_k_score) > 1e-9:
                 was_dampened = True
                 if args.debug: logger.debug(f"Dampening TF-IDF for {uri} ('{pref_label_for_log}'): SBERT={boosted_s_score_for_calc:.4f} < {dampening_sbert_threshold:.4f}. TF-IDF {k_score:.4f} -> {dampened_k_score:.4f} (Factor={dampening_factor}).")
                 dampened_count += 1

        # *** Concept-Specific Alpha Override ***
        effective_alpha = alpha
        alpha_was_overridden = False
        if normalized_concept in concept_alpha_overrides:
            new_alpha = concept_alpha_overrides[normalized_concept]
            if abs(new_alpha - effective_alpha) > 1e-9:
                 if args.debug and i == 0:
                     logger.debug(f"CONCEPT ALPHA OVERRIDE: For '{normalized_concept}', changing alpha from {effective_alpha:.2f} to {new_alpha:.2f} for all candidates.")
                 effective_alpha = new_alpha
                 alpha_was_overridden = True
                 if i == 0:
                     alpha_override_applied = True

        # Normalize scores
        norm_s = max(0.0, min(1.0, boosted_s_score_for_calc))
        norm_k = max(0.0, min(1.0, dampened_k_score))

        # Calculate combined score using EFFECTIVE alpha
        combined = (effective_alpha * norm_k) + ((1.0 - effective_alpha) * norm_s)

        scored_list.append({
            "uri": uri,
            "details": all_details[uri],
            "sim_score": original_s_score_before_boost,
            "boosted_sim_score": boosted_s_score_for_calc if boost_applied else None,
            "kw_score": k_score,
            "dampened_kw_score": dampened_k_score if was_dampened else None,
            "alpha_overridden": alpha_was_overridden,
            "effective_alpha": effective_alpha,
            "combined_score": combined
        })

    logger.info(f"[{normalized_concept}] Excluded {abs_sbert_filtered_count} candidates (SBERT < {abs_min_sbert} after potential boost).")
    logger.info(f"[{normalized_concept}] Conditionally dampened TF-IDF (Factor={dampening_factor}) for {dampened_count} candidates (SBERT < {dampening_sbert_threshold}).")
    if alpha_override_applied:
         override_val = concept_alpha_overrides.get(normalized_concept, 'N/A')
         log_val_str = f"{override_val:.2f}" if isinstance(override_val, (int, float)) else str(override_val)
         logger.info(f"[{normalized_concept}] Applied CONCEPT-SPECIFIC ALPHA OVERRIDE (Value: {log_val_str}).")

    if not scored_list:
        logger.warning(f"[{normalized_concept}] No candidates remaining after filtering/dampening.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, unique_candidates_before_ranking

    # --- Sorting with Tie-breaking ---
    logger.debug(f"[{normalized_concept}] Sorting {len(scored_list)} candidates with tie-breaking (Priority: Combined Score > Namespace > SBERT)...")

    # 1. Fallback sort: Original SBERT score (descending)
    scored_list.sort(key=lambda x: x.get('sim_score', 0.0), reverse=True)

    # 2. Secondary sort: Namespace priority (ascending - lower priority value first)
    #    Uses the helper function get_sort_priority defined above
    scored_list.sort(key=lambda x: get_sort_priority(x.get('uri', '')), reverse=False)

    # 3. Primary sort: Combined score (descending)
    scored_list.sort(key=lambda x: x['combined_score'], reverse=True)
    # --- End Sorting ---


    # Debug logging for Top 5 (Include priority for clarity)
    if args.debug:
        logger.debug(f"--- Top 5 Candidates for '{normalized_concept}' AFTER Sorting (Tie-Breaking Enabled: Score>Prio>SBERT) ---")
        logger.debug(f"    (DefaultAlpha={alpha}, ConceptOverrides={concept_alpha_overrides.get(normalized_concept,'None')}, DampThresh={dampening_sbert_threshold}, DampFactor={dampening_factor})")
        for i, c in enumerate(scored_list[:5]):
            damp_info = f"(Damp TFIDF: {c.get('dampened_kw_score'):.4f})" if c.get('dampened_kw_score') is not None else ""
            boost_info = f"(Boost SBERT: {c.get('boosted_sim_score'):.4f})" if c.get('boosted_sim_score') is not None else ""
            alpha_info = f"(EffAlpha: {c.get('effective_alpha', alpha):.2f})"
            prio_info = f"(Prio: {get_sort_priority(c.get('uri',''))})" # Show priority
            label = c.get('details', {}).get('prefLabel', 'N/A')
            logger.debug(f"{i+1}. URI: {c.get('uri', 'MISSING')} {prio_info} Label: '{label}' "
                         f"Combined: {c.get('combined_score', 0.0):.6f} {alpha_info} "
                         f"(Orig SBERT: {c.get('sim_score', 0.0):.4f} {boost_info}, Raw TFIDF: {c.get('kw_score', 0.0):.4f} {damp_info})")
        logger.debug("--- End Top 5 ---")

    selected = scored_list[:max_cands]
    logger.info(f"[{normalized_concept}] Selected top {len(selected)}/{len(scored_list)} candidates based on combined score and tie-breaking.")

    if not selected:
        logger.warning(f"[{normalized_concept}] No candidates selected after scoring and filtering/dampening.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, unique_candidates_before_ranking

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
                  "keyword_score": c.get("kw_score", 0.0),
                  "combined_score": c.get("combined_score", 0.0),
                  "effective_alpha": c.get("effective_alpha", alpha)
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
             # Fallback construction...
             if anchor_data.get('details'):
                 anchor = {
                     **anchor_data.get('details'),
                     "sbert_score": anchor_data.get("sim_score", 0.0),
                     "keyword_score": anchor_data.get("kw_score", 0.0),
                     "combined_score": anchor_data.get("combined_score", 0.0),
                     "effective_alpha": anchor_data.get("effective_alpha", alpha)
                 }
                 if anchor_data.get("boosted_sim_score") is not None: anchor["boosted_sbert_score"] = anchor_data.get("boosted_sim_score")
                 if anchor_data.get("dampened_kw_score") is not None: anchor["dampened_keyword_score"] = anchor_data.get("dampened_kw_score")
                 logger.warning(f"[{normalized_concept}] Using fallback anchor constructed from scored_list data.")
             else:
                 logger.error(f"[{normalized_concept}] Fallback anchor construction failed - missing details in scored_list top item.")
    else:
        logger.error(f"[{normalized_concept}] Top candidate in 'selected' list has no URI! Data: {anchor_data}")

    # Debug Sanity Check (remains the same)
    if args.debug:
        if anchor and 'uri' in anchor:
            if anchor['uri'] == anchor_uri_from_data:
                logger.debug(f"DEBUG SANITY CHECK [{normalized_concept}]: Anchor URI lookup successful.")
            else:
                logger.error(f"DEBUG SANITY CHECK [{normalized_concept}]: MISMATCH! Anchor URI lookup returned wrong item. Top Scored URI ({anchor_uri_from_data}) != Looked-up Anchor URI ({anchor['uri']})")
        elif anchor_uri_from_data:
            logger.error(f"DEBUG SANITY CHECK [{normalized_concept}]: Anchor URI lookup FAILED. Top Scored URI ({anchor_uri_from_data}) not found in orig_map.")
        else:
             logger.error(f"DEBUG SANITY CHECK [{normalized_concept}]: Cannot perform check because Top Scored URI is missing.")

    # Anchor Logging (remains the same)
    if anchor:
        damp_info_log = f"(Damp TFIDF: {anchor_data.get('dampened_kw_score'):.4f})" if anchor_data.get('dampened_kw_score') is not None else ""
        boost_info_log = f"(Boost SBERT: {anchor_data.get('boosted_sim_score'):.4f})" if anchor_data.get('boosted_sim_score') is not None else ""
        alpha_log = f"(EffAlpha: {anchor_data.get('effective_alpha', alpha):.2f})"
        logger.info(f"[{normalized_concept}] Anchor selected: {anchor.get('prefLabel', anchor.get('uri', 'MISSING URI'))} "
                    f"(Combined: {anchor_data.get('combined_score', 0.0):.6f} {alpha_log}, "
                    f"Orig SBERT: {anchor_data.get('sim_score', 0.0):.4f} {boost_info_log}, "
                    f"Raw TFIDF: {anchor_data.get('kw_score', 0.0):.4f} {damp_info_log})")
    else:
        logger.warning(f"[{normalized_concept}] Could not determine anchor details. Anchor will be null.")
        if anchor_uri_from_data:
             logger.warning(f"    Reason: URI '{anchor_uri_from_data}' (from top candidate) was not found in orig_map and fallback failed.")
        else:
             logger.warning(f"    Reason: Top candidate dictionary lacked a URI.")

    # Return values
    sbert_scores_final = {uri: d["sbert_score"] for uri, d in orig_map.items()}
    return llm_details, orig_map, anchor, sbert_scores_final, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, unique_candidates_before_ranking
# --- END: Stage 1: Evidence Preparation (v33.0.26) ---


# --- Stage 2: Finalization ---
# (apply_rules_and_finalize unchanged from v33.0.22/v33.0.24)
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
    """
    start = time.time(); norm_concept = normalize_concept(input_concept)
    output: Dict[str, Any] = {"applicable_lodging_types": "Both", "travel_category": travel_category or {"uri": None, "name": input_concept, "type": "Unknown"}, "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [], "must_not_have": [], "failed_fallback_themes": {}, "diagnostics": {"theme_processing": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "final_output": {}, "stage2": {"status": "Started", "duration_seconds": 0.0, "error": None}}}
    theme_diag = output["diagnostics"]["theme_processing"]; reprompt_diag = output["diagnostics"]["reprompting_fallback"]
    base_themes = config.get("base_themes", {}); overrides = config.get("concept_overrides", {}).get(norm_concept, {})
    final_cfg = config.get('STAGE2_CONFIG', {}); min_weight = float(final_cfg.get('THEME_ATTRIBUTE_MIN_WEIGHT', THEME_ATTRIBUTE_MIN_WEIGHT)); top_n_attrs = int(final_cfg.get('TOP_N_DEFINING_ATTRIBUTES', TOP_N_DEFINING_ATTRIBUTES_DEFAULT))

    llm_assigns: Dict[str, List[str]] = {}; diag_val = {}
    if llm_call_result and llm_call_result.get("success"):
        validated = validate_llm_assignments(llm_call_result.get("response"), set(original_candidates_map_for_reprompt.keys()), set(base_themes.keys()), norm_concept, diag_val)
        if validated is not None: llm_assigns = validated; logger.debug(f"[{norm_concept}] Validated LLM assignments.")
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
        rule, _, _, _ = get_dynamic_theme_config(norm_concept, name, config)
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
                    logger.debug(f"[{norm_concept}] Added {added} unique URIs via fallback for '{name}'.")
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
        logger.debug(f"[{norm_concept}][{name}] URIs:{len(uris)}, Combined Score Sum:{total_score:.4f}, Theme Weight:{theme_w:.4f}")

        theme_attrs = []
        if uris and theme_w > 0:
            n_uris = len(uris)
            if total_score < 1e-9 and n_uris > 0:
                logger.warning(f"[{norm_concept}][{name}] Zero combined scores for assigned candidates. Using equal weight distribution for {n_uris} URIs.")
                eq_w = (theme_w / n_uris)
                if eq_w >= min_weight:
                    for u in uris:
                        if u not in original_candidates_map_for_reprompt: continue
                        d = original_candidates_map_for_reprompt[u]; is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                        attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(eq_w, 6), "type": d.get('type_labels', [])}
                        if is_fb: attr["comment"] = "Fallback Assignment"
                        theme_attrs.append(attr); all_final_attrs.append(attr)
                else: logger.warning(f"[{norm_concept}][{name}] Equal weight {eq_w:.6f} < {min_weight}, skipping attributes for this theme.")
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

    # --- Apply Final Concept Overrides ---
    output["applicable_lodging_types"] = overrides.get("lodging_type", "Both")
    tc = output["travel_category"]
    if isinstance(tc, dict):
        tc["type"] = overrides.get("category_type", tc.get("type", "Unknown"))
        tc["exclusionary_concepts"] = overrides.get("exclusionary_concepts", [])
    else: logger.error(f"[{norm_concept}] Travel category structure invalid during override application?"); output["travel_category"] = {"uri": None, "name": norm_concept, "type": overrides.get("category_type", "Uncategorized"), "exclusionary_concepts": overrides.get("exclusionary_concepts", [])}
    mnh = overrides.get("must_not_have", []); mnh_uris = set(i["uri"] for i in mnh if isinstance(i, dict) and "uri" in i) if isinstance(mnh, list) else set()
    output["must_not_have"] = [{"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "scope": None} for u in sorted(list(mnh_uris))]
    add_scores = overrides.get("additional_subscores", []); output["additional_relevant_subscores"] = add_scores if isinstance(add_scores, list) else []

    # --- Final Diagnostic Counts ---
    final_diag = output["diagnostics"]["final_output"]
    final_diag["must_not_have_count"] = len(output["must_not_have"]); final_diag["additional_subscores_count"] = len(output["additional_relevant_subscores"])
    final_diag["themes_count"] = len(output["themes"]); output["failed_fallback_themes"] = { n: r for n, r in failed_rules.items() if n not in fixed }
    final_diag["failed_fallback_themes_count"] = len(output["failed_fallback_themes"]); final_diag["top_defining_attributes_count"] = len(output['top_defining_attributes'])
    output["diagnostics"]["stage2"]["status"] = "Completed"; output["diagnostics"]["stage2"]["duration_seconds"] = round(time.time() - start, 2)
    return output

# --- Main Processing Loop ---
# (generate_affinity_definitions_loop - CORRECTED for get_concept_embedding call)
def generate_affinity_definitions_loop(
    concepts_to_process: List[str], config: Dict, args: argparse.Namespace,
    sbert_model: SentenceTransformer, primary_embeddings_map: Dict[str, np.ndarray],
    taxonomy_concepts_cache: Dict[str, Dict], keyword_label_index: Optional[Dict[str, Set[str]]],
    tfidf_vectorizer: Optional[TfidfVectorizer], tfidf_matrix: Optional[Any],
    tfidf_corpus_uris: Optional[List[str]]
) -> List[Dict]:
    """ Main loop processing each concept. Reduced logging."""
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
                "stage1": { "status": "Not Started", "error": None, "selection_method": "Combined TFIDF+SBERT", "expansion": {}, "sbert_candidate_count_initial": 0, "tfidf_candidate_count_initial": 0, "unique_candidates_before_ranking": 0, "llm_candidate_count": 0 },
                "llm_slotting": {"status": "Not Started", "error": None},
                "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0},
                "stage2": {"status": "Not Started", "error": None},
                "theme_processing": {}, "final_output": {}, "error_details": None
            }
        }
        diag1 = affinity_def["diagnostics"]["stage1"]; diag_llm = affinity_def["diagnostics"]["llm_slotting"]

        try:
            # Use the function defined within the script or imported
            concept_emb = get_concept_embedding(norm_concept, sbert_model)
            if concept_emb is None:
                logger.error(f"[{norm_concept}] Embedding failed."); diag1["error"] = "Embedding failed"; diag1["status"] = "Failed"

            stage1_start = time.time()
            cand_details, orig_map, anchor, sbert_scores, exp_diag, sbert_init_count, tfidf_init_count, unique_count = prepare_evidence(
                concept, concept_emb, primary_embeddings_map, config, args,
                _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris,
                _keyword_label_index, _taxonomy_concepts_cache
            )
            stage1_dur = time.time() - stage1_start

            diag1["expansion"] = exp_diag
            diag1.update({
                "status": "Completed" if diag1.get("error") is None and concept_emb is not None else "Failed",
                "duration_seconds": round(stage1_dur, 2),
                "sbert_candidate_count_initial": sbert_init_count,
                "tfidf_candidate_count_initial": tfidf_init_count,
                "unique_candidates_before_ranking": unique_count,
                "llm_candidate_count": len(cand_details)
                })
            logger.info(f"[{norm_concept}] Stage 1 done ({stage1_dur:.2f}s). Status: {diag1['status']}. LLM Cands: {len(cand_details)}. Initial SBERT: {sbert_init_count}, TFIDF: {tfidf_init_count}.")

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
    global _config_data, _taxonomy_concepts_cache, _taxonomy_embeddings_cache, \
           _keyword_label_index, _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris, args # Declare args global

    parser = argparse.ArgumentParser(description=f"Generate Travel Concept Affinity Definitions ({SCRIPT_VERSION})")
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
    args = parser.parse_args() # Assign parsed args to the global variable

    config = load_affinity_config(args.config)
    if config is None: sys.exit(1)
    _config_data = config

    # Ensure necessary params are loaded or defaulted
    config["COMBINED_SCORE_ALPHA"] = config.get("COMBINED_SCORE_ALPHA", COMBINED_SCORE_ALPHA)
    config.setdefault("STAGE1_CONFIG", {})
    config["STAGE1_CONFIG"]["ABSOLUTE_MIN_SBERT_SCORE"] = config["STAGE1_CONFIG"].get("ABSOLUTE_MIN_SBERT_SCORE", ABSOLUTE_MIN_SBERT_SCORE)
    config["STAGE1_CONFIG"]["DAMPENING_SBERT_THRESHOLD"] = config["STAGE1_CONFIG"].get("DAMPENING_SBERT_THRESHOLD", DAMPENING_SBERT_THRESHOLD)
    config["STAGE1_CONFIG"]["DAMPENING_FACTOR"] = config["STAGE1_CONFIG"].get("DAMPENING_FACTOR", DAMPENING_FACTOR)

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
    logger.info(f"ALPHA MECHANISM: Concept-Specific Alpha Override (e.g., Aircon=0.05)")
    logger.info(f"SORTING: Tie-Breaking Enabled (Score > Namespace Prio > SBERT)")
    logger.info(f"TF-IDF STRATEGY: Weighted Field Concatenation (prefLabel x5, altLabel x3, definition x1)") # Added Log
    logger.info(f"ABSOLUTE_MIN_SBERT_SCORE: {config['STAGE1_CONFIG']['ABSOLUTE_MIN_SBERT_SCORE']}")
    logger.info(f"DAMPENING_SBERT_THRESHOLD: {config['STAGE1_CONFIG']['DAMPENING_SBERT_THRESHOLD']}")
    logger.info(f"DAMPENING_FACTOR: {config['STAGE1_CONFIG']['DAMPENING_FACTOR']}")
    logger.info(f"ENABLE_KW_EXPANSION (Default): {config.get('ENABLE_KW_EXPANSION', ENABLE_KW_EXPANSION)}")
    tfidf_enabled = config.get("TFIDF_CONFIG", {}).get("enabled", False)
    logger.info(f"TF-IDF Enabled: {tfidf_enabled}")
    if tfidf_enabled:
         tfidf_params_log = config.get("TFIDF_CONFIG", {}).get("TFIDF_VECTORIZER_PARAMS", config.get("TFIDF_VECTORIZER_PARAMS", {}))
         logger.info(f"TF-IDF N-grams: Unigrams (ngram_range removed/default). Config Params: {tfidf_params_log}")
    logger.info(f"Specific Handling: Aikido (Seed/NoExp/Boost), Aircon/AllInc (NoExp), Aircon (Manual Query Split), Aircon (Specific Filters)")
    logger.info(f"--- End Key Parameters ---")

    if args.llm_provider is None: args.llm_provider = config.get('LLM_PROVIDER', 'none')
    else: logger.warning(f"Overriding LLM provider from command line: '{args.llm_provider}'")
    if args.llm_model is None: args.llm_model = config.get('LLM_MODEL', None)
    else: logger.warning(f"Overriding LLM model from command line: '{args.llm_model}'")
    logger.info(f"Effective LLM Provider: {args.llm_provider}, Model: {args.llm_model or 'N/A'}")

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
        # build_keyword_index now uses global 'args' for debug_mode check
        build_keyword_index(config, _taxonomy_concepts_cache, args.cache_dir, cache_ver, args.rebuild_cache)
        if _tfidf_vectorizer is not None: logger.info(f"TF-IDF Index ready.")
        elif config.get("TFIDF_CONFIG", {}).get("enabled", False): logger.warning("TF-IDF is enabled in config but index failed to load/build.")
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
    results = generate_affinity_definitions_loop(input_concepts, config, args, sbert_model, primary_embs, _taxonomy_concepts_cache, _keyword_label_index, _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris)
    end_loop = time.time(); logger.info(f"Generation loop finished ({end_loop - start_loop:.2f}s). Generated {len(results)} definitions.")

    if results: save_results_json(results, out_file)
    else: logger.warning("No results were generated to save.")
    logger.info(f"Script finished. Final log messages in: {log_file}");

# --- Standard Python Entry Point ---
if __name__ == "__main__":
    if not all([UTILS_ST_AVAILABLE, UTILS_SK_AVAILABLE, RDFLIB_AVAILABLE]):
         print("CRITICAL: Required libraries (SentenceTransformers, Scikit-learn, RDFlib) are unavailable. Check imports in utils.py and main script.", file=sys.stderr); sys.exit(1)
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