#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate affinity definitions for travel concepts using combined scoring (v33.0.15).

Implements:
- Added ABSOLUTE_MIN_SBERT_SCORE = 0.15 constant.
- In prepare_evidence:
- Added loading of abs_min_sbert from stage1_cfg with fallback to the constant.
- Added logging of the threshold value being used.
- Inside the loop iterating all_uris:
- Added an if s_score < abs_min_sbert: check.
- If the check is true, the candidate is skipped using continue, and a counter abs_sbert_filtered_count is incremented. Debug logging is included.
- Added a log message after the loop indicating how many candidates were filtered by this new check.
- Added checks for scored_list being empty after filtering.
- Updated docstring and initial log message to mention the absolute SBERT filter.
- Script version mentioned in docstring updated to v33.0.15
- Combined TF-IDF Keyword + SBERT Similarity score for candidate selection.
- TF-IDF indexing and search functionality.
- Optional LLM keyword expansion for weak concepts (with enhanced prompt).
- LLM-assisted theme slotting with prompt enhancements (v33.0.15).
- Relies on shared utility functions in utils.py.
- Includes mandatory splitting of input concepts for TF-IDF keyword search.
- Corrected diagnostic reporting for Stage 1 counts.
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
from typing import List, Dict, Optional, Tuple, Any, Set, Union

# --- Third-Party Imports ---
try:
    import numpy as np
except ImportError: print("CRITICAL ERROR: numpy not found.", file=sys.stderr); sys.exit(1)

try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF
    from rdflib import util as rdflib_util
except ImportError: pass # Checked via utils import

try:
    from sentence_transformers import SentenceTransformer
except ImportError: pass # Checked via utils import

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError: pass # Checked via utils import

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found.", file=sys.stderr) # Print on one line
    def tqdm(iterable, *args, **kwargs): return iterable # Define dummy on the next line

# --- LLM Imports & Placeholders ---
# Define placeholders FIRST
class APITimeoutError(Exception): pass
class APIConnectionError(Exception): pass
class RateLimitError(Exception): pass
class DummyLLMClient: pass

# Assign placeholder types and availability flags
OpenAI_Type = DummyLLMClient
GoogleAI_Type = DummyLLMClient
OpenAI = None
OPENAI_AVAILABLE = False
genai = None
GOOGLE_AI_AVAILABLE = False

# --- Try importing real LLM libraries ---
try:
    from openai import OpenAI as RealOpenAI, APITimeoutError as RealAPITimeoutError, APIConnectionError as RealAPIConnectionError, RateLimitError as RealRateLimitError
    OpenAI = RealOpenAI; OpenAI_Type = RealOpenAI
    APITimeoutError = RealAPITimeoutError # Overwrite placeholder
    APIConnectionError = RealAPIConnectionError # Overwrite placeholder
    RateLimitError = RealRateLimitError # Overwrite placeholder
    OPENAI_AVAILABLE = True; logging.debug("Imported OpenAI library.")
except ImportError: logging.warning("OpenAI library not found.")
except Exception as e: logging.error(f"OpenAI import error: {e}")

try:
    import google.generativeai as genai_import
    genai = genai_import; GoogleAI_Type = genai_import.GenerativeModel
    GOOGLE_AI_AVAILABLE = True; logging.debug("Imported google.generativeai library.")
except ImportError: logging.warning("google.generativeai library not found.")
except Exception as e: GOOGLE_AI_AVAILABLE = False; genai = None; logging.error(f"Google AI import error: {e}")


# --- Utility Function Imports ---
try:
    from utils import (
        setup_logging, normalize_concept, get_primary_label, get_concept_type_labels,
        get_sbert_model, load_affinity_config, get_cache_filename, load_cache, save_cache,
        load_taxonomy_concepts, precompute_taxonomy_embeddings, get_concept_embedding,
        get_batch_embedding_similarity, get_kg_data,
        build_keyword_label_index, # Use correct name
        save_results_json,
        RDFLIB_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE as UTILS_ST_AVAILABLE,
        SKLEARN_AVAILABLE as UTILS_SK_AVAILABLE
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Failed import from 'utils.py': {e}", file=sys.stderr); sys.exit(1)
except Exception as e:
     print(f"CRITICAL ERROR: Unexpected error importing from 'utils.py': {e}", file=sys.stderr); sys.exit(1)

# --- Config Defaults ---
SCRIPT_VERSION = "affinity-rule-engine-v33.0.15 (ABSOLUTE_MIN_SBERT_SCORE = 0.15 constant and Added loading of abs_min_sbert from stage1_cfg)" # Updated version marker
DEFAULT_CACHE_VERSION = "v20250422.affinity.33.0.13.1" # Update cache version if desired
DEFAULT_TAXONOMY_DIR = "./datasources/"
DEFAULT_CACHE_DIR = "./cache_v33/"
DEFAULT_CONFIG_FILE = "./affinity_config_v33.0.json"
OUTPUT_FILE_TEMPLATE = "affinity_definitions_{cache_version}.json"
LOG_FILE_TEMPLATE = "affinity_generation_{cache_version}.log"
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
COMBINED_SCORE_ALPHA = 0.6
KEYWORD_TOP_N_SEARCH = 500
ENABLE_KW_EXPANSION = True
KW_EXPANSION_TEMPERATURE = 0.5
TOP_N_DEFINING_ATTRIBUTES_DEFAULT = 25
ABSOLUTE_MIN_SBERT_SCORE = 0.15


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

# --- Helper Functions Retained in Main Script ---

def get_theme_definition_for_prompt(theme_name: str, theme_data: Dict) -> str:
    """Gets a theme description suitable for an LLM prompt."""
    desc = theme_data.get("description")
    if isinstance(desc, str) and desc.strip(): return desc.strip()
    logger.warning(f"Theme '{theme_name}' missing description. Using fallback.")
    return f"Theme related to {theme_name} ({theme_data.get('type', 'general')} aspects)."

def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: Dict) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    """Gets theme config considering base and concept-specific overrides."""
    base_themes = config.get("base_themes", {}); overrides = config.get("concept_overrides", {})
    base_data = base_themes.get(theme_name)
    if not base_data: logger.error(f"Base theme '{theme_name}' not found!"); return "Optional", 0.0, None, None
    concept_override = overrides.get(normalized_concept, {})
    theme_override = concept_override.get("theme_overrides", {}).get(theme_name, {})
    merged = {**base_data, **theme_override}
    rule = merged.get("rule_applied", merged.get("rule", "Optional")); rule = "Optional" if rule not in ["Must have 1", "Optional"] else rule
    weight = merged.get("weight", 0.0); weight = 0.0 if not isinstance(weight, (int, float)) or weight < 0 else float(weight)
    return rule, weight, merged.get("subScore"), merged.get("fallback_logic")

def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a dictionary of weights to sum to 1.0."""
    if not weights_dict: return {}
    total = sum(weights_dict.values()) # Calculate total first
    if total <= 0: return {k: 0.0 for k in weights_dict} # Handle non-positive total
    return {k: v / total for k, v in weights_dict.items()}

def deduplicate_attributes(attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates attribute list, keeping entry with highest weight for each URI."""
    best: Dict[str, Dict[str, Any]] = {};
    for attr in attributes:
        uri = attr.get("uri");
        if not uri or not isinstance(uri, str): continue
        try: weight = float(attr.get("concept_weight", 0.0))
        except (ValueError, TypeError): weight = 0.0
        current = best.get(uri);
        try: current_weight = float(current.get("concept_weight", -1.0)) if current else -1.0
        except (ValueError, TypeError): current_weight = -1.0
        if current is None or weight > current_weight: best[uri] = attr
    return list(best.values())

def validate_llm_assignments(llm_response_data: Optional[Dict[str, Any]], uris_sent: Set[str], valid_themes: Set[str], concept_label: str, diag_llm: Dict) -> Optional[Dict[str, List[str]]]:
     """Validates the structure and content of the LLM theme assignments."""
     if not llm_response_data or "theme_assignments" not in llm_response_data: logger.error(f"[{concept_label}] LLM response missing key."); diag_llm["error"] = "Missing 'theme_assignments'."; return None
     assigns = llm_response_data["theme_assignments"]
     if not isinstance(assigns, dict): logger.error(f"[{concept_label}] LLM assignments not dict."); diag_llm["error"] = "Assignments structure not dict."; return None
     validated: Dict[str, List[str]] = {}; uris_resp = set(assigns.keys()); extra = uris_resp - uris_sent; missing = uris_sent - uris_resp
     if extra: logger.warning(f"[{concept_label}] LLM returned {len(extra)} extra URIs. Ignoring.")
     if missing: logger.warning(f"[{concept_label}] LLM response missing {len(missing)} URIs. Adding empty."); [assigns.setdefault(uri, []) for uri in missing]
     parsed_count = 0; uris_proc = 0
     for uri, themes in assigns.items():
         if uri not in uris_sent: continue
         uris_proc += 1
         if not isinstance(themes, list): logger.warning(f"[{concept_label}] Invalid themes format for {uri}."); validated[uri] = []; continue
         valid = [t for t in themes if isinstance(t, str) and t in valid_themes]; invalid = set(themes) - set(valid)
         if invalid: logger.warning(f"[{concept_label}] Invalid themes ignored for {uri}: {invalid}")
         validated[uri] = valid; parsed_count += len(valid)
     diag_llm["assignments_parsed_count"] = parsed_count; diag_llm["uris_in_response_count"] = uris_proc
     return validated

# --- TF-IDF Indexing & Search Functions ---

def build_keyword_index(
    config: Dict, taxonomy_concepts_cache: Dict[str, Dict], cache_dir: str,
    cache_version: str, rebuild_cache: bool
) -> Tuple[Optional[TfidfVectorizer], Optional[Any], Optional[List[str]], Optional[Dict[str, Set[str]]]]:
    """ Builds or loads TF-IDF index and simple label index. """
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris, _keyword_label_index
    logger.info("Building/Loading keyword indexes (TF-IDF & Simple Label)...")
    vectorizer, tfidf_matrix, corpus_uris, label_index = None, None, None, None
    tfidf_cfg = config.get("TFIDF_CONFIG", {}); use_tfidf = tfidf_cfg.get("enabled", False) and UTILS_SK_AVAILABLE # Use flag from utils

    if use_tfidf:
        logger.info("TF-IDF Indexing enabled.")
        # --- Consider setting min_df=1 here if needed for rare terms like 'aikido' ---
        tfidf_params_cfg = config.get("TFIDF_VECTORIZER_PARAMS", {})
        tfidf_params = {"max_df": 0.95, "min_df": 2, "stop_words": "english", **tfidf_params_cfg} # Example: min_df=1
        logger.info(f"Using TF-IDF Vectorizer Params: {tfidf_params}")

        matrix_cache_file = get_cache_filename("tfidf_matrix", cache_version, cache_dir, tfidf_params, ".pkl")
        vec_cache_file = get_cache_filename("tfidf_vectorizer", cache_version, cache_dir, tfidf_params, ".pkl")
        uris_cache_file = get_cache_filename("tfidf_corpus_uris", cache_version, cache_dir, tfidf_params, ".pkl")

        cache_valid = False
        if not rebuild_cache:
            logger.info(f"Attempting load TF-IDF cache (Version: {cache_version})...")
            cached_matrix = load_cache(matrix_cache_file, 'pickle'); cached_vec = load_cache(vec_cache_file, 'pickle'); cached_uris = load_cache(uris_cache_file, 'pickle')
            if cached_matrix is not None and cached_vec is not None and isinstance(cached_uris, list):
                if hasattr(cached_matrix, 'shape') and cached_matrix.shape[0] == len(cached_uris):
                    tfidf_matrix, vectorizer, corpus_uris, cache_valid = cached_matrix, cached_vec, cached_uris, True
                    logger.info(f"TF-IDF loaded from cache ({len(corpus_uris)} URIs).")
                else: logger.warning("TF-IDF cache dim/URI mismatch. Rebuilding.")
            else: logger.info("TF-IDF cache incomplete/invalid. Rebuilding.")

        if not cache_valid:
            logger.info("Rebuilding TF-IDF index...")
            try:
                docs, doc_uris = [], []; logger.info("Preparing docs for TF-IDF...")
                for uri, data in tqdm(sorted(taxonomy_concepts_cache.items()), desc="Prepare TF-IDF Docs"):
                    texts = []; props = ["skos:prefLabel", "rdfs:label", "skos:altLabel", "skos:definition"]
                    for k in props: vals = data.get(k, []); texts.extend(str(v) for v in (vals if isinstance(vals, list) else [vals]) if v)
                    norm_doc = normalize_concept(" ".join(texts)) # From utils
                    if norm_doc: docs.append(norm_doc); doc_uris.append(uri)
                if not docs: logger.warning("No documents for TF-IDF build.")
                else:
                    logger.info(f"Fitting TF-IDF Vectorizer on {len(docs)} documents with params: {tfidf_params}...")
                    vectorizer = TfidfVectorizer(**tfidf_params)
                    tfidf_matrix = vectorizer.fit_transform(docs); corpus_uris = doc_uris
                    logger.info(f"TF-IDF Matrix: {tfidf_matrix.shape}, Corpus URIs: {len(corpus_uris)}")
                    # --- Optional Debugging: Check vocabulary ---
                    # if args.debug and vectorizer and hasattr(vectorizer, 'vocabulary_'):
                    #     vocab = vectorizer.vocabulary_
                    #     logger.debug(f"TF-IDF Vocab check: 'aikido' in vocab: {'aikido' in vocab}")
                    #     logger.debug(f"TF-IDF Vocab check: 'american' in vocab: {'american' in vocab}")
                    #     logger.debug(f"TF-IDF Vocab check: 'breakfast' in vocab: {'breakfast' in vocab}")
                    # --- End Optional Debugging ---
                    save_cache(tfidf_matrix, matrix_cache_file, 'pickle'); save_cache(vectorizer, vec_cache_file, 'pickle'); save_cache(corpus_uris, uris_cache_file, 'pickle')
                    logger.info("TF-IDF components rebuilt and saved.")
            except Exception as e: logger.error(f"TF-IDF build error: {e}", exc_info=True); tfidf_matrix, vectorizer, corpus_uris = None, None, None

        logger.debug("--- TF-IDF Index Build/Load Diagnostics ---")
        logger.debug(f"Vectorizer Type: {type(vectorizer)}")
        if vectorizer and hasattr(vectorizer, 'vocabulary_'): logger.debug(f"Vocab Size: {len(vectorizer.vocabulary_)}")
        else: logger.debug("Vocab: Not available.")
        logger.debug(f"Matrix Shape: {tfidf_matrix.shape if tfidf_matrix is not None else 'None'}")
        logger.debug(f"Corpus URIs Count: {len(corpus_uris) if corpus_uris is not None else 'None'}")
        logger.debug("--- End TF-IDF Diagnostics ---")
    else: logger.info("TF-IDF Indexing disabled or sklearn unavailable.")

    logger.info("Building/Loading simple keyword label index...");
    label_index = build_keyword_label_index(taxonomy_concepts_cache) # From utils
    if label_index is None: logger.warning("Failed simple label index build.")
    else: logger.info(f"Simple label index ready ({len(label_index)} keywords).")

    _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris, _keyword_label_index = vectorizer, tfidf_matrix, corpus_uris, label_index
    return vectorizer, tfidf_matrix, corpus_uris, label_index

def get_candidate_concepts_keyword(
    query_texts: List[str], vectorizer: TfidfVectorizer, tfidf_matrix: Any,
    corpus_uris: List[str], top_n: int, min_score: float = 0.05
) -> List[Dict[str, Any]]:
    """ Finds candidate concepts using TF-IDF similarity. """
    if not query_texts or tfidf_matrix is None or vectorizer is None or not corpus_uris: return []
    if tfidf_matrix.shape[0] != len(corpus_uris): logger.error(f"TF-IDF shape/URI mismatch."); return []
    start_time = time.time(); logger.debug(f"TF-IDF search query: {query_texts[:5]}...")
    try:
        query_matrix = vectorizer.transform(query_texts); logger.debug(f"Query Matrix: {query_matrix.shape}")
        if query_matrix.shape[1] != tfidf_matrix.shape[1]: logger.error(f"Vocab mismatch!"); return []
        similarities = cosine_similarity(query_matrix, tfidf_matrix); logger.debug(f"Similarities Shape: {similarities.shape}")
        agg_scores = np.max(similarities, axis=0) if similarities.shape[0] > 1 else similarities[0] if similarities.shape[0] == 1 else np.array([])
        if agg_scores.size == 0: logger.debug("Aggregated TF-IDF scores empty."); return []
        logger.debug(f"Agg Scores Shape: {agg_scores.shape}, Min: {np.min(agg_scores):.4f}, Max: {np.max(agg_scores):.4f}")
        num_cands = len(agg_scores); top_n = min(top_n, num_cands)
        if top_n <= 0: return []
        if top_n < num_cands // 2 : idx_unsorted = np.argpartition(agg_scores, -top_n)[-top_n:]; indices = idx_unsorted[np.argsort(agg_scores[idx_unsorted])[::-1]]
        else: indices = np.argsort(agg_scores)[::-1][:top_n]
        logger.debug(f"Top {len(indices)} indices. Scores: {agg_scores[indices[:5]] if len(indices)>0 else 'N/A'}")
        candidates = []; filtered = 0
        for i in indices:
            score = float(agg_scores[i]);
            if score >= min_score: candidates.append({"uri": corpus_uris[i], "score": score, "method": "keyword_tfidf"})
            else: filtered += 1; # break # Optional early stop
        logger.debug(f"Filtered {filtered} TF-IDF candidates < {min_score}.")
        logger.info(f"TF-IDF search found {len(candidates)} candidates >= {min_score} ({time.time() - start_time:.2f}s).")
        return candidates
    except Exception as e: logger.error(f"TF-IDF search error: {e}", exc_info=True); return []

# --- LLM Client Initialization & Call ---
def get_openai_client()  -> Optional[OpenAI_Type]:
    """Initializes and returns the OpenAI client."""
    global _openai_client, _config_data
    if not OPENAI_AVAILABLE: return None
    if _openai_client is None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY");
            if not api_key: logger.warning("OPENAI_API_KEY env var not set."); return None
            timeout = _config_data.get("LLM_API_CONFIG", {}).get("REQUEST_TIMEOUT", 60) if _config_data else 60
            _openai_client = OpenAI(api_key=api_key, timeout=timeout); logger.info("OpenAI client initialized.")
        except Exception as e: logger.error(f"OpenAI client init failed: {e}"); return None
    return _openai_client

def get_google_client() -> Optional[Any]:
    """Configures and returns the Google AI client (genai module)."""
    global _google_client
    if not GOOGLE_AI_AVAILABLE: return None
    if _google_client is None:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY");
            if not api_key: logger.warning("GOOGLE_API_KEY env var not set."); return None
            genai.configure(api_key=api_key); logger.info("Google AI client configured."); _google_client = genai
        except Exception as e: logger.error(f"Google AI config failed: {e}"); return None
    return _google_client

def call_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float, provider: str) -> Dict[str, Any]:
    """Handles API calls to the specified LLM provider with retries."""
    logger.info(f"LLM call: {provider}, Model: {model_name}")
    if not model_name: logger.error(f"LLM model missing."); return {"success": False, "error": "LLM model missing"}
    result = {"success": False, "response": None, "error": None, "attempts": 0}
    client = get_openai_client() if provider == "openai" else get_google_client()
    if not client: result["error"] = f"{provider} client unavailable."; logger.error(result["error"]); return result
    delay = _config_data.get("LLM_API_CONFIG", {}).get("RETRY_DELAY_SECONDS", LLM_RETRY_DELAY_SECONDS) if _config_data else LLM_RETRY_DELAY_SECONDS

    for attempt in range(max_retries + 1):
        result["attempts"] = attempt + 1; logger.info(f"{provider} Attempt {attempt + 1}/{max_retries + 1}")
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
                if resp.prompt_feedback and resp.prompt_feedback.block_reason: block_reason = getattr(resp.prompt_feedback, 'block_reason', 'N/A'); logger.warning(f"Gemini blocked: {block_reason}."); result["error"] = f"Blocked: {block_reason}"; return result
                if resp.candidates: content = resp.text
                else: logger.warning("Gemini response no candidates."); content = None
            else: result["error"] = f"Unsupported provider: {provider}"; logger.error(result["error"]); return result

            if content:
                try:
                    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
                    cleaned = json_match.group(1) if json_match else content[content.find('{'):content.rfind('}')+1] if '{' in content else content.strip('`')
                    output = json.loads(cleaned)
                    if isinstance(output, dict): result["success"] = True; result["response"] = output; logger.info(f"{provider} OK ({time.time() - start:.2f}s)."); return result
                    else: logger.error(f"{provider} response not dict. Type: {type(output)}.")
                except json.JSONDecodeError as e: logger.error(f"{provider} JSON parse error: {e}. Raw: {content[:500]}..."); result["error"] = f"JSON Parse Error: {e}"
                except Exception as e: logger.error(f"{provider} response processing error: {e}. Raw: {content[:500]}..."); result["error"] = f"Response Processing Error: {e}"
            else: logger.warning(f"{provider} response empty."); result["error"] = "Empty response from LLM"

        except (APITimeoutError, APIConnectionError, RateLimitError) as e: logger.warning(f"{provider} API Error: {type(e).__name__}"); result["error"] = f"{type(e).__name__}"
        except Exception as e: logger.error(f"{provider} Call Error: {e}", exc_info=True); result["error"] = str(e); return result

        should_retry = attempt < max_retries and ("Error" in str(result.get("error")) or "Empty response" in str(result.get("error")))
        if not should_retry: logger.error(f"LLM call failed permanently after {attempt + 1} attempts.");
        if not result["error"]: result["error"] = "Failed after retries."; return result
        wait = delay * (2**attempt) + np.random.uniform(0, delay*0.5); logger.info(f"Retrying in {wait:.2f}s... (Error: {result.get('error')})"); time.sleep(wait)
    return result

# --- Prompt Building Functions ---

# --- Updated Prompt Building Function ---
def construct_keyword_expansion_prompt(input_concept: str) -> Tuple[str, str]:
    """Creates prompts for Stage 1 LLM keyword expansion (v33.0.15 - Enhanced)."""
    # System prompt remains focused on JSON output and travel domain expertise
    system_prompt = """You are a helpful assistant specializing in travel concepts and keyword analysis for search retrieval. You understand nuances like compound words and relationships between concepts. Your goal is to generate relevant keywords that will improve search results within a travel taxonomy. Output ONLY a valid JSON object containing a single key "keywords" with a list of strings as its value. Do not include any explanations or introductory text outside the JSON structure.
    """

    # User prompt is enhanced to guide the LLM towards better keyword generation for search
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
    # Ensure prompts are clean strings
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
    desc = theme_config.get('description', 'N/A'); hints = theme_config.get("hints", {}); hints_str = "";
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
def expand_keywords_with_llm(concept_label: str, config: Dict, args: argparse.Namespace) -> List[str]:
    """Uses LLM to generate keywords. Returns unique list including original normalized words."""
    llm_api_cfg = config.get("LLM_API_CONFIG", {}); temp = llm_api_cfg.get("KW_EXPANSION_TEMPERATURE", KW_EXPANSION_TEMPERATURE)
    timeout = llm_api_cfg.get("REQUEST_TIMEOUT", LLM_TIMEOUT); retries = llm_api_cfg.get("MAX_RETRIES", LLM_MAX_RETRIES)
    # Note: Using the *input* concept label for expansion, not normalized yet
    logger.info(f"[{concept_label}] LLM keyword expansion triggered (Temp: {temp})...")
    sys_prompt, user_prompt = construct_keyword_expansion_prompt(concept_label)
    result = call_llm(sys_prompt, user_prompt, args.llm_model, timeout, retries, temp, args.llm_provider)
    final_keyword_terms = set() # Use a set to store individual terms for uniqueness
    raw_phrases = []

    if result and result.get("success"):
        raw_phrases = result.get("response", {}).get("keywords", [])
        if isinstance(raw_phrases, list):
            # Process phrases: normalize and split into terms
            for kw_phrase in raw_phrases:
                if isinstance(kw_phrase, str) and kw_phrase.strip():
                    normalized_phrase = normalize_concept(kw_phrase)
                    # Add individual terms longer than 2 characters
                    final_keyword_terms.update(term for term in normalized_phrase.split() if len(term) > 2)
            logger.info(f"[{concept_label}] LLM generated {len(final_keyword_terms)} unique keyword terms from {len(raw_phrases)} phrases.")
        else:
            logger.error(f"[{concept_label}] LLM expansion response 'keywords' field was not a list.")
            exp_diag = {"error": "LLM response format invalid"} # Add error info if possible
    else:
        err_msg = result.get('error', 'N/A') if result else 'No result object'
        logger.warning(f"[{concept_label}] LLM keyword expansion failed: {err_msg}")
        exp_diag = {"error": err_msg} # Add error info if possible

    # Always include the normalized terms from the original input concept
    original_normalized_words = set(w for w in normalize_concept(concept_label).split() if len(w) > 2)
    final_keyword_terms.update(original_normalized_words)

    return list(final_keyword_terms) # Return the unique terms


# --- Updated Stage 1: Evidence Preparation (Combined Score) ---

# --- Updated Stage 1: Evidence Preparation (Combined Score + Absolute SBERT Filter) ---
def prepare_evidence(
    input_concept: str, concept_embedding: Optional[np.ndarray], primary_embeddings: Dict[str, np.ndarray],
    config: Dict, args: argparse.Namespace, tfidf_vectorizer: Optional[TfidfVectorizer],
    tfidf_matrix: Optional[Any], tfidf_corpus_uris: Optional[List[str]],
    keyword_label_index: Optional[Dict[str, Set[str]]], taxonomy_concepts_cache: Dict[str, Dict]
) -> Tuple[List[Dict], Dict[str, Dict], Optional[Dict], Dict[str, float], Dict[str, Any], int, int, int]: # Added counts to return tuple
    """
    Prepares evidence candidates using combined score (SBERT + TF-IDF Keyword)
    and filters candidates below an absolute minimum SBERT threshold.
    v33.0.15 - Includes mandatory splitting, abs SBERT filter.
    Returns candidate details for LLM, original candidates map, anchor candidate,
            final SBERT scores, expansion diagnostics, initial SBERT count,
            initial TFIDF count, and unique candidates count before ranking.
    """
    normalized_concept = normalize_concept(input_concept); logger.info(f"Starting evidence prep for: '{normalized_concept}' (Combined Scoring + Abs SBERT Filter)")
    stage1_cfg = config.get('STAGE1_CONFIG', {}); max_cands = int(stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', MAX_CANDIDATES_FOR_LLM))
    min_sim = float(stage1_cfg.get('EVIDENCE_MIN_SIMILARITY', EVIDENCE_MIN_SIMILARITY)); min_kw = float(stage1_cfg.get('KEYWORD_MIN_SCORE', KEYWORD_MIN_SCORE))
    # Load ABSOLUTE_MIN_SBERT_SCORE from config, defaulting to the constant
    abs_min_sbert = float(stage1_cfg.get('ABSOLUTE_MIN_SBERT_SCORE', ABSOLUTE_MIN_SBERT_SCORE))
    alpha = float(config.get('COMBINED_SCORE_ALPHA', COMBINED_SCORE_ALPHA)); kw_exp_enabled = config.get("ENABLE_KW_EXPANSION", ENABLE_KW_EXPANSION) and args.llm_provider != "none"
    kw_trigger = int(stage1_cfg.get('MIN_KEYWORD_CANDIDATES_FOR_EXPANSION', MIN_KEYWORD_CANDIDATES_FOR_EXPANSION)); kw_top_n = int(stage1_cfg.get('KEYWORD_TOP_N', KEYWORD_TOP_N_SEARCH))
    sim_scores, kw_scores = {}, {}; exp_diag = {"attempted": False, "successful": False, "count": 0, "terms": [], "tfidf_count": 0, "error": None}
    logger.info(f"Using Absolute Min SBERT Threshold: {abs_min_sbert}") # Log the threshold being used

    # --- Keyword Preparation (Mandatory Split + Optional Expansion) ---
    base_split_kws = set(kw for kw in normalized_concept.split() if len(kw) > 2)
    logger.debug(f"Base split keywords for '{normalized_concept}': {list(base_split_kws)}")
    query_texts_set = set(base_split_kws)
    initial_kw_count = 0
    if keyword_label_index:
        initial_kw_count = len(set().union(*[keyword_label_index.get(kw, set()) for kw in base_split_kws]))
        logger.info(f"Initial simple matches (based on split words): {initial_kw_count}.")
    else:
        logger.warning("Simple keyword index unavailable.")

    needs_exp = kw_exp_enabled and (initial_kw_count < kw_trigger or normalized_concept in ABSTRACT_CONCEPTS_LIST)
    expanded_kws = set()
    if needs_exp:
        exp_diag["attempted"] = True
        logger.info(f"Attempting LLM keyword expansion for '{normalized_concept}' (Trigger: initial matches {initial_kw_count} < {kw_trigger} or abstract concept).")
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
                logger.info(f"LLM expansion added {len(newly_expanded_kws)} new keyword terms.")
                query_texts_set.update(newly_expanded_kws)
            else:
                logger.info("LLM expansion did not add new useful keyword terms.")
                exp_diag["error"] = exp_diag.get("error", "LLM returned no new terms")
        except Exception as llm_exp_err:
            logger.error(f"LLM Keyword Expansion failed for '{input_concept}': {llm_exp_err}", exc_info=True)
            exp_diag["successful"] = False
            exp_diag["error"] = str(llm_exp_err) # Capture the error message
    else:
        logger.info("Skipping LLM KW expansion.")

    final_query_texts = list(query_texts_set)
    exp_diag["count"] = len(final_query_texts)
    exp_diag["terms"] = final_query_texts if args.debug else ["..."]
    logger.info(f"Using {len(final_query_texts)} unique keywords for TF-IDF search: {final_query_texts if args.debug else '...'}")

    # --- TF-IDF Search ---
    tfidf_candidate_count_initial = 0
    if tfidf_vectorizer and tfidf_matrix is not None and tfidf_corpus_uris:
        if final_query_texts:
            kw_cands = get_candidate_concepts_keyword(final_query_texts, tfidf_vectorizer, tfidf_matrix, tfidf_corpus_uris, kw_top_n, min_kw)
            kw_scores = {c['uri']: c['score'] for c in kw_cands}
            tfidf_candidate_count_initial = len(kw_scores)
            exp_diag["tfidf_count"] = tfidf_candidate_count_initial
            logger.info(f"Found {tfidf_candidate_count_initial} TF-IDF candidates >= {min_kw}.")
        else:
            logger.warning(f"No query keywords derived for TF-IDF search for '{normalized_concept}'.")
            kw_scores = {}; exp_diag["tfidf_count"] = 0
    else:
        logger.warning(f"TF-IDF components missing or unavailable. Skipping TF-IDF search.")
        kw_scores = {}; exp_diag["tfidf_count"] = 0

    # --- SBERT Similarity Calculation ---
    sbert_candidate_count_initial = 0
    if concept_embedding is None:
        logger.error(f"No anchor embedding for '{normalized_concept}'. Skipping SBERT similarity.");
        sim_scores = {}
    else:
        logger.debug(f"Calculating SBERT similarities for '{normalized_concept}'...")
        sim_scores = get_batch_embedding_similarity(concept_embedding, primary_embeddings)
        # Keep only those >= EVIDENCE_MIN_SIMILARITY for initial count, but retain all for potential TFIDF rescue
        sbert_initial_candidates = {uri: s for uri, s in sim_scores.items() if s >= min_sim}
        sbert_candidate_count_initial = len(sbert_initial_candidates)
        logger.info(f"Found {sbert_candidate_count_initial} SBERT candidates >= {min_sim}.")
        # Note: We keep the full sim_scores dict for the combination step below

    # --- Candidate Combination, Filtering, and Ranking ---
    all_uris = set(kw_scores.keys()) | set(sim_scores.keys())
    unique_candidates_before_ranking = len(all_uris)
    if not all_uris:
        logger.warning(f"No candidates found from either TF-IDF or SBERT for '{normalized_concept}'.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, 0

    logger.info(f"Processing {unique_candidates_before_ranking} unique candidates from TF-IDF and SBERT.")
    all_details = get_kg_data(list(all_uris), taxonomy_concepts_cache)
    scored_list: List[Dict] = []
    abs_sbert_filtered_count = 0

    for uri in all_uris:
        if uri not in all_details:
            logger.debug(f"Details not found for candidate URI: {uri}. Skipping.")
            continue

        s_score = sim_scores.get(uri, 0.0) # Get SBERT score (could be low or 0 if only found by TFIDF)
        k_score = kw_scores.get(uri, 0.0) # Get TFIDF score (could be 0 if only found by SBERT)

        # *** NEW FILTER STEP ***
        # Check if SBERT score is below the absolute minimum threshold
        if s_score < abs_min_sbert:
            # Optionally log more details if debugging this filter specifically
            if args.debug:
                 logger.debug(f"Excluding candidate {uri} ('{all_details[uri].get('prefLabel', 'N/A')}') due to SBERT score {s_score:.4f} < {abs_min_sbert:.4f}. (TFIDF: {k_score:.4f})")
            abs_sbert_filtered_count += 1
            continue # Skip this candidate entirely, do not add to scored_list

        # Normalize scores (optional but good practice if alpha is used for blending)
        norm_s = max(0.0, min(1.0, s_score))
        norm_k = max(0.0, min(1.0, k_score))

        # Calculate combined score ONLY for candidates passing the filter
        combined = (alpha * norm_k) + ((1.0 - alpha) * norm_s)

        scored_list.append({
            "uri": uri,
            "details": all_details[uri],
            "sim_score": s_score,
            "kw_score": k_score,
            "combined_score": combined
        })

    logger.info(f"Excluded {abs_sbert_filtered_count} candidates with SBERT score < {abs_min_sbert}.")

    if not scored_list:
        logger.warning(f"No candidates remaining after absolute SBERT filtering for '{normalized_concept}'.")
        # Return empty/None values along with counts (unique_candidates_before_ranking is still valid)
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, unique_candidates_before_ranking

    # Sort by combined score and select top N
    scored_list.sort(key=lambda x: x['combined_score'], reverse=True)
    selected = scored_list[:max_cands]
    logger.info(f"Selected top {len(selected)}/{len(scored_list)} candidates based on combined score (alpha={alpha}) after filtering.")

    if not selected:
        # This case should technically be covered by the previous check, but added for safety
        logger.warning(f"No candidates selected after scoring and filtering for '{normalized_concept}'.")
        return [], {}, None, {}, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, unique_candidates_before_ranking

    # Prepare outputs
    llm_details = [c["details"] for c in selected]
    # Ensure all scores are included for downstream weighting/analysis
    orig_map = { c['uri']: {**c["details"], "sbert_score": c["sim_score"], "keyword_score": c["kw_score"], "combined_score": c["combined_score"]} for c in selected }
    anchor_data = selected[0] # Anchor is the top candidate after combined scoring AND filtering
    anchor = orig_map.get(anchor_data['uri'])
    if anchor:
        logger.info(f"Anchor selected: {anchor.get('prefLabel', anchor['uri'])} (Score: {anchor_data['combined_score']:.4f}, SBERT: {anchor_data['sim_score']:.4f}, TFIDF: {anchor_data['kw_score']:.4f})")
    else:
        logger.warning(f"Could not determine anchor details for '{normalized_concept}'.")

    # SBERT scores of the *final selected* candidates (for potential use in Stage 2 if needed)
    sbert_scores_final = {uri: d["sbert_score"] for uri, d in orig_map.items()}

    # Return the calculated counts along with other results
    return llm_details, orig_map, anchor, sbert_scores_final, exp_diag, sbert_candidate_count_initial, tfidf_candidate_count_initial, unique_candidates_before_ranking




# --- Stage 2: Finalization (Ensure this function exists and is correct) ---
# --- Updated Stage 2: Finalization (Using Combined Score for Weighting) ---
def apply_rules_and_finalize(
    input_concept: str, llm_call_result: Optional[Dict[str, Any]], config: Dict,
    travel_category: Optional[Dict], anchor_candidate: Optional[Dict],
    original_candidates_map_for_reprompt: Dict[str, Dict], # Includes sbert_score, keyword_score, combined_score
    candidate_evidence_scores: Dict[str, float], # SBERT scores (retained for potential future use, but NOT used for weighting now)
    args: argparse.Namespace, taxonomy_concepts_cache: Dict[str, Dict] # Pass cache
) -> Dict[str, Any]:
    """
    Applies rules, handles fallbacks, weights attributes, structures final output.
    v33.0.14 - Uses combined_score for proportional attribute weighting within themes.
    """
    start = time.time(); norm_concept = normalize_concept(input_concept); logger.info(f"Starting finalization for: {norm_concept}")
    output: Dict[str, Any] = {"applicable_lodging_types": "Both", "travel_category": travel_category or {"uri": None, "name": input_concept, "type": "Unknown"}, "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [], "must_not_have": [], "failed_fallback_themes": {}, "diagnostics": {"theme_processing": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "final_output": {}, "stage2": {"status": "Started", "duration_seconds": 0.0, "error": None}}}
    theme_diag = output["diagnostics"]["theme_processing"]; reprompt_diag = output["diagnostics"]["reprompting_fallback"]
    base_themes = config.get("base_themes", {}); overrides = config.get("concept_overrides", {}).get(norm_concept, {})
    final_cfg = config.get('STAGE2_CONFIG', {}); min_weight = float(final_cfg.get('THEME_ATTRIBUTE_MIN_WEIGHT', THEME_ATTRIBUTE_MIN_WEIGHT)); top_n_attrs = int(final_cfg.get('TOP_N_DEFINING_ATTRIBUTES', TOP_N_DEFINING_ATTRIBUTES_DEFAULT))

    llm_assigns: Dict[str, List[str]] = {}; diag_val = {}
    if llm_call_result and llm_call_result.get("success"):
        validated = validate_llm_assignments(llm_call_result.get("response"), set(original_candidates_map_for_reprompt.keys()), set(base_themes.keys()), norm_concept, diag_val)
        if validated is not None: llm_assigns = validated; logger.debug(f"Validated LLM assignments.")
        else: logger.warning(f"LLM validation failed."); output["diagnostics"]["stage2"]["error"] = diag_val.get("error", "LLM Validation Failed")
    elif llm_call_result: logger.warning(f"LLM call unsuccessful: {llm_call_result.get('error')}"); output["diagnostics"]["stage2"]["error"] = f"LLM Call Failed: {llm_call_result.get('error')}"
    else: logger.info(f"No LLM result.")

    theme_map = defaultdict(list)
    for uri, themes in llm_assigns.items():
        if uri in original_candidates_map_for_reprompt:
            for t in themes:
                if t in base_themes: theme_map[t].append(uri)

    failed_rules: Dict[str, Dict] = {};
    for name in base_themes.keys():
        diag = theme_diag[name] = {"llm_assigned_count": len(theme_map.get(name, [])), "attributes_after_weighting": 0, "status": "Pending", "rule_failed": False}
        rule, _, _, _ = get_dynamic_theme_config(norm_concept, name, config)
        if rule == "Must have 1" and not theme_map.get(name): logger.warning(f"Rule FAILED: '{name}'"); failed_rules[name] = {"reason": "No assigns."}; diag.update({"status": "Failed Rule", "rule_failed": True})

    fixed = set(); fallback_adds = []
    if failed_rules and original_candidates_map_for_reprompt and args.llm_provider != "none":
        logger.info(f"Attempting fallback for {len(failed_rules)} themes.")
        for name in list(failed_rules.keys()):
            reprompt_diag["attempts"] += 1; base_cfg = base_themes.get(name)
            if not base_cfg: logger.error(f"No config for fallback '{name}'."); reprompt_diag["failures"] += 1; continue
            sys_p, user_p = build_reprompt_prompt(input_concept, name, base_cfg, original_candidates_map_for_reprompt)
            fb_result = call_llm(sys_p, user_p, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)
            if fb_result and fb_result.get("success"):
                fb_assigns = fb_result.get("response", {}).get("theme_assignments", {}); new_uris = set(uri for uri, ts in fb_assigns.items() if isinstance(ts, list) and name in ts and uri in original_candidates_map_for_reprompt) if isinstance(fb_assigns, dict) else set()
                if new_uris:
                    logger.info(f"Fallback SUCCESS for '{name}': Assigned {len(new_uris)} URIs."); reprompt_diag["successes"] += 1; fixed.add(name); added = 0
                    for uri in new_uris:
                        if uri not in theme_map.get(name, []): theme_map[name].append(uri); fallback_adds.append({"uri": uri, "assigned_theme": name}); added += 1
                    logger.debug(f"Added {added} unique URIs via fallback for '{name}'.")
                    if name in theme_diag: theme_diag[name].update({"status": "Passed (Fallback)", "rule_failed": False})
                else: logger.warning(f"Fallback LLM for '{name}' assigned 0."); reprompt_diag["failures"] += 1; theme_diag[name]["status"] = "Failed (Fallback - No Assigns)"
            else: err = fb_result.get("error", "Unknown") if fb_result else "None"; logger.error(f"Fallback LLM failed for '{name}'. Error: {err}"); reprompt_diag["failures"] += 1; theme_diag[name]["status"] = "Failed (Fallback - API Error)"
    elif failed_rules: logger.warning(f"Cannot attempt fallback, LLM provider is 'none'.")

    final_themes_out = []; all_final_attrs = []
    theme_w_cfg = {n: get_dynamic_theme_config(norm_concept, n, config)[1] for n in base_themes.keys()}; norm_w = normalize_weights(theme_w_cfg)

    for name, base_data in base_themes.items():
        diag = theme_diag[name]; rule, _, subscore, _ = get_dynamic_theme_config(norm_concept, name, config)
        theme_w = norm_w.get(name, 0.0); uris = theme_map.get(name, [])
        diag["llm_assigned_count"] = len(uris)

        # *** MODIFICATION START: Use combined_score for weighting ***
        # Get COMBINED scores from the original map for weighting
        scores = {u: original_candidates_map_for_reprompt.get(u, {}).get("combined_score", 0.0)
                  for u in uris if u in original_candidates_map_for_reprompt}
        total_score = sum(scores.values())
        logger.debug(f"[{norm_concept}][{name}] URIs:{len(uris)}, Combined Score Sum:{total_score:.4f}, Theme Weight:{theme_w:.4f}")
        # *** MODIFICATION END ***

        theme_attrs = []
        if uris and theme_w > 0:
            n_uris = len(uris)
            # *** MODIFICATION START: Check total_score (sum of combined scores) ***
            if total_score < 1e-9 and n_uris > 0:
                # Fallback to equal weighting if total COMBINED score is effectively zero
                logger.warning(f"Zero Combined scores for '{name}'. Using equal weight distribution for {n_uris} URIs.")
                # *** MODIFICATION END ***
                eq_w = (theme_w / n_uris)
                if eq_w >= min_weight:
                    for u in uris:
                        if u not in original_candidates_map_for_reprompt: continue
                        d = original_candidates_map_for_reprompt[u]; is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                        attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(eq_w, 6), "type": d.get('type_labels', [])}; # Use util
                        if is_fb: attr["comment"] = "Fallback"
                        theme_attrs.append(attr); all_final_attrs.append(attr)
                else: logger.warning(f"Equal weight {eq_w:.6f} < {min_weight} for '{name}', skipping attributes.")
            elif total_score > 1e-9:
                 # *** MODIFICATION START: Weight by proportional COMBINED score ***
                 # Weight by proportional combined score
                 for u in uris:
                      if u not in original_candidates_map_for_reprompt: continue
                      d = original_candidates_map_for_reprompt[u]
                      # 's' now represents the combined_score fetched earlier
                      s = scores.get(u, 0.0)
                      # *** MODIFICATION END ***
                      prop = s / total_score; attr_w = theme_w * prop
                      if attr_w >= min_weight:
                           is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                           attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(attr_w, 6), "type": d.get('type_labels', [])}; # Use util
                           if is_fb: attr["comment"] = "Fallback"
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
             unique_attrs[uri] = {k: v for k, v in attr.items() if k != 'comment'} # Exclude comment field from top attributes

    sorted_top = sorted(unique_attrs.values(), key=lambda x: x.get('concept_weight', 0.0), reverse=True)
    output['top_defining_attributes'] = sorted_top[:top_n_attrs] # Use variable top_n_attrs

    # --- Apply Final Concept Overrides ---
    output["applicable_lodging_types"] = overrides.get("lodging_type", "Both")
    tc = output["travel_category"]
    if isinstance(tc, dict): tc["type"] = overrides.get("category_type", tc.get("type", "Uncategorized")); tc["exclusionary_concepts"] = overrides.get("exclusionary_concepts", [])
    else: logger.error(f"travel_category invalid?"); output["travel_category"] = {"uri": None, "name": norm_concept, "type": overrides.get("category_type", "Uncategorized"), "exclusionary_concepts": overrides.get("exclusionary_concepts", [])}
    mnh = overrides.get("must_not_have", []); mnh_uris = set(i["uri"] for i in mnh if isinstance(i, dict) and "uri" in i) if isinstance(mnh, list) else set()
    output["must_not_have"] = [{"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "scope": None} for u in sorted(list(mnh_uris))] # Pass cache
    add_scores = overrides.get("additional_subscores", []); output["additional_relevant_subscores"] = add_scores if isinstance(add_scores, list) else []

    # --- Final Diagnostic Counts ---
    final_diag = output["diagnostics"]["final_output"]
    final_diag["must_not_have_count"] = len(output["must_not_have"]); final_diag["additional_subscores_count"] = len(output["additional_relevant_subscores"])
    final_diag["themes_count"] = len(output["themes"]); output["failed_fallback_themes"] = { n: r for n, r in failed_rules.items() if n not in fixed }
    final_diag["failed_fallback_themes_count"] = len(output["failed_fallback_themes"]); final_diag["top_defining_attributes_count"] = len(output['top_defining_attributes'])
    output["diagnostics"]["stage2"]["status"] = "Completed"; output["diagnostics"]["stage2"]["duration_seconds"] = round(time.time() - start, 2)
    logger.info(f"[{norm_concept}] Finalization complete ({output['diagnostics']['stage2']['duration_seconds']:.2f}s).")
    return output






# --- Updated Main Processing Loop ---
def generate_affinity_definitions_loop(
    concepts_to_process: List[str], config: Dict, args: argparse.Namespace,
    sbert_model: SentenceTransformer, primary_embeddings_map: Dict[str, np.ndarray],
    taxonomy_concepts_cache: Dict[str, Dict], keyword_label_index: Optional[Dict[str, Set[str]]],
    tfidf_vectorizer: Optional[TfidfVectorizer], tfidf_matrix: Optional[Any],
    tfidf_corpus_uris: Optional[List[str]]
) -> List[Dict]:
    """ Main loop processing each concept (v33.0.15 version with diagnostics update). """ # Updated version marker
    print(">>> DEBUG: generate_affinity_definitions_loop entered.") # Debug Print
    all_definitions = []; cache_ver = config.get("CACHE_VERSION", DEFAULT_CACHE_VERSION)
    if not taxonomy_concepts_cache: logger.critical("FATAL: Taxonomy concepts cache empty."); return []
    limit = args.limit if args.limit and args.limit > 0 else len(concepts_to_process)
    concepts_subset = concepts_to_process[:limit]; logger.info(f"Processing {len(concepts_subset)} concepts.")
    print(f">>> DEBUG: Subset size for loop: {len(concepts_subset)}") # Debug Print
    if not concepts_subset: print(">>> DEBUG: concepts_subset is EMPTY!"); return [] # Debug Print

    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    print(f">>> DEBUG: Starting concept iteration...") # Debug Print

    for concept in tqdm(concepts_subset, desc="Processing Concepts", disable=disable_tqdm):
        print(f"\n>>> DEBUG: START LOOP ITERATION for concept: '{concept}'") # Debug Print
        start_time = time.time(); norm_concept = normalize_concept(concept)
        logger.info(f"=== Processing Concept: '{concept}' ('{norm_concept}') ===")

        # Initialize affinity definition structure
        affinity_def = {
            "input_concept": concept,
            "normalized_concept": norm_concept,
            "applicable_lodging_types": "Both",
            "travel_category": {},
            "top_defining_attributes": [],
            "themes": [],
            "additional_relevant_subscores": [],
            "must_not_have": [],
            "failed_fallback_themes": {},
            "processing_metadata": {
                "status": "Started",
                "version": SCRIPT_VERSION, # Use updated SCRIPT_VERSION global
                "timestamp": None,
                "total_duration_seconds": 0.0,
                "cache_version": cache_ver,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model if args.llm_provider != "none" else None
            },
            "diagnostics": {
                "stage1": {
                    "status": "Not Started",
                    "error": None,
                    "selection_method": "Combined TFIDF+SBERT",
                    "expansion": {},
                    # Add placeholders for detailed counts
                    "sbert_candidate_count_initial": 0,
                    "tfidf_candidate_count_initial": 0,
                    "unique_candidates_before_ranking": 0,
                    "llm_candidate_count": 0
                },
                "llm_slotting": {
                    "status": "Not Started",
                    "error": None
                },
                "reprompting_fallback": {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0
                },
                "stage2": {
                    "status": "Not Started",
                    "error": None
                },
                "theme_processing": {},
                "final_output": {},
                "error_details": None
            }
        }
        diag1 = affinity_def["diagnostics"]["stage1"]; diag_llm = affinity_def["diagnostics"]["llm_slotting"]

        try:
            # --- Stage 1: Embedding and Evidence Preparation ---
            embed_text = f"{TRAVEL_CONTEXT}{norm_concept}"; logger.debug(f"Embedding: '{embed_text}'")
            concept_emb = get_concept_embedding(embed_text, sbert_model) # Util
            if concept_emb is None:
                logger.error(f"Embedding failed for '{norm_concept}'.")
                diag1["error"] = "Embedding failed"
                diag1["status"] = "Failed"
                # Continue to finalize, but stages depending on embedding will be skipped or empty
            else:
                 logger.debug(f"Successfully generated embedding for '{norm_concept}'.")

            stage1_start = time.time()
            # Call prepare_evidence and unpack ALL 8 return values
            cand_details, orig_map, anchor, sbert_scores, exp_diag, sbert_init_count, tfidf_init_count, unique_count = prepare_evidence(
                concept, concept_emb, primary_embeddings_map, config, args,
                _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris,
                _keyword_label_index, _taxonomy_concepts_cache
            ) # Calls the updated function
            stage1_dur = time.time() - stage1_start

            # Update Stage 1 diagnostics using the returned counts
            diag1["expansion"] = exp_diag # Update expansion details
            diag1.update({
                "status": "Completed" if diag1.get("error") is None else "Failed", # Update status based on embedding success/failure
                "duration_seconds": round(stage1_dur, 2),
                "sbert_candidate_count_initial": sbert_init_count,
                "tfidf_candidate_count_initial": tfidf_init_count,
                "unique_candidates_before_ranking": unique_count,
                "llm_candidate_count": len(cand_details)          # Count of candidates *actually selected*
                })
            logger.info(f"Stage 1 done ({stage1_dur:.2f}s). LLM Cands: {len(cand_details)}. Initial SBERT: {sbert_init_count}, TFIDF: {tfidf_init_count}.")

            # Set travel category based on anchor
            affinity_def["travel_category"] = anchor if anchor and anchor.get('uri') else {"uri": None, "name": concept, "type": "Unknown"}
            if not anchor: logger.warning(f"No anchor candidate found for '{norm_concept}'.")

            # --- Stage 1.5: LLM Theme Slotting ---
            llm_result = None
            llm_start = time.time()
            diag_llm.update({"llm_provider": args.llm_provider, "llm_model": args.llm_model if args.llm_provider != "none" else None})

            if not cand_details:
                logger.warning(f"No candidates generated for LLM slotting for concept '{concept}'.")
                if affinity_def["processing_metadata"]["status"] == "Started": # Don't overwrite previous errors
                    affinity_def["processing_metadata"]["status"] = "Warning - No LLM Candidates"
                diag_llm["status"] = "Skipped (No Candidates)"
            elif args.llm_provider == "none":
                logger.info(f"LLM provider is 'none'. Skipping LLM slotting.")
                diag_llm["status"] = "Skipped (Provider None)"
            else:
                diag_llm["status"] = "Started"
                diag_llm["llm_call_attempted"] = True
                # Prepare themes for the prompt
                themes_for_prompt = [
                    {"name": name,
                     "description": get_theme_definition_for_prompt(name, data),
                     "is_must_have": get_dynamic_theme_config(norm_concept, name, config)[0] == "Must have 1"}
                    for name, data in config.get("base_themes", {}).items()
                ]
                # Construct and call LLM
                sys_p, user_p = construct_llm_slotting_prompt(concept, themes_for_prompt, cand_details, args)
                llm_result = call_llm(sys_p, user_p, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)

                # Update diagnostics based on LLM call result
                diag_llm["attempts_made"] = llm_result.get("attempts", 0) if llm_result else 0
                if llm_result and llm_result.get("success"):
                    diag_llm["llm_call_success"] = True
                    diag_llm["status"] = "Completed"
                    logger.info(f"LLM slotting call successful for '{concept}'.")
                else:
                    diag_llm["llm_call_success"] = False
                    diag_llm["status"] = "Failed"
                    diag_llm["error"] = llm_result.get("error", "Unknown Error") if llm_result else "LLM Call resulted in None"
                    logger.warning(f"LLM slotting call failed for '{concept}'. Error: {diag_llm['error']}")

            diag_llm["duration_seconds"] = round(time.time() - llm_start, 2)
            logger.info(f"LLM Slotting process took {diag_llm['duration_seconds']:.2f}s. Status: {diag_llm['status']}")

            # --- Stage 2: Finalization ---
            stage2_start = time.time()
            # Ensure taxonomy_concepts_cache is passed to finalize function
            stage2_out = apply_rules_and_finalize(
                concept, llm_result, config, affinity_def["travel_category"], anchor,
                orig_map, sbert_scores, args, _taxonomy_concepts_cache # Pass cache here
            )
            stage2_dur = time.time() - stage2_start

            # Update main affinity_def with results from stage 2
            affinity_def.update({k: v for k, v in stage2_out.items() if k != 'diagnostics'})

            # Merge diagnostics from stage 2
            if "diagnostics" in stage2_out:
                s2d = stage2_out["diagnostics"]
                affinity_def["diagnostics"]["theme_processing"] = s2d.get("theme_processing", {})
                affinity_def["diagnostics"]["reprompting_fallback"].update(s2d.get("reprompting_fallback", {})) # Use update to merge counts
                affinity_def["diagnostics"]["final_output"] = s2d.get("final_output", {})
                affinity_def["diagnostics"]["stage2"] = s2d.get("stage2", {"status": "Unknown"}) # Ensure stage2 dict exists
            else:
                affinity_def["diagnostics"]["stage2"] = {"status": "Unknown", "error": "Stage 2 diagnostics missing"}

            # Ensure duration is recorded correctly for stage 2
            affinity_def["diagnostics"]["stage2"]["duration_seconds"] = round(stage2_dur, 2)
            logger.info(f"Stage 2 finalization completed in {affinity_def['diagnostics']['stage2']['duration_seconds']:.2f}s.")

            # Determine final processing status
            final_status = affinity_def["processing_metadata"]["status"]
            if final_status == "Started": # Only update if not already set to a warning/error state earlier
                if affinity_def.get("failed_fallback_themes"): final_status = "Success with Failed Rules"
                elif diag_llm["status"].startswith("Failed"): final_status = "Warning - LLM Slotting Failed"
                elif diag_llm["status"] == "Skipped (No Candidates)": final_status = "Warning - No LLM Candidates"
                elif diag_llm["status"] == "Skipped (Provider None)": final_status = "Success (LLM Skipped)"
                elif affinity_def["diagnostics"]["stage2"].get("error"): final_status = f"Warning - Finalization Error"
                elif diag1["status"] == "Failed": final_status = "Failed - Stage 1 Error" # Check if Stage 1 failed
                else: final_status = "Success"
            affinity_def["processing_metadata"]["status"] = final_status
            log_func = logger.warning if "Warning" in final_status or "Failed" in final_status else logger.info
            log_func(f"Concept '{concept}' finished status: {final_status}")

        except Exception as e:
            logger.error(f"Core processing loop failed unexpectedly for concept '{concept}': {e}", exc_info=True)
            affinity_def["processing_metadata"]["status"] = f"FATAL ERROR"
            affinity_def["diagnostics"]["error_details"] = traceback.format_exc()
            # Mark stages as failed if not already completed/failed
            if diag1["status"] not in ["Completed", "Failed"]: diag1["status"] = "Failed"
            if diag_llm["status"] not in ["Completed", "Failed", "Skipped (No Candidates)", "Skipped (Provider None)"]: diag_llm["status"] = "Failed (Exception)"
            if affinity_def["diagnostics"]["stage2"]["status"] not in ["Completed", "Failed"]: affinity_def["diagnostics"]["stage2"]["status"] = "Failed (Exception)"

        finally:
            # Finalize metadata regardless of success or failure
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            affinity_def["processing_metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            affinity_def["processing_metadata"]["total_duration_seconds"] = duration
            all_definitions.append(affinity_def)
            logger.info(f"--- Finished '{norm_concept}' ({duration:.2f}s). Status: {affinity_def['processing_metadata']['status']} ---")
            print(f">>> DEBUG: END LOOP ITERATION for concept: '{concept}'") # Debug Print

    print(f">>> DEBUG: Exiting generate_affinity_definitions_loop.") # Debug Print
    return all_definitions


# --- Main Execution ---
def main():
    global _config_data, _taxonomy_concepts_cache, _taxonomy_embeddings_cache, _keyword_label_index, _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris
    print("--- DEBUG: main() started ---") # Debug Print
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
    args = parser.parse_args()

    config = load_affinity_config(args.config); # Util
    if config is None: sys.exit(1)
    _config_data = config; print(f">>> DEBUG: Config loaded.")

    cache_ver = config.get("CACHE_VERSION", DEFAULT_CACHE_VERSION)
    os.makedirs(args.output_dir, exist_ok=True); os.makedirs(args.cache_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, LOG_FILE_TEMPLATE.format(cache_version=cache_ver))
    out_file = os.path.join(args.output_dir, OUTPUT_FILE_TEMPLATE.format(cache_version=cache_ver))
    log_level = logging.DEBUG if args.debug else logging.INFO; setup_logging(log_level, log_file, args.debug) # Util
    print(f">>> DEBUG: Logging setup.")

    logger.info(f"Starting {SCRIPT_VERSION}"); logger.info(f"Cmd: {' '.join(sys.argv)}")
    logger.info(f"Config: {args.config}, Taxo: {args.taxonomy_dir}, Input: {args.input_concepts_file}")
    logger.info(f"Output: {args.output_dir}, Cache: {args.cache_dir}, Log: {log_file}, Outfile: {out_file}")
    logger.info(f"Rebuild: {args.rebuild_cache}, Limit: {args.limit or 'None'}, Debug: {args.debug}")

    if args.llm_provider is None: args.llm_provider = config.get('LLM_PROVIDER', 'none')
    else: logger.warning(f"Override LLM provider: '{args.llm_provider}'")
    if args.llm_model is None: args.llm_model = config.get('LLM_MODEL', None)
    else: logger.warning(f"Override LLM model: '{args.llm_model}'")
    logger.info(f"LLM Provider: {args.llm_provider}, Model: {args.llm_model or 'N/A'}")

    input_concepts = []
    try:
        with open(args.input_concepts_file, 'r', encoding='utf-8') as f: input_concepts = [l.strip() for l in f if l.strip()]
        if not input_concepts: raise ValueError("Input file empty.")
        logger.info(f"Loaded {len(input_concepts)} concepts from '{args.input_concepts_file}'.")
        print(f">>> DEBUG: Input concepts loaded. Count: {len(input_concepts)}")
    except Exception as e: logger.critical(f"Read input file failed: {e}"); sys.exit(1)

    concepts_cache_f = get_cache_filename('concepts', cache_ver, args.cache_dir, extension=".json") # Util
    concepts_data = load_taxonomy_concepts(args.taxonomy_dir, concepts_cache_f, args.rebuild_cache, cache_ver, args.debug) # Util
    if concepts_data is None: logger.critical("Taxonomy load failed."); sys.exit(1)
    _taxonomy_concepts_cache = concepts_data; logger.info(f"Taxonomy concepts ready ({len(_taxonomy_concepts_cache)})."); print(f">>> DEBUG: Taxonomy concepts ready.")

    try: # Build indexes
        build_keyword_index(config, _taxonomy_concepts_cache, args.cache_dir, cache_ver, args.rebuild_cache) # Local fn
        print(f">>> DEBUG: Index build function called.")
        if _tfidf_vectorizer is not None: logger.info(f"TF-IDF Index ready.")
        elif config.get("TFIDF_CONFIG", {}).get("enabled", False): logger.warning("TF-IDF failed but enabled.")
        if _keyword_label_index is not None: logger.info(f"Simple Label Index ready.")
        else: logger.warning("Simple Label Index failed.")
    except Exception as e: logger.critical(f"Index building failed: {e}", exc_info=True); sys.exit(1)

    logger.info("Loading SBERT model..."); print(">>> DEBUG: Loading SBERT model...")
    try:
        sbert_name = config.get("SBERT_MODEL_NAME"); sbert_model = get_sbert_model(sbert_name) # Util
        if sbert_model is None: raise RuntimeError("SBERT model load failed")
        logger.info(f"SBERT model load successful. Dimension: {sbert_model.get_sentence_embedding_dimension()}") # Corrected log
        print(">>> DEBUG: SBERT model loaded.")
    except Exception as e: logger.critical(f"SBERT load failed: {e}", exc_info=True); sys.exit(1)

    print(">>> DEBUG: Loading/Building embeddings...")
    embed_cache_params = {"model": sbert_name or "default"}
    embed_cache_f = get_cache_filename('embeddings', cache_ver, args.cache_dir, embed_cache_params, ".pkl") # Util
    embed_data = precompute_taxonomy_embeddings(_taxonomy_concepts_cache, sbert_model, embed_cache_f, cache_ver, args.rebuild_cache, args.debug) # Util
    if embed_data is None: logger.critical("Embeddings load failed."); sys.exit(1)
    _taxonomy_embeddings_cache = embed_data; primary_embs, uris_w_embeds = _taxonomy_embeddings_cache
    logger.info(f"Embeddings ready ({len(uris_w_embeds)} concepts)."); print(f">>> DEBUG: Embeddings ready.")

    if not input_concepts: print(">>> DEBUG: input_concepts list EMPTY before loop! Exiting."); sys.exit(0)
    print(f"\n>>> DEBUG: About to call generate_affinity_definitions_loop with {len(input_concepts)} concepts.")

    logger.info("Starting generation loop..."); start_loop = time.time()
    results = generate_affinity_definitions_loop(input_concepts, config, args, sbert_model, primary_embs, _taxonomy_concepts_cache, _keyword_label_index, _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus_uris) # Local loop
    print(f">>> DEBUG: Returned from loop. Result count: {len(results)}") # Check after return
    end_loop = time.time(); logger.info(f"Generation loop finished ({end_loop - start_loop:.2f}s).")

    if results: save_results_json(results, out_file) # Util
    else: logger.warning("No results generated.")
    logger.info(f"Script finished. Log: {log_file}"); print("--- DEBUG: main() finished ---")

# --- Standard Python Entry Point ---
if __name__ == "__main__":
    # Check required library availability using flags imported from utils
    if not all([UTILS_ST_AVAILABLE, UTILS_SK_AVAILABLE, RDFLIB_AVAILABLE]):
         print("CRITICAL: Required libs unavailable.", file=sys.stderr); sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
         print("Warning: LLM API keys not found.", file=sys.stderr)
    main() # Call the main function