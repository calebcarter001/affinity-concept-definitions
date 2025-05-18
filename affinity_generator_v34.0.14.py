# -*- coding: utf-8 -*-
"""
Generate affinity definitions for travel concepts using combined scoring
(v34.0.14 - Weight Attrs by CombinedScore + Dynamic Lodging Type + v34.0.13 features).

Implements:
- Weights attributes within themes based on their combined_score (BM25+SBERT+Bias)
  instead of just SBERT score, for better reflection of overall relevance.
- Dynamically determines 'applicable_lodging_types' ('CL', 'VR', 'Both') based on
  anchor/attribute analysis and configurable hints.
- Includes detailed debug logging for the LLM negation candidate selection process.
- (Inherited from v34.0.13) Reverted LLM slotting prompt to v34.0.11 version.
- (Inherited from v34.0.12) Captures unthemed high-scoring concepts.
- (Inherited from v34.0.11) Dynamic requires_geo_check flag.
- (Inherited from v34.0.11) LLM Negation Identification & Filtering.
- bm2s library for keyword relevance scoring.
- ACS Enrichment, Concept Overrides, Namespace Biasing, Exact Match Priority, etc.
- Relies on shared utility functions in utils.py (v34.0.2+ expected with NaN checks).

Changes from v34.0.13:
- Modified attribute weighting logic in `apply_rules_and_finalize` to use
  'combined_score' from `original_candidates_map_for_reprompt` for proportionality.
- Added new helper function `determine_lodging_type`.
- Calls `determine_lodging_type` in `apply_rules_and_finalize` to dynamically set
  the `applicable_lodging_types` field in the output.
- Added `lodging_type_hints` section to DEFAULT_CONFIG and parameters to STAGE2_CONFIG.
- Added detailed debug logging around LLM negation candidate selection and call.
- Updated SCRIPT_VERSION and default cache/config versions.
"""

# --- Imports ---
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

    NUMPY_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: numpy not found.", file=sys.stderr)
    sys.exit(1)

try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF
    from rdflib import util as rdflib_util

    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found.", file=sys.stderr)

    def tqdm(iterable, *args, **kwargs):
        return iterable


try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    print(
        "Warning: pandas library not found. ACS Enrichment will be disabled.",
        file=sys.stderr,
    )
    PANDAS_AVAILABLE = False

try:
    import bm25s

    BM25S_AVAILABLE = True
except ImportError:
    print(
        "Warning: bm2s library not found. BM25s scoring will be disabled.",
        file=sys.stderr,
    )
    BM25S_AVAILABLE = False

    class Bm25sDummy:
        class BM25:
            pass

        @staticmethod
        def tokenize(text):
            return [str(t).split() for t in text]

    bm25s = Bm25sDummy()


# --- LLM Imports & Placeholders ---
class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class RateLimitError(Exception):
    pass


class DummyLLMClient:
    pass


OpenAI_Type = DummyLLMClient
GoogleAI_Type = DummyLLMClient
OpenAI = None
OPENAI_AVAILABLE = False
genai = None
GOOGLE_AI_AVAILABLE = False
try:
    from openai import (
        OpenAI as RealOpenAI,
        APITimeoutError as RealAPITimeoutError,
        APIConnectionError as RealAPIConnectionError,
        RateLimitError as RealRateLimitError,
    )

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
    GOOGLE_AI_AVAILABLE = False
    genai = None
    logging.error(f"Google AI import error: {e}")

# --- Utility Function Imports ---
try:
    from utils import (
        setup_logging,
        setup_detailed_loggers,
        normalize_concept,
        get_primary_label,
        get_concept_type_labels,
        get_sbert_model,
        load_affinity_config,
        get_cache_filename,
        load_cache,
        save_cache,
        load_taxonomy_concepts,
        precompute_taxonomy_embeddings,
        get_concept_embedding,
        get_batch_embedding_similarity,
        get_kg_data,
        build_keyword_label_index,
        save_results_json,
        RDFLIB_AVAILABLE as UTILS_RDFLIB_AVAILABLE,
        SENTENCE_TRANSFORMERS_AVAILABLE as UTILS_ST_AVAILABLE,
    )

    if RDFLIB_AVAILABLE != UTILS_RDFLIB_AVAILABLE:
        logging.warning("Mismatch in RDFLIB availability")
    if SENTENCE_TRANSFORMERS_AVAILABLE != UTILS_ST_AVAILABLE:
        logging.warning("Mismatch in SentenceTransformers availability")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed import from 'utils.py': {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(
        f"CRITICAL ERROR: Unexpected error importing from 'utils.py': {e}",
        file=sys.stderr,
    )
    sys.exit(1)
# --- End Imports ---

# --- Config Defaults & Constants ---
SCRIPT_VERSION = "affinity-rule-engine-v34.0.14 (Attr Weight by CombinedScore + Dyn Lodging Type + ...)"
DEFAULT_CACHE_VERSION = "v20250502.affinity.34.0.14" # Incremented version
DEFAULT_CONFIG_FILE = "./affinity_config_v34.12.json" # Expecting updated config
OUTPUT_FILE_TEMPLATE = "affinity_definitions_{cache_version}.json"
LOG_FILE_TEMPLATE = "affinity_generation_{cache_version}.log"

# ***** START: DEFAULT_CONFIG DICTIONARY DEFINITION *****
DEFAULT_CONFIG = {
    "output_dir": "./output_v34.14", # Updated output dir
    "cache_dir": "./cache_v34.14",   # Updated cache dir
    "taxonomy_dir": "./datasources/",
    "cache_version": DEFAULT_CACHE_VERSION,
    "sparql_endpoint": "http://localhost:7200/repositories/your-repo",
    "limit_per_concept": 100,
    "sbert_model_name": "sentence-transformers/all-mpnet-base-v2",
    "LLM_PROVIDER": "none",
    "LLM_MODEL": None,
    "global_alpha": 0.6,
    "min_sbert_score": 0.12,
    "keyword_dampening_threshold": 0.35,
    "keyword_dampening_factor": 0.15,
    "prioritize_exact_prefLabel_match": True,
    "preferred_namespaces": [
        "urn:expediagroup:taxonomies:core",
        "urn:expediagroup:taxonomies:acsPCS",
        "urn:expediagroup:taxonomies:acs",
        "urn:expediagroup:taxonomies:spaces",
        "urn:expediagroup:taxonomies:activities",
        "urn:expe:taxo:amenity-view-property-features",
        "urn:expe:taxo:property-media",
        "urn:expe:taxo:trip-preferences",
        "urn:expediagroup:taxonomies:lcm:",
        "http://schema.org/",
        "ontology.example.com",
    ],
    "NAMESPACE_BIASING": {
        "enabled": True,
        "core_boost_factor": 1.10,
        "boost_factor": 1.05,
        "context_boost_factor": 1.02,
        "penalty_factor": 0.95,
        "strong_penalty_factor": 0.85,
        "metadata_penalty_factor": 0.90,
        "lodging_amenity_ns": [
            "urn:expediagroup:taxonomies:acs",
            "urn:expediagroup:taxonomies:spaces",
            "urn:expe:taxo:hospitality",
            "urn:expe:taxo:amenity-view-property-features",
            "urn:expediagroup:taxonomies:lcm:",
            "urn:expediagroup:taxonomies:acsPCS",
            "urn:expediagroup:taxonomies:acsBaseAttribute",
        ],
        "activity_ns": [
            "urn:expediagroup:taxonomies:activities",
            "urn:expe:taxo:events",
        ],
        "location_context_ns": [
            "urn:expediagroup:taxonomies:gaia",
            "urn:expediagroup:taxonomies:places",
        ],
        "visual_context_ns": [
            "urn:expe:taxo:media-descriptors",
            "urn:expe:taxo:property-media",
        ],
        "preference_context_ns": [
            "urn:expe:taxo:trip-preferences",
            "urn:expe:taxo:personalization",
        ],
        "clearly_wrong_ns": ["urn:expe:taxo:cars:", "urn:expe:taxo:flights:"],
        "metadata_ns": [
            "urn:expe:taxo:checkout:",
            "urn:expe:taxo:payments:",
            "urn:expe:taxo:policies:",
            "urn:expe:taxo:review_categories:",
            "urn:expe:taxo:review_category_values:",
            "urn:expe:taxo:reviews-attributes:",
            "urn:expe:taxo:text:",
            "urn:expe:taxo:data-element-values:",
            "urn:expediagroup:taxonomies:acsDomainType:",
            "urn:expediagroup:taxonomies:acsBaseTerms:",
            "urn:expediagroup:taxonomies:taxonomy_management:",
            "urn:expediagroup:taxonomies:acsEnumerations:",
        ],
        "low_priority_ns": ["urn:expediagroup:taxonomies:tmpt:"],
    },
    "KEYWORD_SCORING_CONFIG": {
        "enabled": True,
        "algorithm": "bm25s",
        "bm25_min_score": 0.01,
        "bm25_top_n": 500,
    },
    "KG_CONFIG": {
        "pref_label_weight": 3,
        "alt_label_weight": 1,
        "definition_weight": 0,
        "acs_name_weight": 4,
        "acs_def_weight": 2,
    },
    "ACS_DATA_CONFIG": {
        "acs_data_path": "./datasources/transformed_acs_tracker.csv",
        "enable_acs_enrichment": True,
        "acs_name_weight": 4,
        "acs_def_weight": 2,
    },
    "STAGE1_CONFIG": {
        "MAX_CANDIDATES_FOR_LLM": 75,
        "EVIDENCE_MIN_SIMILARITY": 0.30,
        "MIN_KEYWORD_CANDIDATES_FOR_EXPANSION": 5,
        "ENABLE_KW_EXPANSION": True,
        "KW_EXPANSION_TEMPERATURE": 0.5,
    },
    "STAGE2_CONFIG": {
        "ENABLE_LLM_REFINEMENT": True,
        "LLM_TEMPERATURE": 0.2,
        "THEME_ATTRIBUTE_MIN_WEIGHT": 0.001,
        "TOP_N_DEFINING_ATTRIBUTES": 25,
        "UNTHEMED_CAPTURE_SCORE_PERCENTILE": 75,
        # --- ADDED DEFAULTS for lodging type determination ---
        "LODGING_TYPE_TOP_ATTR_CHECK": 10, # How many top attributes to check
        "LODGING_TYPE_CONFIDENCE_THRESHOLD": 0.6 # Ratio of CL/VR indicators needed
    },
    "LLM_API_CONFIG": {
        "MAX_RETRIES": 5,
        "RETRY_DELAY_SECONDS": 5,
        "REQUEST_TIMEOUT": 180,
    },
    "LLM_NEGATION_CONFIG": {
        "enabled": True,
        "temperature": 0.3,
        "max_candidates_to_check": 30,
    },
    "vrbo_default_subscore_weights": {
        "SentimentScore": 0.1,
        "GroupIntelligenceScore": 0.1,
    },
    "base_themes": {},
    "concept_overrides": {
        "example_concept": {
            "must_not_have": [{"uri": "urn:config:excluded"}],
            # "applicable_lodging_types": "Both" # Note: This static override is now ignored
        }
    },
    "master_subscore_list": [],
    # --- ADDED Default lodging type hints ---
    "lodging_type_hints": {
        "CL": [
            "urn:expediagroup:taxonomies:core:#Hotel",
            "urn:expediagroup:taxonomies:core:#Motel",
            "urn:expediagroup:taxonomies:core:#Resort",
            "urn:expediagroup:taxonomies:core:#Inn",
            "urn:expediagroup:taxonomies:core:#BedAndBreakfast",
            "urn:expediagroup:taxonomies:lcm:#Hotel",
            "urn:expediagroup:taxonomies:acs:#FrontDesk",
            "urn:expediagroup:taxonomies:acs:#RoomService",
            "urn:expediagroup:taxonomies:acs:#DailyHousekeeping"
        ],
        "VR": [
            "urn:expediagroup:taxonomies:core:#Cabin",
            "urn:expediagroup:taxonomies:core:#Villa",
            "urn:expediagroup:taxonomies:core:#House",
            "urn:expediagroup:taxonomies:core:#VacationRental",
            "urn:expediagroup:taxonomies:core:#Apartment",
            "urn:expediagroup:taxonomies:core:#Condo",
            "urn:expediagroup:taxonomies:lcm:#Cabin",
            "urn:expediagroup:taxonomies:lcm:#Villa",
            "urn:expediagroup:taxonomies:acs:#FullKitchen",
            "urn:expediagroup:taxonomies:acs:#PrivateYard",
            "urn:expediagroup:taxonomies:acs:#WasherDryerInUnit"
        ],
        "Both": [
           "urn:expediagroup:taxonomies:core:#Luxury",
           "urn:expediagroup:taxonomies:core:#PetFriendly",
           "urn:expediagroup:taxonomies:acs:#Pool",
           "urn:expediagroup:taxonomies:acs:#WiFi",
           "urn:expediagroup:taxonomies:core:#Amenity" # Example generic concept
       ]
     },
}
# ***** END: DEFAULT_CONFIG DICTIONARY DEFINITION *****

# --- List of known abstract/package concepts ---
ABSTRACT_CONCEPTS_LIST = ["luxury", "budget", "value"]

# --- Globals ---
_config_data: Optional[Dict] = None
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict[str, np.ndarray], List[str]]] = None
_keyword_label_index: Optional[Dict[str, Set[str]]] = None
_bm25_model: Optional[bm25s.BM25] = None
_keyword_corpus_uris: Optional[List[str]] = None
_acs_data: Optional[pd.DataFrame] = None
_openai_client: Optional[OpenAI_Type] = None
_google_client: Optional[GoogleAI_Type] = None
logger = logging.getLogger(__name__)
args: Optional[argparse.Namespace] = None


# --- Helper Functions ---
# ... (load_acs_data, get_theme_definition_for_prompt, get_dynamic_theme_config,
#      normalize_weights, deduplicate_attributes, validate_llm_assignments remain unchanged) ...
def load_acs_data(path: Optional[str]) -> Optional[pd.DataFrame]:
    # ... (code from v34.0.13) ...
    if not PANDAS_AVAILABLE: logger.error("Pandas library required but not available."); return None
    if not path: logger.warning("ACS data path not configured."); return None
    if not os.path.exists(path): logger.warning(f"ACS data file not found: '{path}'."); return None
    try:
        start_time = time.time(); acs_df = pd.read_csv(path, index_col="URI", low_memory=False)
        if "AttributeNameENG" not in acs_df.columns: acs_df["AttributeNameENG"] = ""
        else: acs_df["AttributeNameENG"] = acs_df["AttributeNameENG"].fillna("")
        if "ACS Definition" not in acs_df.columns: acs_df["ACS Definition"] = ""
        else: acs_df["ACS Definition"] = acs_df["ACS Definition"].fillna("")
        logging.info(f"Loaded ACS data '{path}' ({len(acs_df):,} entries) in {time.time() - start_time:.2f}s."); return acs_df
    except KeyError: logger.error(f"ACS file '{path}' missing 'URI' column."); return None
    except Exception as e: logger.error(f"Error loading ACS data '{path}': {e}."); return None

def get_theme_definition_for_prompt(theme_name: str, theme_data: Dict) -> str:
    # ... (code from v34.0.13) ...
    desc = theme_data.get("description")
    if isinstance(desc, str) and desc.strip(): return desc.strip()
    else:
        logger.warning(f"Theme '{theme_name}' missing desc.");
        return f"Theme related to {theme_name} ({theme_data.get('type', 'general')} aspects)."

def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: Dict) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    # ... (code from v34.0.13) ...
    base_themes = config.get("base_themes", {}); concept_overrides = config.get("concept_overrides", {})
    base_data = base_themes.get(theme_name);
    if not base_data: logger.error(f"Base theme '{theme_name}' not found!"); return "Optional", 0.0, None, None
    concept_override_data = concept_overrides.get(normalized_concept, {}); theme_override = concept_override_data.get("themes", {}).get(theme_name, {})
    merged = {**base_data, **theme_override}; rule = merged.get("rule_applied", merged.get("rule", "Optional"))
    if rule not in ["Must have 1", "Optional"]: rule = "Optional"
    weight = merged.get("weight", base_data.get("weight", 0.0))
    if not isinstance(weight, (int, float)) or weight < 0: weight = 0.0
    else: weight = float(weight)
    return rule, weight, merged.get("subScore"), merged.get("fallback_logic")

def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
    # ... (code from v34.0.13) ...
    total = sum(weights_dict.values())
    if total > 0: return {k: v / total for k, v in weights_dict.items()}
    else: return {k: 0.0 for k in weights_dict}

def deduplicate_attributes(attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # ... (code from v34.0.13) ...
    best: Dict[str, Dict[str, Any]] = {};
    for attr in attributes:
        uri = attr.get("uri"); weight = 0.0; current_weight = -1.0
        if not uri or not isinstance(uri, str): continue
        try: weight = float(attr.get("concept_weight", 0.0))
        except (ValueError, TypeError): pass
        current = best.get(uri)
        try: current_weight = (float(current.get("concept_weight", -1.0)) if current else -1.0)
        except (ValueError, TypeError): pass
        if current is None or weight > current_weight: best[uri] = attr
    return list(best.values())

def validate_llm_assignments(llm_response_data: Optional[Dict[str, Any]], uris_sent: Set[str], valid_themes: Set[str], concept_label: str, diag_llm: Dict) -> Optional[Dict[str, List[str]]]:
    # ... (code from v34.0.13) ...
    if not llm_response_data or "theme_assignments" not in llm_response_data: logger.error(f"[{concept_label}] LLM resp missing 'theme_assignments'."); diag_llm["error"] = "Missing 'theme_assignments'."; return None
    assigns = llm_response_data["theme_assignments"];
    if not isinstance(assigns, dict): logger.error(f"[{concept_label}] LLM assigns not dict."); diag_llm["error"] = "Assignments not dict."; return None
    validated: Dict[str, List[str]] = {}; uris_resp = set(assigns.keys()); extra = uris_resp - uris_sent; missing = uris_sent - uris_resp
    if extra: logger.warning(f"[{concept_label}] LLM returned {len(extra)} extra URIs. Ignoring.")
    if missing: logger.warning(f"[{concept_label}] LLM missing {len(missing)} URIs. Adding empty.");
    for uri in missing: assigns.setdefault(uri, [])
    parsed_count = 0; uris_proc = 0
    for uri, themes in assigns.items():
        if uri not in uris_sent: continue
        uris_proc += 1
        if not isinstance(themes, list): validated[uri] = []; continue #logger.warning(f"[{concept_label}] Invalid themes format for {uri}.");
        valid = [t for t in themes if isinstance(t, str) and t in valid_themes]; invalid = set(themes) - set(valid)
        if invalid: logger.warning(f"[{concept_label}] Invalid themes for {uri}: {invalid}")
        validated[uri] = valid; parsed_count += len(valid)
    diag_llm["assignments_parsed_count"] = parsed_count; diag_llm["uris_in_response_count"] = uris_proc
    return validated


# --- Indexing ---
# ... (build_keyword_index, get_candidate_concepts_keyword remain unchanged) ...
def build_keyword_index(config: Dict, taxonomy_concepts_cache: Dict[str, Dict], cache_dir: str, cache_version: str, rebuild_cache: bool) -> Tuple[Optional[bm25s.BM25], Optional[List[str]], Optional[Dict[str, Set[str]]]]:
    # ... (code from v34.0.13) ...
    global _bm25_model, _keyword_corpus_uris, _keyword_label_index, _acs_data, args; bm25_model, corpus_uris, label_index = None, None, None
    kw_scoring_cfg = config.get("KEYWORD_SCORING_CONFIG", {}); use_bm25 = (kw_scoring_cfg.get("enabled", False) and kw_scoring_cfg.get("algorithm", "").lower() == "bm25s" and BM25S_AVAILABLE)
    kg_cfg = config.get("KG_CONFIG", {}); acs_cfg = config.get("ACS_DATA_CONFIG", {}); pref_label_weight = int(kg_cfg.get("pref_label_weight", 3)); alt_label_weight = int(kg_cfg.get("alt_label_weight", 1)); definition_weight = int(kg_cfg.get("definition_weight", 0))
    acs_name_weight = int(kg_cfg.get("acs_name_weight", 0)); acs_def_weight = int(kg_cfg.get("acs_def_weight", 0)); enrichment_enabled = acs_cfg.get("enable_acs_enrichment", False) and (_acs_data is not None)
    if use_bm25:
        logger.info("BM25s Keyword Indexing enabled."); logger.info(f"KG Field Weights: pref x{pref_label_weight}, alt x{alt_label_weight}, def x{definition_weight}")
        if enrichment_enabled: logger.info(f"ACS Enrichment ENABLED: name x{acs_name_weight}, def x{acs_def_weight}")
        else: logger.info("ACS Enrichment DISABLED.")
        kgw_str = f"p{pref_label_weight}a{alt_label_weight}d{definition_weight}";
        if enrichment_enabled: kgw_str += f"_acs_n{acs_name_weight}d{acs_def_weight}"
        cache_params_for_filename = {"alg": "bm25s", "kgw": kgw_str}
        model_cache_file = get_cache_filename("bm25_model", cache_version, cache_dir, cache_params_for_filename, ".pkl"); uris_cache_file = get_cache_filename("bm25_corpus_uris", cache_version, cache_dir, cache_params_for_filename, ".pkl"); cache_valid = False
        if not rebuild_cache:
            cached_model = load_cache(model_cache_file, "pickle"); cached_uris = load_cache(uris_cache_file, "pickle")
            if (cached_model is not None and isinstance(cached_model, bm25s.BM25) and isinstance(cached_uris, list) and len(cached_uris) > 0):
                bm25_model, corpus_uris, cache_valid = (cached_model, cached_uris, True); logger.info(f"BM25s Model loaded from cache ({len(corpus_uris)} URIs).")
            else: logger.info("BM25s cache files missing or invalid. Rebuilding."); cache_valid = False
        if not cache_valid:
            logger.info("Rebuilding BM25s index...");
            try:
                corpus_strings, doc_uris = [], []; debug_mode = args.debug if args else False; disable_tqdm = not logger.isEnabledFor(logging.INFO) or debug_mode; enriched_count = 0
                logger.info("Preparing document strings for BM25s...")
                for uri, data in tqdm(sorted(taxonomy_concepts_cache.items()), desc="Prepare BM25s Docs", disable=disable_tqdm):
                    texts_to_join = [];
                    if not isinstance(data, dict): continue
                    pref_labels_raw = data.get("skos:prefLabel", []); alt_labels_raw = data.get("skos:altLabel", []); definitions_raw = data.get("skos:definition", [])
                    pref_labels = [str(lbl) for lbl in (pref_labels_raw if isinstance(pref_labels_raw, list) else [pref_labels_raw]) if lbl and isinstance(lbl, str)]
                    alt_labels = [str(lbl) for lbl in (alt_labels_raw if isinstance(alt_labels_raw, list) else [alt_labels_raw]) if lbl and isinstance(lbl, str)]
                    kg_definition = None
                    if isinstance(definitions_raw, list):
                        for defin in definitions_raw:
                            if defin and isinstance(defin, str) and defin.strip(): kg_definition = str(defin); break
                    elif isinstance(definitions_raw, str) and definitions_raw.strip(): kg_definition = definitions_raw
                    texts_to_join.extend(pref_labels * pref_label_weight); texts_to_join.extend(alt_labels * alt_label_weight);
                    if kg_definition and definition_weight > 0: texts_to_join.extend([kg_definition] * definition_weight)
                    if (enrichment_enabled and _acs_data is not None and uri in _acs_data.index):
                        try:
                            acs_record = _acs_data.loc[uri];
                            if isinstance(acs_record, pd.DataFrame): acs_record = acs_record.iloc[0]
                            acs_name = (acs_record.get("AttributeNameENG", "") if pd.notna(acs_record.get("AttributeNameENG", "")) else ""); acs_def = (acs_record.get("ACS Definition", "") if pd.notna(acs_record.get("ACS Definition", "")) else "")
                            if acs_name or acs_def: enriched_count += 1
                            if acs_name and acs_name_weight > 0: texts_to_join.extend([acs_name] * acs_name_weight)
                            if acs_def and acs_def_weight > 0: texts_to_join.extend([acs_def] * acs_def_weight)
                        except Exception as acs_lookup_err: logger.warning(f"Error enriching URI '{uri}': {acs_lookup_err}.")
                    raw_doc_text = " ".join(filter(None, texts_to_join)); corpus_strings.append(raw_doc_text); doc_uris.append(uri)
                initial_doc_count = len(corpus_strings)
                if initial_doc_count == 0: logger.warning("No documents generated."); bm25_model = None; corpus_uris = []
                else:
                    valid_indices = [i for i, doc_str in enumerate(corpus_strings) if doc_str and doc_str.strip()]
                    if len(valid_indices) < initial_doc_count:
                        empty_count = initial_doc_count - len(valid_indices); logger.warning(f"Found {empty_count} empty doc strings. Filtering.")
                        if doc_uris is None or len(doc_uris) != initial_doc_count: logger.error("FATAL: URI list mismatch."); bm25_model = None; corpus_uris = []
                        else: filtered_corpus_strings = [corpus_strings[i] for i in valid_indices]; filtered_doc_uris = [doc_uris[i] for i in valid_indices]; logger.info(f"Proceeding with {len(filtered_corpus_strings)} non-empty docs.")
                    else: filtered_corpus_strings = corpus_strings; filtered_doc_uris = doc_uris; logger.info("All generated doc strings non-empty.")
                    if filtered_corpus_strings and bm25_model is None:
                        logger.info(f"Tokenizing {len(filtered_corpus_strings)} documents for BM25s...");
                        try:
                            tokenized_corpus = bm25s.tokenize(filtered_corpus_strings); logger.info(f"Building BM25s index for {len(filtered_corpus_strings)} documents ({enriched_count} enriched).")
                            bm25_model = bm25s.BM25(); bm25_model.index(tokenized_corpus); corpus_uris = filtered_doc_uris; logger.debug(f"BM25s Model created/indexed. Type: {type(bm25_model)}")
                        except Exception as bm25_init_err: logger.error(f"BM25s init/index error: {bm25_init_err}", exc_info=True); bm25_model = None; corpus_uris = []
                    if bm25_model and corpus_uris: save_cache(bm25_model, model_cache_file, "pickle"); save_cache(corpus_uris, uris_cache_file, "pickle"); logger.info("BM25s model/URIs rebuilt/saved.")
                    else: logger.warning("Skipping BM25s cache saving due to build issues.")
            except Exception as e: logger.error(f"BM25s build error: {e}", exc_info=True); bm25_model, corpus_uris = None, None
        logger.debug("--- BM25s Index Diagnostics ---");
        logger.debug(f"BM25s Model: {'Ready' if bm25_model else 'Not available.'}");
        logger.debug(f"Corpus URIs: {len(corpus_uris) if corpus_uris is not None else 'None'}");
        logger.debug("--- End BM25s Diagnostics ---")
    else: logger.info("BM25s Keyword Indexing disabled or library unavailable.")
    try:
        label_index = build_keyword_label_index(taxonomy_concepts_cache)
        if label_index is not None: logger.info(f"Simple label index ready ({len(label_index)} keywords).")
        else: logger.warning("Failed simple label index build.")
    except Exception as e: logger.error(f"Error building label index: {e}", exc_info=True); label_index = None
    _bm25_model = bm25_model; _keyword_corpus_uris = corpus_uris; _keyword_label_index = label_index
    return bm25_model, corpus_uris, label_index

def get_candidate_concepts_keyword(query_texts: List[str], bm25_model: bm25s.BM25, corpus_uris: List[str], top_n: int, min_score: float = 0.01) -> List[Dict[str, Any]]:
    # ... (code from v34.0.13) ...
    global args;
    if not query_texts or bm25_model is None or not corpus_uris or not BM25S_AVAILABLE: return []
    query_string = " ".join(normalize_concept(text) for text in query_texts)
    if not query_string: logger.warning(f"Empty query string from input: {query_texts}"); return []
    #logger.debug(f"BM25s query string: '{query_string}'")
    try:
        tokenized_query_list = bm25s.tokenize([query_string]); results_indices, results_scores = bm25_model.retrieve(tokenized_query_list, k=top_n); candidates = []
        if (results_indices is not None and results_scores is not None and len(results_indices) > 0 and len(results_scores) > 0 and isinstance(results_indices[0], np.ndarray) and isinstance(results_scores[0], np.ndarray) and results_indices[0].size > 0 and results_scores[0].size > 0):
            indices_for_query = results_indices[0]; scores_for_query = results_scores[0]
            if args and args.debug:
                max_s = scores_for_query[0] if scores_for_query.size > 0 else 0.0; min_s = scores_for_query[-1] if scores_for_query.size > 0 else 0.0
                logger.debug(f"BM25s Scores (Top {scores_for_query.size}) - Min: {min_s:.4f}, Max: {max_s:.4f}")
                if max_s < min_score: logger.debug(f"Max BM25s score < min_score threshold ({min_score}). No candidates will be added.")
            for idx, score_val in zip(indices_for_query, scores_for_query):
                score = float(score_val)
                if score >= min_score:
                    corpus_index = int(idx)
                    if 0 <= corpus_index < len(corpus_uris): candidates.append({"uri": corpus_uris[corpus_index], "score": score, "method": "keyword_bm25s",})
                    else: logger.warning(f"BM25s returned invalid index {corpus_index} for corpus size {len(corpus_uris)}")
                else:
                    if args and args.debug:
                        logger.debug(f"Stopping BM25s candidate processing as score ({score:.4f}) is below threshold ({min_score}).")
                    break
        else: logger.debug("BM25s retrieve returned no results or empty arrays."); return []
        return candidates
    except Exception as e: logger.error(f"BM25s search error: {e}", exc_info=True); return []


# --- LLM Handling ---
# ... (get_openai_client, get_google_client, call_llm, construct_keyword_expansion_prompt remain unchanged) ...
def get_openai_client() -> Optional[OpenAI_Type]:
    # ... (code from v34.0.13) ...
    global _openai_client, _config_data;
    if not OPENAI_AVAILABLE: return None
    if _openai_client is None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY");
            if not api_key: logger.warning("OPENAI_API_KEY env var not set."); return None
            timeout = _config_data.get("LLM_API_CONFIG", {}).get("REQUEST_TIMEOUT", 60); _openai_client = OpenAI(api_key=api_key, timeout=timeout)
        except Exception as e: logger.error(f"OpenAI client init failed: {e}"); return None
    return _openai_client

def get_google_client() -> Optional[GoogleAI_Type]:
    # ... (code from v34.0.13) ...
    global _google_client, _config_data;
    if not GOOGLE_AI_AVAILABLE: return None
    if _google_client is None:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY");
            if not api_key: logger.warning("GOOGLE_API_KEY env var not set."); return None
            genai.configure(api_key=api_key); _google_client = genai
        except Exception as e: logger.error(f"Google AI config failed: {e}"); return None
    return _google_client

def call_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float, provider: str) -> Dict[str, Any]:
    # ... (code from v34.0.13) ...
    if not model_name: logger.error("LLM model missing."); return {"success": False, "error": "LLM model missing"}
    result = {"success": False, "response": None, "error": None, "attempts": 0}; client = None
    if provider == "openai":
        client = get_openai_client()
    elif provider == "google":
        client = get_google_client()
    else: result["error"] = f"Unsupported provider: {provider}"; logger.error(result["error"]); return result
    if not client: result["error"] = f"{provider} client unavailable."; logger.error(result["error"]); return result
    delay = _config_data.get("LLM_API_CONFIG", {}).get("RETRY_DELAY_SECONDS", 5)
    for attempt in range(max_retries + 1):
        result["attempts"] = attempt + 1; start_time = time.time();
        try:
            content = None
            if provider == "openai":
                if not isinstance(client, OpenAI_Type): raise RuntimeError("OpenAI client invalid type.")
                resp = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=temperature, timeout=timeout,)
                if resp.choices: content = resp.choices[0].message.content
            elif provider == "google":
                model_id = model_name;
                if not model_id.startswith("models/"): model_id = f"models/{model_id}"
                gen_cfg = client.types.GenerationConfig(candidate_count=1, temperature=temperature, response_mime_type="application/json",)
                safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",]]
                if hasattr(client, "GenerativeModel"):
                    model = client.GenerativeModel(model_id, system_instruction=system_prompt, safety_settings=safety_settings,)
                    resp = model.generate_content([user_prompt], generation_config=gen_cfg, request_options={"timeout": timeout},)
                    if (hasattr(resp, "prompt_feedback") and resp.prompt_feedback and resp.prompt_feedback.block_reason):
                        block_reason = getattr(resp.prompt_feedback, "block_reason", "?");
                        logger.warning(f"Gemini blocked. Reason:{block_reason}"); result["error"] = f"Blocked:{block_reason}"; return result
                    if hasattr(resp, "candidates") and resp.candidates:
                        if hasattr(resp.candidates[0], "content") and hasattr(resp.candidates[0].content, "parts"):
                            if resp.candidates[0].content.parts: content = resp.candidates[0].content.parts[0].text
                            else: logger.warning(f"Gemini response candidate parts are empty. Resp:{resp}"); content = None
                        else: logger.warning(f"Gemini response candidate structure unexpected. Resp:{resp}"); content = None
                    else: logger.warning(f"Gemini response has no candidates or text. Resp:{resp}"); content = None
                else: raise RuntimeError("Google AI client is not configured correctly or is of wrong type.")
            if content:
                try:
                    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
                    if json_match: cleaned_content = json_match.group(1)
                    elif "{" in content and "}" in content:
                        first_brace = content.find("{"); last_brace = content.rfind("}")
                        if (first_brace != -1 and last_brace != -1 and last_brace > first_brace): cleaned_content = content[first_brace : last_brace + 1]
                        else: cleaned_content = content.strip().strip("`")
                    else: cleaned_content = content.strip().strip("`")
                    output_data = json.loads(cleaned_content)
                    if isinstance(output_data, dict): result["success"] = True; result["response"] = output_data; logger.debug(f"{provider} call successful in {time.time() - start_time:.2f}s (Attempt {attempt+1})"); return result
                    else: logger.error(f"{provider} response parsed but is not a dict. Type:{type(output_data)}. Raw:{content[:200]}..."); result["error"] = f"LLM response is not a JSON object (dict), type was {type(output_data)}"
                except json.JSONDecodeError as e: logger.error(f"{provider} JSON parse error: {e}. Raw content snippet: {content[:500]}..."); result["error"] = f"JSON Parse Error: {e}"
                except Exception as e: logger.error(f"{provider} response processing error: {e}. Raw content snippet: {content[:500]}..."); result["error"] = f"Response Processing Error: {e}"
            else: logger.warning(f"{provider} response content was empty."); result["error"] = "Empty response from LLM"
        except (APITimeoutError, APIConnectionError, RateLimitError) as e: logger.warning(f"{provider} API Error on attempt {attempt + 1}: {type(e).__name__}"); result["error"] = f"{type(e).__name__}"
        except Exception as e: logger.error(f"{provider} Call Error on attempt {attempt + 1}: {e}", exc_info=True); result["error"] = str(e); return result
        should_retry = (attempt < max_retries and result["error"] is not None and "Blocked" not in str(result.get("error")));
        if should_retry: wait_time = delay * (2**attempt) + np.random.uniform(0, delay * 0.5); logger.info(f"Retrying LLM call in {wait_time:.2f}s... (Error: {result.get('error')})"); time.sleep(wait_time)
        else:
            if not result.get("success"): final_error = result.get("error", "Unknown error after retries"); logger.error(f"LLM call failed after {attempt + 1} attempts. Final Error: {final_error}")
            if not result.get("error"): result["error"] = "LLM call failed after all retries."
            return result
    return result

def construct_keyword_expansion_prompt(input_concept: str) -> Tuple[str, str]:
    # ... (code from v34.0.13) ...
    system_prompt = """You are a helpful assistant specializing in travel concepts and keyword analysis for search retrieval. You understand nuances like compound words and relationships between concepts. Your goal is to generate relevant keywords that will improve search results within a travel taxonomy. Output ONLY a valid JSON object containing a single key "keywords" with a list of strings as its value. Do not include any explanations or introductory text outside the JSON structure."""
    user_prompt = f"""
    Given the input travel concept: '{input_concept}'
    Your task is to generate a list of related keywords specifically useful for *improving keyword search retrieval* within a large travel taxonomy. Consider the following:
    1.  **Synonyms:** Include direct synonyms. 2.  **Constituent Parts:** If compound, include meaningful parts (e.g., 'americanbreakfast' -> 'american', 'breakfast'). 3.  **Related Activities/Concepts:** List highly relevant associated items (e.g., 'americanbreakfast' -> 'eggs', 'bacon'). 4.  **Hypernyms/Hyponyms (Optional):** Broader categories or specific examples. 5.  **Relevance:** Focus strictly on terms highly relevant to '{input_concept}' in travel context. 6.  **Simplicity:** Prefer single words or common short phrases.
    Example Input: 'budget hotel', Example Output: ```json {{"keywords": ["budget", "hotel", "cheap", "affordable", "economy", "value", "low cost", "inn", "motel"]}} ```
    Example Input: 'scuba diving', Example Output: ```json {{"keywords": ["scuba", "diving", "scuba diving", "underwater", "dive", "reef", "snorkel", "padi", "water sports"]}} ```
    Now, generate the keywords for: '{input_concept}'. Output ONLY the JSON object:"""
    return system_prompt.strip(), user_prompt.strip()

# --- LLM Prompt Construction ---
def construct_llm_slotting_prompt(
    input_concept: str,
    theme_definitions: List[Dict[str, Any]],
    candidate_details: List[Dict[str, Any]],
    args: argparse.Namespace, # Keep args in case needed for future prompt tweaks
) -> Tuple[str, str]:
    """Constructs system and user prompts for LLM theme slotting (v34.0.13 - Reverted Task)."""
    system_prompt = """You are an expert travel taxonomist creating structured definitions focused on lodging. Your task is to understand how various concepts relate to traveler intents and preferences when choosing accommodation. Output ONLY a valid JSON object with key "theme_assignments". Value is a dictionary mapping EACH candidate URI to a list of relevant theme names (empty list if none). Ensure ALL input URIs are keys in the output. Focus on semantic relevance and traveler value within the lodging context."""
    theme_defs_str = "\n".join(
        [
            f"- **{t.get('name', '?')}{' (M)' if t.get('is_must_have') else ''}**: {t.get('description','?')}"
            for t in theme_definitions
        ]
    )
    must_haves = (
        ", ".join([t["name"] for t in theme_definitions if t.get("is_must_have")])
        or "None"
    )
    cand_details_str = (
        "".join(
            f"\n{i+1}. URI:{c.get('uri','?')} L:'{c.get('prefLabel','?')}' Alt:{str(c.get('skos:altLabel',[]))[:50]} Def:'{(c.get('skos:definition', '') or '')[:100]}...' T:{c.get('type_labels',[])}"
            for i, c in enumerate(candidate_details)
        )
        or "\nNo candidates."
    )
    schema_example = '```json\n{\n  "theme_assignments": {\n    "URI_1": ["ThemeA"],\n    "URI_2": [],\n    "URI_3": ["ThemeC"]\n    // ... ALL URIs ...\n  }\n}\n```'

    # --- REVERTED (v34.0.11/v34.0.13) TASK STRING ---
    task = f"""Task: For EACH candidate URI, assess its relevance to the input concept '{input_concept}' **specifically within the context of lodging features, amenities, nearby points of interest, or traveler preferences related to accommodation**. List the applicable themes from the provided list. Ensure candidates semantically matching '{input_concept}' are strongly considered. EVERY URI must be a key in the output JSON."""
    # --- END REVERTED TASK STRING ---

    user_prompt = f"""Concept: '{input_concept}'\nThemes:\n{theme_defs_str}\nMandatory:[{must_haves}]\nCandidates:{cand_details_str}\n{task}\nOutput Format: ONLY JSON matching:\n{schema_example}"""
    return system_prompt.strip(), user_prompt.strip()

# ... (build_reprompt_prompt, construct_llm_negation_prompt remain unchanged) ...
def build_reprompt_prompt(input_concept: str, theme_name: str, theme_config: Dict, original_candidates_details_map: Dict[str, Dict]) -> Tuple[str, str]:
    # ... (code from v34.0.13) ...
    system_prompt = """You are assisting travel definition refinement focused on lodging. A mandatory theme needs assignments. Review original candidates ONLY for relevance to the specific theme **in the context of accommodation features or traveler preferences related to lodging**. Output ONLY JSON like {"theme_assignments": {"URI_Relevant_1": ["ThemeName"], ...}}. If none relevant, output {"theme_assignments": {}}."""
    desc = theme_config.get("description", "?"); hints = theme_config.get("hints", {}); hints_str = ""
    if hints.get("keywords"): hints_str += f"\n    - Keywords: {', '.join(hints['keywords'])}"
    if hints.get("uris"): hints_str += f"\n    - URIs: {', '.join(hints['uris'])}"
    if hints_str: hints_str = "  Hints:" + hints_str
    cand_list = ("".join(f"\n{i+1}. URI:{uri} L:'{cand.get('prefLabel','?')}' T:{cand.get('type_labels',[])}" for i, (uri, cand) in enumerate(original_candidates_details_map.items())) or "\nNo candidates.")
    schema = f'```json\n{{\n  "theme_assignments": {{\n    "URI_Relevant_1": ["{theme_name}"],\n // ONLY relevant URIs\n  }}\n}}\n```\nIf none: {{"theme_assignments": {{}} }}'
    instructions = f"""Instructions: Identify candidates relevant to the mandatory theme '{theme_name}', **considering how this theme relates to lodging features, amenities, or nearby points of interest important for a traveler's stay**. Assign at least one candidate if plausible within this context. Output ONLY valid JSON per schema."""
    user_prompt = f"""Re-evaluating '{input_concept}' for MANDATORY theme '{theme_name}'.\nTheme:\n- Name:{theme_name}\n- Desc:{desc}\n{hints_str}\nCandidates:{cand_list}\n{instructions}\nSchema:\n{schema}\nOutput:"""
    return system_prompt.strip(), user_prompt.strip()

def construct_llm_negation_prompt(input_concept: str, anchor_prefLabel: str, candidate_details: List[Dict[str, Any]]) -> Tuple[str, str]:
    # ... (code from v34.0.13) ...
    system_prompt = """You are a travel domain expert focused on identifying contradictory or negating concepts within a taxonomy. Your task is to review a list of candidate concepts and identify any that are semantically opposite, mutually exclusive, or strongly contradictory to the main input concept in a travel context. Output ONLY a valid JSON object containing a single key "negating_uris" with a list of URI strings as its value. Do not include URIs that are merely *unrelated*."""
    cand_list_str = "\n".join(f"- URI: {c.get('uri', '?')} Label: '{c.get('prefLabel', '?')}'" for c in candidate_details)
    schema_example = '```json\n{\n  "negating_uris": [\n    "URI_of_contradictory_concept_1",\n    "URI_of_opposite_concept_2"\n  ]\n}\n```\nIf none found: `{"negating_uris": []}`'
    user_prompt = f"""
Input Concept: '{input_concept}' (Anchor Label: '{anchor_prefLabel}')
Task: Review the following candidate concepts. Identify any URIs that represent concepts clearly **contradictory, negating, or opposite** to '{input_concept}' / '{anchor_prefLabel}' in the context of travel. Do NOT list concepts that are simply unrelated or different. Focus on direct opposites or strong incompatibilities.
Examples of Contradictions:
- If input is 'Luxury', 'Budget' or 'Economy' would be contradictory.
- If input is 'Beachfront', 'Mountain View' or 'City Center' would be contradictory.
- If input is 'Pet-friendly', 'Pets Not Allowed' would be contradictory.
- If input is 'Quiet', 'Nightlife Nearby' would be contradictory.
Candidates to review:
{cand_list_str}
Instructions: Return ONLY a JSON object containing a list of URIs for the concepts identified as contradictory or negating. Use the key "negating_uris". If no candidates are contradictory, return an empty list.
Output Format:
{schema_example}
"""
    return system_prompt.strip(), user_prompt.strip()


# --- Domain Inference and Biasing ---
def infer_concept_domain(candidate_uris: List[str], config: Dict, top_n_check: int = 50) -> str:
    # ... (code from v34.0.13 - Corrected version) ...
    if not candidate_uris: return "Unknown"
    bias_config = config.get("NAMESPACE_BIASING", {}); lodging_amenity_ns = tuple(bias_config.get("lodging_amenity_ns", ())); activity_ns = tuple(bias_config.get("activity_ns", ()))
    core_ns_prefix = "urn:expediagroup:taxonomies:core:"; lodging_count = 0; activity_count = 0; checked_count = 0
    for uri in candidate_uris[:top_n_check]:
        checked_count += 1
        if uri.startswith(core_ns_prefix): lodging_count += 1
        elif lodging_amenity_ns and uri.startswith(lodging_amenity_ns): lodging_count += 1
        elif activity_ns and uri.startswith(activity_ns): activity_count += 1
    if checked_count == 0: return "Unknown"
    lodging_ratio = lodging_count / checked_count; activity_ratio = activity_count / checked_count
    if lodging_ratio >= 0.4: return "Lodging/Amenity"
    elif activity_ratio >= 0.4: return "Activity"
    else: return "General"

# --- CORRECTED function ---
def apply_namespace_bias(score: float, candidate_uri: str, inferred_domain: str, config: Dict) -> Tuple[float, str]:
    """Applies namespace-based biasing to a candidate's score. Corrected in v34.0.13."""
    bias_config = config.get("NAMESPACE_BIASING", {})
    if not bias_config.get("enabled", False): return score, "Neutral (Biasing Disabled)"
    core_boost_factor = float(bias_config.get("core_boost_factor", 1.10)); boost_factor = float(bias_config.get("boost_factor", 1.05)); context_boost_factor = float(bias_config.get("context_boost_factor", 1.02))
    penalty_factor = float(bias_config.get("penalty_factor", 0.95)); strong_penalty_factor = float(bias_config.get("strong_penalty_factor", 0.85)); metadata_penalty_factor = float(bias_config.get("metadata_penalty_factor", 0.90))
    core_ns_prefix = "urn:expediagroup:taxonomies:core:"; lodging_amenity_ns = tuple(ns for ns in bias_config.get("lodging_amenity_ns", ()) if not ns.startswith(core_ns_prefix))
    activity_ns = tuple(bias_config.get("activity_ns", ())); location_context_ns = tuple(bias_config.get("location_context_ns", ())); visual_context_ns = tuple(bias_config.get("visual_context_ns", ()))
    preference_context_ns = tuple(bias_config.get("preference_context_ns", ())); clearly_wrong_ns = tuple(bias_config.get("clearly_wrong_ns", ())); metadata_ns = tuple(bias_config.get("metadata_ns", ()))
    low_priority_ns = tuple(bias_config.get("low_priority_ns", ()))
    reason = "Neutral"; new_score = score; initial_score = score; penalty_applied = False; domain_bias_applied = False
    if metadata_ns and candidate_uri.startswith(metadata_ns): new_score = initial_score * metadata_penalty_factor; reason = f"Penalty (Metadata NS {metadata_penalty_factor:.2f})"; return max(1e-9, new_score), reason
    if clearly_wrong_ns and candidate_uri.startswith(clearly_wrong_ns): new_score = initial_score * strong_penalty_factor; reason = f"Penalty (Wrong Domain {strong_penalty_factor:.2f})"; penalty_applied = True
    if low_priority_ns and candidate_uri.startswith(low_priority_ns):
        current_factor = new_score / initial_score if abs(initial_score) > 1e-9 else 1.0; new_score = initial_score * current_factor * penalty_factor
        reason += f" + LowPrio NS ({penalty_factor:.2f})" if penalty_applied else f"Penalty (LowPrio NS {penalty_factor:.2f})"; penalty_applied = True
    if not penalty_applied:
        if inferred_domain == "Lodging/Amenity":
            if candidate_uri.startswith(core_ns_prefix): new_score = initial_score * core_boost_factor; reason = f"Boost (Core NS Match {core_boost_factor:.2f})"; domain_bias_applied = True
            elif lodging_amenity_ns and candidate_uri.startswith(lodging_amenity_ns): new_score = initial_score * boost_factor; reason = f"Boost (Lodging Domain Match {boost_factor:.2f})"; domain_bias_applied = True
            elif location_context_ns and candidate_uri.startswith(location_context_ns): new_score = initial_score * context_boost_factor; reason = f"Boost (Location Context {context_boost_factor:.2f})"; domain_bias_applied = True
            elif visual_context_ns and candidate_uri.startswith(visual_context_ns): new_score = initial_score * context_boost_factor; reason = f"Boost (Visual Context {context_boost_factor:.2f})"; domain_bias_applied = True
            elif preference_context_ns and candidate_uri.startswith(preference_context_ns): new_score = initial_score * context_boost_factor; reason = f"Boost (Preference Context {context_boost_factor:.2f})"; domain_bias_applied = True
            elif activity_ns and candidate_uri.startswith(activity_ns): new_score = initial_score * penalty_factor; reason = f"Penalty (Activity NS vs Lodging {penalty_factor:.2f})"; domain_bias_applied = True
        elif inferred_domain == "Activity":
            if activity_ns and candidate_uri.startswith(activity_ns): new_score = initial_score * boost_factor; reason = f"Boost (Activity Domain Match {boost_factor:.2f})"; domain_bias_applied = True
            elif location_context_ns and candidate_uri.startswith(location_context_ns): new_score = initial_score * context_boost_factor; reason = f"Boost (Location Context {context_boost_factor:.2f})"; domain_bias_applied = True
            elif preference_context_ns and candidate_uri.startswith(preference_context_ns): new_score = initial_score * context_boost_factor; reason = f"Boost (Preference Context {context_boost_factor:.2f})"; domain_bias_applied = True
            else: # Check for lodging NS penalty within the Activity domain block
                is_lodging_ns = candidate_uri.startswith(core_ns_prefix) or (lodging_amenity_ns and candidate_uri.startswith(lodging_amenity_ns))
                if is_lodging_ns: new_score = initial_score * penalty_factor; reason = f"Penalty (Lodging NS vs Activity {penalty_factor:.2f})"; domain_bias_applied = True
    if not domain_bias_applied and not penalty_applied: reason = "Neutral"
    if new_score <= 0: logger.warning(f"Bias resulted in non-positive score ({new_score:.4f}) for {candidate_uri}. Resetting to small positive value."); new_score = 1e-9
    return new_score, reason


# --- Keyword Expansion Helper ---
# ... (expand_keywords_with_llm remains unchanged) ...
def expand_keywords_with_llm(concept_label: str, config: Dict, args: argparse.Namespace) -> Tuple[List[str], bool, Optional[str]]:
    # ... (code from v34.0.13) ...
    llm_api_cfg = config.get("LLM_API_CONFIG", {}); llm_stage1_cfg = config.get("STAGE1_CONFIG", {}); llm_stage2_cfg = config.get("STAGE2_CONFIG", {})
    timeout = int(llm_api_cfg.get("REQUEST_TIMEOUT", 180)); retries = int(llm_api_cfg.get("MAX_RETRIES", 5)); temp = float(llm_stage1_cfg.get("KW_EXPANSION_TEMPERATURE", llm_stage2_cfg.get("LLM_TEMPERATURE", 0.5)))
    success = False; error_message = None; final_keyword_terms = set(); original_normalized_words = set(w for w in normalize_concept(concept_label).split() if len(w) > 2); final_keyword_terms.update(original_normalized_words)
    try:
        sys_prompt, user_prompt = construct_keyword_expansion_prompt(concept_label)
        result = call_llm(sys_prompt, user_prompt, args.llm_model, timeout, retries, temp, args.llm_provider,)
        if result and result.get("success"):
            raw_phrases = result.get("response", {}).get("keywords", [])
            if isinstance(raw_phrases, list):
                added = set(term for kw in raw_phrases if isinstance(kw, str) and kw.strip() for term in normalize_concept(kw).split() if len(term) > 2)
                newly_expanded = added - original_normalized_words
                if newly_expanded: final_keyword_terms.update(newly_expanded); success = True; logger.debug(f"[{concept_label}] LLM KW expansion added {len(newly_expanded)} terms.")
                else: error_message = "LLM returned no new terms"; logger.info(f"[{concept_label}] LLM KW expansion: No new terms.")
            else: error_message = "LLM response invalid format (keywords not a list)"; logger.error(f"[{concept_label}] LLM KW expansion invalid response format.")
        else: err = result.get("error", "?") if result else "No result object"; error_message = f"LLM API Call Failed:{err}"; logger.warning(f"[{concept_label}] LLM KW expansion failed: {err}")
    except Exception as e: error_message = f"Exception during keyword expansion: {e}"; logger.error(f"[{concept_label}] KW expansion exception: {e}", exc_info=True)
    return list(final_keyword_terms), success, error_message



# --- Stage 1: Evidence Preparation (Corrected v34.0.14.1 - Restore v34.0.13 Filtering Logic) ---
def prepare_evidence(
    input_concept: str,
    concept_embedding: Optional[np.ndarray],
    primary_embeddings: Dict[str, np.ndarray],
    config: Dict,
    args: argparse.Namespace,
    bm25_model: Optional[bm25s.BM25],
    keyword_corpus_uris: Optional[List[str]],
    keyword_label_index: Optional[Dict[str, Set[str]]],
    taxonomy_concepts_cache: Dict[str, Dict],
) -> Tuple[
    List[Dict],
    Dict[str, Dict],
    Optional[Dict],
    Dict[str, float], # NOTE: This Dict[str, float] will be empty as we now rely on orig_map for combined_score
    Dict[str, Any],
    int,
    int,
    int,
]:
    """
    Prepares evidence candidates using combined score (BM25s + SBERT), config-driven overrides,
    refined contextual namespace biasing, and optional exact prefLabel match priority.
    (v34.0.14.1 - Restored v34.0.13 candidate filtering logic, kept debug logs and boost_thresh fix).
    """
    start_time_stage1 = time.time()
    normalized_input_concept = normalize_concept(input_concept)
    selection_method_log = "Combined BM25s+SBERT"  # Default

    def get_sort_priority(item_uri: str) -> int:
        preferred_ns = config.get("preferred_namespaces", [])
        for i, ns_prefix in enumerate(preferred_ns):
            if item_uri.startswith(ns_prefix):
                return i
        return len(preferred_ns) + 10

    # Load config values
    stage1_cfg = config.get("STAGE1_CONFIG", {})
    kw_scoring_cfg = config.get("KEYWORD_SCORING_CONFIG", {})
    damp_thresh = float(config.get("keyword_dampening_threshold", 0.35))
    damp_factor = float(config.get("keyword_dampening_factor", 0.15))
    max_cands = int(stage1_cfg.get("MAX_CANDIDATES_FOR_LLM", 75))
    min_sim_initial = float(stage1_cfg.get("EVIDENCE_MIN_SIMILARITY", 0.30)) # Renamed for clarity
    min_kw = float(kw_scoring_cfg.get("bm25_min_score", 0.01))
    kw_trigger = int(stage1_cfg.get("MIN_KEYWORD_CANDIDATES_FOR_EXPANSION", 5))
    kw_top_n = int(kw_scoring_cfg.get("bm25_top_n", 500))
    abs_min_sbert = float(config.get("min_sbert_score", 0.12)) # Absolute filter threshold
    global_alpha = float(config.get("global_alpha", 0.6))
    global_kw_exp_enabled = stage1_cfg.get("ENABLE_KW_EXPANSION", True)
    ns_bias_enabled = config.get("NAMESPACE_BIASING", {}).get("enabled", True)
    prioritize_exact_match = config.get("prioritize_exact_prefLabel_match", True)

    overrides = config.get("concept_overrides", {}).get(normalized_input_concept, {})
    skip_expansion = overrides.get("skip_expansion", False)
    kw_exp_enabled = (global_kw_exp_enabled and args.llm_provider != "none" and not skip_expansion)
    seed_uris = overrides.get("seed_uris", [])
    boost_cfg = overrides.get("boost_seeds_config", {"enabled": False})
    filter_uris = overrides.get("filter_uris", [])
    effective_alpha = float(overrides.get("effective_alpha", global_alpha))
    manual_query = overrides.get("manual_query_split", None)

    # Logging
    if overrides: logger.info(f"[{normalized_input_concept}] Applying concept overrides.")
    # ... (rest of initial logging identical to v34.0.14) ...
    logger.info(f"[{normalized_input_concept}] Effective Alpha:{effective_alpha:.2f} ({'Override' if abs(effective_alpha-global_alpha)>1e-9 else 'Default'})")
    logger.info(f"[{normalized_input_concept}] Abs Min SBERT:{abs_min_sbert}")
    logger.info(f"[{normalized_input_concept}] KW Dampen: Thresh={damp_thresh}, Factor={damp_factor}")
    if boost_cfg.get("enabled"): logger.info(f"[{normalized_input_concept}] Seed Boost: ON (Thresh:{boost_cfg.get('threshold','?')}) URIs:{seed_uris}")
    logger.info(f"[{normalized_input_concept}] LLM KW Expansion:{kw_exp_enabled}")
    logger.info(f"[{normalized_input_concept}] Namespace Biasing Enabled: {ns_bias_enabled}")
    logger.info(f"[{normalized_input_concept}] Prioritize Exact prefLabel Match (w/ Plurals): {prioritize_exact_match}")
    logger.info(f"[{normalized_input_concept}] Tie-Break Sort (Combined): Score>Namespace>SBERT")
    if prioritize_exact_match: logger.info(f"[{normalized_input_concept}] Tie-Break Sort (Exact Match): Namespace>SBERT")

    sim_scores, kw_scores = {}, {}
    exp_diag = {"attempted": False, "successful": False, "count": 0, "terms": [], "keyword_count": 0, "error": None,}

    # Keyword Expansion (identical to v34.0.14)
    # ...
    base_kws = set(kw for kw in normalized_input_concept.split() if len(kw) > 2)
    final_query_texts = list(base_kws); initial_kw_count = 0
    if keyword_label_index: initial_kw_count = len(set().union(*[keyword_label_index.get(kw, set()) for kw in base_kws]))
    needs_exp = kw_exp_enabled and (initial_kw_count < kw_trigger or normalized_input_concept in ABSTRACT_CONCEPTS_LIST)
    if kw_exp_enabled and needs_exp:
        exp_diag["attempted"] = True;
        logger.info(f"[{normalized_input_concept}] Attempting LLM KW expansion (Initial:{initial_kw_count}<{kw_trigger} or abstract).")
        final_query_texts, exp_success, exp_error = expand_keywords_with_llm(input_concept, config, args); exp_diag["successful"] = exp_success; exp_diag["error"] = exp_error
    if manual_query:
        if isinstance(manual_query, list) and all(isinstance(t, str) for t in manual_query):
            final_query_texts_set = set(normalize_concept(t) for t in final_query_texts); manual_query_set = set(normalize_concept(t) for t in manual_query)
            if final_query_texts_set != manual_query_set: logger.info(f"Applying manual query split, OVERRIDING: {manual_query}"); final_query_texts = [normalize_concept(t) for t in manual_query]; exp_diag["notes"] = "Manual query split."
        else: logger.warning(f"Invalid 'manual_query_split' for {normalized_input_concept}. Ignored.")
    exp_diag["terms"] = final_query_texts; exp_diag["count"] = len(final_query_texts)
    #if args.debug: logger.debug(f"[BM25s QUERY] '{normalized_input_concept}', Input Texts: {final_query_texts}")


    # Keyword Search (identical to v34.0.14)
    # ...
    kw_cand_count_init = 0; kw_cands_raw = []
    if bm25_model and keyword_corpus_uris:
        if final_query_texts:
            kw_cands_raw = get_candidate_concepts_keyword(final_query_texts, bm25_model, keyword_corpus_uris, kw_top_n, min_kw); kw_scores = {c["uri"]: c["score"] for c in kw_cands_raw}; kw_cand_count_init = len(kw_scores); exp_diag["keyword_count"] = kw_cand_count_init
          #  logger.debug(f"[{normalized_input_concept}] Found {kw_cand_count_init} BM25s candidates >= {min_kw} (retrieved top {kw_top_n}).")
        else:
            logger.warning(f"[{normalized_input_concept}] No keywords for BM25s search."); kw_scores = {}; exp_diag["keyword_count"] = 0
    else:
        logger.warning(f"[{normalized_input_concept}] BM25s unavailable."); kw_scores = {}; exp_diag["keyword_count"] = 0


    # SBERT Search (identical to v34.0.14)
    # ...
    sbert_cand_count_init = 0; sbert_cands_raw_map = {}
    if concept_embedding is None: logger.error(f"[{normalized_input_concept}] No anchor embed. Skipping SBERT."); sim_scores = {}
    else:
        sbert_cands_raw_map = get_batch_embedding_similarity(concept_embedding, primary_embeddings)
        sim_scores = {u: s for u, s in sbert_cands_raw_map.items() if s >= min_sim_initial} # Use initial threshold here
        sbert_cand_count_init = len(sim_scores)
        logger.debug(f"[{normalized_input_concept}] Found {sbert_cand_count_init} SBERT candidates >= {min_sim_initial:.2f} (out of {len(sbert_cands_raw_map)} raw).")


    # Combine, Seed, Filter (identical to v34.0.14)
    # ...
    all_uris_for_exact_match = set(kw_scores.keys()) | set(sbert_cands_raw_map.keys()); all_uris = set(kw_scores.keys()) | set(sim_scores.keys()); initial_unique = len(all_uris)
    if seed_uris:
        added = set(seed_uris) - all_uris_for_exact_match;
        if added: logger.info(f"SEEDING '{normalized_input_concept}' with {len(added)} URIs: {added}")
        all_uris.update(added); all_uris_for_exact_match.update(added); needed_sbert = added - set(sbert_cands_raw_map.keys())
        if needed_sbert and concept_embedding is not None:
            seed_embs = {u: primary_embeddings[u] for u in needed_sbert if u in primary_embeddings}
            if seed_embs:
                new_sims = get_batch_embedding_similarity(concept_embedding, seed_embs); sbert_cands_raw_map.update(new_sims);
                sim_scores.update({u: s for u, s in new_sims.items() if s >= min_sim_initial}) # Use initial threshold
                logger.debug(f"Calculated SBERT for {len(new_sims)} seeds.")
            else: logger.warning(f"No embeddings for seeds: {needed_sbert}")
    if filter_uris:
        to_remove = set(u for u in filter_uris if u in all_uris_for_exact_match);
        if to_remove: logger.info(f"Applying filter: Removing {len(to_remove)} URIs: {to_remove}")
        all_uris.difference_update(to_remove); all_uris_for_exact_match.difference_update(to_remove)


    # Exact Match Check (identical to v34.0.14 - checks against abs_min_sbert)
    # ...
    exact_match_uris = []
    if prioritize_exact_match:
        logger.debug(f"[{normalized_input_concept}] Checking for exact prefLabel matches (with simple plural check)..."); match_count = 0; norm_input = normalized_input_concept
        for uri in all_uris_for_exact_match:
            if uri not in all_uris: continue # Check against the potentially filtered URI list
            concept_data = taxonomy_concepts_cache.get(uri);
            if not concept_data: continue
            pref_label = get_primary_label(uri, taxonomy_concepts_cache);
            if pref_label:
                norm_label = normalize_concept(pref_label); is_match = False; match_type = ""
                if norm_input == norm_label: is_match = True; match_type = "Exact"
                elif (norm_input.endswith("s") and len(norm_input) > 1 and norm_input[:-1] == norm_label): is_match = True; match_type = "Plural input"
                elif (norm_label.endswith("s") and len(norm_label) > 1 and norm_input == norm_label[:-1]): is_match = True; match_type = "Plural label"
                if is_match:
                    sbert_score_for_exact = sbert_cands_raw_map.get(uri, 0.0)
                    if sbert_score_for_exact >= abs_min_sbert:
                        exact_match_uris.append(uri); match_count += 1;
                        if args.debug: logger.debug(f"  {match_type} match found AND passed SBERT threshold: {uri} (Input: '{norm_input}', Label: '{norm_label}', Score: {sbert_score_for_exact:.4f})")
                    elif args.debug: logger.debug(f"  {match_type} match found BUT FAILED SBERT threshold: {uri} (Input: '{norm_input}', Label: '{norm_label}', Score: {sbert_score_for_exact:.4f} < {abs_min_sbert})")
        logger.info(f"[{normalized_input_concept}] Found {match_count} exact prefLabel matches (incl. simple plurals) passing SBERT threshold {abs_min_sbert}.")


    all_uris_list = list(all_uris); unique_cands_before_rank = len(all_uris_list)
    if not all_uris_list: logger.warning(f"[{normalized_input_concept}] No candidates after filter/seed."); return [], {}, None, {}, exp_diag, sbert_cand_count_init, kw_cand_count_init, 0

    # Infer Domain (identical to v34.0.14)
    # ...
    inferred_domain = "Unknown"
    if ns_bias_enabled: initial_candidate_uris_for_domain = list(set([c["uri"] for c in kw_cands_raw] + list(sbert_cands_raw_map.keys()))); inferred_domain = infer_concept_domain(initial_candidate_uris_for_domain, config); logger.info(f"[{normalized_input_concept}] Inferred domain (based on raw cands): {inferred_domain}")


    all_details = get_kg_data(all_uris_list, taxonomy_concepts_cache)
    scored_list = []
    abs_filt = 0; damp_cnt = 0; bias_cnt = 0
    alpha_overridden = abs(effective_alpha - global_alpha) > 1e-9
    # --- FIX: Initialize boost_thresh BEFORE the loop ---
    boost_thresh = float(boost_cfg.get("threshold", 0.80)) if boost_cfg.get("enabled", False) else 1.1 # Set default high if not enabled

    for uri in all_uris_list:
        if uri not in all_details: continue
        s_orig = sbert_cands_raw_map.get(uri, 0.0); k_raw = kw_scores.get(uri, 0.0); s_boosted = s_orig; boosted = False
        if uri in seed_uris and boost_cfg.get("enabled", False):
            # Re-assign boost_thresh if boosting is active for this concept
            boost_thresh = float(boost_cfg.get("threshold", 0.80))
            if s_orig < boost_thresh: s_boosted = boost_thresh; boosted = True

        current_sbert_for_filter = s_boosted if boosted else s_orig

        # --- Debug Log Added in v34.0.14 ---
       # logger.debug(f"[{normalized_input_concept}] PRE-FILTER CHECK for URI {uri}: current_sbert_for_filter={current_sbert_for_filter:.6f}, abs_min_sbert={abs_min_sbert:.6f}, boosted={boosted}, s_orig={s_orig:.6f}")

        # Apply absolute SBERT filter (Ensure this logic is IDENTICAL to v34.0.13)
        if current_sbert_for_filter < abs_min_sbert:
            abs_filt += 1
          #  if args.debug:
            #    logger.debug(f"[{normalized_input_concept}] URI {uri} filtered by abs_min_sbert ({current_sbert_for_filter:.4f} < {abs_min_sbert})")
            continue # Skip to next URI

        # Apply Dampening (identical to v34.0.13)
        k_damp = k_raw; dampened = False
        if current_sbert_for_filter < damp_thresh: # Use SBERT *after* potential boost
            k_damp *= damp_factor;
            if k_raw > 0 and abs(k_raw - k_damp) > 1e-9: dampened = True; damp_cnt += 1

        # Combine Scores (identical to v34.0.13)
        norm_s = max(0.0, min(1.0, current_sbert_for_filter)); norm_k = max(0.0, min(1.0, k_damp)); combined_unbiased = (effective_alpha * norm_k) + ((1.0 - effective_alpha) * norm_s)

        # Apply Bias (identical to v34.0.13)
        combined = combined_unbiased; bias_reason = "Neutral"; biased = False
        if ns_bias_enabled and not boosted: # Don't bias boosted seeds
             combined, bias_reason = apply_namespace_bias(combined_unbiased, uri, inferred_domain, config);
             if abs(combined - combined_unbiased) > 1e-9: biased = True; bias_cnt += 1

        # Append to scored_list (Ensure this happens if filter is passed)
        scored_list.append({
            "uri": uri, "details": all_details[uri], "sim_score": s_orig, "boosted_sim_score": s_boosted if boosted else None,
            "kw_score": k_raw, "dampened_kw_score": k_damp if dampened else None, "alpha_overridden": alpha_overridden,
            "effective_alpha": effective_alpha, "combined_score_unbiased": combined_unbiased, "combined_score": combined,
            "biased": biased, "bias_reason": bias_reason,
        })
        # --- END of scoring loop ---

    # Logging & Sorting (identical to v34.0.13)
    logger.info(f"[{normalized_input_concept}] Excluded {abs_filt} (SBERT<{abs_min_sbert}). Dampened KW for {damp_cnt} (SBERT<{damp_thresh}). Biased {bias_cnt} scores (Domain:{inferred_domain}).")
    if alpha_overridden: logger.info(f"[{normalized_input_concept}] Applied alpha override: {effective_alpha:.2f}.")
    if not scored_list: logger.warning(f"[{normalized_input_concept}] No candidates remain after scoring."); return ([], {}, None, {}, exp_diag, sbert_cand_count_init, kw_cand_count_init, unique_cands_before_rank,)
    logger.debug(f"[{normalized_input_concept}] Sorting {len(scored_list)} candidates by combined score..."); scored_list.sort(key=lambda x: x.get("sim_score", 0.0), reverse=True); scored_list.sort(key=lambda x: get_sort_priority(x.get("uri", "")), reverse=False); scored_list.sort(key=lambda x: x["combined_score"], reverse=True)
    if args.debug: # Debug logs for top 5 candidates (identical to v34.0.13)
        logger.debug(f"--- Top 5 for '{normalized_input_concept}' (Sorted by Combined Score) ---"); logger.debug(f"    (EffAlpha:{effective_alpha:.2f}, DampThresh:{damp_thresh}, DampFactor:{damp_factor}), Domain: {inferred_domain}, Bias:{ns_bias_enabled}")
        for i, c in enumerate(scored_list[:5]):
            damp = (f"(Damp:{c.get('dampened_kw_score'):.4f})" if c.get("dampened_kw_score") is not None else ""); boost = (f"(Boost:{c.get('boosted_sim_score'):.4f})" if c.get("boosted_sim_score") is not None else ""); alpha = f"(EffA:{c.get('effective_alpha'):.2f})"; prio = f"(P:{get_sort_priority(c.get('uri',''))})"; lbl = get_primary_label(c.get("uri", "?"), taxonomy_concepts_cache, c.get("uri", "?")); bias_info = (f" -> Biased:{c.get('combined_score'):.6f} ({c.get('bias_reason','?')})" if c.get("biased") else "")
           # logger.debug(f"{i+1}. URI:{c.get('uri','?')} {prio} L:'{lbl}' FinalScore:{c.get('combined_score'):.6f} (Unbiased:{c.get('combined_score_unbiased'):.6f}{alpha}) (SBERT:{c.get('sim_score'):.4f}{boost}, KW:{c.get('kw_score'):.4f}{damp}) {bias_info}")
        logger.debug("--- End Top 5 (Combined Score) ---")

    # Select candidates for LLM (identical to v34.0.13)
    selected = scored_list[:max_cands];
    logger.info(f"[{normalized_input_concept}] Selected top {len(selected)} candidates for LLM based on combined score.")
    if not selected: logger.warning(f"[{normalized_input_concept}] No candidates selected for LLM."); return ([], {}, None, {}, exp_diag, sbert_cand_count_init, kw_cand_count_init, unique_cands_before_rank,)

    # Prepare data for LLM (orig_map now contains combined_score, needed for Stage 2 weighting)
    llm_details = [c["details"] for c in selected]; orig_map = {}
    for c in selected:
        uri = c.get("uri"); details = c.get("details")
        if uri and details:
            entry = {**details, "sbert_score": c.get("sim_score"), "keyword_score": c.get("kw_score"), "combined_score": c.get("combined_score"), "combined_score_unbiased": c.get("combined_score_unbiased"), "biased": c.get("biased"), "effective_alpha": c.get("effective_alpha"),}
            if c.get("boosted_sim_score"): entry["boosted_sbert_score"] = c.get("boosted_sim_score")
            if c.get("dampened_kw_score"): entry["dampened_keyword_score"] = c.get("dampened_kw_score")
            if c.get("bias_reason") != "Neutral": entry["bias_reason"] = c.get("bias_reason")
            orig_map[uri] = entry

    # Select anchor candidate (identical to v34.0.13)
    anchor_uri = None; anchor_data_dict = None
    if prioritize_exact_match and exact_match_uris:
        # Exact match URIs already pre-filtered by abs_min_sbert
        selection_method_log = "Exact prefLabel Match (Plural Check)"
        # Need to sort the filtered exact_match_uris based on the full scored_list ranking
        # Create a map of URI to its rank in scored_list
        score_rank_map = {item['uri']: rank for rank, item in enumerate(scored_list)}
        # Filter exact_match_uris to only those present in scored_list (should be all after abs_min_sbert fix, but safety check)
        valid_exact_uris = [uri for uri in exact_match_uris if uri in score_rank_map]
        # Sort the valid exact matches by their rank in the combined score list
        valid_exact_uris.sort(key=lambda uri: score_rank_map[uri])

        if valid_exact_uris:
            selected_exact_match_uri = valid_exact_uris[0]
            # Find the corresponding entry in scored_list
            anchor_data_dict = next((item for item in scored_list if item['uri'] == selected_exact_match_uri), None)
            if anchor_data_dict:
                anchor_uri = selected_exact_match_uri
                exact_match_details = scored_list[score_rank_map[anchor_uri]] # Get details from full list
                logger.info(f"[{normalized_input_concept}] Selected exact match anchor URI: {anchor_uri} (Rank: {score_rank_map[anchor_uri]+1}, SBERT: {exact_match_details.get('sim_score'):.4f})")
            else:
                logger.error(f"[{normalized_input_concept}] CRITICAL: Could not find data in scored_list for exact match anchor {selected_exact_match_uri}! Falling back.")
                anchor_uri = None; selection_method_log = "Combined BM25s+SBERT (Exact Match Inconsistency Fallback)"
        else:
             logger.warning(f"[{normalized_input_concept}] No exact matches remain after filtering/ranking. Falling back.")
             selection_method_log = "Combined BM25s+SBERT (Exact Match Fallback)"

    # Fallback to top combined score (identical to v34.0.13)
    if anchor_uri is None:
        if not scored_list: logger.error(f"[{normalized_input_concept}] No candidates in scored_list for fallback anchor selection."); anchor_data_dict = None
        else:
            anchor_data_dict = scored_list[0]; anchor_uri = anchor_data_dict.get("uri"); selection_method_log = "Combined BM25s+SBERT"
            if prioritize_exact_match and exact_match_uris: logger.info(f"[{normalized_input_concept}] Using fallback anchor (combined score): {anchor_uri}")

    # Prepare final anchor object (identical to v34.0.13)
    anchor = None
    if anchor_data_dict and anchor_uri:
        anchor = {"uri": anchor_uri, **anchor_data_dict.get("details", {}), "sbert_score": anchor_data_dict.get("sim_score"), "keyword_score": anchor_data_dict.get("kw_score"), "combined_score": anchor_data_dict.get("combined_score"), "combined_score_unbiased": anchor_data_dict.get("combined_score_unbiased"), "biased": anchor_data_dict.get("biased"), "effective_alpha": anchor_data_dict.get("effective_alpha"),}
        if anchor_data_dict.get("boosted_sim_score") is not None: anchor["boosted_sbert_score"] = anchor_data_dict.get("boosted_sim_score")
        if anchor_data_dict.get("dampened_kw_score") is not None: anchor["dampened_keyword_score"] = anchor_data_dict.get("dampened_kw_score")
        if anchor_data_dict.get("bias_reason") != "Neutral": anchor["bias_reason"] = anchor_data_dict.get("bias_reason")
        anchor_label = get_primary_label(anchor_uri, taxonomy_concepts_cache, anchor_uri)
        score_display = (f"Exact Match (Rank: {score_rank_map.get(anchor_uri, -1)+1}, SBERT: {anchor.get('sbert_score', 0.0):.4f})" if selection_method_log.startswith("Exact prefLabel Match") else f"Score: {anchor.get('combined_score'):.6f}{' - Biased' if anchor.get('biased') else ''}")
        logger.info(f"[{normalized_input_concept}] Anchor ({selection_method_log}): {anchor_label} ({score_display})")
    elif anchor_uri: logger.error(f"[{normalized_input_concept}] Anchor URI '{anchor_uri}' selected but data missing!"); anchor = {"uri": anchor_uri, "name": input_concept, "type": "Unknown", "error": "Data missing",}
    else: logger.error(f"[{normalized_input_concept}] Failed to select any anchor."); anchor = {"uri": None, "name": input_concept, "type": "Unknown", "error": "No anchor selected",}

    # sbert_scores_final no longer needed as primary input for Stage 2
    exp_diag["selection_method"] = selection_method_log; stage1_duration = time.time() - start_time_stage1; exp_diag["duration_seconds"] = round(stage1_duration, 2)
    return (llm_details, orig_map, anchor, {}, exp_diag, sbert_cand_count_init, kw_cand_count_init, unique_cands_before_rank,)



# --- NEW Function: Determine Lodging Type (v34.0.14) ---
# ... (Function code identical to previous regeneration) ...
def determine_lodging_type(travel_category: Optional[Dict], top_defining_attributes: List[Dict], config: Dict, taxonomy_concepts_cache: Dict[str, Dict]) -> str:
    hints = config.get("lodging_type_hints", {}); cl_hints = set(hints.get("CL", [])); vr_hints = set(hints.get("VR", [])); both_hints = set(hints.get("Both", []))
    stage2_cfg = config.get("STAGE2_CONFIG", {}); top_n_check = int(stage2_cfg.get("LODGING_TYPE_TOP_ATTR_CHECK", 10)); threshold_ratio = float(stage2_cfg.get("LODGING_TYPE_CONFIDENCE_THRESHOLD", 0.6))
    cl_score = 0; vr_score = 0; checked_count = 0; decision_reason = []; final_type = "Both"
    if travel_category and travel_category.get("uri"):
        anchor_uri = travel_category["uri"]; checked_count += 1
        if anchor_uri in cl_hints: cl_score += 2; decision_reason.append(f"Anchor URI ({get_primary_label(anchor_uri, taxonomy_concepts_cache, anchor_uri)}) in CL hints."); final_type = "CL"
        elif anchor_uri in vr_hints: vr_score += 2; decision_reason.append(f"Anchor URI ({get_primary_label(anchor_uri, taxonomy_concepts_cache, anchor_uri)}) in VR hints."); final_type = "VR"
        elif anchor_uri in both_hints: decision_reason.append(f"Anchor URI ({get_primary_label(anchor_uri, taxonomy_concepts_cache, anchor_uri)}) in Both hints.")
    if final_type == "Both" and top_defining_attributes:
         attributes_to_check = top_defining_attributes[:top_n_check]; attr_cl_score = 0; attr_vr_score = 0; attr_checked_count = 0
         for attr in attributes_to_check:
             uri = attr.get("uri");
             if not uri: continue
             attr_checked_count += 1
             if uri in cl_hints: attr_cl_score += 1; decision_reason.append(f"Attr URI ({get_primary_label(uri, taxonomy_concepts_cache, uri)}) in CL hints.")
             elif uri in vr_hints: attr_vr_score += 1; decision_reason.append(f"Attr URI ({get_primary_label(uri, taxonomy_concepts_cache, uri)}) in VR hints.")
             elif uri in both_hints: decision_reason.append(f"Attr URI ({get_primary_label(uri, taxonomy_concepts_cache, uri)}) in Both hints.")
         if attr_checked_count > 0:
             total_attr_score = attr_cl_score + attr_vr_score
             if total_attr_score > 0:
                 cl_attr_ratio = attr_cl_score / total_attr_score; vr_attr_ratio = attr_vr_score / total_attr_score
                 decision_reason.append(f"Attr scores: CL={attr_cl_score}, VR={attr_vr_score}. Ratios: CL={cl_attr_ratio:.2f}, VR={vr_attr_ratio:.2f}")
                 if cl_attr_ratio >= threshold_ratio: final_type = "CL"
                 elif vr_attr_ratio >= threshold_ratio: final_type = "VR"
             else: decision_reason.append("No decisive CL/VR attributes found in top N.")
         else: decision_reason.append("No top attributes to check for hints.")
    elif final_type != "Both": decision_reason.append("Decision made based on anchor hint.")
    else: decision_reason.append("No anchor or attributes to check.")
    logger.debug(f"Lodging Type Check: Final Type={final_type}. Reasons: {'; '.join(decision_reason)}")
    return final_type



# --- NEW Function: Determine Lodging Type ---
def determine_lodging_type(
    travel_category: Optional[Dict],
    top_defining_attributes: List[Dict],
    config: Dict,
    taxonomy_concepts_cache: Dict[str, Dict],
) -> str:
    """Determines if concept applies more to CL, VR, or Both."""
    hints = config.get("lodging_type_hints", {})
    cl_hints = set(hints.get("CL", []))
    vr_hints = set(hints.get("VR", []))
    both_hints = set(hints.get("Both", [])) # Not strictly needed for scoring but good for logging

    stage2_cfg = config.get("STAGE2_CONFIG", {})
    top_n_check = int(stage2_cfg.get("LODGING_TYPE_TOP_ATTR_CHECK", 10))
    threshold_ratio = float(stage2_cfg.get("LODGING_TYPE_CONFIDENCE_THRESHOLD", 0.6))

    cl_score = 0
    vr_score = 0
    checked_count = 0
    decision_reason = []

    # Check Anchor Concept
    if travel_category and travel_category.get("uri"):
        anchor_uri = travel_category["uri"]
        checked_count += 1 # Count anchor check
        if anchor_uri in cl_hints:
            cl_score += 2 # Strong signal
            decision_reason.append(f"Anchor URI ({anchor_uri}) in CL hints.")
        elif anchor_uri in vr_hints:
            vr_score += 2 # Strong signal
            decision_reason.append(f"Anchor URI ({anchor_uri}) in VR hints.")
        elif anchor_uri in both_hints:
             # Neutral signal, doesn't strongly sway
             decision_reason.append(f"Anchor URI ({anchor_uri}) in Both hints.")
        else:
             decision_reason.append(f"Anchor URI ({anchor_uri}) not in hints.")
             # Optional: Check anchor type labels (less reliable)
             # anchor_types = travel_category.get("type_labels", [])
             # if any(t in cl_type_hints for t in anchor_types): cl_score += 0.5
             # if any(t in vr_type_hints for t in anchor_types): vr_score += 0.5

    # Check Top Defining Attributes
    attributes_to_check = top_defining_attributes[:top_n_check]
    for attr in attributes_to_check:
        uri = attr.get("uri")
        if not uri: continue
        checked_count += 1 # Count attribute check
        if uri in cl_hints:
            cl_score += 1
            decision_reason.append(f"Attr URI ({uri}) in CL hints.")
        elif uri in vr_hints:
            vr_score += 1
            decision_reason.append(f"Attr URI ({uri}) in VR hints.")
        elif uri in both_hints:
             decision_reason.append(f"Attr URI ({uri}) in Both hints.")
        # Optional: Check attribute type labels
        # attr_types = attr.get("type", []) # Assumes type_labels are nested as 'type' here
        # if any(t in cl_type_hints for t in attr_types): cl_score += 0.25
        # if any(t in vr_type_hints for t in attr_types): vr_score += 0.25

    # Make decision
    final_type = "Both" # Default
    if checked_count > 0: # Avoid division by zero if no anchor/attrs
        cl_ratio = cl_score / checked_count
        vr_ratio = vr_score / checked_count

        if cl_ratio >= threshold_ratio and cl_ratio > vr_ratio:
            final_type = "CL"
        elif vr_ratio >= threshold_ratio and vr_ratio > cl_ratio:
            final_type = "VR"

    logger.debug(f"Lodging Type Check: CL Score={cl_score}, VR Score={vr_score}, Checked={checked_count}. Threshold Ratio={threshold_ratio}. Result={final_type}.")
    logger.debug(f"Lodging Type Decision Reasons: {'; '.join(decision_reason)}")
    return final_type


# --- Stage 2: Finalization (v34.0.14 - Attr Weight by CombinedScore + Dyn Lodging Type) ---
def apply_rules_and_finalize(
    input_concept: str,
    llm_call_result: Optional[Dict[str, Any]],
    config: Dict,
    travel_category: Optional[Dict],
    anchor_candidate: Optional[Dict],
    original_candidates_map_for_reprompt: Dict[str, Dict], # Now includes combined_score
    # sbert_scores_final: Dict[str, float], # Removed, use original_candidates_map_for_reprompt
    args: argparse.Namespace,
    taxonomy_concepts_cache: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Applies rules, handles fallbacks, identifies LLM negations,
    populates must_not_have, FILTERS definition attributes,
    weights attributes based on COMBINED SCORE, calculates subscores,
    DYNAMICALLY sets requires_geo_check & applicable_lodging_types,
    CAPTURES unthemed high-scoring concepts, and structures final output.
    """
    # ... (Initial setup identical to v34.0.14 previous generation) ...
    start_stage2 = time.time(); norm_concept = normalize_concept(input_concept); concept_overrides = config.get("concept_overrides", {}).get(norm_concept, {}); base_themes_config = config.get("base_themes", {}); final_cfg = config.get("STAGE2_CONFIG", {})
    min_weight_attr = float(final_cfg.get("THEME_ATTRIBUTE_MIN_WEIGHT", 0.001)); top_n_attrs = int(final_cfg.get("TOP_N_DEFINING_ATTRIBUTES", 25)); llm_refinement_enabled = final_cfg.get("ENABLE_LLM_REFINEMENT", True)
    vrbo_defaults = config.get("vrbo_default_subscore_weights", {}); vrbo_sentiment_min = 0.10; vrbo_groupintel_min = 0.05; llm_negation_cfg = config.get("LLM_NEGATION_CONFIG", {}); llm_negation_enabled = llm_negation_cfg.get("enabled", False)
    property_type = concept_overrides.get("property_type", "Unknown"); logger.info(f"[{norm_concept}] Determined property_type (for VRBO rules): {property_type}")
    output: Dict[str, Any] = {"applicable_lodging_types": "Both", "travel_category": travel_category or {"uri": None, "name": input_concept, "type": "Unknown"}, "top_defining_attributes": [], "themes": [], "key_associated_concepts_unthemed": [], "additional_relevant_subscores": [], "must_not_have": [], "requires_geo_check": False, "failed_fallback_themes": {}, "affinity_score_total_allocated": 0.0, "diagnostics": {"lodging_type_determination": {"reason": "Not run yet"}, "llm_negation": {"attempted": False, "success": False, "uris_found": 0, "error": None, "candidates_checked": [], "identified_negating_uris": [],}, "theme_processing": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "final_output": {}, "stage2": {"status": "Started", "duration_seconds": 0.0, "error": None},},}
    theme_diag = output["diagnostics"]["theme_processing"]; reprompt_diag = output["diagnostics"]["reprompting_fallback"]; neg_diag = output["diagnostics"]["llm_negation"]; fallback_adds = []

    # ... (LLM Theme Slotting, Fallback Logic, LLM Negation Identification, Must Not Have population/filtering, Theme Weight Redistribution - all identical to v34.0.14 previous generation) ...
    # LLM Theme Slotting
    llm_assigns: Dict[str, List[str]] = {}; diag_val = {}
    if llm_call_result and llm_call_result.get("success"):
        validated = validate_llm_assignments(llm_call_result.get("response"), set(original_candidates_map_for_reprompt.keys()), set(base_themes_config.keys()), norm_concept, diag_val,)
        if validated is not None: llm_assigns = validated
        else: logger.warning(f"[{norm_concept}] LLM validation failed."); output["diagnostics"]["stage2"]["error"] = diag_val.get("error", "LLM Validation Failed")
    elif llm_call_result: logger.warning(f"[{norm_concept}] LLM slotting unsuccessful: {llm_call_result.get('error')}"); output["diagnostics"]["stage2"]["error"] = f"LLM Call Failed: {llm_call_result.get('error')}"
    theme_map = defaultdict(list)
    for uri, themes in llm_assigns.items():
        if uri in original_candidates_map_for_reprompt:
            for t in themes:
                if t in base_themes_config: theme_map[t].append(uri)
    # Fallback Logic
    failed_rules: Dict[str, Dict] = {};
    for name in base_themes_config.keys():
        diag = theme_diag[name] = {"llm_assigned_count": len(theme_map.get(name, [])),"attributes_after_weighting": 0, "status": "Pending", "rule_failed": False,}
        rule, _, _, _ = get_dynamic_theme_config(norm_concept, name, config)
        if rule == "Must have 1" and not theme_map.get(name): logger.warning(f"[{norm_concept}] Rule FAILED: Mandatory theme '{name}' has no assigns initially."); failed_rules[name] = {"reason": "No assigns."}; diag.update({"status": "Failed Rule", "rule_failed": True})
    fixed = set()
    if (failed_rules and original_candidates_map_for_reprompt and llm_refinement_enabled and args.llm_provider != "none"):
        logger.info(f"[{norm_concept}] Attempting LLM fallback for {len(failed_rules)} themes: {list(failed_rules.keys())}"); llm_api_cfg = config.get("LLM_API_CONFIG", {}); llm_stage2_cfg = config.get("STAGE2_CONFIG", {})
        fb_timeout = int(llm_api_cfg.get("REQUEST_TIMEOUT", 180)); fb_retries = int(llm_api_cfg.get("MAX_RETRIES", 5)); fb_temp = float(llm_stage2_cfg.get("LLM_TEMPERATURE", 0.2))
        for name in list(failed_rules.keys()):
            reprompt_diag["attempts"] += 1; base_cfg = base_themes_config.get(name)
            if not base_cfg: logger.error(f"[{norm_concept}] No config for fallback theme '{name}'."); reprompt_diag["failures"] += 1; continue
            sys_p, user_p = build_reprompt_prompt(input_concept, name, base_cfg, original_candidates_map_for_reprompt); fb_result = call_llm(sys_p, user_p, args.llm_model, fb_timeout, fb_retries, fb_temp, args.llm_provider,)
            if fb_result and fb_result.get("success"):
                fb_assigns = fb_result.get("response", {}).get("theme_assignments", {}); new_uris = (set(uri for uri, ts in fb_assigns.items() if isinstance(ts, list) and name in ts and uri in original_candidates_map_for_reprompt) if isinstance(fb_assigns, dict) else set())
                if new_uris:
                    logger.info(f"[{norm_concept}] Fallback SUCCESS for '{name}': Assigned {len(new_uris)} URIs."); reprompt_diag["successes"] += 1; fixed.add(name)
                    for uri in new_uris:
                        if uri not in theme_map.get(name, []): theme_map[name].append(uri); fallback_adds.append({"uri": uri, "assigned_theme": name})
                    theme_diag[name].update({"status": "Passed (Fallback)", "rule_failed": False});
                    if name in failed_rules: del failed_rules[name]
                else: logger.warning(f"[{norm_concept}] Fallback LLM for '{name}' assigned 0 candidates."); reprompt_diag["failures"] += 1; theme_diag[name]["status"] = "Failed (Fallback - No Assigns)"
            else: err = fb_result.get("error", "?") if fb_result else "?"; logger.error(f"[{norm_concept}] Fallback LLM failed for '{name}'. Error:{err}"); reprompt_diag["failures"] += 1; theme_diag[name]["status"] = "Failed (Fallback - API Error)"
    elif failed_rules: logger.warning(f"[{norm_concept}] Cannot attempt fallback ({list(failed_rules.keys())}), LLM refinement disabled or provider 'none'.")
    # LLM Negation ID + Debug Logs
    mnh_uris_llm = set()
    if (llm_negation_enabled and args.llm_provider != "none" and travel_category and travel_category.get("uri")):
        neg_diag["attempted"] = True; logger.info(f"[{norm_concept}] Attempting LLM Negation Identification...");
        neg_api_cfg = config.get("LLM_API_CONFIG", {}); neg_llm_cfg = config.get("LLM_NEGATION_CONFIG", {})
        neg_timeout = int(neg_api_cfg.get("REQUEST_TIMEOUT", 180)); neg_retries = int(neg_api_cfg.get("MAX_RETRIES", 3)); neg_temp = float(neg_llm_cfg.get("temperature", 0.3)); max_cands_check = int(neg_llm_cfg.get("max_candidates_to_check", 30))
        anchor_label = travel_category.get("prefLabel", input_concept)
        logger.debug(f"[{norm_concept}] Preparing candidates for negation check (Max: {max_cands_check}).") # Debug Log 1
        candidates_sorted_unbiased = sorted(original_candidates_map_for_reprompt.values(), key=lambda x: x.get("combined_score_unbiased", 0.0), reverse=True,)
        candidates_for_negation_check = [{"uri": c.get("uri"), "prefLabel": c.get("prefLabel")} for c in candidates_sorted_unbiased[:max_cands_check] if c.get("uri") and c.get("prefLabel")]
        neg_diag["candidates_checked"] = [c['uri'] for c in candidates_for_negation_check]
        logger.debug(f"[{norm_concept}] Found {len(candidates_for_negation_check)} candidates for negation check:") # Debug Log 2
       # if args.debug:
         #    for idx, cand_neg in enumerate(candidates_for_negation_check):
           #      logger.debug(f"  Neg Cand {idx+1}: {cand_neg.get('uri')} - '{cand_neg.get('prefLabel')}'")
        if candidates_for_negation_check:
            try:
                neg_sys_p, neg_user_p = construct_llm_negation_prompt(input_concept, anchor_label, candidates_for_negation_check)
               # logger.debug(f"[{norm_concept}] Sending Negation Prompt to LLM:\n--- SYSTEM ---\n{neg_sys_p}\n--- USER ---\n{neg_user_p[:1000]}...\n----------") # Debug Log 3
                neg_result = call_llm(neg_sys_p, neg_user_p, args.llm_model, neg_timeout, neg_retries, neg_temp, args.llm_provider,)
               # logger.debug(f"[{norm_concept}] Received Negation LLM Result: {neg_result}") # Debug Log 4
                if neg_result and neg_result.get("success"):
                    neg_response = neg_result.get("response", {}); found_uris = neg_response.get("negating_uris", [])
                    if isinstance(found_uris, list):
                        valid_neg_uris = {uri for uri in found_uris if isinstance(uri, str) and uri in taxonomy_concepts_cache}; invalid_identified = set(found_uris) - valid_neg_uris
                        if invalid_identified: logger.warning(f"[{norm_concept}] LLM negation identified unknown URIs: {invalid_identified}")
                        mnh_uris_llm = valid_neg_uris; neg_diag["success"] = True; neg_diag["uris_found"] = len(mnh_uris_llm); neg_diag["identified_negating_uris"] = sorted(list(mnh_uris_llm))
                        logger.info(f"[{norm_concept}] LLM identified {len(mnh_uris_llm)} negating URIs: {mnh_uris_llm}")
                    else: logger.warning(f"[{norm_concept}] LLM negation response 'negating_uris' not a list."); neg_diag["error"] = "Invalid LLM response format (not list)"
                else: err = neg_result.get("error", "?") if neg_result else "?"; logger.warning(f"[{norm_concept}] LLM negation call failed: {err}"); neg_diag["error"] = f"LLM Call Failed: {err}"
            except Exception as e: logger.error(f"[{norm_concept}] LLM negation exception: {e}", exc_info=True); neg_diag["error"] = f"Exception: {e}"
        else: logger.info(f"[{norm_concept}] Skipping LLM negation: No candidates provided to check.")
    elif llm_negation_enabled: logger.info(f"[{norm_concept}] Skipping LLM negation (Provider 'none' or no anchor URI).")
    else: logger.info(f"[{norm_concept}] LLM negation disabled in config.")
    # Must Not Have processing
    mnh_uris_manual = set()
    mnh_config_override = concept_overrides.get("must_not_have", [])
    if isinstance(mnh_config_override, list): mnh_uris_manual = set(i["uri"] for i in mnh_config_override if isinstance(i, dict) and "uri" in i)
    all_uris_to_exclude = mnh_uris_manual.union(mnh_uris_llm)
    output["must_not_have"] = [{"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "scope": "Config Override" if u in mnh_uris_manual else "LLM Identified",} for u in sorted(list(all_uris_to_exclude))]
    if all_uris_to_exclude: logger.info(f"[{norm_concept}] Populated 'must_not_have' with {len(output['must_not_have'])} URIs ({len(mnh_uris_manual)} from config, {len(mnh_uris_llm)} from LLM). These will be excluded from definition attributes.")
    # Filter theme_map
    if all_uris_to_exclude:
        logger.debug(f"[{norm_concept}] Filtering theme assignments to remove excluded URIs...")
        for theme_name in list(theme_map.keys()):
            original_uris = theme_map[theme_name]; filtered_uris = [uri for uri in original_uris if uri not in all_uris_to_exclude]
            if not filtered_uris: del theme_map[theme_name]; logger.debug(f"[{norm_concept}] Theme '{theme_name}' became empty after removing excluded URIs.")
            elif len(original_uris) != len(filtered_uris): removed_count = len(original_uris) - len(filtered_uris); theme_map[theme_name] = filtered_uris; logger.debug(f"[{norm_concept}] Removed {removed_count} excluded URIs from theme '{theme_name}'.")
    else: logger.debug(f"[{norm_concept}] No URIs identified for exclusion from definition attributes.")
    # Theme Weight Redistribution
    active_themes = set(theme_map.keys()); logger.debug(f"[{norm_concept}] Active themes after filtering excluded URIs: {active_themes}")
    initial_theme_weights = {}; total_initial_theme_weight = 0.0
    for name, theme_data in base_themes_config.items(): _, weight, _, _ = get_dynamic_theme_config(norm_concept, name, config); initial_theme_weights[name] = weight; total_initial_theme_weight += weight
    normalized_initial_theme_weights = {name: (w / total_initial_theme_weight) if total_initial_theme_weight > 0 else 0 for name, w in initial_theme_weights.items()}
    weight_to_redistribute = sum(normalized_initial_theme_weights.get(name, 0) for name in base_themes_config if name not in active_themes)
    total_weight_of_active_themes = sum(normalized_initial_theme_weights.get(name, 0) for name in active_themes); final_theme_weights = {}
    for name in base_themes_config:
        if name in active_themes and total_weight_of_active_themes > 0:
            initial_norm_w = normalized_initial_theme_weights.get(name, 0); redistributed_share = ((initial_norm_w / total_weight_of_active_themes) * weight_to_redistribute if total_weight_of_active_themes > 0 else 0); final_theme_weights[name] = initial_norm_w + redistributed_share
        else: final_theme_weights[name] = 0.0
    #logger.debug(f"[{norm_concept}] Final theme weights after filtering and redistribution: { {k: round(v, 4) for k, v in final_theme_weights.items()} }")

    # --- Theme/Attribute Processing (NOW USES COMBINED SCORE) ---
    final_themes_out = []; all_final_attrs = []; total_attribute_weight_sum = 0.0
    for name, base_data in base_themes_config.items():
        final_norm_w = final_theme_weights.get(name, 0.0); rule, _, subscore_affinity_name, _ = get_dynamic_theme_config(norm_concept, name, config)
        uris = theme_map.get(name, []) # Use filtered URIs
        theme_attrs = []
        if uris and final_norm_w > 1e-9:
            # --- Use combined_score for weighting ---
            scores = {}
            for u in uris:
                 score_val = original_candidates_map_for_reprompt.get(u, {}).get("combined_score", 0.0) # Use final combined score
                 scores[u] = max(1e-9, score_val) # Ensure score is positive
            total_score = sum(scores.values())
         #   logger.debug(f"[{norm_concept}] Theme '{name}' - Total combined_score for weighting: {total_score:.4f} across {len(uris)} URIs.")
            # --- End score source change ---
            if total_score > 1e-9:
                for u in uris:
                    s = scores.get(u, 1e-9); prop = s / total_score; attr_w = final_norm_w * prop
                    if attr_w >= min_weight_attr:
                        d = original_candidates_map_for_reprompt.get(u, {}); is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                        attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(attr_w, 6), "type": d.get("type_labels", []),}
                        if is_fb: attr["comment"] = "Fallback Assignment"
                        # Store the score used for weighting for potential debugging/analysis
                        attr["_weighting_score"] = round(s, 6)
                        theme_attrs.append(attr); all_final_attrs.append(attr); total_attribute_weight_sum += attr_w
            elif len(uris) > 0: # Fallback to equal weight
                eq_w = final_norm_w / len(uris)
                if eq_w >= min_weight_attr:
                    logger.debug(f"[{norm_concept}] Theme '{name}' - Using equal weight fallback ({eq_w:.4f}) as total score was zero or negligible.")
                    for u in uris:
                        d = original_candidates_map_for_reprompt.get(u, {}); is_fb = any(f["uri"] == u and f["assigned_theme"] == name for f in fallback_adds)
                        attr = {"uri": u, "skos:prefLabel": get_primary_label(u, taxonomy_concepts_cache, u), "concept_weight": round(eq_w, 6), "type": d.get("type_labels", []),}
                        if is_fb: attr["comment"] = "Fallback Assignment"
                        theme_attrs.append(attr); all_final_attrs.append(attr); total_attribute_weight_sum += eq_w

        # Sort attributes within theme by weight
        theme_attrs.sort(key=lambda x: x["concept_weight"], reverse=True)
        # Remove temporary weighting score before finalizing output
        for attr in theme_attrs: attr.pop("_weighting_score", None)

        theme_diag[name]["attributes_after_weighting"] = len(theme_attrs)
        if name not in active_themes and theme_diag[name].get("status") == "Pending": theme_diag[name]["status"] = "Filtered Out (Excluded URI)"
        elif theme_diag[name].get("status") == "Pending": theme_diag[name]["status"] = ("Processed (Initial)" if not theme_diag[name].get("rule_failed") else theme_diag[name].get("status", "Failed Rule"))
        final_themes_out.append({"theme_name": name, "theme_type": base_data.get("type", "?"), "rule_applied": rule, "normalized_theme_weight": round(final_norm_w, 6), "subScore": subscore_affinity_name or f"{name}Affinity", "llm_summary": None, "attributes": theme_attrs,})
    output["themes"] = final_themes_out
    logger.debug(f"[{norm_concept}] Sum of final attribute weights after filtering: {total_attribute_weight_sum:.4f}")

    # --- Top Defining Attributes --- (unchanged from v34.0.13)
    unique_attrs: Dict[str, Dict[str, Any]] = {};
    for attr in all_final_attrs:
        uri = attr.get("uri");
        if not uri or uri not in original_candidates_map_for_reprompt: continue
        combined_score = original_candidates_map_for_reprompt.get(uri, {}).get("combined_score", 0.0)
        if uri not in unique_attrs or combined_score > unique_attrs[uri].get("_score_for_dedupe", -1.0):
            attr_copy = {k: v for k, v in attr.items() if k != "comment" and k != "_weighting_score"}; attr_copy["_score_for_dedupe"] = combined_score; unique_attrs[uri] = attr_copy
    sorted_top = sorted(unique_attrs.values(), key=lambda x: x.get("_score_for_dedupe", 0.0), reverse=True,)
    for attr in sorted_top: attr.pop("_score_for_dedupe", None)
    output["top_defining_attributes"] = sorted_top[:top_n_attrs]

    # --- Dynamically Determine Applicable Lodging Types (v34.0.14) ---
    try:
        determined_lodging_type = determine_lodging_type(travel_category, output["top_defining_attributes"], config, taxonomy_concepts_cache)
        output["applicable_lodging_types"] = determined_lodging_type
        logger.info(f"[{norm_concept}] Dynamically determined Applicable Lodging Type: {determined_lodging_type}")
        output["diagnostics"]["lodging_type_determination"] = {"result": determined_lodging_type}
    except Exception as e:
        logger.error(f"[{norm_concept}] Error during lodging type determination: {e}", exc_info=True)
        output["applicable_lodging_types"] = "Error"; output["diagnostics"]["lodging_type_determination"] = {"error": str(e)}

    # --- Capture Unthemed High-Relevance Concepts --- (unchanged from v34.0.13)
    output["key_associated_concepts_unthemed"] = []
    unthemed_percentile_threshold = float(final_cfg.get("UNTHEMED_CAPTURE_SCORE_PERCENTILE", 75))
    logger.info(f"[{norm_concept}] Checking for unthemed concepts above {unthemed_percentile_threshold}th percentile score...")
    llm_candidate_uris = list(original_candidates_map_for_reprompt.keys()); llm_candidate_scores = [original_candidates_map_for_reprompt.get(uri, {}).get("combined_score", 0.0) for uri in llm_candidate_uris]
    score_threshold = 0.0
    if llm_candidate_scores:
        try: score_threshold = np.percentile(llm_candidate_scores, unthemed_percentile_threshold); logger.debug(f"[{norm_concept}] Score threshold for unthemed capture ({unthemed_percentile_threshold}th percentile): {score_threshold:.4f}")
        except Exception as e: logger.warning(f"[{norm_concept}] Could not calculate percentile threshold: {e}. Using 0.0."); score_threshold = 0.0
    else: logger.warning(f"[{norm_concept}] No candidate scores available to calculate percentile.")
    assigned_uris = set();
    for theme_name in theme_map: assigned_uris.update(theme_map[theme_name])
    logger.debug(f"[{norm_concept}] Total URIs assigned to themes: {len(assigned_uris)}")
    unthemed_concepts_found = []
    for uri, details in original_candidates_map_for_reprompt.items():
        if uri not in assigned_uris and uri not in all_uris_to_exclude:
            score = details.get("combined_score", 0.0)
            if score >= score_threshold and score > 0: unthemed_concepts_found.append({"uri": uri, "skos:prefLabel": get_primary_label(uri, taxonomy_concepts_cache, uri), "combined_score": round(score, 6), "type": details.get("type_labels", [])})
    unthemed_concepts_found.sort(key=lambda x: x["combined_score"], reverse=True); output["key_associated_concepts_unthemed"] = unthemed_concepts_found
    logger.info(f"[{norm_concept}] Found {len(unthemed_concepts_found)} high-scoring concepts not assigned to any theme (threshold: {score_threshold:.4f}).")

    # --- Subscore Calculation --- (unchanged from v34.0.13)
    final_subscore_weights = defaultdict(float); logger.debug(f"[{norm_concept}] Calculating subscore weights from active themes (post-filtering)...")
    for name, theme_data in base_themes_config.items():
        if name in active_themes:
            theme_final_weight = final_theme_weights.get(name, 0.0); relevant_subscores_cfg = theme_data.get("relevant_subscores", {})
            if isinstance(relevant_subscores_cfg, dict):
                for (subscore_name, base_subscore_weight,) in relevant_subscores_cfg.items(): contribution = theme_final_weight * float(base_subscore_weight); final_subscore_weights[subscore_name] += contribution; logger.debug(f"  Theme '{name}' (w={theme_final_weight:.3f}) adds {contribution:.4f} to '{subscore_name}'")
    if property_type == "VRBO":
        logger.info(f"[{norm_concept}] Applying VRBO subscore rules."); sentiment_default = vrbo_defaults.get("SentimentScore", 1e-6); groupintel_default = vrbo_defaults.get("GroupIntelligenceScore", 1e-6)
        if "SentimentScore" not in final_subscore_weights: final_subscore_weights["SentimentScore"] = sentiment_default; logger.debug(f"  Added missing SentimentScore for VRBO (initial weight: {sentiment_default:.2e})")
        if "GroupIntelligenceScore" not in final_subscore_weights: final_subscore_weights["GroupIntelligenceScore"] = groupintel_default; logger.debug(f"  Added missing GroupIntelligenceScore for VRBO (initial weight: {groupintel_default:.2e})")
    total_subscore_weight_before_vrbo_min = sum(final_subscore_weights.values()); normalized_subscores = {}
    if total_subscore_weight_before_vrbo_min > 1e-9:
        for name, weight in final_subscore_weights.items(): normalized_subscores[name] = weight / total_subscore_weight_before_vrbo_min
       # logger.debug(f"[{norm_concept}] Normalized subscores BEFORE VRBO min enforcement: { {k: round(v, 4) for k, v in normalized_subscores.items()} }")
    else: logger.info(f"[{norm_concept}] No relevant subscores found or calculated total weight is zero.")
    if property_type == "VRBO" and normalized_subscores:
        made_adjustments = False
        if "SentimentScore" not in normalized_subscores: normalized_subscores["SentimentScore"] = 0.0
        if "GroupIntelligenceScore" not in normalized_subscores: normalized_subscores["GroupIntelligenceScore"] = 0.0
        if normalized_subscores["SentimentScore"] < vrbo_sentiment_min: logger.debug(f"  Boosting SentimentScore weight from {normalized_subscores['SentimentScore']:.4f} to minimum {vrbo_sentiment_min}"); normalized_subscores["SentimentScore"] = vrbo_sentiment_min; made_adjustments = True
        if normalized_subscores["GroupIntelligenceScore"] < vrbo_groupintel_min: logger.debug(f"  Boosting GroupIntelligenceScore weight from {normalized_subscores['GroupIntelligenceScore']:.4f} to minimum {vrbo_groupintel_min}"); normalized_subscores["GroupIntelligenceScore"] = vrbo_groupintel_min; made_adjustments = True
        if made_adjustments:
            current_sum = sum(normalized_subscores.values())
            if current_sum > 1e-9 and abs(current_sum - 1.0) > 1e-9: logger.debug(f"  Re-normalizing subscore weights after VRBO minimum enforcement (Current sum: {current_sum:.4f})"); renorm_factor = 1.0 / current_sum; normalized_subscores = {name: weight * renorm_factor for name, weight in normalized_subscores.items()}
    output["additional_relevant_subscores"] = [{"subscore_name": name, "weight": round(weight, 6)} for name, weight in sorted(normalized_subscores.items(), key=lambda item: item[1], reverse=True) if weight > 1e-9]
    logger.debug(f"[{norm_concept}] Final subscores: {output['additional_relevant_subscores']}")

    # --- Calculate Final Score --- (unchanged from v34.0.13)
    subscore_component = sum(item["weight"] for item in output["additional_relevant_subscores"]); attribute_component = total_attribute_weight_sum
    subscore_component = min(1.0, subscore_component) if subscore_component > 0 else 0; attribute_component = (min(1.0, attribute_component) if attribute_component > 0 else 0)
    output["affinity_score_total_allocated"] = round((0.60 * subscore_component) + (0.40 * attribute_component), 6)
    logger.info(f"[{norm_concept}] Calculated Affinity Score (Total Allocated): {output['affinity_score_total_allocated']:.4f} (Subscores: {subscore_component:.3f}*0.6, Attributes: {attribute_component:.3f}*0.4)")

    # --- Determine requires_geo_check dynamically --- (unchanged from v34.0.13)
    geo_check = False; geo_subscore_names = {"GeospatialAffinityScore", "WalkabilityScore"}
    if any(ss["subscore_name"] in geo_subscore_names for ss in output["additional_relevant_subscores"]): geo_check = True; logger.debug(f"[{norm_concept}] Setting requires_geo_check=True (Geo subscore found).")
    if not geo_check:
        for theme in output["themes"]:
            if theme.get("theme_name") == "Location" and theme.get("attributes"): geo_check = True; logger.debug(f"[{norm_concept}] Setting requires_geo_check=True (Location theme active)."); break
    if not geo_check and travel_category and travel_category.get("uri"):
        anchor_uri = travel_category["uri"]; bias_config = config.get("NAMESPACE_BIASING", {}); location_ns = tuple(bias_config.get("location_context_ns", ()))
        if location_ns and anchor_uri.startswith(location_ns): geo_check = True; logger.debug(f"[{norm_concept}] Setting requires_geo_check=True (Anchor URI in location namespace: {anchor_uri}).")
    output["requires_geo_check"] = geo_check

    # --- Final Diagnostics & Cleanup --- (unchanged from v34.0.13)
    final_diag = output["diagnostics"]["final_output"]; final_diag["must_not_have_count"] = len(output["must_not_have"]); final_diag["additional_subscores_count"] = len(output["additional_relevant_subscores"])
    final_diag["themes_count"] = len(output["themes"]); final_diag["unthemed_concepts_captured_count"] = len(output["key_associated_concepts_unthemed"])
    output["failed_fallback_themes"] = {n: r for n, r in failed_rules.items() if n not in fixed}; final_diag["failed_fallback_themes_count"] = len(output["failed_fallback_themes"])
    final_diag["top_defining_attributes_count"] = len(output["top_defining_attributes"]); output["diagnostics"]["stage2"]["status"] = "Completed"; output["diagnostics"]["stage2"]["duration_seconds"] = round(time.time() - start_stage2, 2)

    return output


# --- Main Processing Loop ---
def generate_affinity_definitions_loop(concepts_to_process: List[str], config: Dict, args: argparse.Namespace, sbert_model: SentenceTransformer, primary_embeddings_map: Dict[str, np.ndarray], taxonomy_concepts_cache: Dict[str, Dict], keyword_label_index: Optional[Dict[str, Set[str]]], bm25_model: Optional[bm25s.BM25], keyword_corpus_uris: Optional[List[str]]) -> List[Dict]:
    # ... (Loop structure unchanged, but passes orig_map instead of sbert_scores_final to finalize) ...
    all_definitions = []; cache_ver = config.get("cache_version")
    if not taxonomy_concepts_cache: logger.critical("FATAL: Concepts cache empty."); return []
    limit = args.limit if args.limit and args.limit > 0 else len(concepts_to_process); concepts_subset = concepts_to_process[:limit];
    logger.info(f"Processing {len(concepts_subset)}/{len(concepts_to_process)} concepts.")
    if not concepts_subset: logger.warning("Concept subset empty!"); return []
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    for concept in tqdm(concepts_subset, desc="Processing Concepts", disable=disable_tqdm):
        start_time = time.time(); norm_concept = normalize_concept(concept);
        logger.info(f"=== Processing Concept: '{concept}' ('{norm_concept}') ===")
        affinity_def = { # Initialize structure v34.0.14
            "input_concept": concept, "normalized_concept": norm_concept, "applicable_lodging_types": "Both", "travel_category": {},
            "top_defining_attributes": [], "themes": [], "key_associated_concepts_unthemed": [], "additional_relevant_subscores": [],
            "must_not_have": [], "requires_geo_check": False, "failed_fallback_themes": {}, "affinity_score_total_allocated": 0.0,
            "processing_metadata": {"status": "Started", "version": SCRIPT_VERSION, "timestamp": None, "total_duration_seconds": 0.0, "cache_version": cache_ver, "llm_provider": args.llm_provider, "llm_model": args.llm_model if args.llm_provider != "none" else None,},
            "diagnostics": {"lodging_type_determination": {}, "llm_negation": {"attempted": False, "success": False, "uris_found": 0, "error": None, "candidates_checked": [], "identified_negating_uris": [],}, "stage1": {"status": "Not Started", "error": None, "selection_method": "?", "expansion": {}, "sbert_candidate_count_initial": 0, "keyword_candidate_count_initial": 0, "unique_candidates_before_ranking": 0, "llm_candidate_count": 0,}, "llm_slotting": {"status": "Not Started", "error": None}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "stage2": {"status": "Not Started", "error": None}, "theme_processing": {}, "final_output": {}, "error_details": None,},
        }
        diag1 = affinity_def["diagnostics"]["stage1"]; diag_llm = affinity_def["diagnostics"]["llm_slotting"]
        try:
            concept_emb = get_concept_embedding(norm_concept, sbert_model)
            if concept_emb is None: logger.error(f"[{norm_concept}] Embedding failed."); diag1["error"] = "Embedding failed"; diag1["status"] = "Failed"
            cand_details, orig_map, anchor, _, exp_diag, sbert_init, kw_init, unique_count = [], {}, None, {}, {}, 0, 0, 0 # Initialize defaults
            if diag1["status"] != "Failed":
                stage1_start_call = time.time(); (cand_details, orig_map, anchor, _, exp_diag, sbert_init, kw_init, unique_count,) = prepare_evidence(concept, concept_emb, primary_embeddings_map, config, args, bm25_model, keyword_corpus_uris, keyword_label_index, taxonomy_concepts_cache,)
                stage1_dur = time.time() - stage1_start_call; diag1.update({"status": "Completed", "error": diag1.get("error"), "selection_method": exp_diag.get("selection_method", "Combined BM25s+SBERT"), "expansion": exp_diag, "sbert_candidate_count_initial": sbert_init, "keyword_candidate_count_initial": kw_init, "unique_candidates_before_ranking": unique_count, "llm_candidate_count": len(cand_details), "duration_seconds": round(stage1_dur, 2),})
                logger.info(f"[{norm_concept}] Stage 1 done ({stage1_dur:.2f}s). Status:{diag1['status']}. LLM Cands:{len(cand_details)}. SBERT:{sbert_init}, KW:{kw_init}.")
                if not anchor or not anchor.get("uri"): logger.warning(f"[{norm_concept}] Stage 1 completed but no valid anchor.")
            else: logger.warning(f"[{norm_concept}] Skipping Stages 1b and 2 due to embedding failure.")
            llm_result = None; llm_start = time.time(); diag_llm.update({"llm_provider": args.llm_provider, "llm_model": (args.llm_model if args.llm_provider != "none" else None),})
            if diag1["status"] == "Failed": logger.warning(f"[{norm_concept}] Skipping LLM (Stage 1 failed)."); diag_llm["status"] = "Skipped (Stage 1 Failed)"
            elif not cand_details: logger.warning(f"[{norm_concept}] Skipping LLM (No candidates)."); affinity_def["processing_metadata"]["status"] = "Warning - No LLM Candidates"; diag_llm["status"] = "Skipped (No Candidates)"
            elif args.llm_provider == "none": logger.info(f"[{norm_concept}] Skipping LLM (Provider 'none')."); diag_llm["status"] = "Skipped (Provider None)"
            else:
                diag_llm["status"] = "Started"; diag_llm["llm_call_attempted"] = True
                themes_for_prompt = [{"name": name, "description": get_theme_definition_for_prompt(name, data), "is_must_have": get_dynamic_theme_config(norm_concept, name, config)[0] == "Must have 1",} for name, data in config.get("base_themes", {}).items()]
                sys_p, user_p = construct_llm_slotting_prompt(concept, themes_for_prompt, cand_details, args)
                llm_api_cfg = config.get("LLM_API_CONFIG", {}); llm_stage2_cfg = config.get("STAGE2_CONFIG", {}); slot_timeout = int(llm_api_cfg.get("REQUEST_TIMEOUT", 180)); slot_retries = int(llm_api_cfg.get("MAX_RETRIES", 5)); slot_temp = float(llm_stage2_cfg.get("LLM_TEMPERATURE", 0.2))
                llm_result = call_llm(sys_p, user_p, args.llm_model, slot_timeout, slot_retries, slot_temp, args.llm_provider,)
                diag_llm["attempts_made"] = (llm_result.get("attempts", 0) if llm_result else 0)
                if llm_result and llm_result.get("success"): diag_llm["llm_call_success"] = True; diag_llm["status"] = "Completed"
                else: diag_llm["llm_call_success"] = False; diag_llm["status"] = "Failed"; diag_llm["error"] = (llm_result.get("error", "?") if llm_result else "?"); logger.warning(f"[{norm_concept}] LLM slotting failed. Error:{diag_llm['error']}")
            diag_llm["duration_seconds"] = round(time.time() - llm_start, 2); logger.info(f"[{norm_concept}] LLM Slotting:{diag_llm['duration_seconds']:.2f}s. Status:{diag_llm['status']}")
            if diag1["status"] != "Failed":
                stage2_out = apply_rules_and_finalize(input_concept=concept, llm_call_result=llm_result, config=config, travel_category=anchor, anchor_candidate=anchor, original_candidates_map_for_reprompt=orig_map, args=args, taxonomy_concepts_cache=_taxonomy_concepts_cache,) # Pass orig_map here
                affinity_def.update({k: v for k, v in stage2_out.items() if k != "diagnostics"})
                if "diagnostics" in stage2_out:
                    s2d = stage2_out["diagnostics"]; affinity_def["diagnostics"]["lodging_type_determination"] = s2d.get("lodging_type_determination", {}); affinity_def["diagnostics"]["llm_negation"].update(s2d.get("llm_negation", {})); affinity_def["diagnostics"]["theme_processing"] = s2d.get("theme_processing", {}); affinity_def["diagnostics"]["reprompting_fallback"].update(s2d.get("reprompting_fallback", {})); affinity_def["diagnostics"]["final_output"] = s2d.get("final_output", {}); affinity_def["diagnostics"]["stage2"] = s2d.get("stage2", {"status": "Unknown"})
                else: affinity_def["diagnostics"]["stage2"] = {"status": "Unknown", "error": "Stage 2 diagnostics missing"}
            else: affinity_def["diagnostics"]["stage2"]["status"] = "Skipped (Stage 1 Failed)"
            final_status = affinity_def["processing_metadata"]["status"]
            if (final_status == "Started"):
                if diag1["status"] == "Failed": final_status = "Failed - Stage 1 Error"
                elif affinity_def.get("failed_fallback_themes"): final_status = "Success with Failed Rules"
                elif diag_llm["status"] == "Failed": final_status = "Warning - LLM Slotting Failed"
                elif affinity_def["diagnostics"]["llm_negation"].get("error"): final_status = "Warning - LLM Negation Failed"
                elif affinity_def["processing_metadata"]["status"] != "Warning - No LLM Candidates":
                    if affinity_def["diagnostics"]["stage2"].get("error"): final_status = f"Warning - Finalization Error ({affinity_def['diagnostics']['stage2']['error']})"
                    elif diag_llm["status"] == "Skipped (Provider None)": final_status = "Success (LLM Skipped)"
                    else: final_status = "Success"
            affinity_def["processing_metadata"]["status"] = final_status
        except Exception as e:
            logger.error(f"Core loop failed for '{concept}': {e}", exc_info=True); affinity_def["processing_metadata"]["status"] = "FATAL ERROR"; affinity_def["diagnostics"]["error_details"] = traceback.format_exc()
            if diag1["status"] not in ["Completed", "Failed"]: diag1["status"] = "Failed (Exception)"
            if diag_llm["status"] not in ["Completed", "Failed", "Skipped"]: diag_llm["status"] = "Failed (Exception)"
            if affinity_def["diagnostics"]["stage2"]["status"] not in ["Completed", "Failed", "Skipped",]: affinity_def["diagnostics"]["stage2"]["status"] = "Failed (Exception)"
        finally:
            end_time = time.time(); duration = round(end_time - start_time, 2); affinity_def["processing_metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); affinity_def["processing_metadata"]["total_duration_seconds"] = duration
            all_definitions.append(affinity_def); log_func = (logger.warning if "Warning" in affinity_def["processing_metadata"]["status"] or "Failed" in affinity_def["processing_metadata"]["status"] else logger.info)
            log_func(f"--- Finished '{norm_concept}' ({duration:.2f}s). Status: {affinity_def['processing_metadata']['status']} ---")
    return all_definitions



# --- Main Execution ---
def main():
    # ... (Argument parsing remains unchanged) ...
    global _config_data, _taxonomy_concepts_cache, _taxonomy_embeddings_cache, _keyword_label_index, _bm25_model, _keyword_corpus_uris, _acs_data, args
    parser = argparse.ArgumentParser(description=f"Generate Travel Concept Affinity Definitions ({SCRIPT_VERSION})")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_FILE, help="Config JSON path.")
    parser.add_argument("-t", "--taxonomy-dir", dest="taxonomy_dir", type=str, help="Override Taxonomy RDF directory.")
    parser.add_argument("-i", "--input-concepts-file", type=str, required=True, help="Input concepts file path.")
    parser.add_argument("-o", "--output-dir", type=str, help="Override Output directory.")
    parser.add_argument("--cache-dir", type=str, help="Override Cache directory.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild caches.")
    parser.add_argument("--limit", type=int, default=None, help="Limit concepts processed.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--llm-provider", type=str, choices=["openai", "google", "none"], default=None, help="Override LLM provider.")
    parser.add_argument("--llm-model", type=str, default=None, help="Override LLM model.")
    args = parser.parse_args(); user_config = load_affinity_config(args.config)
    if user_config is None: sys.exit(1)
    config = DEFAULT_CONFIG.copy()
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d.get(k), dict): d[k] = recursive_update(d.get(k, {}), v)
            else: d[k] = v
        return d
    config = recursive_update(config, user_config)
    if args.output_dir: config["output_dir"] = args.output_dir
    if args.cache_dir: config["cache_dir"] = args.cache_dir
    if args.taxonomy_dir: config["taxonomy_dir"] = args.taxonomy_dir
    if args.llm_provider is not None: config["LLM_PROVIDER"] = args.llm_provider
    if args.llm_model is not None: config["LLM_MODEL"] = args.llm_model
    _config_data = config; args.llm_provider = config["LLM_PROVIDER"]; args.llm_model = config["LLM_MODEL"]
    output_dir = config["output_dir"]; cache_dir = config["cache_dir"]; cache_ver = config.get("cache_version", DEFAULT_CACHE_VERSION)
    os.makedirs(output_dir, exist_ok=True); os.makedirs(cache_dir, exist_ok=True)
    log_file = os.path.join(output_dir, LOG_FILE_TEMPLATE.format(cache_version=cache_ver)); out_file = os.path.join(output_dir, OUTPUT_FILE_TEMPLATE.format(cache_version=cache_ver))
    log_level = logging.DEBUG if args.debug else logging.INFO; setup_logging(log_level, log_file, args.debug); setup_detailed_loggers(output_dir, cache_ver)
    logger.info(f"Starting {SCRIPT_VERSION}"); logger.info(f"Cmd: {' '.join(sys.argv)}"); logger.info(f"Using Config: {args.config}")
    logger.info(f"Effective Output Dir: {output_dir}, Cache Dir: {cache_dir}, Taxonomy Dir: {config['taxonomy_dir']}")
    logger.info(f"Effective Log File: {log_file}, Outfile: {out_file}"); logger.info(f"Rebuild: {args.rebuild_cache}, Limit: {args.limit or 'None'}, Debug: {args.debug}")
    logger.info("--- Key Parameters (Effective) ---")
    logger.info(f"Global Alpha: {config.get('global_alpha', 'N/A')}")
    kw_scoring_cfg = config.get("KEYWORD_SCORING_CONFIG", {}); kw_enabled = kw_scoring_cfg.get("enabled", False); kw_alg = kw_scoring_cfg.get("algorithm", "N/A").upper(); logger.info(f"Keyword Scoring: {kw_alg} (Enabled: {kw_enabled})")
    kg_config = config.get("KG_CONFIG", {}); acs_config = config.get("ACS_DATA_CONFIG", {}); acs_enabled_log = acs_config.get("enable_acs_enrichment", False)
    logger.info(f"BM25 Doc Assembly: KG(pref x{kg_config.get('pref_label_weight', '?')}, alt x{kg_config.get('alt_label_weight', '?')}, def x{kg_config.get('definition_weight', '?')}) + ACS(En:{acs_enabled_log}, name x{kg_config.get('acs_name_weight', '?')}, def x{kg_config.get('acs_def_weight', '?')})")
    logger.info(f"Abs Min SBERT: {config.get('min_sbert_score', 'N/A')}"); logger.info(f"KW Dampening Thresh: {config.get('keyword_dampening_threshold','N/A')}, Factor: {config.get('keyword_dampening_factor','N/A')}")
    logger.info(f"Prioritize Exact prefLabel Match: {config.get('prioritize_exact_prefLabel_match', False)}")
    ns_bias_cfg = config.get("NAMESPACE_BIASING", {}); logger.info(f"Namespace Biasing: {ns_bias_cfg.get('enabled', False)} (Core Boost: {ns_bias_cfg.get('core_boost_factor', 'N/A')})")
    stage2_cfg = config.get("STAGE2_CONFIG", {}); logger.info(f"Unthemed Concept Capture Percentile: {stage2_cfg.get('UNTHEMED_CAPTURE_SCORE_PERCENTILE', 'N/A (Default 75)')}")
    logger.info(f"Dynamic Lodging Type Check: Top {stage2_cfg.get('LODGING_TYPE_TOP_ATTR_CHECK', '?')} Attrs, Threshold {stage2_cfg.get('LODGING_TYPE_CONFIDENCE_THRESHOLD', '?')}")
    logger.info(f"LLM Negation Check Enabled: {config.get('LLM_NEGATION_CONFIG', {}).get('enabled', False)}")
    logger.info(f"LLM Provider: {args.llm_provider}, Model: {args.llm_model or 'N/A'}"); logger.info("--- End Key Parameters ---")
    if not PANDAS_AVAILABLE and acs_enabled_log: logger.critical("FATAL: ACS Enrichment enabled, but 'pandas' not installed."); sys.exit(1)
    if not BM25S_AVAILABLE and kw_enabled and kw_alg == "BM25S": logger.critical("FATAL: BM25s enabled, but 'bm25s' library not installed."); sys.exit(1)
    input_concepts = []
    try:
        with open(args.input_concepts_file, 'r', encoding='utf-8') as f: input_concepts = [l.strip() for l in f if l.strip()]
        if not input_concepts: raise ValueError("Input concepts file is empty.")
        logger.info(f"Loaded {len(input_concepts)} concepts from '{args.input_concepts_file}'.")
    except Exception as e: logger.critical(f"Failed to read input file '{args.input_concepts_file}': {e}"); sys.exit(1)
    concepts_cache_f = get_cache_filename("concepts", cache_ver, cache_dir, extension=".json"); concepts_data = load_taxonomy_concepts(config["taxonomy_dir"], concepts_cache_f, args.rebuild_cache, cache_ver, args.debug,)
    if concepts_data is None: logger.critical("Taxonomy concepts loading failed."); sys.exit(1)
    _taxonomy_concepts_cache = concepts_data; logger.info(f"Taxonomy concepts ready ({len(_taxonomy_concepts_cache)} concepts).")
    try: _acs_data = load_acs_data(config.get("ACS_DATA_CONFIG", {}).get("acs_data_path")); logger.info(f"ACS data status: {'Loaded' if _acs_data is not None else 'Load Failed or Disabled'}")
    except Exception as e: logger.error(f"ACS data loading exception: {e}", exc_info=True); _acs_data = None
    try:
        build_keyword_index(config, _taxonomy_concepts_cache, cache_dir, cache_ver, args.rebuild_cache)
        if kw_enabled and kw_alg == "BM25S": logger.info(f"BM25s Index Status: {'Ready' if _bm25_model is not None else 'Failed/Unavailable'}")
        if _keyword_label_index is not None: logger.info("Simple Label Index ready.")
        else: logger.warning("Simple Label Index failed.")
    except Exception as e: logger.critical(f"Index building failed: {e}", exc_info=True); sys.exit(1)
    try:
        sbert_name = config.get("sbert_model_name"); sbert_model = get_sbert_model(sbert_name)
        if sbert_model is None: raise RuntimeError("SBERT model failed to load.")
        logger.info(f"SBERT model '{sbert_name or 'default'}' loaded.")
    except Exception as e: logger.critical(f"SBERT loading failed: {e}", exc_info=True); sys.exit(1)
    embed_cache_params = {"model": sbert_name or "default"}; embed_cache_f = get_cache_filename("embeddings", cache_ver, cache_dir, embed_cache_params, ".pkl")
    embed_data = precompute_taxonomy_embeddings(_taxonomy_concepts_cache, sbert_model, embed_cache_f, cache_ver, args.rebuild_cache, args.debug,)
    if embed_data is None: logger.critical("Embeddings failed."); sys.exit(1)
    _taxonomy_embeddings_cache = embed_data; primary_embs, uris_w_embeds = _taxonomy_embeddings_cache; logger.info(f"Embeddings ready ({len(uris_w_embeds)} concepts).")
    logger.info("Starting affinity definition generation loop...")
    start_loop = time.time(); results = generate_affinity_definitions_loop(input_concepts, config, args, sbert_model, primary_embs, _taxonomy_concepts_cache, _keyword_label_index, _bm25_model, _keyword_corpus_uris,)
    end_loop = time.time(); logger.info(f"Loop finished ({end_loop - start_loop:.2f}s). Generated {len(results)} definitions.")
    if results: save_results_json(results, out_file)
    else: logger.warning("No results generated to save.")
    logger.info(f"Script finished. Log: {log_file}")


if __name__ == "__main__":
    # Check essential library availability
    libs_ok = True
    if not NUMPY_AVAILABLE:
        print("CRITICAL: numpy missing.", file=sys.stderr)
        libs_ok = False
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("CRITICAL: sentence-transformers missing.", file=sys.stderr)
        libs_ok = False
    if not RDFLIB_AVAILABLE:
        print("Warning: rdflib missing.", file=sys.stderr)  # Warn only for rdflib
    if not UTILS_ST_AVAILABLE:
        print(
            "CRITICAL: sentence-transformers not available in utils.", file=sys.stderr
        )
        libs_ok = False
    if not UTILS_RDFLIB_AVAILABLE:
        print("Warning: rdflib not available in utils.", file=sys.stderr)  # Warn only
    if not libs_ok:
        sys.exit(1)

    if not BM25S_AVAILABLE:
        print(
            "Warning: bm2s library not found. Keyword scoring may be affected.",
            file=sys.stderr,
        )

    # Check for LLM API keys based on effective provider
    llm_keys_found = False
    llm_provider_in_use = DEFAULT_CONFIG.get("LLM_PROVIDER", "none")
    # Basic check in sys.argv without full argparse
    temp_args = sys.argv[1:]
    if "--llm-provider" in temp_args:
        try:
            provider_index = temp_args.index("--llm-provider")
            if provider_index + 1 < len(temp_args):
                llm_provider_in_use = temp_args[provider_index + 1]
        except (ValueError, IndexError):
            pass

    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if openai_key:
        print("OpenAI Key found.")
        llm_keys_found = True
    if google_key:
        print("Google Key found.")
        llm_keys_found = True

    if not llm_keys_found and llm_provider_in_use != "none":
        print(
            f"Warning: LLM provider is '{llm_provider_in_use}' but no corresponding API key found in environment variables (GOOGLE_API_KEY or OPENAI_API_KEY). LLM features will likely fail.",
            file=sys.stderr,
        )
    elif llm_provider_in_use == "openai" and not openai_key:
        print(
            f"Warning: LLM provider is 'openai' but OPENAI_API_KEY is not set.",
            file=sys.stderr,
        )
    elif llm_provider_in_use == "google" and not google_key:
        print(
            f"Warning: LLM provider is 'google' but GOOGLE_API_KEY is not set.",
            file=sys.stderr,
        )

    main() 