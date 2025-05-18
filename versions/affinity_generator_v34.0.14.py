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
def load_acs_data(path: Optional[str]) -> Optional[pd.DataFrame]:
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
    desc = theme_data.get("description")
    if isinstance(desc, str) and desc.strip(): return desc.strip()
    else:
        logger.warning(f"Theme '{theme_name}' missing desc.");
        return f"Theme related to {theme_name} ({theme_data.get('type', 'general')} aspects)."

def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: Dict) -> Tuple[str, float, Optional[str], Optional[Dict]]:
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
    total = sum(weights_dict.values())
    if total > 0: return {k: v / total for k, v in weights_dict.items()}
    else: return {k: 0.0 for k in weights_dict}

def deduplicate_attributes(attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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