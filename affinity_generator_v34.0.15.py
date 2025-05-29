# -*- coding: utf-8 -*-
"""
Generate affinity definitions for travel concepts using combined scoring
(v34.0.15 - Enhanced LLM Slotting Prompt + Core Definitional Attributes + New Theme Support).

Implements:
- Enhanced LLM slotting prompt to consider traveler preferences, intents, offerings, and sentiment.
- Logic to identify "Core Definitional URIs" (anchor + strong textual variants) and ensure
  their presence in top_defining_attributes.
- Support for new themes like Pricing, TravelerIntent_Preference, etc.
- (Inherited from v34.0.14) Weights attributes within themes based on combined_score.
- (Inherited from v34.0.14) Dynamically determines 'applicable_lodging_types'.
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
    NUMPY_AVAILABLE = False # type: ignore

try:
    from rdflib import Graph, URIRef, Literal, Namespace # type: ignore
    from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF # type: ignore
    from rdflib import util as rdflib_util # type: ignore
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False # type: ignore

try:
    from tqdm import tqdm # type: ignore
except ImportError:
    print("Warning: tqdm not found.", file=sys.stderr)
    def tqdm(iterable, *args, **kwargs): # type: ignore
        return iterable

try:
    import pandas as pd # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas library not found. ACS Enrichment will be disabled.", file=sys.stderr)
    PANDAS_AVAILABLE = False

try:
    import bm25s # type: ignore
    BM25S_AVAILABLE = True
except ImportError:
    print("Warning: bm2s library not found. BM25s scoring will be disabled.", file=sys.stderr)
    BM25S_AVAILABLE = False # type: ignore
    class Bm25sDummy: # type: ignore
        class BM25: # type: ignore
            pass
        @staticmethod
        def tokenize(text): # type: ignore
            return [str(t).split() for t in text]
    bm25s = Bm25sDummy() # type: ignore

# --- LLM Imports & Placeholders ---
class APITimeoutError(Exception): # type: ignore
    pass
class APIConnectionError(Exception): # type: ignore
    pass
class RateLimitError(Exception): # type: ignore
    pass
class DummyLLMClient: # type: ignore
    pass

OpenAI_Type = DummyLLMClient # type: ignore
GoogleAI_Type = DummyLLMClient # type: ignore
OpenAI = None # type: ignore
OPENAI_AVAILABLE = False
genai = None # type: ignore
GOOGLE_AI_AVAILABLE = False
try:
    from openai import ( # type: ignore
        OpenAI as RealOpenAI, # type: ignore
        APITimeoutError as RealAPITimeoutError, # type: ignore
        APIConnectionError as RealAPIConnectionError, # type: ignore
        RateLimitError as RealRateLimitError, # type: ignore
    )
    OpenAI = RealOpenAI # type: ignore
    OpenAI_Type = RealOpenAI # type: ignore
    APITimeoutError = RealAPITimeoutError # type: ignore
    APIConnectionError = RealAPIConnectionError # type: ignore
    RateLimitError = RealRateLimitError # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    logging.debug("OpenAI library not found (optional).")
except Exception as e:
    logging.error(f"OpenAI import error: {e}")
try:
    import google.generativeai as genai_import # type: ignore
    genai = genai_import # type: ignore
    GoogleAI_Type = genai_import.GenerativeModel # type: ignore
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    logging.debug("google.generativeai library not found (optional).")
except Exception as e:
    GOOGLE_AI_AVAILABLE = False
    genai = None # type: ignore
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
        logging.warning("Mismatch in RDFLIB availability between script and utils.")
    if SENTENCE_TRANSFORMERS_AVAILABLE != UTILS_ST_AVAILABLE:
        logging.warning("Mismatch in SentenceTransformers availability between script and utils.")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed import from 'utils.py': {e}. Ensure utils.py is in the same directory or Python path.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Unexpected error importing from 'utils.py': {e}", file=sys.stderr)
    sys.exit(1)

# --- Config Defaults & Constants ---
SCRIPT_VERSION = "affinity-rule-engine-v34.0.15 (Enhanced LLM Slotting + Core Def Attrs)"
DEFAULT_CACHE_VERSION = "v20250502.affinity.34.0.15"
DEFAULT_CONFIG_FILE = "./affinity_config_v34.13.json" # Expecting updated config with new themes
OUTPUT_FILE_TEMPLATE = "affinity_definitions_{cache_version}.json"
LOG_FILE_TEMPLATE = "affinity_generation_{cache_version}.log"

DEFAULT_CONFIG = {
    "output_dir": "./output_v34.15",
    "cache_dir": "./cache_v34.15",
    "taxonomy_dir": "./datasources/",
    "cache_version": DEFAULT_CACHE_VERSION,
    "sparql_endpoint": "http://localhost:7200/repositories/your-repo", # Example
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
        "urn:expediagroup:taxonomies:core", "urn:expediagroup:taxonomies:acsPCS",
        "urn:expediagroup:taxonomies:acs", "urn:expediagroup:taxonomies:spaces",
        "urn:expediagroup:taxonomies:activities", "urn:expe:taxo:amenity-view-property-features",
        "urn:expe:taxo:property-media", "urn:expe:taxo:trip-preferences",
        "urn:expediagroup:taxonomies:lcm:", "http://schema.org/"
    ],
    "NAMESPACE_BIASING": {
        "enabled": True, "core_boost_factor": 1.10, "boost_factor": 1.05,
        "context_boost_factor": 1.02, "penalty_factor": 0.95,
        "strong_penalty_factor": 0.85, "metadata_penalty_factor": 0.90,
        "lodging_amenity_ns": [
            "urn:expediagroup:taxonomies:acs", "urn:expediagroup:taxonomies:spaces",
            "urn:expe:taxo:hospitality", "urn:expe:taxo:amenity-view-property-features",
            "urn:expediagroup:taxonomies:lcm:", "urn:expediagroup:taxonomies:acsPCS",
            "urn:expediagroup:taxonomies:acsBaseAttribute"
        ],
        "activity_ns": ["urn:expediagroup:taxonomies:activities", "urn:expe:taxo:events"],
        "location_context_ns": ["urn:expediagroup:taxonomies:gaia", "urn:expediagroup:taxonomies:places"],
        "visual_context_ns": ["urn:expe:taxo:media-descriptors", "urn:expe:taxo:property-media"],
        "preference_context_ns": ["urn:expe:taxo:trip-preferences", "urn:expe:taxo:personalization"],
        "clearly_wrong_ns": ["urn:expe:taxo:cars:", "urn:expe:taxo:flights:"],
        "metadata_ns": [
            "urn:expe:taxo:checkout:", "urn:expe:taxo:payments:", "urn:expe:taxo:policies:",
            "urn:expe:taxo:review_categories:", "urn:expe:taxo:review_category_values:",
            "urn:expe:taxo:reviews-attributes:", "urn:expe:taxo:text:",
            "urn:expe:taxo:data-element-values:", "urn:expediagroup:taxonomies:acsDomainType:",
            "urn:expediagroup:taxonomies:acsBaseTerms:", "urn:expediagroup:taxonomies:taxonomy_management:",
            "urn:expediagroup:taxonomies:acsEnumerations:"
        ],
        "low_priority_ns": ["urn:expediagroup:taxonomies:tmpt:"]
    },
    "KEYWORD_SCORING_CONFIG": {
        "enabled": True, "algorithm": "bm25s", "bm25_min_score": 0.01, "bm25_top_n": 500
    },
    "KG_CONFIG": {
        "pref_label_weight": 3, "alt_label_weight": 1, "definition_weight": 0,
        "acs_name_weight": 4, "acs_def_weight": 2
    },
    "ACS_DATA_CONFIG": {
        "acs_data_path": "./datasources/transformed_acs_tracker.csv", "enable_acs_enrichment": True,
    },
    "STAGE1_CONFIG": {
        "MAX_CANDIDATES_FOR_LLM": 75, "EVIDENCE_MIN_SIMILARITY": 0.30,
        "MIN_KEYWORD_CANDIDATES_FOR_EXPANSION": 5, "ENABLE_KW_EXPANSION": True,
        "KW_EXPANSION_TEMPERATURE": 0.5,
        "CORE_DEF_TEXT_SIMILARITY_THRESHOLD": 0.90, # Jaro-Winkler for normalized labels
        "CORE_DEF_MAX_VARIANTS": 3
    },
    "STAGE2_CONFIG": {
        "ENABLE_LLM_REFINEMENT": True, "LLM_TEMPERATURE": 0.2,
        "THEME_ATTRIBUTE_MIN_WEIGHT": 0.001, "TOP_N_DEFINING_ATTRIBUTES": 25,
        "UNTHEMED_CAPTURE_SCORE_PERCENTILE": 75,
        "LODGING_TYPE_TOP_ATTR_CHECK": 10, "LODGING_TYPE_CONFIDENCE_THRESHOLD": 0.6,
        "CORE_DEF_FORCED_ATTRIBUTE_WEIGHT": 0.05
    },
    "LLM_API_CONFIG": {"MAX_RETRIES": 5, "RETRY_DELAY_SECONDS": 5, "REQUEST_TIMEOUT": 180},
    "LLM_NEGATION_CONFIG": {"enabled": True, "temperature": 0.3, "max_candidates_to_check": 30},
    "vrbo_default_subscore_weights": {"SentimentScore": 0.1, "GroupIntelligenceScore": 0.1},
    "base_themes": {
      "Location": { "description": "Spatial proximity to points of interest, neighborhoods, accessibility, walkability.", "type": "decision", "weight": 10, "subScore": "LocationAffinity", "rule_applied": "Optional", "relevant_subscores": {"WalkabilityScore": 0.6, "GeospatialAffinityScore": 0.4}},
      "Privacy": { "description": "Seclusion, lack of shared spaces, intimacy, personal space.", "type": "comfort", "weight": 5, "subScore": "PrivacyAffinity", "rule_applied": "Optional", "relevant_subscores": {"PrivacyAffinityScore": 1.0}},
      "Indoor Amenities": { "description": "Features inside the lodging unit or main building (e.g., kitchen, fireplace, specific room types).", "type": "structural", "weight": 9, "rule_applied": "Optional"},
      "Outdoor Amenities": { "description": "Features outside the main unit but on the property (e.g., pool, yard, hot tub, BBQ).", "type": "structural", "weight": 4, "rule_applied": "Optional"},
      "Activities": { "description": "Recreational activities available on-site or very nearby (e.g., skiing, swimming, hiking, nightlife).", "type": "preference", "weight": 8, "rule_applied": "Optional"},
      "Imagery": { "description": "Visual appeal, aesthetics, design style, views.", "type": "imagery", "weight": 10, "subScore": "ImageAffinity", "rule_applied": "Optional", "relevant_subscores": {"ImageAffinityScore": 1.0}},
      "Spaces": { "description": "Types of rooms, size, layout, capacity (e.g., multiple bedrooms, large kitchen, balcony).", "type": "structural", "weight": 6, "rule_applied": "Optional"},
      "Seasonality": { "description": "Relevance tied to specific seasons, weather, or times of year (e.g., skiing, beach access).", "type": "temporal", "weight": 6, "subScore": "SeasonalityAffinity", "rule_applied": "Optional", "relevant_subscores": {"SeasonalityAffinityScore": 1.0}},
      "Group Relevance": { "description": "Suitability for specific traveler groups, their common preferences, or typical trip purposes (e.g., family-friendly, romantic getaway, business travel, adventure trip).", "type": "preference", "weight": 5, "subScore": "GroupIntelligenceAffinity", "rule_applied": "Optional", "relevant_subscores": {"GroupIntelligenceScore": 1.0}},
      "Technology": { "description": "Availability and quality of tech features (e.g., Wi-Fi, smart home tech, entertainment systems).", "type": "technological", "weight": 7, "rule_applied": "Optional", "relevant_subscores": {"TechnologyScore": 1.0}},
      "Sentiment": { "description": "Overall feeling, vibe, or subjective quality described (e.g., cozy, luxurious, rustic, modern).", "type": "comfort", "weight": 10, "subScore": "SentimentAffinity", "rule_applied": "Optional", "relevant_subscores": {"SentimentScore": 1.0}},
      "Pricing": { "description": "Concepts related to the cost, value, affordability, payment, discounts, and overall financial aspect of the lodging or travel experience. Includes pricing tiers, deals, and perceived value for money.", "type": "decision", "weight": 9, "subScore": "PricingAffinity", "rule_applied": "Optional", "hints": {"keywords": ["price", "cost", "value", "deal", "discount", "affordable", "expensive", "rate", "fee", "payment", "budget", "luxury tier"]}},
      "TravelerIntent_Preference": { "description": "Specific traveler intentions, preferences, needs, or desired experiences directly related to or sought after with the input concept. Captures why a traveler chooses something or what they hope to achieve/feel. (e.g., 'relaxation' for 'spa', 'adventure seeking' for 'mountain cabin', 'budget consciousness' for 'hostel').", "type": "preference", "weight": 7, "subScore": "TravelerIntentAffinity", "rule_applied": "Optional", "hints": {"keywords": ["intent", "preference", "goal", "need", "experience seeking", "purpose", "motivation"]}},
      "Offering_Service_Package": { "description": "Distinct services, packages, plans, or types of offerings that are either defined by the input concept or are commonly bundled with it. (e.g., 'all-inclusive plan', 'guided tour package', 'pet-sitting service', 'early check-in option').", "type": "structural", "weight": 6, "subScore": "OfferingAffinity", "rule_applied": "Optional", "hints": {"keywords": ["service", "package", "plan", "offering", "bundle", "add-on"]}},
      "GeneralRelevance_CoreAspect": { "description": "Captures concepts that are highly relevant and definitional to the input concept but may not fit neatly into more specific feature, amenity, or activity themes. Use for core characteristics or very broad associations when other themes are not suitable.", "type": "preference", "weight": 3, "subScore": "GeneralRelevanceAffinity", "rule_applied": "Optional"}
    },
    "concept_overrides": {},
    "master_subscore_list": [
      "WalkabilityScore", "SentimentScore", "SeasonalityAffinityScore", "GroupIntelligenceScore",
      "PrivacyAffinityScore", "AccessibilityAffinityScore", "SustainabilityAffinityScore",
      "ImageAffinityScore", "GeospatialAffinityScore", "UniquePropertyAffinityScore",
      "TrendingPropertyAffinityScore", "VrboLongStaysAffinityScore", "TechnologyScore",
      "LocationScore", "PetFriendlyAffinityScore",
      "PricingAffinity", "TravelerIntentAffinity", "OfferingAffinity", "GeneralRelevanceAffinity"
    ],
    "lodging_type_hints": {
        "CL": ["urn:expediagroup:taxonomies:core:#Hotel", "urn:expediagroup:taxonomies:acs:#FrontDesk"],
        "VR": ["urn:expediagroup:taxonomies:core:#VacationRental", "urn:expediagroup:taxonomies:acs:#FullKitchen"],
        "Both": ["urn:expediagroup:taxonomies:core:#Amenity", "urn:expediagroup:taxonomies:acs:#WiFi"]
     }
}
ABSTRACT_CONCEPTS_LIST = ["luxury", "budget", "value"]
_config_data: Optional[Dict] = None
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict[str, np.ndarray], List[str]]] = None # type: ignore
_keyword_label_index: Optional[Dict[str, Set[str]]] = None
_bm25_model: Optional[bm25s.BM25] = None # type: ignore
_keyword_corpus_uris: Optional[List[str]] = None
_acs_data: Optional[pd.DataFrame] = None # type: ignore
_openai_client: Optional[OpenAI_Type] = None
_google_client: Optional[GoogleAI_Type] = None
logger = logging.getLogger(__name__)
args: Optional[argparse.Namespace] = None


# --- NEW HELPER for Text Similarity ---
try:
    import jellyfish # type: ignore
    JELLYFISH_AVAILABLE = True
    logger.info("jellyfish library found for text similarity.")
except ImportError:
    JELLYFISH_AVAILABLE = False
    logger.warning("jellyfish library not found. Text similarity for Core Definitional URIs will use a basic fallback.")

def get_text_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    if JELLYFISH_AVAILABLE:
        return jellyfish.jaro_winkler_similarity(s1, s2) # type: ignore
    else:
        # Basic fallback: normalized common words ratio
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 and not words2: return 1.0
        if not words1 or not words2: return 0.0
        common = len(words1.intersection(words2))
        total = len(words1.union(words2))
        return common / total if total > 0 else 0.0
# --- END NEW HELPER ---


# --- Helper Functions ---
def load_acs_data(path: Optional[str]) -> Optional[pd.DataFrame]: # type: ignore
    if not PANDAS_AVAILABLE: logger.error("Pandas library required but not available."); return None
    if not path: logger.warning("ACS data path not configured."); return None
    if not os.path.exists(path): logger.warning(f"ACS data file not found: '{path}'."); return None
    try:
        start_time = time.time(); acs_df = pd.read_csv(path, index_col="URI", low_memory=False) # type: ignore
        if "AttributeNameENG" not in acs_df.columns: acs_df["AttributeNameENG"] = "" # type: ignore
        else: acs_df["AttributeNameENG"] = acs_df["AttributeNameENG"].fillna("") # type: ignore
        if "ACS Definition" not in acs_df.columns: acs_df["ACS Definition"] = "" # type: ignore
        else: acs_df["ACS Definition"] = acs_df["ACS Definition"].fillna("") # type: ignore
        logging.info(f"Loaded ACS data '{path}' ({len(acs_df):,} entries) in {time.time() - start_time:.2f}s."); return acs_df # type: ignore
    except KeyError: logger.error(f"ACS file '{path}' missing 'URI' column."); return None
    except Exception as e: logger.error(f"Error loading ACS data '{path}': {e}."); return None

def get_theme_definition_for_prompt(theme_name: str, theme_data: Dict) -> str:
    desc = theme_data.get("description")
    if isinstance(desc, str) and desc.strip(): return desc.strip()
    else:
        logger.warning(f"Theme '{theme_name}' missing description.");
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
    for uri_val in missing: assigns.setdefault(uri_val, [])
    parsed_count = 0; uris_proc = 0
    for uri, themes in assigns.items():
        if uri not in uris_sent: continue
        uris_proc += 1
        if not isinstance(themes, list): validated[uri] = []; continue
        valid = [t for t in themes if isinstance(t, str) and t in valid_themes]; invalid = set(themes) - set(valid)
        if invalid: logger.warning(f"[{concept_label}] Invalid themes for {uri}: {invalid}")
        validated[uri] = valid; parsed_count += len(valid)
    diag_llm["assignments_parsed_count"] = parsed_count; diag_llm["uris_in_response_count"] = uris_proc
    return validated

# --- Indexing ---
def build_keyword_index(config: Dict, taxonomy_concepts_cache: Dict[str, Dict], cache_dir: str, cache_version: str, rebuild_cache: bool) -> Tuple[Optional[bm25s.BM25], Optional[List[str]], Optional[Dict[str, Set[str]]]]: # type: ignore
    global _bm25_model, _keyword_corpus_uris, _keyword_label_index, _acs_data, args; bm25_model, corpus_uris, label_index = None, None, None # type: ignore
    kw_scoring_cfg = config.get("KEYWORD_SCORING_CONFIG", {}); use_bm25 = (kw_scoring_cfg.get("enabled", False) and kw_scoring_cfg.get("algorithm", "").lower() == "bm25s" and BM25S_AVAILABLE)
    kg_cfg = config.get("KG_CONFIG", {}); acs_cfg = config.get("ACS_DATA_CONFIG", {}); pref_label_weight = int(kg_cfg.get("pref_label_weight", 3)); alt_label_weight = int(kg_cfg.get("alt_label_weight", 1)); definition_weight = int(kg_cfg.get("definition_weight", 0))
    acs_name_weight = int(kg_cfg.get("acs_name_weight", acs_cfg.get("acs_name_weight",0))); acs_def_weight = int(kg_cfg.get("acs_def_weight", acs_cfg.get("acs_def_weight",0))); enrichment_enabled = acs_cfg.get("enable_acs_enrichment", False) and (_acs_data is not None)
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
            if (cached_model is not None and isinstance(cached_model, bm25s.BM25) and isinstance(cached_uris, list) and len(cached_uris) > 0): # type: ignore
                bm25_model, corpus_uris, cache_valid = (cached_model, cached_uris, True); logger.info(f"BM25s Model loaded from cache ({len(corpus_uris)} URIs).") # type: ignore
            else: logger.info("BM25s cache files missing or invalid. Rebuilding."); cache_valid = False
        if not cache_valid:
            logger.info("Rebuilding BM25s index...");
            try:
                corpus_strings: List[str] = []; doc_uris_list: List[str] = []
                debug_mode = args.debug if args else False; disable_tqdm = not logger.isEnabledFor(logging.INFO) or debug_mode; enriched_count = 0
                logger.info("Preparing document strings for BM25s...")
                for uri_str, data_dict in tqdm(sorted(taxonomy_concepts_cache.items()), desc="Prepare BM25s Docs", disable=disable_tqdm):
                    texts_to_join: List[str] = [];
                    if not isinstance(data_dict, dict): continue
                    pref_labels_raw = data_dict.get("skos:prefLabel", []); alt_labels_raw = data_dict.get("skos:altLabel", []); definitions_raw = data_dict.get("skos:definition", [])
                    pref_labels = [str(lbl) for lbl in (pref_labels_raw if isinstance(pref_labels_raw, list) else [pref_labels_raw]) if lbl and isinstance(lbl, str)]
                    alt_labels = [str(lbl) for lbl in (alt_labels_raw if isinstance(alt_labels_raw, list) else [alt_labels_raw]) if lbl and isinstance(lbl, str)]
                    kg_definition = None
                    if isinstance(definitions_raw, list):
                        for defin in definitions_raw:
                            if defin and isinstance(defin, str) and defin.strip(): kg_definition = str(defin); break
                    elif isinstance(definitions_raw, str) and definitions_raw.strip(): kg_definition = definitions_raw
                    texts_to_join.extend(pref_labels * pref_label_weight); texts_to_join.extend(alt_labels * alt_label_weight);
                    if kg_definition and definition_weight > 0: texts_to_join.extend([kg_definition] * definition_weight)
                    if (enrichment_enabled and _acs_data is not None and uri_str in _acs_data.index):
                        try:
                            acs_record = _acs_data.loc[uri_str]; # type: ignore
                            if isinstance(acs_record, pd.DataFrame): acs_record = acs_record.iloc[0] # type: ignore
                            acs_name = (acs_record.get("AttributeNameENG", "") if pd.notna(acs_record.get("AttributeNameENG", "")) else ""); acs_def = (acs_record.get("ACS Definition", "") if pd.notna(acs_record.get("ACS Definition", "")) else "") # type: ignore
                            if acs_name or acs_def: enriched_count += 1
                            if acs_name and acs_name_weight > 0: texts_to_join.extend([acs_name] * acs_name_weight)
                            if acs_def and acs_def_weight > 0: texts_to_join.extend([acs_def] * acs_def_weight)
                        except Exception as acs_lookup_err: logger.warning(f"Error enriching URI '{uri_str}': {acs_lookup_err}.")
                    raw_doc_text = " ".join(filter(None, texts_to_join)); corpus_strings.append(raw_doc_text); doc_uris_list.append(uri_str)
                initial_doc_count = len(corpus_strings)
                if initial_doc_count == 0: logger.warning("No documents generated."); bm25_model = None; corpus_uris = [] # type: ignore
                else:
                    valid_indices = [i for i, doc_str_val in enumerate(corpus_strings) if doc_str_val and doc_str_val.strip()]
                    if len(valid_indices) < initial_doc_count:
                        empty_count = initial_doc_count - len(valid_indices); logger.warning(f"Found {empty_count} empty doc strings. Filtering.")
                        if not doc_uris_list or len(doc_uris_list) != initial_doc_count: logger.error("FATAL: URI list mismatch."); bm25_model = None; corpus_uris = [] # type: ignore
                        else: filtered_corpus_strings = [corpus_strings[i] for i in valid_indices]; filtered_doc_uris_val = [doc_uris_list[i] for i in valid_indices]; logger.info(f"Proceeding with {len(filtered_corpus_strings)} non-empty docs.") # type: ignore
                    else: filtered_corpus_strings = corpus_strings; filtered_doc_uris_val = doc_uris_list; logger.info("All generated doc strings non-empty.") # type: ignore
                    if filtered_corpus_strings and bm25_model is None: # type: ignore
                        logger.info(f"Tokenizing {len(filtered_corpus_strings)} documents for BM25s...");
                        try:
                            tokenized_corpus = bm25s.tokenize(filtered_corpus_strings); logger.info(f"Building BM25s index for {len(filtered_corpus_strings)} documents ({enriched_count} enriched).") # type: ignore
                            bm25_model = bm25s.BM25(); bm25_model.index(tokenized_corpus); corpus_uris = filtered_doc_uris_val; logger.debug(f"BM25s Model created/indexed. Type: {type(bm25_model)}") # type: ignore
                        except Exception as bm25_init_err: logger.error(f"BM25s init/index error: {bm25_init_err}", exc_info=True); bm25_model = None; corpus_uris = [] # type: ignore
                    if bm25_model and corpus_uris: save_cache(bm25_model, model_cache_file, "pickle"); save_cache(corpus_uris, uris_cache_file, "pickle"); logger.info("BM25s model/URIs rebuilt/saved.") # type: ignore
                    else: logger.warning("Skipping BM25s cache saving due to build issues.")
            except Exception as e_bm: logger.error(f"BM25s build error: {e_bm}", exc_info=True); bm25_model, corpus_uris = None, None # type: ignore
        logger.debug("--- BM25s Index Diagnostics ---");
        logger.debug(f"BM25s Model: {'Ready' if bm25_model else 'Not available.'}"); # type: ignore
        logger.debug(f"Corpus URIs: {len(corpus_uris) if corpus_uris is not None else 'None'}"); # type: ignore
        logger.debug("--- End BM25s Diagnostics ---")
    else: logger.info("BM25s Keyword Indexing disabled or library unavailable.")
    try:
        label_index = build_keyword_label_index(taxonomy_concepts_cache) # type: ignore
        if label_index is not None: logger.info(f"Simple label index ready ({len(label_index)} keywords).") # type: ignore
        else: logger.warning("Failed simple label index build.")
    except Exception as e_li: logger.error(f"Error building label index: {e_li}", exc_info=True); label_index = None # type: ignore
    _bm25_model = bm25_model; _keyword_corpus_uris = corpus_uris; _keyword_label_index = label_index # type: ignore
    return bm25_model, corpus_uris, label_index # type: ignore

def get_candidate_concepts_keyword(query_texts: List[str], bm25_model: bm25s.BM25, corpus_uris: List[str], top_n: int, min_score: float = 0.01) -> List[Dict[str, Any]]: # type: ignore
    global args;
    if not query_texts or bm25_model is None or not corpus_uris or not BM25S_AVAILABLE: return []
    query_string = " ".join(normalize_concept(text) for text in query_texts)
    if not query_string: logger.warning(f"Empty query string from input: {query_texts}"); return []
    try:
        tokenized_query_list = bm25s.tokenize([query_string]); results_indices, results_scores = bm25_model.retrieve(tokenized_query_list, k=top_n); candidates: List[Dict[str, Any]] = [] # type: ignore
        if (results_indices is not None and results_scores is not None and len(results_indices) > 0 and len(results_scores) > 0 and isinstance(results_indices[0], np.ndarray) and isinstance(results_scores[0], np.ndarray) and results_indices[0].size > 0 and results_scores[0].size > 0): # type: ignore
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
    global _openai_client, _config_data;
    if not OPENAI_AVAILABLE: return None
    if _openai_client is None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY");
            if not api_key: logger.warning("OPENAI_API_KEY env var not set."); return None
            timeout_val = _config_data.get("LLM_API_CONFIG", {}).get("REQUEST_TIMEOUT", 60) if _config_data else 60 # type: ignore
            _openai_client = OpenAI(api_key=api_key, timeout=timeout_val) if OpenAI else None # type: ignore
        except Exception as e: logger.error(f"OpenAI client init failed: {e}"); return None
    return _openai_client

def get_google_client() -> Optional[GoogleAI_Type]:
    global _google_client;
    if not GOOGLE_AI_AVAILABLE or not genai: return None
    if _google_client is None:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY");
            if not api_key: logger.warning("GOOGLE_API_KEY env var not set."); return None
            genai.configure(api_key=api_key); _google_client = genai # type: ignore
        except Exception as e: logger.error(f"Google AI config failed: {e}"); return None
    return _google_client

def call_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float, provider: str, input_concept_for_logging: Optional[str] = None) -> Dict[str, Any]:
    if not model_name: logger.error("LLM model missing."); return {"success": False, "error": "LLM model missing"}
    result: Dict[str, Any] = {"success": False, "response": None, "error": None, "attempts": 0}; client: Any = None
    if provider == "openai":
        client = get_openai_client()
    elif provider == "google":
        client = get_google_client()
    else: result["error"] = f"Unsupported provider: {provider}"; logger.error(result["error"]); return result
    if not client: result["error"] = f"{provider} client unavailable."; logger.error(result["error"]); return result
    delay_val = _config_data.get("LLM_API_CONFIG", {}).get("RETRY_DELAY_SECONDS", 5) if _config_data else 5
    for attempt in range(max_retries + 1):
        result["attempts"] = attempt + 1; start_time = time.time();
        try:
            content = None
            if provider == "openai":
                if not isinstance(client, OpenAI_Type): raise RuntimeError("OpenAI client invalid type.") # type: ignore
                resp = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=temperature, timeout=timeout,) # type: ignore
                if resp.choices: content = resp.choices[0].message.content # type: ignore
            elif provider == "google":
                model_id_str = model_name;
                if not model_id_str.startswith("models/"): model_id_str = f"models/{model_id_str}"
                gen_cfg_obj = client.types.GenerationConfig(candidate_count=1, temperature=temperature, response_mime_type="application/json",) # type: ignore
                safety_settings_list = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",]]
                if hasattr(client, "GenerativeModel"):
                    model_obj = client.GenerativeModel(model_id_str, system_instruction=system_prompt, safety_settings=safety_settings_list,) # type: ignore
                    resp_obj = model_obj.generate_content([user_prompt], generation_config=gen_cfg_obj, request_options={"timeout": timeout},) # type: ignore
                    if (hasattr(resp_obj, "prompt_feedback") and resp_obj.prompt_feedback and resp_obj.prompt_feedback.block_reason): # type: ignore
                        block_reason_str = getattr(resp_obj.prompt_feedback, "block_reason", "?"); # type: ignore
                        logger.warning(f"Gemini blocked. Reason:{block_reason_str}"); result["error"] = f"Blocked:{block_reason_str}"; return result
                    if hasattr(resp_obj, "candidates") and resp_obj.candidates: # type: ignore
                        if hasattr(resp_obj.candidates[0], "content") and hasattr(resp_obj.candidates[0].content, "parts"): # type: ignore
                            if resp_obj.candidates[0].content.parts: content = resp_obj.candidates[0].content.parts[0].text # type: ignore
                            else: logger.warning(f"Gemini response candidate parts are empty. Resp:{resp_obj}"); content = None # type: ignore
                        else: logger.warning(f"Gemini response candidate structure unexpected. Resp:{resp_obj}"); content = None # type: ignore
                    else: logger.warning(f"Gemini response has no candidates or text. Resp:{resp_obj}"); content = None # type: ignore
                else: raise RuntimeError("Google AI client is not configured correctly or is of wrong type.")
            if content:
                # --- ADDED DEBUG LOGGING ---
                active_concept_for_log = input_concept_for_logging or "Unknown Concept"
                logger.debug(f"LLM Raw Response Content (Provider: {provider}, Concept: {active_concept_for_log}):\\n{content}")
                # --- END ADDED DEBUG LOGGING ---
                try:
                    json_match_obj = re.search(r"```json\\s*(\{.*?\})\\s*```", content, re.DOTALL | re.IGNORECASE)
                    if json_match_obj: cleaned_content_str = json_match_obj.group(1)
                    elif "{" in content and "}" in content:
                        first_brace_idx = content.find("{"); last_brace_idx = content.rfind("}")
                        if (first_brace_idx != -1 and last_brace_idx != -1 and last_brace_idx > first_brace_idx): cleaned_content_str = content[first_brace_idx : last_brace_idx + 1]
                        else: cleaned_content_str = content.strip().strip("`")
                    else: cleaned_content_str = content.strip().strip("`")
                    # --- ADDED DEBUG LOGGING ---
                    logger.debug(f"LLM Cleaned Content (Provider: {provider}, Concept: {active_concept_for_log}):\\n{cleaned_content_str}")
                    # --- END ADDED DEBUG LOGGING ---
                    output_data = json.loads(cleaned_content_str)
                    if isinstance(output_data, dict): result["success"] = True; result["response"] = output_data; logger.debug(f"{provider} call successful in {time.time() - start_time:.2f}s (Attempt {attempt+1})"); return result
                    else: logger.error(f"{provider} response parsed but is not a dict. Type:{type(output_data)}. Raw:{content[:200]}..."); result["error"] = f"LLM response is not a JSON object (dict), type was {type(output_data)}"
                except json.JSONDecodeError as e_json: logger.error(f"{provider} JSON parse error: {e_json}. Raw content snippet: {content[:500]}..."); result["error"] = f"JSON Parse Error: {e_json}"
                except Exception as e_proc: logger.error(f"{provider} response processing error: {e_proc}. Raw content snippet: {content[:500]}..."); result["error"] = f"Response Processing Error: {e_proc}"
            else: logger.warning(f"{provider} response content was empty."); result["error"] = "Empty response from LLM"
        except (APITimeoutError, APIConnectionError, RateLimitError) as e_api: logger.warning(f"{provider} API Error on attempt {attempt + 1}: {type(e_api).__name__}"); result["error"] = f"{type(e_api).__name__}" # type: ignore
        except Exception as e_call: logger.error(f"{provider} Call Error on attempt {attempt + 1}: {e_call}", exc_info=True); result["error"] = str(e_call); return result
        should_retry = (attempt < max_retries and result["error"] is not None and "Blocked" not in str(result.get("error")));
        if should_retry:
            current_delay = delay_val * (2**attempt)
            wait_time_val = current_delay + (np.random.uniform(0, current_delay * 0.5) if NUMPY_AVAILABLE else (current_delay * 0.1)) # type: ignore
            logger.info(f"Retrying LLM call in {wait_time_val:.2f}s... (Error: {result.get('error')})"); time.sleep(wait_time_val)
        else:
            if not result.get("success"): final_error_str = result.get("error", "Unknown error after retries"); logger.error(f"LLM call failed after {attempt + 1} attempts. Final Error: {final_error_str}")
            if not result.get("error"): result["error"] = "LLM call failed after all retries."
            return result
    return result

def construct_keyword_expansion_prompt(input_concept: str) -> Tuple[str, str]:
    system_prompt = """You are a helpful assistant specializing in travel concepts and keyword analysis for search retrieval. You understand nuances like compound words and relationships between concepts. Your goal is to generate relevant keywords that will improve search results within a travel taxonomy. Output ONLY a valid JSON object containing a single key "keywords" with a list of strings as its value. Do not include any explanations or introductory text outside the JSON structure."""
    user_prompt = f"""
    Given the input travel concept: '{input_concept}'
    Your task is to generate a list of related keywords specifically useful for *improving keyword search retrieval* within a large travel taxonomy. Consider the following:
    1.  **Synonyms:** Include direct synonyms.
    2.  **Constituent Parts:** If compound, include meaningful parts (e.g., 'americanbreakfast' -> 'american', 'breakfast').
    3.  **Related Activities/Concepts/Aspects:** List highly relevant associated items, common traveler goals, or typical services. (e.g., 'budget' -> 'value', 'deals', 'affordable stays', 'backpacker', 'cost saving').
    4.  **Hypernyms/Hyponyms (Optional):** Broader categories or specific examples if very relevant.
    5.  **Relevance:** Focus strictly on terms highly relevant to '{input_concept}' in travel context.
    6.  **Simplicity:** Prefer single words or common short phrases.
    Example Input: 'budget hotel', Example Output: ```json {{"keywords": ["budget", "hotel", "cheap", "affordable", "economy", "value", "low cost", "inn", "motel", "discount stay"]}} ```
    Example Input: 'scuba diving', Example Output: ```json {{"keywords": ["scuba", "diving", "scuba diving", "underwater", "dive", "reef", "snorkel", "padi", "water sports", "marine life"]}} ```
    Now, generate the keywords for: '{input_concept}'. Output ONLY the JSON object:"""
    return system_prompt.strip(), user_prompt.strip()

def construct_llm_slotting_prompt(
    input_concept: str,
    theme_definitions: List[Dict[str, Any]],
    candidate_details: List[Dict[str, Any]], # This list now comes from prepare_evidence -> llm_details_list
    args_namespace: argparse.Namespace,
) -> Tuple[str, str]:
    """Constructs system and user prompts for LLM theme slotting with enhanced guidance (v34.0.15 - more literal example)."""
    system_prompt = """You are an expert travel taxonomist creating structured definitions focused on lodging and travel experiences. Your task is to understand how various concepts (candidates) relate to a primary input travel concept, considering multiple facets of traveler relevance. Output ONLY a valid JSON object with a key "theme_assignments". The value for this key must be a dictionary where EACH candidate URI is a key, and its value is a list of theme names it's relevant to (this list can be empty if no theme applies). Focus on semantic relevance and traveler value. Ensure ALL input URIs are present as keys in your output dictionary."""

    theme_defs_str = "\n".join(
        [
            f"- **{t.get('name', '?')}{' (M)' if t.get('is_must_have') else ''}**: {t.get('description','No description provided.')}"
            for t in theme_definitions
        ]
    )
    must_haves_str = (
        ", ".join([t["name"] for t in theme_definitions if t.get("is_must_have")])
        or "None"
    )
    
    # Build the candidate details string for the prompt
    cand_details_str_for_llm = ""
    actual_candidate_uris_for_example: List[str] = [] 
    for i, c in enumerate(candidate_details): # candidate_details is now the list of dicts
        uri = c.get('uri', f'ERROR_URI_NOT_FOUND_{i+1}')
        label = c.get('label', 'No Label Found') # 'label' is from get_kg_data
        types_list = c.get('types', [])
        types_str = ", ".join(types_list) if types_list else "N/A"
        definition_snippet = (c.get('definitions')[0] if c.get('definitions') and c.get('definitions')[0] else "No definition")[:50] # take first def
        
        cand_details_str_for_llm += f"""
Candidate {i+1}:
  URI_FOR_KEY: {uri}
  Label: '{label}'
  Types: {types_str}
  Definition: '{definition_snippet}...' 
"""
        
        if i < 3 and uri and not uri.startswith("ERROR_URI"): # Store first few actual URIs for the example
            actual_candidate_uris_for_example.append(uri)

    if not cand_details_str_for_llm: cand_details_str_for_llm = "\nNo candidates provided to LLM."

    # Create a very literal schema example using actual URIs if possible
    schema_example_assignments_dict: Dict[str, List[str]] = {}
    # Use a maximum of 2 example URIs or fewer if not enough candidates
    num_example_uris = min(len(actual_candidate_uris_for_example), 2) 

    if num_example_uris > 0:
        schema_example_assignments_dict[actual_candidate_uris_for_example[0]] = ["ThemeNameA", "ThemeNameC"] # Example
        if num_example_uris > 1:
            schema_example_assignments_dict[actual_candidate_uris_for_example[1]] = [] # Example
    else: # Fallback if no actual candidate URIs could be extracted for the example
        schema_example_assignments_dict["urn:example_candidate_uri_from_input_1"] = ["ThemeNameA"]
        schema_example_assignments_dict["urn:example_candidate_uri_from_input_2"] = []
    
    # Manually format the JSON example string for clarity and to guide the LLM
    example_json_lines_list = []
    for ex_uri, ex_themes in schema_example_assignments_dict.items():
        example_json_lines_list.append(f'    "{ex_uri}": {json.dumps(ex_themes)}')
    
    # Add a comment indicating more entries if there were more candidates than shown in example
    if len(candidate_details) > num_example_uris :
        example_json_lines_list.append("    // ... and so on for ALL other candidates, using their actual URI_FOR_KEY values as keys ...")

    schema_example_json_str_formatted = "{\n" + ",\n".join(example_json_lines_list) + "\n  }"


    schema_example_final = f'''```json
{{
  "theme_assignments": {schema_example_json_str_formatted}
}}
```'''

    task_description = f"""
Task: For the input travel concept '{input_concept}', your goal is to categorize the related candidate concepts (provided below) into the predefined themes.

For EACH candidate listed under "Candidates to Slot":
1.  Note its "URI_FOR_KEY" value. This exact URI string MUST be used as the key in your output JSON dictionary under "theme_assignments".
2.  Assess its direct relevance as a **lodging feature, amenity, nearby point of interest, or a characteristic of the accommodation or its environment** that aligns with '{input_concept}'.
3.  Consider if the candidate represents a **traveler preference, interest, or common travel intent** strongly associated with '{input_concept}'.
4.  Evaluate if the candidate describes a **type of offering, experience, package, or service** that is *characterized by* or directly enabled by '{input_concept}'.
5.  Determine if the candidate reflects a **sentiment, vibe, or qualitative aspect** often sought by travelers looking for '{input_concept}'.

Based on these considerations, list ALL applicable theme names from the "Available Themes" list for the candidate.
If a candidate is highly relevant to '{input_concept}' but doesn't fit any specific theme, you may assign it to 'GeneralRelevance_CoreAspect' if available, or leave its theme list empty.

CRITICALLY IMPORTANT:
- The output JSON's "theme_assignments" dictionary MUST include EVERY candidate presented in "Candidates to Slot".
- For each candidate, use its "URI_FOR_KEY" value (e.g., "urn:...") as the literal JSON key.
- If a candidate is not relevant to any theme, its value in the dictionary should be an empty list: [].
"""
    user_prompt = f"""
Primary Input Concept: '{input_concept}'

Available Themes:
{theme_defs_str}
(Mandatory themes, if any: [{must_haves_str}])

Candidates to Slot:
{cand_details_str_for_llm} 

{task_description}

Output Format: Respond ONLY with a valid JSON object matching the schema demonstrated below (using the actual "URI_FOR_KEY" values from the candidates as keys). Do NOT include any explanatory text before or after the JSON.
Schema Example (illustrative, you MUST use actual candidate URIs from "Candidates to Slot" as keys):
{schema_example_final}
"""
    # Log the prompt being sent for one concept for debugging
    if not hasattr(construct_llm_slotting_prompt, "logged_once"):
        logger.debug(f"LLM Slotting Prompt for '{input_concept}':\nSYSTEM: {system_prompt}\nUSER: {user_prompt}")
        construct_llm_slotting_prompt.logged_once = True # type: ignore

    return system_prompt.strip(), user_prompt.strip()

def build_reprompt_prompt(input_concept: str, theme_name: str, theme_config: Dict, original_candidates_details_map: Dict[str, Dict]) -> Tuple[str, str]:
    system_prompt = """You are assisting travel definition refinement focused on lodging. A mandatory theme needs assignments. Review original candidates ONLY for relevance to the specific theme **in the context of accommodation features or traveler preferences related to lodging**. Output ONLY JSON like {"theme_assignments": {"URI_Relevant_1": ["ThemeName"], ...}}. If none relevant, output {"theme_assignments": {}}."""
    desc = theme_config.get("description", "?"); hints = theme_config.get("hints", {}); hints_str = ""
    if hints.get("keywords"): hints_str += f"\n    - Keywords: {', '.join(hints['keywords'])}"
    if hints.get("uris"): hints_str += f"\n    - URIs: {', '.join(hints['uris'])}"
    if hints_str: hints_str = "  Hints:" + hints_str
    cand_list_str = ("".join(f"\n{i+1}. URI:{uri} L:'{cand.get('label','?')}' T:{cand.get('types',[])}" for i, (uri, cand) in enumerate(original_candidates_details_map.items())) or "\nNo candidates.") # Use 'label' and 'types'
    schema_str = f'```json\n{{\n  "theme_assignments": {{\n    "URI_Relevant_1": ["{theme_name}"],\n // ONLY relevant URIs\n  }}\n}}\n```\nIf none: {{"theme_assignments": {{}} }}'
    instructions_str = f"""Instructions: Identify candidates relevant to the mandatory theme '{theme_name}', **considering how this theme relates to lodging features, amenities, or nearby points of interest important for a traveler's stay**. Assign at least one candidate if plausible within this context. Output ONLY valid JSON per schema."""
    user_prompt = f"""Re-evaluating '{input_concept}' for MANDATORY theme '{theme_name}'.\nTheme:\n- Name:{theme_name}\n- Desc:{desc}\n{hints_str}\nCandidates:{cand_list_str}\n{instructions_str}\nSchema:\n{schema_str}\nOutput:"""
    return system_prompt.strip(), user_prompt.strip()

def construct_llm_negation_prompt(input_concept: str, anchor_prefLabel: str, candidate_details: List[Dict[str, Any]]) -> Tuple[str, str]:
    system_prompt = """You are a travel domain expert focused on identifying contradictory or negating concepts within a taxonomy. Your task is to review a list of candidate concepts and identify any that are semantically opposite, mutually exclusive, or strongly contradictory to the main input concept in a travel context. Output ONLY a valid JSON object containing a single key "negating_uris" with a list of URI strings as its value. Do not include URIs that are merely *unrelated*."""
    cand_list_str_neg = "\n".join(f"- URI: {c.get('uri', '?')} Label: '{c.get('prefLabel', '?')}'" for c in candidate_details) # Assumes prefLabel from candidates_for_negation_check_list
    schema_example_neg = '```json\n{\n  "negating_uris": [\n    "URI_of_contradictory_concept_1",\n    "URI_of_opposite_concept_2"\n  ]\n}\n```\nIf none found: `{"negating_uris": []}`'
    user_prompt_neg = f"""
Input Concept: '{input_concept}' (Anchor Label: '{anchor_prefLabel}')
Task: Review the following candidate concepts. Identify any URIs that represent concepts clearly **contradictory, negating, or opposite** to '{input_concept}' / '{anchor_prefLabel}' in the context of travel. Do NOT list concepts that are simply unrelated or different. Focus on direct opposites or strong incompatibilities.
Examples of Contradictions:
- If input is 'Luxury', 'Budget' or 'Economy' would be contradictory.
- If input is 'Beachfront', 'Mountain View' or 'City Center' would be contradictory.
- If input is 'Pet-friendly', 'Pets Not Allowed' would be contradictory.
- If input is 'Quiet', 'Nightlife Nearby' would be contradictory.
Candidates to review:
{cand_list_str_neg}
Instructions: Return ONLY a JSON object containing a list of URIs for the concepts identified as contradictory or negating. Use the key "negating_uris". If no candidates are contradictory, return an empty list.
Output Format:
{schema_example_neg}
"""
    return system_prompt.strip(), user_prompt_neg.strip()

# --- Domain Inference and Biasing ---
def infer_concept_domain(candidate_uris: List[str], config: Dict, top_n_check: int = 50) -> str:
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

def apply_namespace_bias(score: float, candidate_uri: str, inferred_domain: str, config: Dict) -> Tuple[float, str]:
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
        reason_suffix = f" + LowPrio NS ({penalty_factor:.2f})" if penalty_applied else f"Penalty (LowPrio NS {penalty_factor:.2f})"
        reason = (reason if reason != "Neutral" else "") + reason_suffix
        penalty_applied = True
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
            else:
                is_lodging_ns = candidate_uri.startswith(core_ns_prefix) or (lodging_amenity_ns and candidate_uri.startswith(lodging_amenity_ns))
                if is_lodging_ns: new_score = initial_score * penalty_factor; reason = f"Penalty (Lodging NS vs Activity {penalty_factor:.2f})"; domain_bias_applied = True
    if not domain_bias_applied and not penalty_applied: reason = "Neutral"
    if new_score <= 0: logger.warning(f"Bias resulted in non-positive score ({new_score:.4f}) for {candidate_uri}. Resetting to small positive value."); new_score = 1e-9
    return new_score, reason

# --- Keyword Expansion Helper ---
def expand_keywords_with_llm(concept_label: str, config: Dict, args_namespace: argparse.Namespace) -> Tuple[List[str], bool, Optional[str]]:
    llm_api_cfg = config.get("LLM_API_CONFIG", {}); llm_stage1_cfg = config.get("STAGE1_CONFIG", {}); llm_stage2_cfg = config.get("STAGE2_CONFIG", {})
    timeout = int(llm_api_cfg.get("REQUEST_TIMEOUT", 180)); retries = int(llm_api_cfg.get("MAX_RETRIES", 5)); temp = float(llm_stage1_cfg.get("KW_EXPANSION_TEMPERATURE", llm_stage2_cfg.get("LLM_TEMPERATURE", 0.5)))
    success = False; error_message = None; final_keyword_terms: Set[str] = set(); original_normalized_words = set(w for w in normalize_concept(concept_label).split() if len(w) > 2); final_keyword_terms.update(original_normalized_words)
    try:
        sys_prompt_kw, user_prompt_kw = construct_keyword_expansion_prompt(concept_label)
        result_kw = call_llm(sys_prompt_kw, user_prompt_kw, args_namespace.llm_model, timeout, retries, temp, args_namespace.llm_provider,
                             input_concept_for_logging=f"{concept_label} [KW Expansion]")
        if result_kw and result_kw.get("success"):
            raw_phrases = result_kw.get("response", {}).get("keywords", [])
            if isinstance(raw_phrases, list):
                added_terms = set(term for kw_phrase in raw_phrases if isinstance(kw_phrase, str) and kw_phrase.strip() for term in normalize_concept(kw_phrase).split() if len(term) > 2)
                newly_expanded_terms = added_terms - original_normalized_words
                if newly_expanded_terms: final_keyword_terms.update(newly_expanded_terms); success = True; logger.debug(f"[{concept_label}] LLM KW expansion added {len(newly_expanded_terms)} terms.")
                else: error_message = "LLM returned no new terms"; logger.info(f"[{concept_label}] LLM KW expansion: No new terms.")
            else: error_message = "LLM response invalid format (keywords not a list)"; logger.error(f"[{concept_label}] LLM KW expansion invalid response format.")
        else: err_msg = result_kw.get("error", "?") if result_kw else "No result object"; error_message = f"LLM API Call Failed:{err_msg}"; logger.warning(f"[{concept_label}] LLM KW expansion failed: {err_msg}")
    except Exception as e_kw: error_message = f"Exception during keyword expansion: {e_kw}"; logger.error(f"[{concept_label}] KW expansion exception: {e_kw}", exc_info=True)
    return list(final_keyword_terms), success, error_message

# --- Stage 1: Evidence Preparation ---
def prepare_evidence(
    input_concept: str,
    concept_embedding: Optional[np.ndarray], # type: ignore
    primary_embeddings: Dict[str, np.ndarray], # type: ignore
    config: Dict,
    args_namespace: argparse.Namespace,
    bm25_model_in: Optional[bm25s.BM25], # type: ignore
    keyword_corpus_uris_in: Optional[List[str]],
    keyword_label_index_in: Optional[Dict[str, Set[str]]],
    taxonomy_concepts_cache_in: Dict[str, Dict],
) -> Tuple[
    List[Dict],
    Dict[str, Dict],
    Optional[Dict],
    Dict[str, float],
    Dict[str, Any],
    int,
    int,
    int,
]:
    start_time_stage1 = time.time()
    normalized_input_concept = normalize_concept(input_concept)
    selection_method_log = "Combined BM25s+SBERT"

    def get_sort_priority(item_uri: str) -> int:
        preferred_ns = config.get("preferred_namespaces", [])
        for i, ns_prefix in enumerate(preferred_ns):
            if item_uri.startswith(ns_prefix):
                return i
        return len(preferred_ns) + 10

    stage1_cfg = config.get("STAGE1_CONFIG", {})
    kw_scoring_cfg = config.get("KEYWORD_SCORING_CONFIG", {})
    damp_thresh = float(config.get("keyword_dampening_threshold", 0.35))
    damp_factor = float(config.get("keyword_dampening_factor", 0.15))
    max_cands_llm = int(stage1_cfg.get("MAX_CANDIDATES_FOR_LLM", 75))
    min_sim_initial_sbert = float(stage1_cfg.get("EVIDENCE_MIN_SIMILARITY", 0.30))
    min_kw_score_val = float(kw_scoring_cfg.get("bm25_min_score", 0.01))
    kw_trigger_expansion = int(stage1_cfg.get("MIN_KEYWORD_CANDIDATES_FOR_EXPANSION", 5))
    kw_top_n_bm25 = int(kw_scoring_cfg.get("bm25_top_n", 500))
    abs_min_sbert_filter = float(config.get("min_sbert_score", 0.12))
    global_alpha_val = float(config.get("global_alpha", 0.6))
    global_kw_exp_enabled = stage1_cfg.get("ENABLE_KW_EXPANSION", True)
    ns_bias_enabled_flag = config.get("NAMESPACE_BIASING", {}).get("enabled", True)
    prioritize_exact_match_flag = config.get("prioritize_exact_prefLabel_match", True)

    overrides = config.get("concept_overrides", {}).get(normalized_input_concept, {})
    skip_expansion_flag = overrides.get("skip_expansion", False)
    kw_exp_enabled_final = (global_kw_exp_enabled and args_namespace.llm_provider != "none" and not skip_expansion_flag)
    seed_uris_list = overrides.get("seed_uris", [])
    boost_cfg_dict = overrides.get("boost_seeds_config", {"enabled": False})
    filter_uris_list = overrides.get("filter_uris", [])
    effective_alpha_val = float(overrides.get("effective_alpha", global_alpha_val))
    manual_query_list = overrides.get("manual_query_split", None)

    if overrides: logger.info(f"[{normalized_input_concept}] Applying concept overrides.")
    logger.info(f"[{normalized_input_concept}] Effective Alpha:{effective_alpha_val:.2f} ({'Override' if abs(effective_alpha_val-global_alpha_val)>1e-9 else 'Default'})")
    logger.info(f"[{normalized_input_concept}] Abs Min SBERT:{abs_min_sbert_filter}")
    logger.info(f"[{normalized_input_concept}] KW Dampen: Thresh={damp_thresh}, Factor={damp_factor}")
    if boost_cfg_dict.get("enabled"): logger.info(f"[{normalized_input_concept}] Seed Boost: ON (Thresh:{boost_cfg_dict.get('threshold','?')}) URIs:{seed_uris_list}")
    logger.info(f"[{normalized_input_concept}] LLM KW Expansion:{kw_exp_enabled_final}")
    logger.info(f"[{normalized_input_concept}] Namespace Biasing Enabled: {ns_bias_enabled_flag}")
    logger.info(f"[{normalized_input_concept}] Prioritize Exact prefLabel Match (w/ Plurals): {prioritize_exact_match_flag}")
    logger.info(f"[{normalized_input_concept}] Tie-Break Sort (Combined): Score>Namespace>SBERT")
    if prioritize_exact_match_flag: logger.info(f"[{normalized_input_concept}] Tie-Break Sort (Exact Match): Namespace>SBERT")

    sim_scores_map: Dict[str, float] = {}; kw_scores_map: Dict[str, float] = {}
    exp_diag_dict: Dict[str, Any] = {"attempted": False, "successful": False, "count": 0, "terms": [], "keyword_count": 0, "error": None,}

    base_kws_set = set(kw for kw in normalized_input_concept.split() if len(kw) > 2)
    final_query_texts_list = list(base_kws_set); initial_kw_count_val = 0
    if keyword_label_index_in: initial_kw_count_val = len(set().union(*[keyword_label_index_in.get(kw, set()) for kw in base_kws_set]))
    needs_exp_flag = kw_exp_enabled_final and (initial_kw_count_val < kw_trigger_expansion or normalized_input_concept in ABSTRACT_CONCEPTS_LIST)
    if kw_exp_enabled_final and needs_exp_flag:
        exp_diag_dict["attempted"] = True;
        logger.info(f"[{normalized_input_concept}] Attempting LLM KW expansion (Initial:{initial_kw_count_val}<{kw_trigger_expansion} or abstract).")
        final_query_texts_list, exp_success_flag, exp_error_str = expand_keywords_with_llm(input_concept, config, args_namespace); exp_diag_dict["successful"] = exp_success_flag; exp_diag_dict["error"] = exp_error_str
    if manual_query_list:
        if isinstance(manual_query_list, list) and all(isinstance(t, str) for t in manual_query_list):
            final_query_texts_set_check = set(normalize_concept(t) for t in final_query_texts_list); manual_query_set_check = set(normalize_concept(t) for t in manual_query_list)
            if final_query_texts_set_check != manual_query_set_check: logger.info(f"Applying manual query split, OVERRIDING: {manual_query_list}"); final_query_texts_list = [normalize_concept(t) for t in manual_query_list]; exp_diag_dict["notes"] = "Manual query split."
        else: logger.warning(f"Invalid 'manual_query_split' for {normalized_input_concept}. Ignored.")
    exp_diag_dict["terms"] = final_query_texts_list; exp_diag_dict["count"] = len(final_query_texts_list)

    kw_cand_count_init_val = 0; kw_cands_raw_list: List[Dict[str, Any]] = []
    if bm25_model_in and keyword_corpus_uris_in:
        if final_query_texts_list:
            kw_cands_raw_list = get_candidate_concepts_keyword(final_query_texts_list, bm25_model_in, keyword_corpus_uris_in, kw_top_n_bm25, min_kw_score_val); kw_scores_map = {c["uri"]: c["score"] for c in kw_cands_raw_list}; kw_cand_count_init_val = len(kw_scores_map); exp_diag_dict["keyword_count"] = kw_cand_count_init_val
        else:
            logger.warning(f"[{normalized_input_concept}] No keywords for BM25s search."); kw_scores_map = {}; exp_diag_dict["keyword_count"] = 0
    else:
        logger.warning(f"[{normalized_input_concept}] BM25s unavailable."); kw_scores_map = {}; exp_diag_dict["keyword_count"] = 0

    sbert_cand_count_init_val = 0; sbert_cands_raw_map_val: Dict[str, float] = {}
    if concept_embedding is None: logger.error(f"[{normalized_input_concept}] No anchor embed. Skipping SBERT."); sim_scores_map = {}
    else:
        sbert_cands_raw_map_val = get_batch_embedding_similarity(concept_embedding, primary_embeddings)
        sim_scores_map = {u: s for u, s in sbert_cands_raw_map_val.items() if s >= min_sim_initial_sbert}
        sbert_cand_count_init_val = len(sim_scores_map)
        logger.debug(f"[{normalized_input_concept}] Found {sbert_cand_count_init_val} SBERT candidates >= {min_sim_initial_sbert:.2f} (out of {len(sbert_cands_raw_map_val)} raw).")

    all_uris_for_exact_match_set = set(kw_scores_map.keys()) | set(sbert_cands_raw_map_val.keys()); all_uris_set = set(kw_scores_map.keys()) | set(sim_scores_map.keys()); initial_unique_count = len(all_uris_set)
    if seed_uris_list:
        added_seeds = set(seed_uris_list) - all_uris_for_exact_match_set;
        if added_seeds: logger.info(f"SEEDING '{normalized_input_concept}' with {len(added_seeds)} URIs: {added_seeds}")
        all_uris_set.update(added_seeds); all_uris_for_exact_match_set.update(added_seeds); needed_sbert_seeds = added_seeds - set(sbert_cands_raw_map_val.keys())
        if needed_sbert_seeds and concept_embedding is not None:
            seed_embs_map = {u: primary_embeddings[u] for u in needed_sbert_seeds if u in primary_embeddings}
            if seed_embs_map:
                new_sims_map = get_batch_embedding_similarity(concept_embedding, seed_embs_map); sbert_cands_raw_map_val.update(new_sims_map);
                sim_scores_map.update({u: s for u, s in new_sims_map.items() if s >= min_sim_initial_sbert})
                logger.debug(f"Calculated SBERT for {len(new_sims_map)} seeds.")
            else: logger.warning(f"No embeddings for seeds: {needed_sbert_seeds}")
    if filter_uris_list:
        to_remove_set = set(u for u in filter_uris_list if u in all_uris_for_exact_match_set);
        if to_remove_set: logger.info(f"Applying filter: Removing {len(to_remove_set)} URIs: {to_remove_set}")
        all_uris_set.difference_update(to_remove_set); all_uris_for_exact_match_set.difference_update(to_remove_set)

    exact_match_uris_list: List[str] = []
    if prioritize_exact_match_flag:
        logger.debug(f"[{normalized_input_concept}] Checking for exact prefLabel matches (with simple plural check)..."); match_count_val = 0; norm_input_str = normalized_input_concept
        for uri_val_exact in all_uris_for_exact_match_set:
            if uri_val_exact not in all_uris_set: continue
            concept_data_dict = taxonomy_concepts_cache_in.get(uri_val_exact);
            if not concept_data_dict: continue
            pref_label_str = get_primary_label(uri_val_exact, taxonomy_concepts_cache_in);
            if pref_label_str:
                norm_label_str = normalize_concept(pref_label_str); is_match_flag = False; match_type_str = ""
                if norm_input_str == norm_label_str: is_match_flag = True; match_type_str = "Exact"
                elif (norm_input_str.endswith("s") and len(norm_input_str) > 1 and norm_input_str[:-1] == norm_label_str): is_match_flag = True; match_type_str = "Plural input"
                elif (norm_label_str.endswith("s") and len(norm_label_str) > 1 and norm_input_str == norm_label_str[:-1]): is_match_flag = True; match_type_str = "Plural label"
                if is_match_flag:
                    sbert_score_for_exact_val = sbert_cands_raw_map_val.get(uri_val_exact, 0.0)
                    if sbert_score_for_exact_val >= abs_min_sbert_filter:
                        exact_match_uris_list.append(uri_val_exact); match_count_val += 1;
                        if args_namespace.debug: logger.debug(f"  {match_type_str} match found AND passed SBERT threshold: {uri_val_exact} (Input: '{norm_input_str}', Label: '{norm_label_str}', Score: {sbert_score_for_exact_val:.4f})")
                    elif args_namespace.debug: logger.debug(f"  {match_type_str} match found BUT FAILED SBERT threshold: {uri_val_exact} (Input: '{norm_input_str}', Label: '{norm_label_str}', Score: {sbert_score_for_exact_val:.4f} < {abs_min_sbert_filter})")
        logger.info(f"[{normalized_input_concept}] Found {match_count_val} exact prefLabel matches (incl. simple plurals) passing SBERT threshold {abs_min_sbert_filter}.")

    all_uris_list_final = list(all_uris_set); unique_cands_before_rank_val = len(all_uris_list_final)
    if not all_uris_list_final: logger.warning(f"[{normalized_input_concept}] No candidates after filter/seed."); return [], {}, None, {}, exp_diag_dict, sbert_cand_count_init_val, kw_cand_count_init_val, 0

    inferred_domain_str = "Unknown"
    if ns_bias_enabled_flag: initial_candidate_uris_for_domain_list = list(set([c["uri"] for c in kw_cands_raw_list] + list(sbert_cands_raw_map_val.keys()))); inferred_domain_str = infer_concept_domain(initial_candidate_uris_for_domain_list, config); logger.info(f"[{normalized_input_concept}] Inferred domain (based on raw cands): {inferred_domain_str}")

    all_details_map = get_kg_data(all_uris_list_final, taxonomy_concepts_cache_in)
    scored_list_final: List[Dict[str, Any]] = []
    abs_filt_count = 0; damp_cnt_val = 0; bias_cnt_val = 0
    alpha_overridden_flag = abs(effective_alpha_val - global_alpha_val) > 1e-9
    boost_thresh_val = float(boost_cfg_dict.get("threshold", 0.80)) if boost_cfg_dict.get("enabled", False) else 1.1

    for uri_item in all_uris_list_final:
        if uri_item not in all_details_map: continue
        s_orig_val = sbert_cands_raw_map_val.get(uri_item, 0.0); k_raw_val = kw_scores_map.get(uri_item, 0.0); s_boosted_val = s_orig_val; boosted_flag = False
        if uri_item in seed_uris_list and boost_cfg_dict.get("enabled", False):
            boost_thresh_val = float(boost_cfg_dict.get("threshold", 0.80))
            if s_orig_val < boost_thresh_val: s_boosted_val = boost_thresh_val; boosted_flag = True

        current_sbert_for_filter_val = s_boosted_val if boosted_flag else s_orig_val
        if current_sbert_for_filter_val < abs_min_sbert_filter:
            abs_filt_count += 1
            continue

        k_damp_val = k_raw_val; dampened_flag = False
        if current_sbert_for_filter_val < damp_thresh:
            k_damp_val *= damp_factor;
            if k_raw_val > 0 and abs(k_raw_val - k_damp_val) > 1e-9: dampened_flag = True; damp_cnt_val += 1

        norm_s_val = max(0.0, min(1.0, current_sbert_for_filter_val)); norm_k_val = max(0.0, min(1.0, k_damp_val)); combined_unbiased_val = (effective_alpha_val * norm_k_val) + ((1.0 - effective_alpha_val) * norm_s_val)

        combined_val = combined_unbiased_val; bias_reason_str = "Neutral"; biased_flag = False
        if ns_bias_enabled_flag and not boosted_flag:
             combined_val, bias_reason_str = apply_namespace_bias(combined_unbiased_val, uri_item, inferred_domain_str, config);
             if abs(combined_val - combined_unbiased_val) > 1e-9: biased_flag = True; bias_cnt_val += 1

        scored_list_final.append({
            "uri": uri_item, "details": all_details_map[uri_item], "sim_score": s_orig_val, "boosted_sim_score": s_boosted_val if boosted_flag else None,
            "kw_score": k_raw_val, "dampened_kw_score": k_damp_val if dampened_flag else None, "alpha_overridden": alpha_overridden_flag,
            "effective_alpha": effective_alpha_val, "combined_score_unbiased": combined_unbiased_val, "combined_score": combined_val,
            "biased": biased_flag, "bias_reason": bias_reason_str,
        })

    logger.info(f"[{normalized_input_concept}] Excluded {abs_filt_count} (SBERT<{abs_min_sbert_filter}). Dampened KW for {damp_cnt_val} (SBERT<{damp_thresh}). Biased {bias_cnt_val} scores (Domain:{inferred_domain_str}).")
    if alpha_overridden_flag: logger.info(f"[{normalized_input_concept}] Applied alpha override: {effective_alpha_val:.2f}.")
    if not scored_list_final: logger.warning(f"[{normalized_input_concept}] No candidates remain after scoring."); return ([], {}, None, {}, exp_diag_dict, sbert_cand_count_init_val, kw_cand_count_init_val, unique_cands_before_rank_val,)
    logger.debug(f"[{normalized_input_concept}] Sorting {len(scored_list_final)} candidates by combined score..."); scored_list_final.sort(key=lambda x: x.get("sim_score", 0.0), reverse=True); scored_list_final.sort(key=lambda x: get_sort_priority(x.get("uri", "")), reverse=False); scored_list_final.sort(key=lambda x: x["combined_score"], reverse=True)
    if args_namespace.debug:
        logger.debug(f"--- Top 5 for '{normalized_input_concept}' (Sorted by Combined Score) ---"); logger.debug(f"    (EffAlpha:{effective_alpha_val:.2f}, DampThresh:{damp_thresh}, DampFactor:{damp_factor}), Domain: {inferred_domain_str}, Bias:{ns_bias_enabled_flag}")
        for i, c_item in enumerate(scored_list_final[:5]):
            damp_str_debug = (f"(Damp:{c_item.get('dampened_kw_score'):.4f})" if c_item.get("dampened_kw_score") is not None else ""); boost_str_debug = (f"(Boost:{c_item.get('boosted_sim_score'):.4f})" if c_item.get("boosted_sim_score") is not None else ""); alpha_str_debug = f"(EffA:{c_item.get('effective_alpha'):.2f})"; prio_str_debug = f"(P:{get_sort_priority(c_item.get('uri',''))})"; lbl_str_debug = get_primary_label(c_item.get("uri", "?"), taxonomy_concepts_cache_in, c_item.get("uri", "?")); bias_info_str_debug = (f" -> Biased:{c_item.get('combined_score'):.6f} ({c_item.get('bias_reason','?')})" if c_item.get("biased") else "")
            logger.debug(f"{i+1}. URI:{c_item.get('uri','?')} {prio_str_debug} L:'{lbl_str_debug}' FinalScore:{c_item.get('combined_score'):.6f} (Unbiased:{c_item.get('combined_score_unbiased'):.6f}{alpha_str_debug}) (SBERT:{c_item.get('sim_score'):.4f}{boost_str_debug}, KW:{c_item.get('kw_score'):.4f}{damp_str_debug}) {bias_info_str_debug}")
        logger.debug("--- End Top 5 (Combined Score) ---")

    selected_for_llm_list = scored_list_final[:max_cands_llm];
    logger.info(f"[{normalized_input_concept}] Selected top {len(selected_for_llm_list)} candidates for LLM based on combined score.")
    if not selected_for_llm_list: logger.warning(f"[{normalized_input_concept}] No candidates selected for LLM."); return ([], {}, None, {}, exp_diag_dict, sbert_cand_count_init_val, kw_cand_count_init_val, unique_cands_before_rank_val,)

    llm_details_list: List[Dict[str,Any]] = []
    for c_item_llm in selected_for_llm_list:
        # Construct detail dict for LLM prompt, ensure 'label', 'types', 'definitions' are present
        detail_for_llm = {
            "uri": c_item_llm.get("uri"),
            "label": c_item_llm.get("details", {}).get("label", get_primary_label(str(c_item_llm.get("uri")), taxonomy_concepts_cache_in)),
            "types": c_item_llm.get("details", {}).get("types", []),
            "definitions": c_item_llm.get("details", {}).get("definitions", [])
        }
        llm_details_list.append(detail_for_llm)

    orig_map_val: Dict[str, Dict[str,Any]] = {}
    for c_item in selected_for_llm_list:
        uri_c = c_item.get("uri"); details_c = c_item.get("details") # details_c is from get_kg_data
        if uri_c and details_c:
            entry_c = {
                "uri": uri_c, # Ensure URI is top-level
                "label": details_c.get("label", get_primary_label(uri_c, taxonomy_concepts_cache_in)), # from get_kg_data
                "types": details_c.get("types", []), # from get_kg_data
                "definitions": details_c.get("definitions", []), # from get_kg_data
                "sbert_score": c_item.get("sim_score"),
                "keyword_score": c_item.get("kw_score"),
                "combined_score": c_item.get("combined_score"),
                "combined_score_unbiased": c_item.get("combined_score_unbiased"),
                "biased": c_item.get("biased"),
                "effective_alpha": c_item.get("effective_alpha"),
            }
            # Add prefLabel from taxonomy_concepts_cache if available and not already there
            # This is important if get_kg_data's 'label' isn't always prefLabel
            raw_concept_data = taxonomy_concepts_cache_in.get(uri_c, {})
            pref_label_from_raw = get_primary_label(uri_c, taxonomy_concepts_cache_in) # Uses SKOS:prefLabel first
            if pref_label_from_raw:
                entry_c["prefLabel"] = pref_label_from_raw # For Core Def URI check
                if not entry_c["label"] or entry_c["label"] == uri_c: # If get_kg_data only had URI as label
                    entry_c["label"] = pref_label_from_raw

            if c_item.get("boosted_sim_score"): entry_c["boosted_sbert_score"] = c_item.get("boosted_sim_score")
            if c_item.get("dampened_kw_score"): entry_c["dampened_keyword_score"] = c_item.get("dampened_kw_score")
            if c_item.get("bias_reason") != "Neutral": entry_c["bias_reason"] = c_item.get("bias_reason")
            orig_map_val[uri_c] = entry_c

    anchor_uri_val: Optional[str] = None; anchor_data_dict_val: Optional[Dict[str,Any]] = None
    score_rank_map_val: Dict[str, int] = {item['uri']: rank for rank, item in enumerate(scored_list_final)} # Define here for broader scope

    if prioritize_exact_match_flag and exact_match_uris_list:
        selection_method_log = "Exact prefLabel Match (Plural Check)"
        valid_exact_uris_list = [uri_val_exact_check for uri_val_exact_check in exact_match_uris_list if uri_val_exact_check in score_rank_map_val]
        valid_exact_uris_list.sort(key=lambda uri_val_exact_sort: score_rank_map_val[uri_val_exact_sort])

        if valid_exact_uris_list:
            selected_exact_match_uri_val = valid_exact_uris_list[0]
            anchor_data_dict_val = next((item_anchor for item_anchor in scored_list_final if item_anchor['uri'] == selected_exact_match_uri_val), None)
            if anchor_data_dict_val:
                anchor_uri_val = selected_exact_match_uri_val
                exact_match_details_dict = scored_list_final[score_rank_map_val[anchor_uri_val]]
                logger.info(f"[{normalized_input_concept}] Selected exact match anchor URI: {anchor_uri_val} (Rank: {score_rank_map_val[anchor_uri_val]+1}, SBERT: {exact_match_details_dict.get('sim_score'):.4f})")
            else:
                logger.error(f"[{normalized_input_concept}] CRITICAL: Could not find data in scored_list for exact match anchor {selected_exact_match_uri_val}! Falling back.")
                anchor_uri_val = None; selection_method_log = "Combined BM25s+SBERT (Exact Match Inconsistency Fallback)"
        else:
             logger.warning(f"[{normalized_input_concept}] No exact matches remain after filtering/ranking. Falling back.")
             selection_method_log = "Combined BM25s+SBERT (Exact Match Fallback)"

    if anchor_uri_val is None:
        if not scored_list_final: logger.error(f"[{normalized_input_concept}] No candidates in scored_list for fallback anchor selection."); anchor_data_dict_val = None
        else:
            anchor_data_dict_val = scored_list_final[0]; anchor_uri_val = str(anchor_data_dict_val.get("uri")); selection_method_log = "Combined BM25s+SBERT" # Ensure URI is str
            if prioritize_exact_match_flag and exact_match_uris_list: logger.info(f"[{normalized_input_concept}] Using fallback anchor (combined score): {anchor_uri_val}")

    anchor_final_dict: Optional[Dict[str,Any]] = None
    if anchor_data_dict_val and anchor_uri_val:
        # Ensure details for anchor_final_dict come from orig_map_val to have consistent structure with 'label', 'types', 'definitions'
        anchor_details_from_orig = orig_map_val.get(anchor_uri_val, anchor_data_dict_val.get("details", {}))

        anchor_final_dict = {
            "uri": anchor_uri_val,
            "label": anchor_details_from_orig.get("label", get_primary_label(anchor_uri_val, taxonomy_concepts_cache_in, anchor_uri_val)),
            "types": anchor_details_from_orig.get("types", []),
            "definitions": anchor_details_from_orig.get("definitions", []),
            "sbert_score": anchor_data_dict_val.get("sim_score"),
            "keyword_score": anchor_data_dict_val.get("kw_score"),
            "combined_score": anchor_data_dict_val.get("combined_score"),
            "combined_score_unbiased": anchor_data_dict_val.get("combined_score_unbiased"),
            "biased": anchor_data_dict_val.get("biased"),
            "effective_alpha": anchor_data_dict_val.get("effective_alpha"),
        }
        if anchor_data_dict_val.get("boosted_sim_score") is not None: anchor_final_dict["boosted_sbert_score"] = anchor_data_dict_val.get("boosted_sim_score")
        if anchor_data_dict_val.get("dampened_kw_score") is not None: anchor_final_dict["dampened_keyword_score"] = anchor_data_dict_val.get("dampened_kw_score")
        if anchor_data_dict_val.get("bias_reason") != "Neutral": anchor_final_dict["bias_reason"] = anchor_data_dict_val.get("bias_reason")

        anchor_label_str = anchor_final_dict.get("label", anchor_uri_val) # Use the label from the dict
        score_display_str = (f"Exact Match (Rank: {score_rank_map_val.get(anchor_uri_val, -1)+1}, SBERT: {anchor_final_dict.get('sbert_score', 0.0):.4f})" if selection_method_log.startswith("Exact prefLabel Match") else f"Score: {anchor_final_dict.get('combined_score'):.6f}{' - Biased' if anchor_final_dict.get('biased') else ''}")
        logger.info(f"[{normalized_input_concept}] Anchor ({selection_method_log}): {anchor_label_str} ({score_display_str})")
    elif anchor_uri_val: logger.error(f"[{normalized_input_concept}] Anchor URI '{anchor_uri_val}' selected but data missing!"); anchor_final_dict = {"uri": anchor_uri_val, "label": input_concept, "types": ["Unknown"], "definitions": [], "error": "Data missing",}
    else: logger.error(f"[{normalized_input_concept}] Failed to select any anchor."); anchor_final_dict = {"uri": None, "label": input_concept, "types": ["Unknown"], "definitions": [], "error": "No anchor selected",}

    exp_diag_dict["selection_method"] = selection_method_log; stage1_duration_val = time.time() - start_time_stage1; exp_diag_dict["duration_seconds"] = round(stage1_duration_val, 2)
    return (llm_details_list, orig_map_val, anchor_final_dict, {}, exp_diag_dict, sbert_cand_count_init_val, kw_cand_count_init_val, unique_cands_before_rank_val,)

# --- Determine Lodging Type ---
def determine_lodging_type(
    travel_category: Optional[Dict],
    top_defining_attributes: List[Dict],
    config: Dict,
    taxonomy_concepts_cache_in: Dict[str, Dict],
) -> str:
    hints = config.get("lodging_type_hints", {})
    cl_hints = set(hints.get("CL", []))
    vr_hints = set(hints.get("VR", []))
    both_hints = set(hints.get("Both", []))

    stage2_cfg_lodging = config.get("STAGE2_CONFIG", {})
    top_n_check_lodging = int(stage2_cfg_lodging.get("LODGING_TYPE_TOP_ATTR_CHECK", 10))
    threshold_ratio_lodging = float(stage2_cfg_lodging.get("LODGING_TYPE_CONFIDENCE_THRESHOLD", 0.6))

    cl_score_lodging = 0; vr_score_lodging = 0; checked_count_lodging = 0
    decision_reason_list: List[str] = []

    if travel_category and travel_category.get("uri"):
        anchor_uri_lodging = travel_category["uri"]
        checked_count_lodging += 1
        anchor_label_lodging = get_primary_label(anchor_uri_lodging, taxonomy_concepts_cache_in, anchor_uri_lodging)
        if anchor_uri_lodging in cl_hints:
            cl_score_lodging += 2
            decision_reason_list.append(f"Anchor URI ({anchor_label_lodging}) in CL hints.")
        elif anchor_uri_lodging in vr_hints:
            vr_score_lodging += 2
            decision_reason_list.append(f"Anchor URI ({anchor_label_lodging}) in VR hints.")
        elif anchor_uri_lodging in both_hints:
             decision_reason_list.append(f"Anchor URI ({anchor_label_lodging}) in Both hints.")
        else:
             decision_reason_list.append(f"Anchor URI ({anchor_label_lodging}) not in hints.")

    attributes_to_check_lodging = top_defining_attributes[:top_n_check_lodging]
    for attr_lodging in attributes_to_check_lodging:
        uri_lodging = attr_lodging.get("uri")
        if not uri_lodging: continue
        checked_count_lodging += 1
        attr_label_lodging = get_primary_label(uri_lodging, taxonomy_concepts_cache_in, uri_lodging)
        if uri_lodging in cl_hints:
            cl_score_lodging += 1
            decision_reason_list.append(f"Attr URI ({attr_label_lodging}) in CL hints.")
        elif uri_lodging in vr_hints:
            vr_score_lodging += 1
            decision_reason_list.append(f"Attr URI ({attr_label_lodging}) in VR hints.")
        elif uri_lodging in both_hints:
             decision_reason_list.append(f"Attr URI ({attr_label_lodging}) in Both hints.")

    final_type_lodging = "Both"
    if checked_count_lodging > 0 : # Avoid division by zero
        cl_ratio_lodging = cl_score_lodging / checked_count_lodging if checked_count_lodging > 0 else 0
        vr_ratio_lodging = vr_score_lodging / checked_count_lodging if checked_count_lodging > 0 else 0

        # More nuanced decision:
        if cl_score_lodging > vr_score_lodging and cl_ratio_lodging >= threshold_ratio_lodging :
            final_type_lodging = "CL"
        elif vr_score_lodging > cl_score_lodging and vr_ratio_lodging >= threshold_ratio_lodging:
            final_type_lodging = "VR"
        # If scores are equal but meet threshold, or neither meets threshold, default to "Both"
        elif (cl_score_lodging == vr_score_lodging and cl_score_lodging > 0 and cl_ratio_lodging >= threshold_ratio_lodging): # Tie, but strong signal for both
            final_type_lodging = "Both" # Or could be based on a slight preference if needed
        # Default to "Both" if no strong signal or conflicting signals
    else: # No anchor or attributes checked with hints
        decision_reason_list.append("No anchor or attributes with hints checked to make a CL/VR decision.")


    logger.debug(f"Lodging Type Check: CL Score={cl_score_lodging}, VR Score={vr_score_lodging}, Checked={checked_count_lodging}. Threshold Ratio={threshold_ratio_lodging}. Result={final_type_lodging}.")
    # To store reasons in diagnostics (optional, for future debugging)
    # output["diagnostics"]["lodging_type_determination"]["reason"] = "; ".join(decision_reason_list)
    return final_type_lodging

# --- Stage 2: Finalization ---
def apply_rules_and_finalize(
    input_concept: str,
    llm_call_result: Optional[Dict[str, Any]],
    config: Dict,
    travel_category: Optional[Dict],
    original_candidates_map_for_reprompt: Dict[str, Dict],
    args_namespace: argparse.Namespace, # Renamed from args to avoid conflict
    taxonomy_concepts_cache_in: Dict[str, Dict],
) -> Dict[str, Any]:
    start_stage2 = time.time(); norm_concept_str = normalize_concept(input_concept); concept_overrides = config.get("concept_overrides", {}).get(norm_concept_str, {}); base_themes_config = config.get("base_themes", {}); final_cfg = config.get("STAGE2_CONFIG", {})
    min_weight_attr = float(final_cfg.get("THEME_ATTRIBUTE_MIN_WEIGHT", 0.001)); top_n_attrs = int(final_cfg.get("TOP_N_DEFINING_ATTRIBUTES", 25)); llm_refinement_enabled = final_cfg.get("ENABLE_LLM_REFINEMENT", True)
    vrbo_defaults = config.get("vrbo_default_subscore_weights", {}); llm_negation_cfg = config.get("LLM_NEGATION_CONFIG", {}); llm_negation_enabled = llm_negation_cfg.get("enabled", False)
    property_type_str = concept_overrides.get("property_type", "Unknown"); logger.info(f"[{norm_concept_str}] Determined property_type (for VRBO rules): {property_type_str}")

    stage1_cfg_core = config.get("STAGE1_CONFIG", {}) # Get STAGE1_CONFIG for core def params
    core_def_text_sim_thresh = float(stage1_cfg_core.get("CORE_DEF_TEXT_SIMILARITY_THRESHOLD", 0.90))
    core_def_max_variants = int(stage1_cfg_core.get("CORE_DEF_MAX_VARIANTS", 3))
    core_def_forced_weight = float(final_cfg.get("CORE_DEF_FORCED_ATTRIBUTE_WEIGHT", 0.05))

    output: Dict[str, Any] = {
        "applicable_lodging_types": "Both",
        "travel_category": travel_category or {"uri": None, "label": input_concept, "types": [], "definitions": []},
        "top_defining_attributes": [], "themes": [], "key_associated_concepts_unthemed": [],
        "additional_relevant_subscores": [], "must_not_have": [], "requires_geo_check": False,
        "failed_fallback_themes": {}, "affinity_score_total_allocated": 0.0,
        "diagnostics": {
            "lodging_type_determination": {"result": "Not run yet", "reason": "Not run yet"},
            "llm_negation": {"attempted": False, "success": False, "uris_found": 0, "error": None, "candidates_checked": [], "identified_negating_uris": [],},
            "theme_processing": {},
            "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0},
            "core_definitional_uris": {"identified": [], "promoted_to_top_attrs": []}, # NEW
            "final_output": {},
            "stage2": {"status": "Started", "duration_seconds": 0.0, "error": None},
        },
    }
    theme_diag = output["diagnostics"]["theme_processing"]; reprompt_diag = output["diagnostics"]["reprompting_fallback"]; neg_diag = output["diagnostics"]["llm_negation"]; core_def_diag = output["diagnostics"]["core_definitional_uris"]
    fallback_adds_list: List[Dict[str,str]] = []

    llm_assigns_map: Dict[str, List[str]] = {}; diag_val_map: Dict[str,Any] = {}
    if llm_call_result and llm_call_result.get("success"):
        validated_assigns = validate_llm_assignments(llm_call_result.get("response"), set(original_candidates_map_for_reprompt.keys()), set(base_themes_config.keys()), norm_concept_str, diag_val_map,)
        if validated_assigns is not None: llm_assigns_map = validated_assigns
        else: logger.warning(f"[{norm_concept_str}] LLM validation failed."); output["diagnostics"]["stage2"]["error"] = diag_val_map.get("error", "LLM Validation Failed")
    elif llm_call_result: logger.warning(f"[{norm_concept_str}] LLM slotting unsuccessful: {llm_call_result.get('error')}"); output["diagnostics"]["stage2"]["error"] = f"LLM Call Failed: {llm_call_result.get('error')}"

    theme_map_final = defaultdict(list)
    for uri_theme, themes_list in llm_assigns_map.items():
        if uri_theme in original_candidates_map_for_reprompt: # Ensure candidate was part of original list
            for t_name in themes_list:
                if t_name in base_themes_config: theme_map_final[t_name].append(uri_theme)

    failed_rules_map: Dict[str, Dict[str,str]] = {};
    for theme_name_check in base_themes_config.keys():
        diag_entry = theme_diag[theme_name_check] = {"llm_assigned_count": len(theme_map_final.get(theme_name_check, [])),"attributes_after_weighting": 0, "status": "Pending", "rule_failed": False,}
        rule_val, _, _, _ = get_dynamic_theme_config(norm_concept_str, theme_name_check, config)
        if rule_val == "Must have 1" and not theme_map_final.get(theme_name_check): logger.warning(f"[{norm_concept_str}] Rule FAILED: Mandatory theme '{theme_name_check}' has no assigns initially."); failed_rules_map[theme_name_check] = {"reason": "No assigns."}; diag_entry.update({"status": "Failed Rule", "rule_failed": True})

    fixed_themes_set = set()
    if (failed_rules_map and original_candidates_map_for_reprompt and llm_refinement_enabled and args_namespace.llm_provider != "none"):
        logger.info(f"[{norm_concept_str}] Attempting LLM fallback for {len(failed_rules_map)} themes: {list(failed_rules_map.keys())}"); llm_api_cfg_fb = config.get("LLM_API_CONFIG", {}); llm_stage2_cfg_fb = config.get("STAGE2_CONFIG", {})
        fb_timeout_val = int(llm_api_cfg_fb.get("REQUEST_TIMEOUT", 180)); fb_retries_val = int(llm_api_cfg_fb.get("MAX_RETRIES", 5)); fb_temp_val = float(llm_stage2_cfg_fb.get("LLM_TEMPERATURE", 0.2))
        for theme_name_fb in list(failed_rules_map.keys()):
            reprompt_diag["attempts"] += 1; base_cfg_fb = base_themes_config.get(theme_name_fb)
            if not base_cfg_fb: logger.error(f"[{norm_concept_str}] No config for fallback theme '{theme_name_fb}'."); reprompt_diag["failures"] += 1; continue
            sys_p_fb, user_p_fb = build_reprompt_prompt(input_concept, theme_name_fb, base_cfg_fb, original_candidates_map_for_reprompt);
            fb_result_val = call_llm(sys_p_fb, user_p_fb, args_namespace.llm_model, fb_timeout_val, fb_retries_val, fb_temp_val, args_namespace.llm_provider,
                                     input_concept_for_logging=f"{input_concept} [Reprompt Fallback: {theme_name_fb}]")
            if fb_result_val and fb_result_val.get("success"):
                fb_assigns_val = fb_result_val.get("response", {}).get("theme_assignments", {}); new_uris_fb = (set(uri_fb for uri_fb, ts_fb in fb_assigns_val.items() if isinstance(ts_fb, list) and theme_name_fb in ts_fb and uri_fb in original_candidates_map_for_reprompt) if isinstance(fb_assigns_val, dict) else set())
                if new_uris_fb:
                    logger.info(f"[{norm_concept_str}] Fallback SUCCESS for '{theme_name_fb}': Assigned {len(new_uris_fb)} URIs."); reprompt_diag["successes"] += 1; fixed_themes_set.add(theme_name_fb)
                    for uri_new_fb in new_uris_fb:
                        if uri_new_fb not in theme_map_final.get(theme_name_fb, []): theme_map_final[theme_name_fb].append(uri_new_fb); fallback_adds_list.append({"uri": uri_new_fb, "assigned_theme": theme_name_fb})
                    theme_diag[theme_name_fb].update({"status": "Passed (Fallback)", "rule_failed": False});
                    if theme_name_fb in failed_rules_map: del failed_rules_map[theme_name_fb]
                else: logger.warning(f"[{norm_concept_str}] Fallback LLM for '{theme_name_fb}' assigned 0 candidates."); reprompt_diag["failures"] += 1; theme_diag[theme_name_fb]["status"] = "Failed (Fallback - No Assigns)"
            else: err_fb = fb_result_val.get("error", "?") if fb_result_val else "?"; logger.error(f"[{norm_concept_str}] Fallback LLM failed for '{theme_name_fb}'. Error:{err_fb}"); reprompt_diag["failures"] += 1; theme_diag[theme_name_fb]["status"] = "Failed (Fallback - API Error)"
    elif failed_rules_map: logger.warning(f"[{norm_concept_str}] Cannot attempt fallback ({list(failed_rules_map.keys())}), LLM refinement disabled or provider 'none'.")

    mnh_uris_llm_set: Set[str] = set()
    if (llm_negation_enabled and args_namespace.llm_provider != "none" and travel_category and travel_category.get("uri")):
        neg_diag["attempted"] = True; logger.info(f"[{norm_concept_str}] Attempting LLM Negation Identification...");
        neg_api_cfg_val = config.get("LLM_API_CONFIG", {}); neg_llm_cfg_val = config.get("LLM_NEGATION_CONFIG", {})
        neg_timeout_val = int(neg_api_cfg_val.get("REQUEST_TIMEOUT", 180)); neg_retries_val = int(neg_api_cfg_val.get("MAX_RETRIES", 3)); neg_temp_val = float(neg_llm_cfg_val.get("temperature", 0.3)); max_cands_check_neg = int(neg_llm_cfg_val.get("max_candidates_to_check", 30))
        anchor_label_neg = travel_category.get("label", input_concept)
        logger.debug(f"[{norm_concept_str}] Preparing candidates for negation check (Max: {max_cands_check_neg}). Anchor label for prompt: {anchor_label_neg}")
        candidates_sorted_unbiased_neg = sorted(original_candidates_map_for_reprompt.values(), key=lambda x_neg: x_neg.get("combined_score_unbiased", 0.0), reverse=True,)
        candidates_for_negation_check_list: List[Dict[str,str]] = []
        for c_neg in candidates_sorted_unbiased_neg[:max_cands_check_neg]:
            if c_neg.get("uri"):
                 label_for_neg = c_neg.get("prefLabel") or c_neg.get("label")
                 if label_for_neg:
                     candidates_for_negation_check_list.append({"uri": str(c_neg["uri"]), "prefLabel": str(label_for_neg)}) # Ensure strings
        neg_diag["candidates_checked"] = [c_neg_item['uri'] for c_neg_item in candidates_for_negation_check_list]
        logger.debug(f"[{norm_concept_str}] Found {len(candidates_for_negation_check_list)} candidates for negation check.")
        if candidates_for_negation_check_list:
            try:
                neg_sys_p_val, neg_user_p_val = construct_llm_negation_prompt(input_concept, anchor_label_neg, candidates_for_negation_check_list)
                neg_result_val = call_llm(neg_sys_p_val, neg_user_p_val, args_namespace.llm_model, neg_timeout_val, neg_retries_val, neg_temp_val, args_namespace.llm_provider,
                                          input_concept_for_logging=f"{input_concept} [Negation Check]")
                if neg_result_val and neg_result_val.get("success"):
                    neg_response_val = neg_result_val.get("response", {}); found_uris_neg = neg_response_val.get("negating_uris", [])
                    if isinstance(found_uris_neg, list):
                        valid_neg_uris_set = {uri_neg for uri_neg in found_uris_neg if isinstance(uri_neg, str) and uri_neg in taxonomy_concepts_cache_in}; invalid_identified_neg = set(found_uris_neg) - valid_neg_uris_set
                        if invalid_identified_neg: logger.warning(f"[{norm_concept_str}] LLM negation identified unknown URIs: {invalid_identified_neg}")
                        mnh_uris_llm_set = valid_neg_uris_set; neg_diag["success"] = True; neg_diag["uris_found"] = len(mnh_uris_llm_set); neg_diag["identified_negating_uris"] = sorted(list(mnh_uris_llm_set))
                        logger.info(f"[{norm_concept_str}] LLM identified {len(mnh_uris_llm_set)} negating URIs: {mnh_uris_llm_set}")
                    else: logger.warning(f"[{norm_concept_str}] LLM negation response 'negating_uris' not a list."); neg_diag["error"] = "Invalid LLM response format (not list)"
                else: err_neg = neg_result_val.get("error", "?") if neg_result_val else "?"; logger.warning(f"[{norm_concept_str}] LLM negation call failed: {err_neg}"); neg_diag["error"] = f"LLM Call Failed: {err_neg}"
            except Exception as e_neg: logger.error(f"[{norm_concept_str}] LLM negation exception: {e_neg}", exc_info=True); neg_diag["error"] = f"Exception: {e_neg}"
        else: logger.info(f"[{norm_concept_str}] Skipping LLM negation: No candidates provided to check.")
    elif llm_negation_enabled: logger.info(f"[{norm_concept_str}] Skipping LLM negation (Provider 'none' or no anchor URI).")
    else: logger.info(f"[{norm_concept_str}] LLM negation disabled in config.")

    mnh_uris_manual_set: Set[str] = set()
    mnh_config_override_list = concept_overrides.get("must_not_have", [])
    if isinstance(mnh_config_override_list, list): mnh_uris_manual_set = set(i_mnh["uri"] for i_mnh in mnh_config_override_list if isinstance(i_mnh, dict) and "uri" in i_mnh)
    all_uris_to_exclude_set = mnh_uris_manual_set.union(mnh_uris_llm_set)
    output["must_not_have"] = [{"uri": u_mnh, "skos:prefLabel": get_primary_label(u_mnh, taxonomy_concepts_cache_in, u_mnh), "scope": "Config Override" if u_mnh in mnh_uris_manual_set else "LLM Identified",} for u_mnh in sorted(list(all_uris_to_exclude_set))]
    if all_uris_to_exclude_set: logger.info(f"[{norm_concept_str}] Populated 'must_not_have' with {len(output['must_not_have'])} URIs. These will be excluded.")

    if all_uris_to_exclude_set:
        logger.debug(f"[{norm_concept_str}] Filtering theme assignments to remove excluded URIs...")
        for theme_name_filter in list(theme_map_final.keys()):
            original_uris_filter = theme_map_final[theme_name_filter]; filtered_uris_list = [uri_filter for uri_filter in original_uris_filter if uri_filter not in all_uris_to_exclude_set]
            if not filtered_uris_list: del theme_map_final[theme_name_filter]; logger.debug(f"[{norm_concept_str}] Theme '{theme_name_filter}' became empty after removing excluded URIs.")
            elif len(original_uris_filter) != len(filtered_uris_list): removed_count_filter = len(original_uris_filter) - len(filtered_uris_list); theme_map_final[theme_name_filter] = filtered_uris_list; logger.debug(f"[{norm_concept_str}] Removed {removed_count_filter} excluded URIs from theme '{theme_name_filter}'.")
    else: logger.debug(f"[{norm_concept_str}] No URIs identified for exclusion from definition attributes.")

    active_themes_set = set(theme_map_final.keys()); logger.debug(f"[{norm_concept_str}] Active themes after filtering excluded URIs: {active_themes_set}")
    initial_theme_weights_map: Dict[str,float] = {}; total_initial_theme_weight_val = 0.0
    for theme_name_iw, theme_data_iw in base_themes_config.items(): _, weight_iw, _, _ = get_dynamic_theme_config(norm_concept_str, theme_name_iw, config); initial_theme_weights_map[theme_name_iw] = weight_iw; total_initial_theme_weight_val += weight_iw
    normalized_initial_theme_weights_map = {name_niw: (w_niw / total_initial_theme_weight_val) if total_initial_theme_weight_val > 0 else 0 for name_niw, w_niw in initial_theme_weights_map.items()}
    weight_to_redistribute_val = sum(normalized_initial_theme_weights_map.get(name_wr, 0) for name_wr in base_themes_config if name_wr not in active_themes_set)
    total_weight_of_active_themes_val = sum(normalized_initial_theme_weights_map.get(name_active_w, 0) for name_active_w in active_themes_set); final_theme_weights_map: Dict[str, float] = {}
    for name_final_w in base_themes_config:
        if name_final_w in active_themes_set and total_weight_of_active_themes_val > 0:
            initial_norm_w_val = normalized_initial_theme_weights_map.get(name_final_w, 0); redistributed_share_val = ((initial_norm_w_val / total_weight_of_active_themes_val) * weight_to_redistribute_val if total_weight_of_active_themes_val > 0 else 0); final_theme_weights_map[name_final_w] = initial_norm_w_val + redistributed_share_val
        else: final_theme_weights_map[name_final_w] = 0.0
    
    final_themes_out_list: List[Dict[str,Any]] = []; all_final_attrs_list: List[Dict[str,Any]] = [];
    # total_attribute_weight_sum_val = 0.0 # Already initialized
    for theme_name_attr, base_data_attr in base_themes_config.items():
        final_norm_w_attr = final_theme_weights_map.get(theme_name_attr, 0.0); rule_attr, _, subscore_affinity_name_attr, _ = get_dynamic_theme_config(norm_concept_str, theme_name_attr, config)
        uris_attr_list = theme_map_final.get(theme_name_attr, [])
        theme_attrs_list: List[Dict[str,Any]] = []
        if uris_attr_list and final_norm_w_attr > 1e-9:
            scores_map_attr: Dict[str,float] = {}
            for u_attr in uris_attr_list:
                 score_val_attr = original_candidates_map_for_reprompt.get(u_attr, {}).get("combined_score", 0.0)
                 scores_map_attr[u_attr] = max(1e-9, score_val_attr)
            total_score_attr = sum(scores_map_attr.values())
            if total_score_attr > 1e-9:
                for u_s_attr in uris_attr_list:
                    s_val_attr = scores_map_attr.get(u_s_attr, 1e-9); prop_attr = s_val_attr / total_score_attr; attr_w_val = final_norm_w_attr * prop_attr
                    if attr_w_val >= min_weight_attr:
                        d_attr = original_candidates_map_for_reprompt.get(u_s_attr, {}); is_fb_attr = any(f_attr["uri"] == u_s_attr and f_attr["assigned_theme"] == theme_name_attr for f_attr in fallback_adds_list)
                        # Ensure 'type_labels' from d_attr is a list, default to empty list if not present or not list
                        type_labels_val = d_attr.get("types", []) # Use 'types' as per get_kg_data output
                        if not isinstance(type_labels_val, list): type_labels_val = []

                        attr_entry = {"uri": u_s_attr, "skos:prefLabel": get_primary_label(u_s_attr, taxonomy_concepts_cache_in, u_s_attr), "concept_weight": round(attr_w_val, 6), "type": type_labels_val}
                        if is_fb_attr: attr_entry["comment"] = "Fallback Assignment"
                        # attr_entry["_weighting_score"] = round(s_val_attr, 6) # No longer needed to store this
                        theme_attrs_list.append(attr_entry); all_final_attrs_list.append(attr_entry); # total_attribute_weight_sum_val += attr_w_val # This will be summed later from top_defining_attributes
            elif len(uris_attr_list) > 0: # Fallback to equal weight
                eq_w_attr = final_norm_w_attr / len(uris_attr_list)
                if eq_w_attr >= min_weight_attr:
                    logger.debug(f"[{norm_concept_str}] Theme '{theme_name_attr}' - Using equal weight fallback ({eq_w_attr:.4f}) as total score was zero or negligible.")
                    for u_eq_attr in uris_attr_list:
                        d_eq_attr = original_candidates_map_for_reprompt.get(u_eq_attr, {}); is_fb_eq_attr = any(f_eq_attr["uri"] == u_eq_attr and f_eq_attr["assigned_theme"] == theme_name_attr for f_eq_attr in fallback_adds_list)
                        type_labels_eq = d_eq_attr.get("types", [])
                        if not isinstance(type_labels_eq, list): type_labels_eq = []
                        attr_entry_eq = {"uri": u_eq_attr, "skos:prefLabel": get_primary_label(u_eq_attr, taxonomy_concepts_cache_in, u_eq_attr), "concept_weight": round(eq_w_attr, 6), "type": type_labels_eq}
                        if is_fb_eq_attr: attr_entry_eq["comment"] = "Fallback Assignment"
                        theme_attrs_list.append(attr_entry_eq); all_final_attrs_list.append(attr_entry_eq); # total_attribute_weight_sum_val += eq_w_attr

        theme_attrs_list.sort(key=lambda x_sort_attr: x_sort_attr["concept_weight"], reverse=True)
        theme_diag[theme_name_attr]["attributes_after_weighting"] = len(theme_attrs_list)
        if theme_name_attr not in active_themes_set and theme_diag[theme_name_attr].get("status") == "Pending": theme_diag[theme_name_attr]["status"] = "Filtered Out (No Active Assigns)"
        elif theme_diag[theme_name_attr].get("status") == "Pending": theme_diag[theme_name_attr]["status"] = ("Processed (Initial)" if not theme_diag[theme_name_attr].get("rule_failed") else theme_diag[theme_name_attr].get("status", "Failed Rule"))
        final_themes_out_list.append({"theme_name": theme_name_attr, "theme_type": base_data_attr.get("type", "?"), "rule_applied": rule_attr, "normalized_theme_weight": round(final_norm_w_attr, 6), "subScore": subscore_affinity_name_attr or f"{theme_name_attr}Affinity", "llm_summary": None, "attributes": theme_attrs_list,})
    output["themes"] = final_themes_out_list
    # logger.debug(f"[{norm_concept_str}] Sum of final attribute weights from themes: {total_attribute_weight_sum_val:.4f}") # Commented out as sum will be from top_defining_attributes

    core_definitional_uris_set: Set[str] = set()
    promoted_to_top_attrs_list: List[str] = []
    if travel_category and travel_category.get("uri"):
        anchor_uri_core = travel_category["uri"]
        core_definitional_uris_set.add(anchor_uri_core)
        if anchor_uri_core not in core_def_diag["identified"]: core_def_diag["identified"].append(anchor_uri_core)
        logger.debug(f"[{norm_concept_str}] Anchor URI '{anchor_uri_core}' added to Core Definitional URIs.")

        normalized_input_for_core_def = normalize_concept(input_concept)
        sorted_candidates_for_core_def = sorted(
            original_candidates_map_for_reprompt.values(),
            key=lambda c_core: c_core.get("combined_score", 0.0),
            reverse=True
        )
        variants_found_count = 0
        for cand_core in sorted_candidates_for_core_def:
            if variants_found_count >= core_def_max_variants: break
            cand_uri_core = cand_core.get("uri")
            if cand_uri_core and cand_uri_core != anchor_uri_core:
                cand_label_for_norm = cand_core.get("prefLabel") or cand_core.get("label") # Use prefLabel from orig_map
                if not cand_label_for_norm and cand_uri_core: cand_label_for_norm = cand_uri_core.split('/')[-1].split('#')[-1]

                if cand_label_for_norm: # Ensure we have a label to normalize
                    normalized_cand_label_core = normalize_concept(cand_label_for_norm)
                    similarity_score_core = get_text_similarity(normalized_input_for_core_def, normalized_cand_label_core)
                    if similarity_score_core >= core_def_text_sim_thresh:
                        core_definitional_uris_set.add(cand_uri_core)
                        if cand_uri_core not in core_def_diag["identified"]: core_def_diag["identified"].append(cand_uri_core)
                        variants_found_count += 1
                        logger.debug(f"[{norm_concept_str}] Added Core Definitional Variant: {cand_uri_core} (Label: '{cand_label_for_norm}', NormLabel: '{normalized_cand_label_core}', Sim: {similarity_score_core:.2f})")
        logger.info(f"[{norm_concept_str}] Identified {len(core_definitional_uris_set)} Core Definitional URIs: {core_definitional_uris_set}")
    else:
        logger.warning(f"[{norm_concept_str}] No travel_category (anchor) URI found, cannot identify core definitional URIs.")

    unique_attrs_map: Dict[str, Dict[str, Any]] = {};
    for attr_final_item in all_final_attrs_list:
        uri_final_attr = attr_final_item.get("uri");
        if not uri_final_attr or uri_final_attr not in original_candidates_map_for_reprompt: continue
        existing_attr = unique_attrs_map.get(uri_final_attr)
        current_combined_score = original_candidates_map_for_reprompt.get(uri_final_attr, {}).get("combined_score", 0.0)
        current_concept_weight = attr_final_item.get("concept_weight", 0.0)

        if not existing_attr or current_concept_weight > existing_attr.get("concept_weight", 0.0):
            attr_copy_final = {k_final: v_final for k_final, v_final in attr_final_item.items() if k_final not in ["comment", "_weighting_score"]}
            attr_copy_final["_score_for_ranking"] = current_combined_score # Store original combined score for sorting later
            unique_attrs_map[uri_final_attr] = attr_copy_final
        elif current_concept_weight == existing_attr.get("concept_weight", 0.0):
            if current_combined_score > existing_attr.get("_score_for_ranking", -1.0):
                attr_copy_final_tie = {k_final_tie: v_final_tie for k_final_tie, v_final_tie in attr_final_item.items() if k_final_tie not in ["comment", "_weighting_score"]}
                attr_copy_final_tie["_score_for_ranking"] = current_combined_score
                unique_attrs_map[uri_final_attr] = attr_copy_final_tie

    for core_uri in core_definitional_uris_set:
        if core_uri not in unique_attrs_map:
            if core_uri in original_candidates_map_for_reprompt:
                core_details = original_candidates_map_for_reprompt[core_uri]
                type_labels_core = core_details.get("types", []) # Use 'types' from orig_map
                if not isinstance(type_labels_core, list): type_labels_core = []
                forced_attr = {
                    "uri": core_uri,
                    "skos:prefLabel": get_primary_label(core_uri, taxonomy_concepts_cache_in, core_uri),
                    "concept_weight": round(core_def_forced_weight, 6),
                    "type": type_labels_core,
                    "_score_for_ranking": core_details.get("combined_score", 0.0),
                    "_is_core_def_forced": True
                }
                unique_attrs_map[core_uri] = forced_attr
                if core_uri not in promoted_to_top_attrs_list : promoted_to_top_attrs_list.append(core_uri) # Track promotion
                logger.debug(f"[{norm_concept_str}] Forcing Core Definitional URI '{core_uri}' into unique_attrs_map with weight {core_def_forced_weight}.")
            else: logger.warning(f"[{norm_concept_str}] Core URI {core_uri} not in original_candidates_map, cannot force.")
    core_def_diag["promoted_to_top_attrs"] = promoted_to_top_attrs_list

    sorted_top_list = sorted(unique_attrs_map.values(), key=lambda x_sort_top: x_sort_top.get("_score_for_ranking", 0.0), reverse=True)
    for attr_final_clean in sorted_top_list:
        attr_final_clean.pop("_score_for_ranking", None)
        attr_final_clean.pop("_is_core_def_forced", None)
    output["top_defining_attributes"] = sorted_top_list[:top_n_attrs]
    logger.info(f"[{norm_concept_str}] Final `top_defining_attributes` count: {len(output['top_defining_attributes'])}.")

    try:
        determined_lodging_type_val = determine_lodging_type(travel_category, output["top_defining_attributes"], config, taxonomy_concepts_cache_in)
        output["applicable_lodging_types"] = determined_lodging_type_val
        logger.info(f"[{norm_concept_str}] Dynamically determined Applicable Lodging Type: {determined_lodging_type_val}")
        output["diagnostics"]["lodging_type_determination"]["result"] = determined_lodging_type_val
    except Exception as e_lodging:
        logger.error(f"[{norm_concept_str}] Error during lodging type determination: {e_lodging}", exc_info=True)
        output["applicable_lodging_types"] = "Error"; output["diagnostics"]["lodging_type_determination"]["error"] = str(e_lodging)

    output["key_associated_concepts_unthemed"] = []
    unthemed_percentile_threshold = float(final_cfg.get("UNTHEMED_CAPTURE_SCORE_PERCENTILE", 75))
    # logger.info(f"[{norm_concept_str}] Checking for unthemed concepts above {unthemed_percentile_threshold}th percentile score...") # Already logged
    llm_candidate_uris_list = list(original_candidates_map_for_reprompt.keys()); llm_candidate_scores_list = [original_candidates_map_for_reprompt.get(uri_score, {}).get("combined_score", 0.0) for uri_score in llm_candidate_uris_list]
    score_threshold_unthemed = 0.0
    if llm_candidate_scores_list and NUMPY_AVAILABLE: # type: ignore
        try: score_threshold_unthemed = np.percentile(llm_candidate_scores_list, unthemed_percentile_threshold); # type: ignore
        except Exception as e_perc: logger.warning(f"[{norm_concept_str}] Could not calculate percentile: {e_perc}. Using 0.0."); score_threshold_unthemed = 0.0
    elif not NUMPY_AVAILABLE: logger.warning(f"[{norm_concept_str}] Numpy unavailable for percentile. Using 0.0."); score_threshold_unthemed = 0.0 # type: ignore
    else: logger.warning(f"[{norm_concept_str}] No scores for percentile."); score_threshold_unthemed = 0.0

    assigned_uris_in_themes_set = set();
    for theme_name_assigned in theme_map_final: assigned_uris_in_themes_set.update(theme_map_final[theme_name_assigned])
    handled_core_def_uris_set = set(promoted_to_top_attrs_list) # URIs that were forced into top_defining_attributes

    unthemed_concepts_found_list: List[Dict[str,Any]] = []
    for uri_unthemed, details_unthemed in original_candidates_map_for_reprompt.items():
        if uri_unthemed not in assigned_uris_in_themes_set and \
           uri_unthemed not in all_uris_to_exclude_set and \
           uri_unthemed not in handled_core_def_uris_set: # Exclude if already promoted as core
            score_unthemed = details_unthemed.get("combined_score", 0.0)
            type_labels_unthemed = details_unthemed.get("types", []) # Use 'types'
            if not isinstance(type_labels_unthemed, list): type_labels_unthemed = []
            if score_unthemed >= score_threshold_unthemed and score_unthemed > 0:
                unthemed_concepts_found_list.append({"uri": uri_unthemed, "skos:prefLabel": get_primary_label(uri_unthemed, taxonomy_concepts_cache_in, uri_unthemed), "combined_score": round(score_unthemed, 6), "type": type_labels_unthemed})
    unthemed_concepts_found_list.sort(key=lambda x_sort_unthemed: x_sort_unthemed["combined_score"], reverse=True); output["key_associated_concepts_unthemed"] = unthemed_concepts_found_list
    logger.info(f"[{norm_concept_str}] Found {len(unthemed_concepts_found_list)} unthemed concepts (score >= {score_threshold_unthemed:.4f}).")

    final_subscore_weights_map = defaultdict(float); # logger.debug already present
    for name_ss, theme_data_ss in base_themes_config.items():
        if name_ss in active_themes_set:
            theme_final_weight_ss = final_theme_weights_map.get(name_ss, 0.0); relevant_subscores_cfg_ss = theme_data_ss.get("relevant_subscores", {})
            if isinstance(relevant_subscores_cfg_ss, dict):
                for (subscore_name_ss, base_subscore_weight_ss,) in relevant_subscores_cfg_ss.items(): contribution_ss = theme_final_weight_ss * float(base_subscore_weight_ss); final_subscore_weights_map[subscore_name_ss] += contribution_ss; # logger.debug removed for brevity
    if property_type_str == "VRBO":
        vrbo_sentiment_min_val = vrbo_defaults.get("SentimentScore", 1e-6);
        vrbo_groupintel_min_val = vrbo_defaults.get("GroupIntelligenceScore", 1e-6);
        if "SentimentScore" not in final_subscore_weights_map: final_subscore_weights_map["SentimentScore"] = vrbo_sentiment_min_val;
        if "GroupIntelligenceScore" not in final_subscore_weights_map: final_subscore_weights_map["GroupIntelligenceScore"] = vrbo_groupintel_min_val;
    total_subscore_weight_before_vrbo_min_val = sum(final_subscore_weights_map.values()); normalized_subscores_map: Dict[str, float] = {}
    if total_subscore_weight_before_vrbo_min_val > 1e-9:
        for name_norm_ss, weight_norm_ss in final_subscore_weights_map.items(): normalized_subscores_map[name_norm_ss] = weight_norm_ss / total_subscore_weight_before_vrbo_min_val
    if property_type_str == "VRBO" and normalized_subscores_map:
        made_adjustments_flag = False
        vrbo_sentiment_min_cfg = vrbo_defaults.get("SentimentScore", 0.10)
        vrbo_groupintel_min_cfg = vrbo_defaults.get("GroupIntelligenceScore", 0.05)
        if "SentimentScore" not in normalized_subscores_map: normalized_subscores_map["SentimentScore"] = 0.0
        if "GroupIntelligenceScore" not in normalized_subscores_map: normalized_subscores_map["GroupIntelligenceScore"] = 0.0
        if normalized_subscores_map["SentimentScore"] < vrbo_sentiment_min_cfg: normalized_subscores_map["SentimentScore"] = vrbo_sentiment_min_cfg; made_adjustments_flag = True
        if normalized_subscores_map["GroupIntelligenceScore"] < vrbo_groupintel_min_cfg: normalized_subscores_map["GroupIntelligenceScore"] = vrbo_groupintel_min_cfg; made_adjustments_flag = True
        if made_adjustments_flag:
            current_sum_ss = sum(normalized_subscores_map.values())
            if current_sum_ss > 1e-9 and abs(current_sum_ss - 1.0) > 1e-9: renorm_factor_ss = 1.0 / current_sum_ss; normalized_subscores_map = {name_renorm_ss: weight_renorm_ss * renorm_factor_ss for name_renorm_ss, weight_renorm_ss in normalized_subscores_map.items()}
    output["additional_relevant_subscores"] = [{"subscore_name": name_out_ss, "weight": round(weight_out_ss, 6)} for name_out_ss, weight_out_ss in sorted(normalized_subscores_map.items(), key=lambda item_ss: item_ss[1], reverse=True) if weight_out_ss > 1e-9]

    subscore_component_val = sum(item_final_score["weight"] for item_final_score in output["additional_relevant_subscores"]);
    final_total_attr_weight = sum(attr_item.get("concept_weight", 0.0) for attr_item in output["top_defining_attributes"]) # Sum weights from the final list
    subscore_component_val = min(1.0, subscore_component_val) if subscore_component_val > 0 else 0;
    attribute_component_val = (min(1.0, final_total_attr_weight) if final_total_attr_weight > 0 else 0)
    output["affinity_score_total_allocated"] = round((0.60 * subscore_component_val) + (0.40 * attribute_component_val), 6)
    logger.info(f"[{norm_concept_str}] Calculated Affinity Score (Total Allocated): {output['affinity_score_total_allocated']:.4f} (Subscores: {subscore_component_val:.3f}*0.6, Attributes: {attribute_component_val:.3f}*0.4)")

    geo_check_flag = False; geo_subscore_names_set = {"GeospatialAffinityScore", "WalkabilityScore"}
    if any(ss_geo["subscore_name"] in geo_subscore_names_set for ss_geo in output["additional_relevant_subscores"]): geo_check_flag = True;
    if not geo_check_flag:
        for theme_geo in output["themes"]:
            if theme_geo.get("theme_name") == "Location" and theme_geo.get("attributes"): geo_check_flag = True; break
    if not geo_check_flag and travel_category and travel_category.get("uri"):
        anchor_uri_geo = travel_category["uri"]; bias_config_geo = config.get("NAMESPACE_BIASING", {}); location_ns_geo = tuple(bias_config_geo.get("location_context_ns", ()))
        if location_ns_geo and anchor_uri_geo.startswith(location_ns_geo): geo_check_flag = True;
    output["requires_geo_check"] = geo_check_flag

    final_diag_dict = output["diagnostics"]["final_output"]; final_diag_dict["must_not_have_count"] = len(output["must_not_have"]); final_diag_dict["additional_subscores_count"] = len(output["additional_relevant_subscores"])
    final_diag_dict["themes_count"] = len(output["themes"]); final_diag_dict["unthemed_concepts_captured_count"] = len(output["key_associated_concepts_unthemed"])
    output["failed_fallback_themes"] = {n_ff: r_ff for n_ff, r_ff in failed_rules_map.items() if n_ff not in fixed_themes_set}; final_diag_dict["failed_fallback_themes_count"] = len(output["failed_fallback_themes"])
    final_diag_dict["top_defining_attributes_count"] = len(output["top_defining_attributes"]); output["diagnostics"]["stage2"]["status"] = "Completed"; output["diagnostics"]["stage2"]["duration_seconds"] = round(time.time() - start_stage2, 2)
    return output

# --- Main Processing Loop ---
def generate_affinity_definitions_loop(concepts_to_process: List[str], config: Dict, args_namespace: argparse.Namespace, sbert_model_in: SentenceTransformer, primary_embeddings_map_in: Dict[str, np.ndarray], taxonomy_concepts_cache_in: Dict[str, Dict], keyword_label_index_in: Optional[Dict[str, Set[str]]], bm25_model_in: Optional[bm25s.BM25], keyword_corpus_uris_in: Optional[List[str]]) -> List[Dict]: # type: ignore
    all_definitions_list: List[Dict[str,Any]] = []; cache_ver_str = config.get("cache_version")
    if not taxonomy_concepts_cache_in: logger.critical("FATAL: Concepts cache empty."); return []
    limit_val = args_namespace.limit if args_namespace.limit and args_namespace.limit > 0 else len(concepts_to_process); concepts_subset_list = concepts_to_process[:limit_val];
    logger.info(f"Processing {len(concepts_subset_list)}/{len(concepts_to_process)} concepts.")
    if not concepts_subset_list: logger.warning("Concept subset empty!"); return []
    disable_tqdm_flag = not logger.isEnabledFor(logging.INFO) or args_namespace.debug
    for concept_str_loop in tqdm(concepts_subset_list, desc="Processing Concepts", disable=disable_tqdm_flag):
        start_time_loop = time.time(); norm_concept_loop = normalize_concept(concept_str_loop);
        logger.info(f"=== Processing Concept: '{concept_str_loop}' ('{norm_concept_loop}') ===")
        affinity_def_dict: Dict[str, Any] = {
            "input_concept": concept_str_loop, "normalized_concept": norm_concept_loop, "applicable_lodging_types": "Both", "travel_category": {},
            "top_defining_attributes": [], "themes": [], "key_associated_concepts_unthemed": [], "additional_relevant_subscores": [],
            "must_not_have": [], "requires_geo_check": False, "failed_fallback_themes": {}, "affinity_score_total_allocated": 0.0,
            "processing_metadata": {"status": "Started", "version": SCRIPT_VERSION, "timestamp": None, "total_duration_seconds": 0.0, "cache_version": cache_ver_str, "llm_provider": args_namespace.llm_provider, "llm_model": args_namespace.llm_model if args_namespace.llm_provider != "none" else None,},
            "diagnostics": {"lodging_type_determination": {}, "llm_negation": {"attempted": False, "success": False, "uris_found": 0, "error": None, "candidates_checked": [], "identified_negating_uris": [],}, "stage1": {"status": "Not Started", "error": None, "selection_method": "?", "expansion": {}, "sbert_candidate_count_initial": 0, "keyword_candidate_count_initial": 0, "unique_candidates_before_ranking": 0, "llm_candidate_count": 0,}, "llm_slotting": {"status": "Not Started", "error": None}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "core_definitional_uris": {"identified": [], "promoted_to_top_attrs": []}, "stage2": {"status": "Not Started", "error": None}, "theme_processing": {}, "final_output": {}, "error_details": None,},
        }
        diag1_loop = affinity_def_dict["diagnostics"]["stage1"]; diag_llm_loop = affinity_def_dict["diagnostics"]["llm_slotting"]
        try:
            concept_emb_loop = get_concept_embedding(norm_concept_loop, sbert_model_in)
            if concept_emb_loop is None: logger.error(f"[{norm_concept_loop}] Embedding failed."); diag1_loop["error"] = "Embedding failed"; diag1_loop["status"] = "Failed" # type: ignore
            cand_details_loop: List[Dict[str,Any]] = []; orig_map_loop: Dict[str,Dict[str,Any]] = {}; anchor_loop: Optional[Dict[str,Any]] = None; exp_diag_loop: Dict[str,Any] = {}; sbert_init_loop = 0; kw_init_loop = 0; unique_count_loop = 0
            if diag1_loop["status"] != "Failed": # type: ignore
                stage1_start_call_loop = time.time();
                (cand_details_loop, orig_map_loop, anchor_loop, _, exp_diag_loop, sbert_init_loop, kw_init_loop, unique_count_loop,) = prepare_evidence(concept_str_loop, concept_emb_loop, primary_embeddings_map_in, config, args_namespace, bm25_model_in, keyword_corpus_uris_in, keyword_label_index_in, taxonomy_concepts_cache_in,)
                stage1_dur_loop = time.time() - stage1_start_call_loop;
                diag1_loop.update({"status": "Completed", "error": diag1_loop.get("error"), "selection_method": exp_diag_loop.get("selection_method", "Combined BM25s+SBERT"), "expansion": exp_diag_loop, "sbert_candidate_count_initial": sbert_init_loop, "keyword_candidate_count_initial": kw_init_loop, "unique_candidates_before_ranking": unique_count_loop, "llm_candidate_count": len(cand_details_loop), "duration_seconds": round(stage1_dur_loop, 2),}) # type: ignore
                logger.info(f"[{norm_concept_loop}] Stage 1 done ({stage1_dur_loop:.2f}s). Status:{diag1_loop['status']}. LLM Cands:{len(cand_details_loop)}. SBERT:{sbert_init_loop}, KW:{kw_init_loop}.") # type: ignore
                if not anchor_loop or not anchor_loop.get("uri"): logger.warning(f"[{norm_concept_loop}] Stage 1 completed but no valid anchor.")
            else: logger.warning(f"[{norm_concept_loop}] Skipping Stages 1b and 2 due to embedding failure.")

            llm_result_loop = None; llm_start_loop = time.time();
            diag_llm_loop.update({"llm_provider": args_namespace.llm_provider, "llm_model": (args_namespace.llm_model if args_namespace.llm_provider != "none" else None),}) # type: ignore
            if diag1_loop["status"] == "Failed": logger.warning(f"[{norm_concept_loop}] Skipping LLM (Stage 1 failed)."); diag_llm_loop["status"] = "Skipped (Stage 1 Failed)" # type: ignore
            elif not cand_details_loop: logger.warning(f"[{norm_concept_loop}] Skipping LLM (No candidates)."); affinity_def_dict["processing_metadata"]["status"] = "Warning - No LLM Candidates"; diag_llm_loop["status"] = "Skipped (No Candidates)" # type: ignore
            elif args_namespace.llm_provider == "none": logger.info(f"[{norm_concept_loop}] Skipping LLM (Provider 'none')."); diag_llm_loop["status"] = "Skipped (Provider None)" # type: ignore
            else:
                diag_llm_loop["status"] = "Started"; diag_llm_loop["llm_call_attempted"] = True # type: ignore
                themes_for_prompt_list = [{"name": name_theme_prompt, "description": get_theme_definition_for_prompt(name_theme_prompt, data_theme_prompt), "is_must_have": get_dynamic_theme_config(norm_concept_loop, name_theme_prompt, config)[0] == "Must have 1",} for name_theme_prompt, data_theme_prompt in config.get("base_themes", {}).items()]
                # Corrected argument order: themes_for_prompt_list, then cand_details_loop
                sys_p_loop, user_p_loop = construct_llm_slotting_prompt(
                    concept_str_loop, 
                    themes_for_prompt_list,  # Correct: themes go here
                    cand_details_loop,       # Correct: candidates go here
                    args_namespace
                )
                llm_api_cfg_loop = config.get("LLM_API_CONFIG", {}); llm_stage2_cfg_loop = config.get("STAGE2_CONFIG", {}); slot_timeout_loop = int(llm_api_cfg_loop.get("REQUEST_TIMEOUT", 180)); slot_retries_loop = int(llm_api_cfg_loop.get("MAX_RETRIES", 5)); slot_temp_loop = float(llm_stage2_cfg_loop.get("LLM_TEMPERATURE", 0.2))
                llm_result_loop = call_llm(sys_p_loop, user_p_loop, args_namespace.llm_model, slot_timeout_loop, slot_retries_loop, slot_temp_loop, args_namespace.llm_provider,
                                           input_concept_for_logging=f"{concept_str_loop} [Slotting]")
                diag_llm_loop["attempts_made"] = (llm_result_loop.get("attempts", 0) if llm_result_loop else 0) # type: ignore
                if llm_result_loop and llm_result_loop.get("success"): diag_llm_loop["llm_call_success"] = True; diag_llm_loop["status"] = "Completed" # type: ignore
                else: diag_llm_loop["llm_call_success"] = False; diag_llm_loop["status"] = "Failed"; diag_llm_loop["error"] = (llm_result_loop.get("error", "?") if llm_result_loop else "?"); logger.warning(f"[{norm_concept_loop}] LLM slotting failed. Error:{diag_llm_loop['error']}") # type: ignore
            diag_llm_loop["duration_seconds"] = round(time.time() - llm_start_loop, 2); logger.info(f"[{norm_concept_loop}] LLM Slotting:{diag_llm_loop['duration_seconds']:.2f}s. Status:{diag_llm_loop['status']}") # type: ignore

            if diag1_loop["status"] != "Failed": # type: ignore
                stage2_out_dict = apply_rules_and_finalize(input_concept=concept_str_loop, llm_call_result=llm_result_loop, config=config, travel_category=anchor_loop, original_candidates_map_for_reprompt=orig_map_loop, args_namespace=args_namespace, taxonomy_concepts_cache_in=taxonomy_concepts_cache_in,)
                affinity_def_dict.update({k_s2: v_s2 for k_s2, v_s2 in stage2_out_dict.items() if k_s2 != "diagnostics"})
                if "diagnostics" in stage2_out_dict: # Merge diagnostics carefully
                    s2d_loop = stage2_out_dict["diagnostics"];
                    affinity_def_dict["diagnostics"]["lodging_type_determination"] = s2d_loop.get("lodging_type_determination", affinity_def_dict["diagnostics"]["lodging_type_determination"]);
                    affinity_def_dict["diagnostics"]["llm_negation"].update(s2d_loop.get("llm_negation", {}));
                    affinity_def_dict["diagnostics"]["theme_processing"] = s2d_loop.get("theme_processing", {});
                    affinity_def_dict["diagnostics"]["reprompting_fallback"].update(s2d_loop.get("reprompting_fallback", {}));
                    affinity_def_dict["diagnostics"]["core_definitional_uris"] = s2d_loop.get("core_definitional_uris", affinity_def_dict["diagnostics"]["core_definitional_uris"]);
                    affinity_def_dict["diagnostics"]["final_output"] = s2d_loop.get("final_output", {});
                    affinity_def_dict["diagnostics"]["stage2"] = s2d_loop.get("stage2", {"status": "Unknown"})
                else: affinity_def_dict["diagnostics"]["stage2"] = {"status": "Unknown", "error": "Stage 2 diagnostics missing"}
            else: affinity_def_dict["diagnostics"]["stage2"]["status"] = "Skipped (Stage 1 Failed)" # type: ignore

            final_status_str = affinity_def_dict["processing_metadata"]["status"]
            if (final_status_str == "Started"):
                if diag1_loop["status"] == "Failed": final_status_str = "Failed - Stage 1 Error" # type: ignore
                elif affinity_def_dict.get("failed_fallback_themes"): final_status_str = "Success with Failed Rules"
                elif diag_llm_loop["status"] == "Failed": final_status_str = "Warning - LLM Slotting Failed" # type: ignore
                elif affinity_def_dict["diagnostics"]["llm_negation"].get("error"): final_status_str = "Warning - LLM Negation Failed"
                elif affinity_def_dict["processing_metadata"]["status"] != "Warning - No LLM Candidates":
                    if affinity_def_dict["diagnostics"]["stage2"].get("error"): final_status_str = f"Warning - Finalization Error ({affinity_def_dict['diagnostics']['stage2']['error']})" # type: ignore
                    elif diag_llm_loop["status"] == "Skipped (Provider None)": final_status_str = "Success (LLM Skipped)" # type: ignore
                    else: final_status_str = "Success"
            affinity_def_dict["processing_metadata"]["status"] = final_status_str
        except Exception as e_loop:
            logger.error(f"Core loop failed for '{concept_str_loop}': {e_loop}", exc_info=True); affinity_def_dict["processing_metadata"]["status"] = "FATAL ERROR"; affinity_def_dict["diagnostics"]["error_details"] = traceback.format_exc()
            if diag1_loop["status"] not in ["Completed", "Failed"]: diag1_loop["status"] = "Failed (Exception)" # type: ignore
            if diag_llm_loop["status"] not in ["Completed", "Failed", "Skipped"]: diag_llm_loop["status"] = "Failed (Exception)" # type: ignore
            if affinity_def_dict["diagnostics"]["stage2"]["status"] not in ["Completed", "Failed", "Skipped",]: affinity_def_dict["diagnostics"]["stage2"]["status"] = "Failed (Exception)" # type: ignore
        finally:
            end_time_loop = time.time(); duration_loop = round(end_time_loop - start_time_loop, 2); affinity_def_dict["processing_metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); affinity_def_dict["processing_metadata"]["total_duration_seconds"] = duration_loop
            all_definitions_list.append(affinity_def_dict);
            log_func_loop = (logger.warning if "Warning" in affinity_def_dict["processing_metadata"]["status"] or "Failed" in affinity_def_dict["processing_metadata"]["status"] else logger.info)
            log_func_loop(f"--- Finished '{norm_concept_loop}' ({duration_loop:.2f}s). Status: {affinity_def_dict['processing_metadata']['status']} ---")
    return all_definitions_list

# --- Main Execution ---
def main():
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
    config_merged = DEFAULT_CONFIG.copy()
    def recursive_update(d: Dict, u: Dict) -> Dict:
        for k, v_update in u.items():
            if isinstance(v_update, dict) and k in d and isinstance(d.get(k), dict):
                d[k] = recursive_update(d.get(k, {}), v_update)
            else:
                d[k] = v_update
        return d
    config_merged = recursive_update(config_merged, user_config)
    if args.output_dir: config_merged["output_dir"] = args.output_dir
    if args.cache_dir: config_merged["cache_dir"] = args.cache_dir
    if args.taxonomy_dir: config_merged["taxonomy_dir"] = args.taxonomy_dir
    if args.llm_provider is not None: config_merged["LLM_PROVIDER"] = args.llm_provider
    if args.llm_model is not None: config_merged["LLM_MODEL"] = args.llm_model

    _config_data = config_merged;
    args.llm_provider = config_merged["LLM_PROVIDER"] # Ensure args reflect final config
    args.llm_model = config_merged["LLM_MODEL"]

    output_dir_final = config_merged["output_dir"]; cache_dir_final = config_merged["cache_dir"]; cache_ver_final = config_merged.get("cache_version", DEFAULT_CACHE_VERSION)
    os.makedirs(output_dir_final, exist_ok=True); os.makedirs(cache_dir_final, exist_ok=True)
    log_file_final = os.path.join(output_dir_final, LOG_FILE_TEMPLATE.format(cache_version=cache_ver_final)); out_file_final = os.path.join(output_dir_final, OUTPUT_FILE_TEMPLATE.format(cache_version=cache_ver_final))
    log_level_final = logging.DEBUG if args.debug else logging.INFO; setup_logging(log_level_final, log_file_final, args.debug); setup_detailed_loggers(output_dir_final, cache_ver_final) # From utils
    logger.info(f"Starting {SCRIPT_VERSION}"); logger.info(f"Cmd: {' '.join(sys.argv)}"); logger.info(f"Using Config: {args.config}")
    logger.info(f"Effective Output Dir: {output_dir_final}, Cache Dir: {cache_dir_final}, Taxonomy Dir: {config_merged['taxonomy_dir']}")
    logger.info(f"Effective Log File: {log_file_final}, Outfile: {out_file_final}"); logger.info(f"Rebuild: {args.rebuild_cache}, Limit: {args.limit or 'None'}, Debug: {args.debug}")
    logger.info("--- Key Parameters (Effective) ---")
    logger.info(f"Global Alpha: {config_merged.get('global_alpha', 'N/A')}")
    kw_scoring_cfg_log = config_merged.get("KEYWORD_SCORING_CONFIG", {}); kw_enabled_log = kw_scoring_cfg_log.get("enabled", False); kw_alg_log = kw_scoring_cfg_log.get("algorithm", "N/A").upper(); logger.info(f"Keyword Scoring: {kw_alg_log} (Enabled: {kw_enabled_log})")
    kg_config_log = config_merged.get("KG_CONFIG", {}); acs_config_log = config_merged.get("ACS_DATA_CONFIG", {}); acs_enabled_log_val = acs_config_log.get("enable_acs_enrichment", False)
    logger.info(f"BM25 Doc Assembly: KG(pref x{kg_config_log.get('pref_label_weight', '?')}, alt x{kg_config_log.get('alt_label_weight', '?')}, def x{kg_config_log.get('definition_weight', '?')}) + ACS(En:{acs_enabled_log_val}, name x{kg_config_log.get('acs_name_weight', '?')}, def x{kg_config_log.get('acs_def_weight', '?')})")
    logger.info(f"Abs Min SBERT: {config_merged.get('min_sbert_score', 'N/A')}"); logger.info(f"KW Dampening Thresh: {config_merged.get('keyword_dampening_threshold','N/A')}, Factor: {config_merged.get('keyword_dampening_factor','N/A')}")
    logger.info(f"Prioritize Exact prefLabel Match: {config_merged.get('prioritize_exact_prefLabel_match', False)}")
    ns_bias_cfg_log = config_merged.get("NAMESPACE_BIASING", {}); logger.info(f"Namespace Biasing: {ns_bias_cfg_log.get('enabled', False)} (Core Boost: {ns_bias_cfg_log.get('core_boost_factor', 'N/A')})")
    stage1_cfg_log = config_merged.get("STAGE1_CONFIG", {}); logger.info(f"Core Def URI Text Sim Thresh: {stage1_cfg_log.get('CORE_DEF_TEXT_SIMILARITY_THRESHOLD', 'N/A (Default 0.90)')}")
    stage2_cfg_log = config_merged.get("STAGE2_CONFIG", {}); logger.info(f"Unthemed Concept Capture Percentile: {stage2_cfg_log.get('UNTHEMED_CAPTURE_SCORE_PERCENTILE', 'N/A (Default 75)')}")
    logger.info(f"Dynamic Lodging Type Check: Top {stage2_cfg_log.get('LODGING_TYPE_TOP_ATTR_CHECK', '?')} Attrs, Threshold {stage2_cfg_log.get('LODGING_TYPE_CONFIDENCE_THRESHOLD', '?')}")
    logger.info(f"LLM Negation Check Enabled: {config_merged.get('LLM_NEGATION_CONFIG', {}).get('enabled', False)}")
    logger.info(f"LLM Provider: {args.llm_provider}, Model: {args.llm_model or 'N/A'}"); logger.info("--- End Key Parameters ---")

    if not PANDAS_AVAILABLE and acs_enabled_log_val: logger.critical("FATAL: ACS Enrichment enabled, but 'pandas' not installed."); sys.exit(1)
    if not BM25S_AVAILABLE and kw_enabled_log and kw_alg_log == "BM25S": logger.critical("FATAL: BM25s enabled, but 'bm25s' library not installed."); sys.exit(1) # type: ignore

    input_concepts_list: List[str] = []
    try:
        with open(args.input_concepts_file, 'r', encoding='utf-8') as f_in: input_concepts_list = [l_in.strip() for l_in in f_in if l_in.strip()]
        if not input_concepts_list: raise ValueError("Input concepts file is empty.")
        logger.info(f"Loaded {len(input_concepts_list)} concepts from '{args.input_concepts_file}'.")
    except Exception as e_in_file: logger.critical(f"Failed to read input file '{args.input_concepts_file}': {e_in_file}"); sys.exit(1)

    concepts_cache_file_main = get_cache_filename("concepts", cache_ver_final, cache_dir_final, extension=".json");
    concepts_data_main = load_taxonomy_concepts(config_merged["taxonomy_dir"], concepts_cache_file_main, args.rebuild_cache, cache_ver_final, args.debug,)
    if concepts_data_main is None: logger.critical("Taxonomy concepts loading failed."); sys.exit(1)
    _taxonomy_concepts_cache = concepts_data_main; logger.info(f"Taxonomy concepts ready ({len(_taxonomy_concepts_cache)} concepts).")

    try: _acs_data = load_acs_data(config_merged.get("ACS_DATA_CONFIG", {}).get("acs_data_path")); logger.info(f"ACS data status: {'Loaded' if _acs_data is not None else 'Load Failed or Disabled'}")
    except Exception as e_acs_main: logger.error(f"ACS data loading exception: {e_acs_main}", exc_info=True); _acs_data = None

    try:
        build_keyword_index(config_merged, _taxonomy_concepts_cache, cache_dir_final, cache_ver_final, args.rebuild_cache)
        if kw_enabled_log and kw_alg_log == "BM25S": logger.info(f"BM25s Index Status: {'Ready' if _bm25_model is not None else 'Failed/Unavailable'}") # type: ignore
        if _keyword_label_index is not None: logger.info("Simple Label Index ready.")
        else: logger.warning("Simple Label Index failed.")
    except Exception as e_idx_main: logger.critical(f"Index building failed: {e_idx_main}", exc_info=True); sys.exit(1)

    try:
        sbert_name_main = config_merged.get("sbert_model_name"); sbert_model_main = get_sbert_model(sbert_name_main)
        if sbert_model_main is None: raise RuntimeError("SBERT model failed to load.")
        logger.info(f"SBERT model '{sbert_name_main or 'default'}' loaded.")
    except Exception as e_sbert_main: logger.critical(f"SBERT loading failed: {e_sbert_main}", exc_info=True); sys.exit(1)

    embed_cache_params_main = {"model": sbert_name_main or "default"}; embed_cache_file_main = get_cache_filename("embeddings", cache_ver_final, cache_dir_final, embed_cache_params_main, ".pkl")
    embed_data_main = precompute_taxonomy_embeddings(_taxonomy_concepts_cache, sbert_model_main, embed_cache_file_main, cache_ver_final, args.rebuild_cache, args.debug,)
    if embed_data_main is None: logger.critical("Embeddings failed."); sys.exit(1)
    _taxonomy_embeddings_cache = embed_data_main; primary_embs_main, uris_w_embeds_main = _taxonomy_embeddings_cache; logger.info(f"Embeddings ready ({len(uris_w_embeds_main)} concepts).")

    logger.info("Starting affinity definition generation loop...")
    start_loop_main = time.time(); results_list = generate_affinity_definitions_loop(input_concepts_list, config_merged, args, sbert_model_main, primary_embs_main, _taxonomy_concepts_cache, _keyword_label_index, _bm25_model, _keyword_corpus_uris,)
    end_loop_main = time.time(); logger.info(f"Loop finished ({end_loop_main - start_loop_main:.2f}s). Generated {len(results_list)} definitions.")
    if results_list: save_results_json(results_list, out_file_final)
    else: logger.warning("No results generated to save.")
    logger.info(f"Script finished. Log: {log_file_final}")


if __name__ == "__main__":
    libs_ok = True
    if not NUMPY_AVAILABLE: print("CRITICAL: numpy missing.", file=sys.stderr); libs_ok = False # type: ignore
    if not SENTENCE_TRANSFORMERS_AVAILABLE: print("CRITICAL: sentence-transformers missing.", file=sys.stderr); libs_ok = False # type: ignore
    if not RDFLIB_AVAILABLE: print("Warning: rdflib missing.", file=sys.stderr) # type: ignore
    if not UTILS_ST_AVAILABLE: print("CRITICAL: sentence-transformers not available in utils.", file=sys.stderr); libs_ok = False # type: ignore
    if not UTILS_RDFLIB_AVAILABLE: print("Warning: rdflib not available in utils.", file=sys.stderr) # type: ignore
    if not libs_ok: sys.exit(1)
    if not BM25S_AVAILABLE: print("Warning: bm2s library not found. Keyword scoring may be affected.", file=sys.stderr) # type: ignore

    llm_provider_in_use_early_check = "none"
    if "--llm-provider" in sys.argv:
        try:
            provider_index_early = sys.argv.index("--llm-provider")
            if provider_index_early + 1 < len(sys.argv):
                llm_provider_in_use_early_check = sys.argv[provider_index_early + 1]
        except (ValueError, IndexError): pass
    elif os.path.exists(DEFAULT_CONFIG_FILE):
        try:
            with open(DEFAULT_CONFIG_FILE, 'r') as f_cfg_early:
                cfg_early_data = json.load(f_cfg_early)
                llm_provider_in_use_early_check = cfg_early_data.get("LLM_PROVIDER", "none")
        except: pass

    openai_key_main = os.environ.get("OPENAI_API_KEY")
    google_key_main = os.environ.get("GOOGLE_API_KEY")
    keys_found_main = bool(openai_key_main or google_key_main)

    if not keys_found_main and llm_provider_in_use_early_check != "none":
        print(f"Warning: LLM provider appears to be '{llm_provider_in_use_early_check}' but no API keys (GOOGLE_API_KEY or OPENAI_API_KEY) found. LLM features will likely fail.", file=sys.stderr)
    elif llm_provider_in_use_early_check == "openai" and not openai_key_main:
        print(f"Warning: LLM provider is 'openai' but OPENAI_API_KEY is not set.", file=sys.stderr)
    elif llm_provider_in_use_early_check == "google" and not google_key_main:
        print(f"Warning: LLM provider is 'google' but GOOGLE_API_KEY is not set.", file=sys.stderr)

    main()