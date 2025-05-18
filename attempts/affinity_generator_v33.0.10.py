#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate affinity definitions for travel concepts (v33 - Simplified LLM Slotting)
- v33.0.10: Added Refined CoT Prompting for abstract concepts.
- v33.0.8: Added Keyword Matching Boost to Stage 1 (prepare_evidence)
           Builds keyword index from labels for efficiency.
           Uses strict priority ranking (keyword matches first).
- Reverted to simpler prompt (no CoT) and disabled hint boosting (in LLM).
- Fixed LLM diagnostics logging and fallback rule failure reporting.
- Kept fallback re-prompting mechanism and increased retries.
- Includes fixes for weight calculation, config interpretation, variable passing, text input.

Version: affinity-rule-engine-v33.0.10 (Abstract CoT Prompt)
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
from typing import List, Dict, Optional, Tuple, Any, Set, TypedDict, Union

import numpy as np
# Ensure rdflib is imported
try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF
    from rdflib import util as rdflib_util
    RDFLIB_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: rdflib library not found (pip install rdflib).", file=sys.stderr)
    RDFLIB_AVAILABLE = False
    sys.exit(1) # Exit if rdflib is not available

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("CRITICAL ERROR: sentence-transformers library not found (pip install sentence-transformers).", file=sys.stderr)
    sys.exit(1)
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("CRITICAL ERROR: scikit-learn library not found (pip install scikit-learn).", file=sys.stderr)
    sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found (pip install tqdm), progress bars disabled.")
    # Dummy tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

# --- LLM Imports (with corrected optional handling) ---
class APITimeoutError(Exception): pass
class APIConnectionError(Exception): pass
class RateLimitError(Exception): pass

OpenAI = None; OPENAI_AVAILABLE = False
genai = None; GOOGLE_AI_AVAILABLE = False

try:
    from openai import OpenAI as RealOpenAI, APITimeoutError as RealAPITimeoutError, APIConnectionError as RealAPIConnectionError, RateLimitError as RealRateLimitError
    OpenAI = RealOpenAI; APITimeoutError = RealAPITimeoutError; APIConnectionError = RealAPIConnectionError; RateLimitError = RealRateLimitError
    OPENAI_AVAILABLE = True; logging.debug("Imported OpenAI library.")
except ImportError: logging.warning("OpenAI library not found. OpenAI functionality will be disabled.")
except Exception as e: logging.error(f"Unexpected error during OpenAI import: {e}")

try:
    import google.generativeai as genai_import
    genai = genai_import
    GOOGLE_AI_AVAILABLE = True; logging.debug("Imported google.generativeai library.")
except ImportError: logging.warning("google.generativeai library not found. Google AI functionality will be disabled.")
except Exception as e: GOOGLE_AI_AVAILABLE = False; genai = None; logging.error(f"Unexpected error during google.generativeai import: {e}")

# --- Base Utility Import ---
try: from utils import get_sbert_model
except ImportError: logging.critical("FATAL: Failed import 'get_sbert_model' from 'utils.py'. Ensure utils.py is accessible."); sys.exit(1)
except Exception as e: logging.critical(f"FATAL: Error importing from 'utils.py': {e}"); sys.exit(1)

# --- Logging ---
# Configured later in main based on args

# --- Config Defaults ---
SCRIPT_VERSION = "affinity-rule-engine-v33.0.10 (Abstract CoT Prompt)" # <-- Updated Version
CACHE_VERSION = "v20250421.affinity.33.0.10_default" # Default, overridden by config
DEFAULT_TAXONOMY_DIR = "./datasources/"
DEFAULT_CACHE_DIR = "./cache_v33/"
DEFAULT_CONFIG_FILE = "./affinity_config_v32.0.json" # Keep consistent unless changed
OUTPUT_FILE_TEMPLATE = "affinity_definitions_{cache_version}.json"
LOG_FILE_TEMPLATE = "affinity_generation_{cache_version}.log"
MAX_CANDIDATES_FOR_LLM = 75
INITIAL_CANDIDATE_POOL_SIZE = 150
EVIDENCE_MIN_SIMILARITY = 0.30
THEME_ATTRIBUTE_MIN_WEIGHT = 0.001
TRAVEL_CONTEXT = "travel "
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 5
LLM_TEMPERATURE = 0.2
LLM_RETRY_DELAY_SECONDS = 5

# --- List of known abstract/package concepts for tailored prompting ---
ABSTRACT_CONCEPTS_LIST = [
    "allinclusive",
    "allinclusivemeals",
    "allinclusiveresort",
    # Add other known abstract concepts here as needed (lowercase, normalized)
    "luxury",
    "budget",
    "value"
]

# --- Globals ---
_config_data: Optional[Dict] = None
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict[str, np.ndarray], List[str]]] = None
_keyword_label_index: Optional[Dict[str, Set[str]]] = None # Keyword -> Set[URI] index
_openai_client: Optional['OpenAI'] = None
_google_client: Optional[Any] = None
logger = logging.getLogger(__name__) # Define logger globally

# --- Type Hinting --- (Unchanged from v33.0.8)
class ThemeOverride(TypedDict, total=False): weight: float; rule: str; hints: Dict[str, List[str]]; description: str; subScore: str; fallback_logic: Optional[Dict]; rule_applied: str # Allow both rule keys
class ConceptOverride(TypedDict, total=False):
    lodging_type: str; category_type: str; exclusionary_concepts: List[str]
    must_not_have: List[Dict[str, str]] # Reflects config structure
    must_not_have_uris: List[str] # Keep for potential internal use, but populated from must_not_have
    theme_overrides: Dict[str, ThemeOverride]; additional_subscores: List[Dict[str, Union[str, float]]] # Corrected type

class BaseTheme(TypedDict):
    type: str
    rule: str # Internal script key
    rule_applied: str # Key used in config
    weight: float; subScore: Optional[str]; hints: Dict[str, List[str]]; description: Optional[str]; fallback_logic: Optional[Dict]

class AffinityConfig(TypedDict):
    config_version: str; description: str; base_themes: Dict[str, BaseTheme]; concept_overrides: Dict[str, ConceptOverride]
    master_subscore_list: List[str]; LLM_API_CONFIG: Dict; STAGE1_CONFIG: Dict; STAGE2_CONFIG: Dict; CACHE_VERSION: str
    LLM_PROVIDER: Optional[str]; LLM_MODEL: Optional[str]

# --- Utility Functions --- (Unchanged from v33.0.8)
def normalize_concept(concept: Optional[str]) -> str:
    """Normalizes concept text: lowercase, remove punctuation (except internal hyphens/underscores initially), split camelCase."""
    if not isinstance(concept, str) or not concept: return ""
    try:
        # Split camelCase first
        norm = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept)
        # Replace hyphens/underscores with spaces
        norm = norm.replace("-", " ").replace("_", " ")
        # Remove possessive 's and most punctuation, keep alphanumeric and spaces
        norm = re.sub(r'[^\w\s]|(\'s\b)', '', norm)
        # Lowercase and remove extra whitespace
        norm = ' '.join(norm.lower().split())
        return norm
    except Exception as e:
        logger.debug(f"Normalize regex failed: {e}")
        # Fallback to basic normalization
        return concept.lower().strip()

def get_primary_label(uri: str, fallback: Optional[str] = None) -> str:
    label = fallback
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        details = _taxonomy_concepts_cache[uri]
        # Prioritize specific labels
        for prop in ["skos:prefLabel", "rdfs:label"]: # Add rdfsLabel check if potentially used
            val_list = details.get(prop, [])
            # Handle cases where value might not be a list or might be empty
            if isinstance(val_list, list) and val_list and val_list[0]:
                return str(val_list[0]).strip()
            elif isinstance(val_list, str) and val_list.strip(): # Handle if it's just a string
                 return val_list.strip()

        # Fallback to altLabel
        alt_labels = details.get("skos:altLabel", [])
        if isinstance(alt_labels, list) and alt_labels and alt_labels[0]:
             return str(alt_labels[0]).strip()
        elif isinstance(alt_labels, str) and alt_labels.strip():
             return alt_labels.strip()

        # Further fallback to definition snippet if no fallback provided
        if fallback is None:
            definitions = details.get("skos:definition", [])
            if isinstance(definitions, list) and definitions and definitions[0]:
                definition_text = str(definitions[0]).strip()
                if definition_text:
                    label = definition_text[:60] + ("..." if len(definition_text) > 60 else "")
            elif isinstance(definitions, str) and definitions.strip():
                 definition_text = definitions.strip()
                 label = definition_text[:60] + ("..." if len(definition_text) > 60 else "")

    # Final fallback: Parse URI if no better label found
    if label is None or label == fallback:
        try:
            parsed_label = uri
            if '#' in uri: parsed_label = uri.split('#')[-1]
            elif '/' in uri: parsed_label = uri.split('/')[-1]
            # Check if parsing actually changed the string
            if parsed_label and parsed_label != uri:
                # Apply spacing for camelCase and replace underscores/hyphens
                parsed_label = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', parsed_label)
                parsed_label = parsed_label.replace('_', ' ').replace('-', ' ')
                # Capitalize words (simple title case)
                label = ' '.join(word.capitalize() for word in parsed_label.split() if word)
        except Exception as e:
            logger.debug(f"URI parsing failed for {uri}: {e}")
            # If URI parsing fails, the original fallback (or URI itself) remains

    # Return the best label found, or the fallback, or the original URI
    return label if label is not None else fallback if fallback is not None else uri


def get_concept_type_labels(uri: str) -> List[str]:
    type_labels = set()
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        type_uris = _taxonomy_concepts_cache[uri].get("type", [])
        if isinstance(type_uris, list):
            for type_uri_str in type_uris:
                if isinstance(type_uri_str, str): # Ensure it's a string URI
                     label = get_primary_label(type_uri_str) # Get label of the type URI
                     if label != type_uri_str: # Add only if a label was found (not just URI)
                         type_labels.add(label)
    return sorted(list(type_labels))

def get_theme_definition_for_prompt(theme_name: str, theme_data: BaseTheme) -> str:
    desc = theme_data.get("description")
    if isinstance(desc, str) and desc.strip(): return desc.strip()
    logger.warning(f"Theme '{theme_name}' missing description. Using fallback.")
    theme_type = theme_data.get("type", "general")
    return f"Theme related to {theme_name} ({theme_type} aspects of travel)."

def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: AffinityConfig) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    base_themes = config.get("base_themes", {}); concept_overrides_all = config.get("concept_overrides", {})
    base_data = base_themes.get(theme_name)
    if not base_data: logger.error(f"Base theme '{theme_name}' not found!"); return "Optional", 0.0, None, None
    concept_specific_overrides = concept_overrides_all.get(normalized_concept, {})
    theme_specific_override = concept_specific_overrides.get("theme_overrides", {}).get(theme_name, {})
    merged_data = base_data.copy(); merged_data.update(theme_specific_override)
    # Prioritize "rule_applied" from config structure, fallback to "rule"
    rule = merged_data.get("rule_applied", merged_data.get("rule", "Optional"))
    # Check against actual rule value in config: "Must have 1"
    if rule not in ["Must have 1", "Optional"]: rule = "Optional"
    weight = merged_data.get("weight", 0.0)
    if not isinstance(weight, (int, float)) or weight < 0: weight = 0.0
    subscore = merged_data.get("subScore"); fallback = merged_data.get("fallback_logic")
    return rule, float(weight), subscore, fallback

def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
    if not weights_dict: return {}
    total = sum(weights_dict.values())
    if total <= 0: logger.debug("Total weight non-positive."); return {k: 0.0 for k in weights_dict}
    return {k: v / total for k, v in weights_dict.items()}

def deduplicate_attributes(attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not attributes: return []
    best_attr_for_uri: Dict[str, Dict[str, Any]] = {}
    for attr in attributes:
        uri = attr.get("uri");
        try: weight = float(attr.get("concept_weight", 0.0))
        except (ValueError, TypeError): weight = 0.0
        if not uri or not isinstance(uri, str): continue
        current_best = best_attr_for_uri.get(uri)
        try: current_best_weight = float(current_best.get("concept_weight", -1.0)) if current_best else -1.0
        except (ValueError, TypeError): current_best_weight = -1.0
        if current_best is None or weight > current_best_weight: best_attr_for_uri[uri] = attr
    return list(best_attr_for_uri.values())

# --- Keyword Indexing --- (Unchanged from v33.0.8)
def build_keyword_label_index(taxonomy_concepts: Dict[str, Dict]) -> Optional[Dict[str, Set[str]]]:
    """Builds an inverted index from keywords in labels to URIs."""
    global _keyword_label_index
    if _keyword_label_index is not None:
        logger.info("Keyword label index already built.")
        return _keyword_label_index

    if not taxonomy_concepts:
        logger.error("Cannot build keyword index: Taxonomy concepts not loaded.")
        return None

    logger.info("Building keyword index from labels...")
    start_time = time.time()
    index: Dict[str, Set[str]] = defaultdict(set)
    # Properties to index for keywords - prefLabel, altLabel, rdfs:label, hiddenLabel
    label_props_to_index = ["skos:prefLabel", "rdfs:label", "skos:altLabel", "skos:hiddenLabel"]

    for uri, data in tqdm(taxonomy_concepts.items(), desc="Indexing Labels"):
        if not isinstance(data, dict): continue

        for prop in label_props_to_index:
            labels = data.get(prop, [])
            if isinstance(labels, str): # Handle single string value
                labels = [labels]
            if not isinstance(labels, list): continue # Skip if not list or string

            for label_text in labels:
                if not isinstance(label_text, str) or not label_text.strip(): continue
                # Normalize the label text and split into words
                normalized_label = normalize_concept(label_text) # Use the same normalization
                keywords = normalized_label.split()
                for keyword in keywords:
                    if len(keyword) > 2: # Optional: Ignore very short words (e.g., <= 2 chars)
                        index[keyword].add(uri)

    _keyword_label_index = dict(index) # Convert back to regular dict if needed, keep sets inside
    duration = time.time() - start_time
    logger.info(f"Keyword index built in {duration:.2f}s. Indexed {len(_keyword_label_index)} unique keywords.")
    return _keyword_label_index


# --- Loading Functions --- (load_affinity_config, load_taxonomy_concepts remain same as v33.0.8)
def load_affinity_config(config_file: str) -> Optional[AffinityConfig]:
    """Loads, validates slightly, and stores the affinity configuration globally."""
    global _config_data, LLM_MAX_RETRIES, LLM_RETRY_DELAY_SECONDS, INITIAL_CANDIDATE_POOL_SIZE, MAX_CANDIDATES_FOR_LLM, EVIDENCE_MIN_SIMILARITY, THEME_ATTRIBUTE_MIN_WEIGHT, CACHE_VERSION
    if _config_data is not None: return _config_data
    logger.info(f"Loading configuration from: {config_file}")
    if not os.path.exists(config_file): logger.critical(f"FATAL: Config file not found: '{config_file}'"); return None
    try:
        with open(config_file, 'r', encoding='utf-8') as f: config = json.load(f)
        # Validation
        if "base_themes" not in config: raise ValueError("Config missing 'base_themes'")
        if "concept_overrides" not in config: config["concept_overrides"] = {}
        if "LLM_API_CONFIG" not in config: config["LLM_API_CONFIG"] = {}
        if "STAGE1_CONFIG" not in config: config["STAGE1_CONFIG"] = {}
        if "STAGE2_CONFIG" not in config: config["STAGE2_CONFIG"] = {}
        for theme, data in config["base_themes"].items():
            if not data.get("description"): raise ValueError(f"Theme '{theme}' missing mandatory description.")
            # Harmonize rule keys if needed
            if "rule_applied" in data and "rule" not in data: data["rule"] = data["rule_applied"]
            elif "rule" in data and "rule_applied" not in data: data["rule_applied"] = data["rule"]
        # Update runtime params
        llm_api_cfg = config.get('LLM_API_CONFIG', {})
        LLM_MAX_RETRIES = int(llm_api_cfg.get('MAX_RETRIES', LLM_MAX_RETRIES))
        LLM_RETRY_DELAY_SECONDS = float(llm_api_cfg.get('RETRY_DELAY_SECONDS', LLM_RETRY_DELAY_SECONDS))
        stage1_cfg = config.get('STAGE1_CONFIG', {}); INITIAL_CANDIDATE_POOL_SIZE = int(stage1_cfg.get('INITIAL_CANDIDATE_POOL_SIZE', INITIAL_CANDIDATE_POOL_SIZE)); MAX_CANDIDATES_FOR_LLM = int(stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', MAX_CANDIDATES_FOR_LLM)); EVIDENCE_MIN_SIMILARITY = float(stage1_cfg.get('EVIDENCE_MIN_SIMILARITY', EVIDENCE_MIN_SIMILARITY))
        CACHE_VERSION = config.get('CACHE_VERSION', CACHE_VERSION); stage2_cfg = config.get('STAGE2_CONFIG', {}); THEME_ATTRIBUTE_MIN_WEIGHT = float(stage2_cfg.get('THEME_ATTRIBUTE_MIN_WEIGHT', THEME_ATTRIBUTE_MIN_WEIGHT))
        config["concept_overrides"] = {normalize_concept(k): v for k, v in config.get("concept_overrides", {}).items()}
        _config_data = config; logger.info(f"Config loaded. Version: {config.get('config_version', 'N/A')}")
        logger.info(f"Runtime Params: Retries={LLM_MAX_RETRIES}, Pool={INITIAL_CANDIDATE_POOL_SIZE}, LLM_Cand={MAX_CANDIDATES_FOR_LLM}, Cache={CACHE_VERSION}, MinSim={EVIDENCE_MIN_SIMILARITY}, MinAttrW={THEME_ATTRIBUTE_MIN_WEIGHT}")
        return _config_data
    except Exception as e: logger.critical(f"FATAL Error loading/validating config: {e}", exc_info=True); return None

def load_taxonomy_concepts(taxonomy_dir: str, cache_file: str, rebuild_cache: bool, current_cache_version: str, debug_mode: bool) -> Optional[Dict[str, Dict]]:
    """Loads taxonomy concepts from RDF files or cache, populating global cache."""
    global _taxonomy_concepts_cache
    if not RDFLIB_AVAILABLE: logger.critical("rdflib not available, cannot load taxonomy."); return None
    if _taxonomy_concepts_cache is not None: return _taxonomy_concepts_cache
    cache_valid = False
    if not rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Attempting to load concepts from cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f: cached_data = json.load(f)
            if cached_data.get("cache_version") == current_cache_version and isinstance(cached_data.get("data"), dict):
                _taxonomy_concepts_cache = cached_data["data"]; cache_valid = True
                logger.info(f"Loaded {len(_taxonomy_concepts_cache)} concepts from cache (Version: {current_cache_version}).")
            else: logger.info(f"Concept cache invalid/version mismatch. Rebuilding.")
        except Exception as e: logger.warning(f"Concept cache load failed: {e}. Rebuilding.")

    if not cache_valid:
        logger.info(f"Loading concepts from RDF files in directory: {taxonomy_dir}")
        start_time = time.time(); g = Graph() # Use a single graph
        files_parsed_count = 0; files_error_count = 0
        try:
            if not os.path.isdir(taxonomy_dir): raise FileNotFoundError(f"Taxonomy dir not found: {taxonomy_dir}")
            rdf_files_paths = []
            for root, _, files in os.walk(taxonomy_dir):
                 for file in files:
                      # Standard RDF formats + XML as rdflib can often handle RDF/XML
                      if file.endswith(('.ttl', '.rdf', '.owl', '.xml', '.jsonld', '.nt', '.n3')):
                           rdf_files_paths.append(os.path.join(root, file))
            if not rdf_files_paths: raise FileNotFoundError(f"No RDF files found in: {taxonomy_dir}")

            logger.info(f"Found {len(rdf_files_paths)} potential RDF files. Parsing...")
            disable_tqdm = not logger.isEnabledFor(logging.INFO) or debug_mode
            for filepath in tqdm(rdf_files_paths, desc="Parsing RDF", disable=disable_tqdm):
                try:
                    fmt = rdflib_util.guess_format(filepath)
                    g.parse(filepath, format=fmt) # Parse directly into main graph
                    files_parsed_count += 1
                except Exception as e_parse: # Catch generic Exception for parsing errors
                    files_error_count += 1
                    logger.error(f"Error parsing file '{os.path.basename(filepath)}': {e_parse}", exc_info=debug_mode)
                    # Continue to next file

            logger.info(f"Parsed {files_parsed_count}/{len(rdf_files_paths)} files ({files_error_count} errors).")
            if files_parsed_count == 0 and len(rdf_files_paths) > 0: raise RuntimeError("No RDF files parsed successfully.")

            logger.info("Extracting concepts from combined RDF graph...")
            extracted_data = defaultdict(lambda: defaultdict(list))
            all_uris_in_graph = set(s for s in g.subjects() if isinstance(s, URIRef))
            # Ensure all relevant props are included
            props = { "skos:prefLabel": SKOS.prefLabel, "skos:altLabel": SKOS.altLabel,
                      "rdfs:label": RDFS.label, "skos:definition": SKOS.definition,
                      "skos:scopeNote": SKOS.scopeNote, "type": RDF.type,
                      "skos:hiddenLabel": SKOS.hiddenLabel } # Add hiddenLabel

            for uri_ref in tqdm(all_uris_in_graph, desc="Processing URIs", disable=disable_tqdm):
                 if g.value(uri_ref, OWL.deprecated) == Literal(True): continue
                 uri_str = str(uri_ref); has_props = False
                 for key, p_uri in props.items():
                      for obj in g.objects(uri_ref, p_uri):
                           v = str(obj).strip() if isinstance(obj, (Literal, URIRef)) else None
                           if v: extracted_data[uri_str][key].append(v); has_props = True
                 # Add rdfs:label as a fallback if skos:prefLabel is missing but rdfs:label exists
                 if "skos:prefLabel" not in extracted_data[uri_str] and "rdfs:label" in extracted_data[uri_str]:
                     extracted_data[uri_str]["skos:prefLabel"] = extracted_data[uri_str]["rdfs:label"]
                 # Clean up properties that might just be the URI itself (like rdf:type) if no better label exists
                 if "type" in extracted_data[uri_str]:
                     extracted_data[uri_str]["type"] = [t for t in extracted_data[uri_str]["type"] if isinstance(t, str)]

                 if not has_props and uri_str in extracted_data: del extracted_data[uri_str] # Remove if no props found

            # Ensure labels are lists and sorted/unique
            processed_concepts = {}
            for uri, data in extracted_data.items():
                if not data: continue # Skip empty data
                processed_data = {}
                for k, v_list in data.items():
                    # Ensure v_list is actually a list, make unique and sort if appropriate
                    if isinstance(v_list, list):
                         unique_v = sorted(list(set(str(v) for v in v_list if v))) # Ensure string conversion and remove empties
                         if unique_v: processed_data[k] = unique_v
                    elif isinstance(v_list, str) and v_list.strip(): # Handle case where rdflib parsed single value as str
                         processed_data[k] = [v_list.strip()]

                if processed_data: processed_concepts[uri] = processed_data

            _taxonomy_concepts_cache = dict(processed_concepts)
            logger.info(f"Extracted {len(_taxonomy_concepts_cache)} concepts with relevant properties.")

            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_to_save = {"cache_version": current_cache_version, "data": _taxonomy_concepts_cache}
                with open(cache_file, 'w', encoding='utf-8') as f: json.dump(cache_to_save, f, indent=2)
                logger.info(f"Saved concepts cache successfully to: {cache_file}")
            except Exception as e: logger.error(f"Failed writing concepts cache: {e}")

            logger.info(f"Taxonomy loading and processing took {time.time() - start_time:.2f}s.")

        except FileNotFoundError as e: logger.critical(f"FATAL: Taxonomy directory or files error: {e}"); return None
        except RuntimeError as e: logger.critical(f"FATAL: RDF parsing failed: {e}"); return None
        except Exception as e: logger.error(f"Unexpected error during taxonomy load: {e}", exc_info=debug_mode); return None

    if not _taxonomy_concepts_cache: logger.error("Failed to load concepts into cache."); return None
    return _taxonomy_concepts_cache

# --- Embedding/KG Utils --- (precompute_taxonomy_embeddings, get_concept_embedding, get_batch_embedding_similarity, get_kg_data remain same as v33.0.8)
def precompute_taxonomy_embeddings(
    taxonomy_concepts: Dict[str, Dict],
    sbert_model: SentenceTransformer,
    cache_file: str,
    args: argparse.Namespace
) -> Optional[Tuple[Dict[str, np.ndarray], List[str]]]:
    global _taxonomy_embeddings_cache
    if _taxonomy_embeddings_cache is not None: logger.debug("Using cached embeddings from memory."); return _taxonomy_embeddings_cache
    if not taxonomy_concepts: logger.error("Concepts must be loaded first."); return None
    cache_valid = False; rebuild = args.rebuild_cache
    if not rebuild and os.path.exists(cache_file):
        logger.info(f"Attempting to load embeddings from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached = pickle.load(f)
            # Robust check: version, types, and dimensions
            if (cached.get("cache_version") == CACHE_VERSION and
                isinstance(cached.get("primary_embeddings"), dict) and
                isinstance(cached.get("uris_list"), list)):
                p_embs = cached["primary_embeddings"]; u_list = cached["uris_list"]
                sbert_dim = sbert_model.get_sentence_embedding_dimension()
                # Check if list isn't empty AND first element exists in dict AND has correct dimension
                if u_list and u_list[0] in p_embs and p_embs.get(u_list[0]) is not None and p_embs[u_list[0]].shape == (sbert_dim,):
                    _taxonomy_embeddings_cache = (p_embs, u_list); cache_valid = True
                    logger.info(f"Loaded {len(u_list)} primary embeddings from cache.")
                elif not u_list and not p_embs: # Handle empty cache correctly
                     _taxonomy_embeddings_cache = ({}, []); cache_valid = True
                     logger.info("Loaded empty embeddings from cache.")
                else: logger.warning("Embeddings cache dim mismatch/empty/key error. Rebuilding.")
            else: logger.info(f"Embeddings cache version mismatch or structure invalid. Rebuilding.")
        except Exception as e: logger.warning(f"Failed embedding cache load: {e}. Rebuilding.")

    if not cache_valid:
        logger.info("Pre-computing embeddings..."); start_time = time.time()
        primary_embeddings_map: Dict[str, Optional[np.ndarray]] = {}
        all_uris_processed = list(taxonomy_concepts.keys())
        # Map text -> list of (uri, property_key) where it came from
        texts_to_embed_map = defaultdict(list); properties_for_embedding = ["skos:prefLabel", "skos:altLabel", "rdfs:label", "skos:definition"] # Added rdfs:label just in case
        disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug; logger.info("Step 1: Gathering texts...")
        for uri in tqdm(all_uris_processed, desc="Gathering Texts", disable=disable_tqdm):
            data = taxonomy_concepts.get(uri, {}); found_texts_for_uri: Set[Tuple[str, str]] = set()
            for prop_key in properties_for_embedding:
                values = data.get(prop_key, []); value_list = values if isinstance(values, list) else [values] # Ensure list
                for text_value in value_list:
                    if text_value and isinstance(text_value, str):
                        normalized_text = normalize_concept(text_value.strip()) # Use consistent normalization
                        if normalized_text: found_texts_for_uri.add((prop_key, normalized_text))

            # Add unique text-property pairs to the map
            for prop_key, norm_text in found_texts_for_uri: texts_to_embed_map[norm_text].append((uri, prop_key))

        unique_texts_to_embed = list(texts_to_embed_map.keys()); embedded_text_map: Dict[str, Optional[np.ndarray]] = {}
        if unique_texts_to_embed:
            logger.info(f"Step 2: Embeddings for {len(unique_texts_to_embed)} unique texts...")
            try:
                embeddings_array = sbert_model.encode(unique_texts_to_embed, batch_size=128, show_progress_bar=(not disable_tqdm))
                for text, embedding in zip(unique_texts_to_embed, embeddings_array):
                    if embedding is not None and isinstance(embedding, np.ndarray): embedded_text_map[text] = embedding.astype(np.float32)
                    else: embedded_text_map[text] = None
            except Exception as e: logger.error(f"SBERT encoding failed: {e}", exc_info=True); return None
        else: logger.warning("No unique texts found to embed.")

        logger.info("Step 3: Selecting primary embedding..."); uri_embedding_candidates = defaultdict(dict)
        sbert_dim = sbert_model.get_sentence_embedding_dimension()
        # Populate candidates for each URI
        for norm_text, source_infos in texts_to_embed_map.items():
            embedding = embedded_text_map.get(norm_text)
            if embedding is None or embedding.shape != (sbert_dim,): continue
            for uri, prop_key in source_infos: uri_embedding_candidates[uri][prop_key] = embedding

        # Select the best embedding based on priority
        embedding_priority_order = ["skos:prefLabel", "rdfs:label", "skos:altLabel", "skos:definition"]; num_primary_found = 0
        for uri in tqdm(all_uris_processed, desc="Selecting Primary", disable=disable_tqdm):
            candidates = uri_embedding_candidates.get(uri, {}); chosen_embedding = None
            for prop_key in embedding_priority_order:
                if prop_key in candidates and candidates[prop_key] is not None: chosen_embedding = candidates[prop_key]; break
            primary_embeddings_map[uri] = chosen_embedding if (chosen_embedding is not None and isinstance(chosen_embedding, np.ndarray)) else None
            if primary_embeddings_map[uri] is not None: num_primary_found += 1

        # Finalize results
        final_uris_with_embeddings = [uri for uri in all_uris_processed if primary_embeddings_map.get(uri) is not None]
        final_primary_embeddings_dict = {uri: emb for uri, emb in primary_embeddings_map.items() if emb is not None}
        if not final_uris_with_embeddings: logger.warning("No primary embeddings determined."); _taxonomy_embeddings_cache = ({}, []); return _taxonomy_embeddings_cache # Return empty cache
        else: _taxonomy_embeddings_cache = (final_primary_embeddings_dict, final_uris_with_embeddings)

        logger.info(f"Primary embedding complete. {num_primary_found}/{len(all_uris_processed)} concepts have embedding.")
        logger.info(f"Embedding process took {time.time() - start_time:.2f}s.")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True); data_to_cache = {"cache_version": CACHE_VERSION, "primary_embeddings": _taxonomy_embeddings_cache[0], "uris_list": _taxonomy_embeddings_cache[1]}
            with open(cache_file, 'wb') as f: pickle.dump(data_to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved embeddings cache: {cache_file}")
        except Exception as e: logger.error(f"Failed writing embeddings cache: {e}")

    if not _taxonomy_embeddings_cache: logger.error("Embedding cache failed."); return None
    return _taxonomy_embeddings_cache


def get_concept_embedding(normalized_concept: str, model: Optional[SentenceTransformer]) -> Optional[np.ndarray]:
    if not model: logger.error("SBERT model unavailable."); return None;
    if not normalized_concept: return None
    try:
        logger.debug(f"Encoding: '{normalized_concept}'"); embedding = model.encode([normalized_concept], show_progress_bar=False)
        if embedding is not None and embedding.ndim == 2 and embedding.shape[0]==1: return embedding[0].astype(np.float32)
        else: logger.warning(f"SBERT invalid output shape for '{normalized_concept}'."); return None
    except Exception as e: logger.error(f"Embedding failed: {e}", exc_info=True); return None

def get_batch_embedding_similarity(concept_embedding: Optional[np.ndarray], candidate_embeddings_map: Dict[str, Optional[np.ndarray]]) -> Dict[str, float]:
    scores: Dict[str, float] = {};
    if concept_embedding is None or concept_embedding.ndim != 1: return scores;
    if not candidate_embeddings_map: return scores
    try:
        target_dim = concept_embedding.shape[0]; valid_embs, valid_uris = [], []
        for uri, emb in candidate_embeddings_map.items():
            if emb is not None and emb.ndim == 1 and emb.shape[0] == target_dim: valid_embs.append(emb); valid_uris.append(uri)
        if not valid_uris: return {}
        cand_array = np.array(valid_embs, dtype=np.float32); concept_2d = concept_embedding.reshape(1, -1)
        if concept_2d.shape[1] != cand_array.shape[1]: logger.error("Dim mismatch!"); return {}
        sims = cosine_similarity(concept_2d, cand_array)[0]
        return {uri: float(score) for uri, score in zip(valid_uris, sims)}
    except Exception as e: logger.error(f"Batch similarity error: {e}", exc_info=True); return {}

def get_kg_data(uris: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch structured data for a list of URIs from the KG cache."""
    results: Dict[str, Dict[str, Any]] = {}
    if not uris or _taxonomy_concepts_cache is None: return results
    for uri in uris:
        if not isinstance(uri, str): continue
        if uri in _taxonomy_concepts_cache:
            details = _taxonomy_concepts_cache[uri];
            pref_label = get_primary_label(uri, fallback=uri) # Use the refined get_primary_label
            results[uri] = {
                "uri": uri,
                "prefLabel": pref_label, # Keep simple key for easy access
                "skos:prefLabel": details.get("skos:prefLabel", [pref_label if pref_label != uri else None])[0], # Best guess prefLabel
                "skos:altLabel": details.get("skos:altLabel", []),
                "skos:definition": (details.get("skos:definition", []) or [None])[0], # First definition or None
                "skos:scopeNote": (details.get("skos:scopeNote", []) or [None])[0], # First scopeNote or None
                "type_labels": get_concept_type_labels(uri) # Get labels for types
            }
        # else: logger.debug(f"URI {uri} not found in taxonomy cache.") # Optional: log misses
    return results

# --- LLM Utils --- (get_openai_client, get_google_client, call_llm remain same as v33.0.8)
def get_openai_client() -> Optional['OpenAI']:
    global _openai_client
    if not OPENAI_AVAILABLE: return None
    if _openai_client is None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY");
            if not api_key: logger.warning("OPENAI_API_KEY missing."); return None
            _openai_client = OpenAI(api_key=api_key); logger.info("OpenAI client initialized.")
        except Exception as e: logger.error(f"OpenAI init failed: {e}"); return None
    return _openai_client

def get_google_client() -> Optional[Any]:
    global _google_client
    if not GOOGLE_AI_AVAILABLE: return None
    if _google_client is None:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY");
            if not api_key: logger.warning("GOOGLE_API_KEY missing."); return None
            genai.configure(api_key=api_key); logger.info("Google AI client configured."); _google_client = genai
        except Exception as e: logger.error(f"Google AI config failed: {e}"); return None
    return _google_client

def call_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float, provider: str) -> Dict[str, Any]:
    logger.info(f"LLM call via {provider}, model: {model_name}")
    llm_result = {"success": False, "response": None, "error": None, "attempts": 0}
    client_func = get_openai_client if provider == "openai" else get_google_client; client = client_func()
    if not client: llm_result["error"] = f"{provider} client unavailable"; logger.error(llm_result["error"]); return llm_result
    for attempt in range(max_retries + 1):
        llm_result["attempts"] = attempt + 1; logger.info(f"{provider} Call Attempt {attempt + 1}/{max_retries + 1}")
        try:
            content = None
            if provider == "openai":
                if not OPENAI_AVAILABLE or not isinstance(client, OpenAI): raise RuntimeError("OpenAI client unavailable.")
                response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=temperature, timeout=timeout)
                if response.choices and response.choices[0].message: content = response.choices[0].message.content
            elif provider == "google":
                if not GOOGLE_AI_AVAILABLE or client is None: raise RuntimeError("Google AI client unavailable.")
                model_name_full = model_name if model_name.startswith("models/") else f"models/{model_name}"
                gen_config = client.types.GenerationConfig(candidate_count=1, temperature=temperature, response_mime_type="application/json")
                safety = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
                model = client.GenerativeModel(model_name_full, system_instruction=system_prompt, safety_settings=safety)
                response = model.generate_content([user_prompt], generation_config=gen_config, request_options={'timeout': timeout})
                if not response.candidates:
                    block_reason = getattr(response.prompt_feedback, 'block_reason', 'N/A');
                    block_message = getattr(response.prompt_feedback, 'block_reason_message', '')
                    logger.warning(f"Gemini blocked: {block_reason} - {block_message}.");
                    llm_result["error"] = f"Blocked: {block_reason}";
                    return llm_result # Don't retry if blocked
                content = response.text
            else: llm_result["error"] = f"Unsupported provider: {provider}"; logger.error(llm_result["error"]); return llm_result
            if content:
                try:
                    cleaned = content.strip().strip('`').replace("```json","").replace("```","").strip(); llm_output = json.loads(cleaned)
                    if isinstance(llm_output, dict): llm_result["success"] = True; llm_result["response"] = llm_output; logger.info(f"{provider} OK (Attempt {attempt+1})."); return llm_result
                    else: logger.error(f"{provider} parsed non-dict (Attempt {attempt+1}). Type: {type(llm_output)}.")
                except json.JSONDecodeError as json_e: logger.error(f"{provider} JSON parse error (Attempt {attempt+1}): {json_e}")
            else: logger.warning(f"{provider} response empty (Attempt {attempt+1}).")
        except (APITimeoutError, APIConnectionError, RateLimitError) as api_e: logger.warning(f"{provider} API Error (Attempt {attempt+1}): {type(api_e).__name__}")
        except Exception as e: logger.error(f"{provider} Call Error (Attempt {attempt+1}): {e}", exc_info=True); llm_result["error"] = str(e); return llm_result # Return on unexpected errors
        if attempt >= max_retries: logger.error(f"Max retries reached."); llm_result["error"] = "Max retries reached"; return llm_result
        wait_time = LLM_RETRY_DELAY_SECONDS * (2**attempt) + np.random.uniform(0, 1); logger.info(f"Retrying in {wait_time:.2f}s..."); time.sleep(wait_time)
    return llm_result

# --- Prompt Building ---
# --- <<< MODIFIED function signature (v33.0.10) >>> ---
def construct_llm_slotting_prompt(
    input_concept: str,
    theme_definitions: List[Dict[str, Any]],
    candidate_details: List[Dict[str, Any]],
    args: argparse.Namespace,
    is_abstract_concept: bool = False # Add new parameter
) -> Tuple[str, str]:
# --- <<< END MODIFIED signature >>> ---

    # --- <<< System prompt (reverted to base) >>> ---
    system_prompt = """You are an expert travel taxonomist and semantic analyst tasked with defining travel concepts based on related evidence.
Your goal is to assign relevant 'Theme' labels to 'Candidate Evidence' concepts that help define the input 'Concept'.
You MUST adhere strictly to the provided list of Themes and Candidate URIs.
You MUST output ONLY a valid JSON object containing a single key "theme_assignments".
The value of "theme_assignments" MUST be a JSON object where keys are the Candidate URIs and values are lists of assigned Theme names.
Assign themes based on semantic relevance between the candidate's details (label, definition, type) and the theme's description.
If a theme is marked as (Mandatory), you MUST try to assign at least one candidate to it if semantically plausible.
Include ALL candidate URIs in the output JSON, even if they have no assigned themes (use an empty list []).
Do not include explanations or conversational text outside the JSON object."""
    # --- <<< END System prompt >>> ---

    # --- User prompt construction ---
    theme_defs_str = "\n".join([f"- **{t['name']}{' (Mandatory)' if t.get('is_must_have') else ''}**: {t['description']}" for t in theme_definitions])
    must_haves_str = ", ".join([t['name'] for t in theme_definitions if t.get('is_must_have')]) or "None"
    candidate_details_str = "\n".join([f"\n{i+1}. URI: {c['uri']}\n   PrefLabel: \"{c.get('prefLabel', 'N/A')}\"\n   AltLabels: {', '.join(c.get('skos:altLabel', []))}\n   Definition: \"{(c.get('skos:definition', '') or '')[:200]}\"\n   Types: {', '.join(c.get('type_labels', []))}" for i, c in enumerate(candidate_details)]) or "\nNo candidates."
    output_schema_example = """```json\n{\n  "theme_assignments": {\n    "URI_Candidate_1": ["ThemeName_A"],\n    "URI_Candidate_2": ["ThemeName_B", "ThemeName_C"],\n    "URI_Candidate_3": []\n    // ... include ALL candidate URIs ...\n  }\n}\n```"""

    # --- <<< MODIFIED: Conditional Task Instruction (v33.0.10) >>> ---
    if is_abstract_concept:
        task_instruction = f"""Task:
The input concept '{input_concept}' is an abstract travel style or package deal.
1. First, briefly list the typical amenities, activities, or characteristics commonly implied by '{input_concept}'. (Do this thinking step internally, DO NOT include it in the final JSON output).
2. Then, using both these implied characteristics AND the Candidate Evidence details (label, definition, types), determine which of the Available Themes are semantically relevant for EACH candidate URI provided above.
3. Construct a JSON object where each key is a candidate URI from the list above. The value for each URI key should be a list of strings, where each string is the name of a relevant theme from the 'Available Themes' list.
4. If no themes are relevant for a candidate URI, use an empty list `[]`.
5. Ensure EVERY candidate URI listed in the 'Candidate Evidence' section is included as a key in the output JSON."""
    else:
        task_instruction = f"""Task:
For EACH candidate URI provided above, determine which of the Available Themes are semantically relevant based on the candidate's details (label, definition, types) and the theme descriptions.
Construct a JSON object where each key is a candidate URI from the list above.
The value for each URI key should be a list of strings, where each string is the name of a relevant theme from the 'Available Themes' list.
If no themes are relevant for a candidate URI, use an empty list `[]`.
Ensure every candidate URI listed in the 'Candidate Evidence' section is included as a key in the output JSON."""
    # --- <<< END MODIFIED Task Instruction >>> ---


    user_prompt = f"""Please analyze the input travel concept: '{input_concept}'

Available Themes:
{theme_defs_str}
Mandatory themes that require at least one assignment if possible: [{must_haves_str}]

Candidate Evidence (Concepts related to '{input_concept}'):
{candidate_details_str}

{task_instruction} # Insert the chosen task instruction

Output ONLY the JSON object adhering to this schema:
{output_schema_example}"""
    return system_prompt.strip(), user_prompt.strip()


# --- build_reprompt_prompt function remains same as v33.0.8 ---
def build_reprompt_prompt(input_concept: str, theme_name: str, theme_config: BaseTheme, original_candidates_details_map: Dict[str, Dict]) -> Tuple[str, str]:
    system_prompt = """You are assisting in refining travel concept definitions. A previous step failed to assign any candidates to a mandatory theme.
Your task is to re-evaluate the original list of candidates specifically for the given mandatory theme ONLY.
Identify ANY candidates from the list that are semantically relevant to this single theme based on its description and hints.
Output ONLY a valid JSON object containing a single key "theme_assignments".
The value of "theme_assignments" MUST be a JSON object where keys are the URIs of the candidates relevant to this specific theme, and the value for each key MUST be a list containing ONLY the single target theme name.
If NO candidates are relevant, output `{"theme_assignments": {}}`.
Do not include explanations or conversational text outside the JSON object."""
    theme_desc = theme_config.get('description', 'N/A'); hints = theme_config.get("hints", {})
    hints_str = f"  Hints: {json.dumps(hints)}" if isinstance(hints, dict) and (hints.get("keywords") or hints.get("uris")) else ""
    cand_list = "\n".join([f"\n{i+1}. URI: {uri}\n   Label: \"{cand.get('prefLabel', 'N/A')}\"\n   Types: {', '.join(cand.get('type_labels', []))}" for i, (uri, cand) in enumerate(original_candidates_details_map.items())]) or "\nNo candidates."
    reprompt_output_schema = f"""```json\n{{\n  "theme_assignments": {{\n    "URI_Relevant_1": ["{theme_name}"]\n    // ... include ONLY candidates relevant to {theme_name} ...\n  }}\n}}\n```"""
    user_prompt = f"""Re-evaluating concept '{input_concept}' specifically for the MANDATORY theme '{theme_name}'.

Theme Details:
- Name: {theme_name}
- Description: {theme_desc}
{hints_str}

Original Candidate List:
{cand_list}

Instructions:
Carefully review the 'Original Candidate List'. Identify ALL candidate URIs that are semantically relevant to the theme '{theme_name}' based on its description and any hints provided.
You MUST assign at least one candidate if semantically plausible.
Output ONLY a JSON object containing the key "theme_assignments". The value should be an object mapping the relevant candidate URIs to a list containing only the theme name '{theme_name}'.
If absolutely no candidates are relevant to '{theme_name}', output `{{ "theme_assignments": {{}} }}`.

Example Output (if URI_Relevant_1 and URI_Relevant_3 are relevant):
{reprompt_output_schema}

Output:"""
    return system_prompt.strip(), user_prompt.strip()


# --- Stage 1: Evidence Preparation --- (Unchanged from v33.0.8)
def prepare_evidence(
    input_concept: str,
    concept_embedding: Optional[np.ndarray],
    primary_embeddings: Dict[str, np.ndarray],
    config: AffinityConfig,
    args: argparse.Namespace
) -> Tuple[List[Dict], Dict[str, Dict], Optional[Dict], Dict[str, float]]:
    """
    Prepares evidence candidates using embedding similarity and keyword matching.
    v33.0.8: Adds keyword boost using pre-built index, prioritizes keyword matches.
    """
    global _keyword_label_index # Access the pre-built index
    normalized_concept = normalize_concept(input_concept)
    logger.info(f"Starting evidence prep for: '{normalized_concept}' (Keyword Boost Enabled)")

    stage1_cfg = config.get('STAGE1_CONFIG', {})
    initial_pool_size = int(stage1_cfg.get('INITIAL_CANDIDATE_POOL_SIZE', 150))
    max_candidates_for_llm = int(stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', 75))
    min_similarity = float(stage1_cfg.get('EVIDENCE_MIN_SIMILARITY', 0.3))

    candidates_for_llm_details: List[Dict] = []
    original_candidates_map: Dict[str, Dict] = {} # Will store details of candidates selected for LLM
    anchor_candidate: Optional[Dict] = None
    candidate_evidence_scores: Dict[str, float] = {} # URI -> Similarity Score

    # --- Step 1: Embedding Similarity Pool ---
    if concept_embedding is None:
        logger.error("No embedding for input concept. Cannot use similarity.")
        similarity_candidates_uris = set()
    else:
        # Get similarity scores for all concepts with embeddings
        candidate_evidence_scores = get_batch_embedding_similarity(concept_embedding, primary_embeddings)
        # Filter by minimum similarity
        filtered_similarity_candidates = {uri: score for uri, score in candidate_evidence_scores.items() if score >= min_similarity}
        if not filtered_similarity_candidates:
            logger.warning(f"No candidates found via similarity >= {min_similarity}.")
            similarity_candidates_uris = set()
        else:
            # Sort by similarity and take the top N for the initial pool
            sorted_uris_by_sim = sorted(filtered_similarity_candidates, key=filtered_similarity_candidates.get, reverse=True)
            similarity_candidates_uris = set(sorted_uris_by_sim[:initial_pool_size])
            logger.info(f"Found {len(similarity_candidates_uris)} candidates in initial similarity pool (top {initial_pool_size} >= {min_similarity}).")

    # --- Step 2: Keyword Matching Pool ---
    keyword_matched_uris = set()
    if _keyword_label_index is None:
        logger.warning("Keyword label index not built. Skipping keyword matching.")
    else:
        # Extract keywords from the input concept
        input_keywords = set(kw for kw in normalized_concept.split() if len(kw) > 2) # Use same length filter as index build
        logger.debug(f"Input keywords for matching: {input_keywords}")
        if not input_keywords:
            logger.warning(f"No usable keywords extracted from '{normalized_concept}'.")
        else:
            # Find all URIs matching any of the input keywords
            for keyword in input_keywords:
                if keyword in _keyword_label_index:
                    keyword_matched_uris.update(_keyword_label_index[keyword])
            logger.info(f"Found {len(keyword_matched_uris)} candidates via keyword matching.")

    # --- Step 3: Combine Pools and Select Candidates ---
    combined_candidate_uris = similarity_candidates_uris.union(keyword_matched_uris)
    if not combined_candidate_uris:
        logger.warning(f"No candidates found from either similarity or keyword matching for '{normalized_concept}'.")
        return [], {}, None, candidate_evidence_scores # Return empty

    logger.info(f"Total unique candidates from similarity & keywords: {len(combined_candidate_uris)}")

    # Fetch details for all combined candidates
    all_candidate_details_map = get_kg_data(list(combined_candidate_uris))

    # Prepare a list of candidates with scores and keyword match status
    processed_candidates = []
    for uri in combined_candidate_uris:
        if uri in all_candidate_details_map:
            details = all_candidate_details_map[uri].copy()
            details['similarity_score'] = candidate_evidence_scores.get(uri, 0.0) # Use 0.0 if only found by keyword
            details['keyword_match'] = uri in keyword_matched_uris
            processed_candidates.append(details)
        # else: logger.debug(f"Details not found in KG for candidate URI: {uri}") # Optional

    if not processed_candidates:
        logger.warning("Could not retrieve KG details for any candidates.")
        return [], {}, None, candidate_evidence_scores

    # --- Step 4: Rank and Select (Option A: Strict Priority for Keywords) ---
    keyword_candidates = [c for c in processed_candidates if c['keyword_match']]
    similarity_only_candidates = [c for c in processed_candidates if not c['keyword_match']]

    # Sort keyword matches by similarity (descending)
    keyword_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
    # Sort similarity-only matches by similarity (descending)
    similarity_only_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

    selected_candidates_details: List[Dict] = []
    selected_uris: Set[str] = set()

    # Add keyword candidates first, up to the limit
    for cand in keyword_candidates:
        if len(selected_candidates_details) >= max_candidates_for_llm: break
        if cand['uri'] not in selected_uris: # Should always be true if combined_candidate_uris was unique
            selected_candidates_details.append(cand)
            selected_uris.add(cand['uri'])

    logger.info(f"Selected {len(selected_candidates_details)} candidates based on keyword match (sorted by similarity).")

    # Fill remaining slots with top similarity-only candidates
    fill_needed = max_candidates_for_llm - len(selected_candidates_details)
    if fill_needed > 0:
        added_from_sim = 0
        for cand in similarity_only_candidates:
            if len(selected_candidates_details) >= max_candidates_for_llm: break
            if cand['uri'] not in selected_uris:
                selected_candidates_details.append(cand)
                selected_uris.add(cand['uri'])
                added_from_sim += 1
        logger.info(f"Filled remaining {added_from_sim} slots with top similarity-only candidates.")

    # --- Step 5: Determine Anchor and Finalize ---
    # Re-sort the final selected list purely by similarity for anchor selection (optional, but consistent)
    selected_candidates_details.sort(key=lambda x: x['similarity_score'], reverse=True)
    anchor_candidate = selected_candidates_details[0] if selected_candidates_details else None

    logger.info(f"Final selection: {len(selected_candidates_details)} candidates for LLM. Anchor: {anchor_candidate['uri'] if anchor_candidate else 'None'}")

    # original_candidates_map should contain details ONLY for the candidates sent to LLM
    original_candidates_map = {c['uri']: c for c in selected_candidates_details}

    return selected_candidates_details, original_candidates_map, anchor_candidate, candidate_evidence_scores


# --- Finalization Function --- (apply_rules_and_finalize remains same as v33.0.8)
def apply_rules_and_finalize(
    input_concept: str,
    llm_call_result: Optional[Dict[str, Any]], # Corrected type: The whole result dict
    config: AffinityConfig,
    travel_category: Optional[Dict],
    anchor_candidate: Optional[Dict], # Anchor candidate details
    original_candidates_map_for_reprompt: Dict[str, Dict],
    candidate_evidence_scores: Dict[str, float],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Applies rules, handles fallbacks (reporting), calculates weights, and structures final output.
    v33.0.8: No changes needed here, relies on inputs from modified prepare_evidence.
    """
    start_time = time.time(); normalized_concept = normalize_concept(input_concept)
    logger.info(f"Starting rule application and finalization for: {normalized_concept}")

    stage2_output: Dict[str, Any] = {
        "applicable_lodging_types": "Both",
        "travel_category": travel_category or {"uri": None, "name": input_concept, "type": "Unknown"}, # Initialize correctly
        "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [],
        "must_not_have": [], "failed_fallback_themes": {},
        "diagnostics": {"theme_processing": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "final_output": {}, "stage2": {"status": "Started", "duration_seconds": 0.0, "error": None} } }
    theme_processing_diagnostics = stage2_output["diagnostics"]["theme_processing"]
    reprompt_diag = stage2_output["diagnostics"]["reprompting_fallback"]

    base_themes = config.get("base_themes", {}); concept_overrides = config.get("concept_overrides", {}).get(normalized_concept, {})
    finalization_cfg = config.get('STAGE2_CONFIG', {}); theme_attribute_min_weight = float(finalization_cfg.get('THEME_ATTRIBUTE_MIN_WEIGHT', 0.001)); top_n_defining_attributes = int(finalization_cfg.get('TOP_N_DEFINING_ATTRIBUTES', 25))

    parsed_llm_assignments: Dict[str, List[str]] = {}
    # Check success status *before* trying to access response
    if llm_call_result and llm_call_result.get("success"):
        response_data = llm_call_result.get("response", {}); raw_assignments = response_data.get("theme_assignments", {})
        if isinstance(raw_assignments, dict):
            validated_assignments = {}; valid_themes_set = set(base_themes.keys()); uris_sent = set(original_candidates_map_for_reprompt.keys())
            for uri, themes in raw_assignments.items():
                if uri in uris_sent and isinstance(themes, list): validated_assignments[uri] = [t for t in themes if isinstance(t, str) and t in valid_themes_set]
            parsed_llm_assignments = validated_assignments; logger.debug(f"[{normalized_concept}] Parsed LLM Assignments: {json.dumps(parsed_llm_assignments, indent=2)}")
        else: logger.warning(f"[{normalized_concept}] LLM response['theme_assignments'] not a dict.")
    elif llm_call_result: # If the result exists but wasn't successful
        logger.warning(f"[{normalized_concept}] LLM call was not successful, cannot parse assignments. Error: {llm_call_result.get('error')}")
    else: # If no LLM call was made (e.g., provider 'none' or no candidates)
        logger.info(f"[{normalized_concept}] No LLM call result available for parsing assignments.")

    # Build initial map based *only* on successfully parsed LLM assignments
    theme_to_assigned_uris = defaultdict(list)
    for uri, themes_list in parsed_llm_assignments.items():
        if uri in original_candidates_map_for_reprompt: # Ensure URI was one we sent
            for theme_name in themes_list:
                if theme_name in base_themes: theme_to_assigned_uris[theme_name].append(uri)
    logger.debug(f"[{normalized_concept}] Initial Theme to Assigned URIs Map (Pre-Fallback): {json.dumps(dict(theme_to_assigned_uris), indent=2)}")

    failed_must_have_initial: Dict[str, Dict] = {}
    for theme_name in base_themes.keys():
        # Initialize diagnostics for every theme
        diag = theme_processing_diagnostics[theme_name] = {"llm_assigned_count": len(theme_to_assigned_uris.get(theme_name, [])), "attributes_after_weighting": 0, "status": "Pending", "rule_failed": False}
        rule, _, _, _ = get_dynamic_theme_config(normalized_concept, theme_name, config)
        # Check for 'Must have 1' from config
        # Check if the rule is 'Must have 1' AND if the theme is *not* in the initial assignment map
        if rule == "Must have 1" and theme_name not in theme_to_assigned_uris:
            logger.warning(f"[{normalized_concept}] Initial Rule Check FAILED: '{rule}' theme '{theme_name}' has no assignments.")
            failed_must_have_initial[theme_name] = {"reason": "No attributes assigned by initial LLM."}; diag.update({"status": "Failed Rule (Initial)", "rule_failed": True})

    themes_fixed_by_fallback = set(); fallback_added_attributes_details = []
    if failed_must_have_initial and original_candidates_map_for_reprompt:
        logger.info(f"[{normalized_concept}] Attempting fallback for {len(failed_must_have_initial)} themes: {list(failed_must_have_initial.keys())}")
        for theme_name_to_fix in list(failed_must_have_initial.keys()): # Iterate over a copy
            reprompt_diag["attempts"] += 1; theme_base_config = base_themes.get(theme_name_to_fix)
            if not theme_base_config: logger.error(f"[{normalized_concept}] Base theme config not found for fallback theme '{theme_name_to_fix}'."); reprompt_diag["failures"] += 1; continue
            logger.info(f"[{normalized_concept}] Building fallback re-prompt for: {theme_name_to_fix}")
            reprompt_sys, reprompt_user = build_reprompt_prompt(input_concept, theme_name_to_fix, theme_base_config, original_candidates_map_for_reprompt)
            fallback_llm_result = call_llm(reprompt_sys, reprompt_user, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)
            if fallback_llm_result and fallback_llm_result.get("success"):
                fallback_assignments = fallback_llm_result.get("response", {}).get("theme_assignments", {}); newly_assigned_uris = set()
                if isinstance(fallback_assignments, dict):
                    for uri, assigned_themes in fallback_assignments.items():
                        # Check if the list contains the specific theme we asked for and uri is valid
                        if isinstance(assigned_themes, list) and theme_name_to_fix in assigned_themes and uri in original_candidates_map_for_reprompt:
                            newly_assigned_uris.add(uri)
                if newly_assigned_uris:
                    logger.info(f"[{normalized_concept}] Fallback SUCCESS for '{theme_name_to_fix}': Assigned {len(newly_assigned_uris)} URIs.")
                    reprompt_diag["successes"] += 1; themes_fixed_by_fallback.add(theme_name_to_fix)
                    existing_uris_for_theme = set(theme_to_assigned_uris.get(theme_name_to_fix, [])); added_count = 0
                    for uri in newly_assigned_uris:
                         # Add to the main map if not already present for this theme
                         if uri not in existing_uris_for_theme:
                              theme_to_assigned_uris[theme_name_to_fix].append(uri) # Add URI to the list for this theme
                              if uri in original_candidates_map_for_reprompt:
                                  details = original_candidates_map_for_reprompt[uri]
                                  # Store details needed for later weighting, only if added
                                  fallback_added_attributes_details.append({"uri": uri, "skos:prefLabel": details.get('prefLabel', get_primary_label(uri, uri)), "concept_weight": candidate_evidence_scores.get(uri, 0.01), "type": details.get('type_labels', []), "assigned_theme": theme_name_to_fix, "comment": "Added via Fallback"})
                              added_count +=1
                         # else: logger.debug(f"URI {uri} already present for {theme_name_to_fix}, not adding again via fallback.")
                    logger.debug(f"[{normalized_concept}] Added {added_count} unique URIs via fallback for '{theme_name_to_fix}'.")
                    if theme_name_to_fix in theme_processing_diagnostics: theme_processing_diagnostics[theme_name_to_fix].update({"status": "Passed (Fallback)", "rule_failed": False})
                else:
                    logger.warning(f"[{normalized_concept}] Fallback LLM call for '{theme_name_to_fix}' succeeded but assigned 0 relevant candidates."); reprompt_diag["failures"] += 1
                    if theme_name_to_fix in theme_processing_diagnostics: theme_processing_diagnostics[theme_name_to_fix]["status"] = "Failed (Fallback - No Assigns)"
            else:
                fallback_error = fallback_llm_result.get("error", "Unknown error") if fallback_llm_result else "Fallback LLM result was None"
                logger.error(f"[{normalized_concept}] Fallback LLM call failed for '{theme_name_to_fix}'. Error: {fallback_error}"); reprompt_diag["failures"] += 1
                if theme_name_to_fix in theme_processing_diagnostics: theme_processing_diagnostics[theme_name_to_fix]["status"] = "Failed (Fallback - API Error)"

    final_themes_output = []; all_final_attributes = []
    final_theme_weights_config = {name: get_dynamic_theme_config(normalized_concept, name, config)[1] for name in base_themes.keys()}
    normalized_final_theme_weights = normalize_weights(final_theme_weights_config)

    # Now iterate through all themes again to build the final output
    for theme_name, base_theme_data in base_themes.items():
        diag = theme_processing_diagnostics[theme_name]; final_rule, _, final_subscore, _ = get_dynamic_theme_config(normalized_concept, theme_name, config)
        normalized_weight_for_theme = normalized_final_theme_weights.get(theme_name, 0.0)
        assigned_uris_final = theme_to_assigned_uris.get(theme_name, []) # Get the final list of URIs post-fallback
        diag["llm_assigned_count"] = len(assigned_uris_final) # Update count post-fallback

        # Recalculate scores only for the URIs *currently* assigned to this theme
        scores_in_theme = {uri: candidate_evidence_scores.get(uri, 0.0) for uri in assigned_uris_final if uri in candidate_evidence_scores}
        total_initial_score_in_theme = sum(scores_in_theme.values())
        logger.debug(f"[{normalized_concept}][{theme_name}] Final URIs: {len(assigned_uris_final)}, TotalScore: {total_initial_score_in_theme:.4f}, ThemeWeight: {normalized_weight_for_theme:.4f}")

        final_attributes_for_theme = []
        if assigned_uris_final and normalized_weight_for_theme > 0 and total_initial_score_in_theme > 0:
            num_uris_in_theme = len(assigned_uris_final) # Needed for fallback weighting if score is 0
            for uri in assigned_uris_final: # Iterate over all URIs now associated with the theme
                if uri not in original_candidates_map_for_reprompt: continue # Skip if we lost details somehow

                details = original_candidates_map_for_reprompt[uri]; initial_score = scores_in_theme.get(uri, 0.0)
                # Calculate proportion based on score within this theme
                proportion = (initial_score / total_initial_score_in_theme) if total_initial_score_in_theme > 0 else (1.0 / num_uris_in_theme) # Equal split if total score is 0
                final_attribute_weight = normalized_weight_for_theme * proportion
                # logger.debug(f"  [{uri}] Score:{initial_score:.4f}, Prop:{proportion:.4f}, FinalW:{final_attribute_weight:.6f}") # Verbose logging

                if final_attribute_weight >= theme_attribute_min_weight:
                     # Check if this URI was added via fallback for *this* theme
                     is_fallback_add = any(fb_attr["uri"] == uri and fb_attr["assigned_theme"] == theme_name for fb_attr in fallback_added_attributes_details)
                     final_attr = {
                         "uri": uri,
                         "skos:prefLabel": details.get('prefLabel', get_primary_label(uri, uri)),
                         "concept_weight": round(final_attribute_weight, 6),
                         "type": details.get('type_labels', [])
                     }
                     if is_fallback_add:
                         final_attr["comment"] = "Added via Fallback"
                         # logger.debug(f"    -> Attrib added (Fallback): {final_attr}")
                     # else: logger.debug(f"    -> Attrib added (Initial): {final_attr}")

                     final_attributes_for_theme.append(final_attr);
                     # Add to the *overall* list for top defining attributes calculation
                     all_final_attributes.append(final_attr)
                # else: logger.debug(f"    -> Attrib skipped (Weight < {theme_attribute_min_weight})")

        final_attributes_for_theme.sort(key=lambda x: x['concept_weight'], reverse=True)
        diag["attributes_after_weighting"] = len(final_attributes_for_theme)
        if diag["status"] == "Pending": # If rule didn't fail initially and wasn't fixed by fallback
             diag["status"] = "Processed (Initial)" if not diag.get("rule_failed") else diag["status"] # Keep failed status if it failed initially and wasn't fixed


        # Add the theme block to the output
        final_themes_output.append({
            "theme_name": theme_name, "theme_type": base_theme_data.get("type", "unknown"),
            "rule_applied": final_rule, "normalized_theme_weight": round(normalized_weight_for_theme, 6),
            "subScore": final_subscore or f"{theme_name}Affinity", "llm_summary": None, # llm_summary not implemented here
            "attributes": final_attributes_for_theme })

    stage2_output["themes"] = final_themes_output

    # Calculate Top Defining Attributes from the *combined* list across all themes
    unique_attributes_map: Dict[str, Dict[str, Any]] = {}
    for attr in all_final_attributes:
         uri = attr["uri"]; current_weight = attr["concept_weight"]
         # Keep the version of the attribute with the highest weight if it appears in multiple themes
         if uri not in unique_attributes_map or current_weight > unique_attributes_map[uri].get("concept_weight", -1):
              unique_attributes_map[uri] = {k: v for k, v in attr.items() if k != 'comment'} # Don't include comment in top list
    sorted_top_attributes = sorted(unique_attributes_map.values(), key=lambda x: x['concept_weight'], reverse=True)
    stage2_output['top_defining_attributes'] = sorted_top_attributes[:top_n_defining_attributes]

    # --- Final Overrides & Counts ---
    stage2_output["applicable_lodging_types"] = concept_overrides.get("lodging_type", "Both")
    # Apply overrides to the travel_category object
    tc = stage2_output["travel_category"] # Use the initialized object
    if isinstance(tc, dict):
         tc["type"] = concept_overrides.get("category_type", tc.get("type", "Uncategorized"))
         tc["exclusionary_concepts"] = concept_overrides.get("exclusionary_concepts", [])
    else: # Should not happen, but defensively re-initialize
         logger.error(f"[{normalized_concept}] travel_category lost its dict structure? Re-initializing.")
         stage2_output["travel_category"] = {"uri": None, "name": normalized_concept, "type": concept_overrides.get("category_type", "Uncategorized"), "exclusionary_concepts": concept_overrides.get("exclusionary_concepts", [])}

    # Read must_not_have correctly
    must_not_have_list_of_dicts = concept_overrides.get("must_not_have", [])
    must_not_have_uris = set()
    if isinstance(must_not_have_list_of_dicts, list):
        for item in must_not_have_list_of_dicts:
            if isinstance(item, dict) and "uri" in item: must_not_have_uris.add(item["uri"])
    else: logger.warning(f"[{normalized_concept}] Invalid must_not_have structure in config.")
    stage2_output["must_not_have"] = [{"uri": uri, "skos:prefLabel": get_primary_label(uri, uri), "scope": None} for uri in sorted(list(must_not_have_uris))]

    additional_scores_config = concept_overrides.get("additional_subscores", [])
    stage2_output["additional_relevant_subscores"] = additional_scores_config if isinstance(additional_scores_config, list) else []

    # Final diagnostic counts
    final_diag_counts = stage2_output["diagnostics"]["final_output"]
    final_diag_counts["must_not_have_count"] = len(stage2_output["must_not_have"])
    final_diag_counts["additional_subscores_count"] = len(stage2_output["additional_relevant_subscores"])
    final_diag_counts["themes_count"] = len(stage2_output["themes"])
    # Record themes that failed the 'Must have 1' rule initially and were *not* fixed by fallback
    stage2_output["failed_fallback_themes"] = { name: reason for name, reason in failed_must_have_initial.items() if name not in themes_fixed_by_fallback }
    final_diag_counts["failed_fallback_themes_count"] = len(stage2_output["failed_fallback_themes"])
    final_diag_counts["top_defining_attributes_count"] = len(stage2_output['top_defining_attributes'])

    stage2_output["diagnostics"]["stage2"]["status"] = "Completed"; stage2_output["diagnostics"]["stage2"]["duration_seconds"] = round(time.time() - start_time, 2)
    logger.info(f"[{normalized_concept}] Rule application/finalization finished in {stage2_output['diagnostics']['stage2']['duration_seconds']:.2f}s.")
    return stage2_output

# --- Main Loop ---
def generate_affinity_definitions_loop(
    concepts_to_process: List[str],
    config: AffinityConfig,
    args: argparse.Namespace,
    sbert_model: SentenceTransformer,
    primary_embeddings: Dict[str, np.ndarray] # Only need the map
):
    """Main loop processing each concept through evidence prep, LLM slotting, and finalization."""
    all_definitions = [] # <-- This will store the final results for each concept
    effective_cache_version = config.get("CACHE_VERSION", "unknown_v33")

    if _taxonomy_concepts_cache is None: logger.critical("FATAL: Taxonomy concepts cache not loaded."); return []
    if _keyword_label_index is None: logger.warning("Keyword label index not available. Stage 1 keyword boost will be skipped.") # Warn if index missing

    limit = args.limit if args.limit is not None and args.limit > 0 else len(concepts_to_process)
    concepts_subset = concepts_to_process[:limit]; logger.info(f"Will process {len(concepts_subset)} concepts.")

    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    for concept in tqdm(concepts_subset, desc="Processing Concepts", disable=disable_tqdm):
        start_concept_time = time.time(); normalized_concept = normalize_concept(concept)
        logger.info(f"=== Processing Concept: '{normalized_concept}' ===")

        affinity_definition: Dict[str, Any] = {
            "input_concept": concept, "normalized_concept": normalized_concept,
            "applicable_lodging_types": "Both", "travel_category": {},
            "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [],
            "must_not_have": [], "failed_fallback_themes": {},
            "processing_metadata": {"status": "Started", "version": SCRIPT_VERSION, "timestamp": None, "duration_seconds": 0.0, "total_duration_seconds": 0.0, "cache_version": effective_cache_version, "llm_provider": args.llm_provider, "llm_model": args.llm_model if args.llm_provider != "none" else None},
            "diagnostics": { "stage1": {"status": "Not Started", "error": None}, "llm_slotting": {"status": "Not Started", "error": None}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "stage2": {"status": "Not Started", "error": None}, "theme_processing": {}, "final_output": {}, "error_details": None } }
        diag_llm = affinity_definition["diagnostics"]["llm_slotting"] # Local ref

        try:
            logger.debug(f"Getting embedding for anchor: '{TRAVEL_CONTEXT + normalized_concept}'")
            concept_embedding = get_concept_embedding(TRAVEL_CONTEXT + normalized_concept, sbert_model)
            if concept_embedding is None: raise ValueError(f"Embedding failed for '{normalized_concept}'")

            # --- Stage 1: Prepare Evidence (Now uses keyword boost) ---
            stage1_start_time = time.time()
            candidates_for_llm_details, original_candidates_map, anchor_candidate, candidate_evidence_scores = prepare_evidence(
                 concept, concept_embedding, primary_embeddings, config, args )
            stage1_duration = time.time() - stage1_start_time
            affinity_definition["diagnostics"]["stage1"] = {
                "status": "Completed", "duration_seconds": round(stage1_duration, 2),
                "candidate_evidence_count_initial_pool": len(candidate_evidence_scores), # Might be larger now due to keywords
                "candidate_evidence_count_for_llm": len(candidates_for_llm_details),
                "error": None }
            logger.info(f"Stage 1 completed in {stage1_duration:.2f}s. Found {len(candidates_for_llm_details)} candidates for LLM.")

            # Initialize travel_category based on anchor
            if anchor_candidate and anchor_candidate.get('uri'):
                 anchor_details_kg = get_kg_data([anchor_candidate['uri']]).get(anchor_candidate['uri'])
                 affinity_definition["travel_category"] = anchor_details_kg if anchor_details_kg else {"uri": anchor_candidate['uri'], "name": anchor_candidate.get('prefLabel', concept), "type": "Unknown"} # Fallback with label if KG fetch fails
            else:
                 affinity_definition["travel_category"] = {"uri": None, "name": concept, "type": "Unknown"}
                 logger.warning(f"No anchor candidate found for {normalized_concept}. Initial travel_category is basic.")


            # --- Stage 1.5: LLM Slotting ---
            llm_call_result = None # Initialize result for this concept
            llm_stage_start_time = time.time()
            diag_llm.update({"llm_provider": args.llm_provider, "llm_model": args.llm_model if args.llm_provider != "none" else None})

            if not candidates_for_llm_details:
                logger.warning(f"No candidates available for LLM processing for '{concept}'.");
                affinity_definition["processing_metadata"]["status"] = "Warning - No LLM Candidates";
                diag_llm["status"] = "Skipped (No Candidates)"
            elif args.llm_provider == "none":
                logger.info(f"LLM provider set to 'none'. Skipping LLM call for '{concept}'.");
                diag_llm["status"] = "Skipped (Provider None)"
            else:
                diag_llm["status"] = "Started"; diag_llm["llm_call_attempted"] = True
                # --- <<< Check if concept is abstract >>> ---
                is_abstract = normalized_concept in ABSTRACT_CONCEPTS_LIST
                if is_abstract:
                    logger.info(f"Concept '{normalized_concept}' identified as abstract, using tailored prompt instructions.")
                # --- <<< END NEW >>> ---

                # Prepare theme definitions for the prompt
                active_theme_configs_prompt = {}
                for theme_name, base_data in config.get("base_themes", {}).items():
                     rule, _, _, _ = get_dynamic_theme_config(normalized_concept, theme_name, config) # Get rule specific to this concept
                     active_theme_configs_prompt[theme_name] = {
                         "name": theme_name,
                         "description": get_theme_definition_for_prompt(theme_name, base_data),
                         "is_must_have": rule == "Must have 1" # Dynamically set mandatory flag
                     }
                active_themes_prompt_list = list(active_theme_configs_prompt.values())

                # Construct and make the LLM call
                # --- <<< MODIFIED: Pass is_abstract flag >>> ---
                system_prompt, user_prompt = construct_llm_slotting_prompt(
                    concept,
                    active_themes_prompt_list,
                    candidates_for_llm_details,
                    args,
                    is_abstract_concept=is_abstract # Pass the flag
                )
                # --- <<< END MODIFIED >>> ---
                llm_call_result = call_llm(system_prompt, user_prompt, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)
                diag_llm["attempts_made"] = llm_call_result.get("attempts", 0)

                # Process LLM result
                if llm_call_result and llm_call_result.get("success"):
                    diag_llm["llm_call_success"] = True; raw_assignments = llm_call_result.get("response", {}).get("theme_assignments", {})
                    if isinstance(raw_assignments, dict):
                        # Basic validation counts - more detailed validation happens in finalize
                        parsed_count = 0; uris_in_response = 0
                        for uri, themes in raw_assignments.items():
                             uris_in_response += 1
                             parsed_count += len(themes) if isinstance(themes, list) else 0
                        diag_llm["assignments_parsed_count"] = parsed_count; diag_llm["uris_in_response_count"] = uris_in_response; diag_llm["status"] = "Completed"
                        logger.info(f"LLM call OK for '{concept}'. Found {parsed_count} assignments across {uris_in_response} URIs in response.")
                    else:
                        diag_llm["status"] = "Failed (Invalid Format)"; diag_llm["error"] = "LLM response['theme_assignments'] not a dict."; logger.error(f"LLM format error for '{concept}': 'theme_assignments' not dict.")
                else:
                    diag_llm["llm_call_success"] = False; diag_llm["status"] = "Failed (API Error/Parse)"; diag_llm["error"] = llm_call_result.get("error", "Unknown"); logger.warning(f"LLM call failed for '{concept}'. Error: {diag_llm['error']}")

            diag_llm["duration_seconds"] = round(time.time() - llm_stage_start_time, 2)
            logger.info(f"LLM Slotting stage took {diag_llm['duration_seconds']:.2f}s. Status: {diag_llm['status']}")

            # --- Stage 2: Apply Rules and Finalize ---
            stage2_start_time = time.time()
            # Pass the *entire* result dict from call_llm
            stage2_output_data = apply_rules_and_finalize(
                 concept,
                 llm_call_result,
                 config,
                 affinity_definition["travel_category"], # Pass the category initialized earlier
                 anchor_candidate,
                 original_candidates_map, # Map of candidates sent to LLM
                 candidate_evidence_scores, # Similarity scores for weighting
                 args )
            stage2_duration = time.time() - stage2_start_time

            # --- Merge Stage 2 results into the main definition ---
            affinity_definition.update({ k: v for k, v in stage2_output_data.items() if k != 'diagnostics' }) # Update top-level keys

            # Carefully merge diagnostics from stage 2
            if "diagnostics" in stage2_output_data:
                stage2_diags = stage2_output_data["diagnostics"]
                affinity_definition["diagnostics"]["theme_processing"] = stage2_diags.get("theme_processing", {})
                affinity_definition["diagnostics"]["reprompting_fallback"] = stage2_diags.get("reprompting_fallback", {"attempts": 0, "successes": 0, "failures": 0})
                affinity_definition["diagnostics"]["final_output"] = stage2_diags.get("final_output", {})
                affinity_definition["diagnostics"]["stage2"] = stage2_diags.get("stage2", {"status": "Completed", "duration_seconds": round(stage2_duration, 2), "error": None}) # Use the dict from stage2 directly
                # Ensure duration is updated if stage2 provides it correctly
                affinity_definition["diagnostics"]["stage2"]["duration_seconds"] = round(stage2_duration, 2)

            else:
                affinity_definition["diagnostics"]["stage2"] = {"status": "Completed", "duration_seconds": round(stage2_duration, 2), "error": None }
            logger.info(f"Stage 2 (Finalization) completed in {stage2_duration:.2f}s.")

            # --- Final Status Determination ---
            current_status = affinity_definition["processing_metadata"]["status"]
            if current_status == "Started": # Only update if not already set to a warning/error
                if affinity_definition.get("failed_fallback_themes"):
                    affinity_definition["processing_metadata"]["status"] = "Success with Failed Rules"
                    logger.warning(f"Concept '{concept}' completed but failed mandatory rules: {list(affinity_definition['failed_fallback_themes'].keys())}")
                elif diag_llm["status"].startswith("Failed"):
                    affinity_definition["processing_metadata"]["status"] = "Warning - LLM Slotting Failed"
                    logger.warning(f"Concept '{concept}' completed but LLM slotting failed: {diag_llm.get('error')}")
                elif diag_llm["status"].startswith("Skipped"):
                    affinity_definition["processing_metadata"]["status"] = "Success (LLM Skipped)"
                    logger.info(f"Concept '{concept}' completed successfully, LLM was skipped.")
                else:
                    affinity_definition["processing_metadata"]["status"] = "Success"
                    logger.info(f"Concept '{concept}' completed successfully.")


        except Exception as e:
            logger.error(f"Core processing failed unexpectedly for '{concept}': {e}", exc_info=True)
            affinity_definition["processing_metadata"]["status"] = f"FATAL - Exception: {type(e).__name__}"
            if "diagnostics" not in affinity_definition: affinity_definition["diagnostics"] = {}
            affinity_definition["diagnostics"]["error_details"] = traceback.format_exc()
            # Mark stages as failed if they hadn't started/finished
            if affinity_definition["diagnostics"].get("stage1", {}).get("status") == "Not Started": affinity_definition["diagnostics"]["stage1"] = {"status": "Failed", "error": str(e)}
            if affinity_definition["diagnostics"].get("llm_slotting", {}).get("status") == "Not Started": affinity_definition["diagnostics"]["llm_slotting"] = {"status": "Failed", "error": str(e)}
            if affinity_definition["diagnostics"].get("stage2", {}).get("status") == "Not Started": affinity_definition["diagnostics"]["stage2"] = {"status": "Failed", "error": str(e)}

        finally:
            # --- Final Metadata Update ---
            end_concept_time = time.time(); total_duration = round(end_concept_time - start_concept_time, 2)
            affinity_definition["processing_metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # duration_seconds is deprecated, use total_duration_seconds
            affinity_definition["processing_metadata"]["duration_seconds"] = total_duration
            affinity_definition["processing_metadata"]["total_duration_seconds"] = total_duration
            # Ensure LLM diagnostics exist even if skipped/failed early
            if "llm_slotting" not in affinity_definition["diagnostics"]:
                 affinity_definition["diagnostics"]["llm_slotting"] = diag_llm if 'diag_llm' in locals() else {"status": "Not Run"}

            # Append the completed definition (or error state) to the list
            all_definitions.append(affinity_definition)
            logger.info(f"--- Finished processing '{normalized_concept}' in {total_duration:.2f}s. Status: {affinity_definition['processing_metadata']['status']} ---")

    return all_definitions

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description=f"Generate Travel Concept Affinity Definitions ({SCRIPT_VERSION})")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to the JSON configuration file.")
    parser.add_argument("-t", "--taxonomy_dir", type=str, default=DEFAULT_TAXONOMY_DIR, help="Directory containing RDF taxonomy files.")
    # Updated help text for input file type
    parser.add_argument("-i", "--input_concepts_file", type=str, required=True, help="Path to a text file containing one concept per line.")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to save the output JSON and log files.")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory for storing/reading cache files.")
    parser.add_argument("--rebuild_cache", action='store_true', help="Force rebuild of concept and embedding caches.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of concepts to process.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
    # LLM specific overrides (optional)
    parser.add_argument("--llm_provider", type=str, choices=['openai', 'google', 'none'], default=None, help="Override LLM provider (openai, google, none). Uses config if not set.")
    parser.add_argument("--llm_model", type=str, default=None, help="Override LLM model name. Uses config if not set.")

    args = parser.parse_args()

    # --- Setup Output Paths & Logging ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Determine final CACHE_VERSION (config takes precedence)
    global CACHE_VERSION # Allow modification
    temp_config = load_affinity_config(args.config) # Load config early to get cache version
    if temp_config and temp_config.get("CACHE_VERSION"):
        CACHE_VERSION = temp_config["CACHE_VERSION"]
        logger.info(f"Using CACHE_VERSION from config: {CACHE_VERSION}")
    else:
        logger.warning(f"CACHE_VERSION not found in config, using default: {CACHE_VERSION}")
    _config_data = None # Reset config cache so it reloads with potentially updated runtime params

    log_filename = LOG_FILE_TEMPLATE.format(cache_version=CACHE_VERSION)
    log_filepath = os.path.join(args.output_dir, log_filename)
    output_filename = OUTPUT_FILE_TEMPLATE.format(cache_version=CACHE_VERSION)
    output_filepath = os.path.join(args.output_dir, output_filename)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Remove existing handlers before adding new ones
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format,
                        handlers=[logging.FileHandler(log_filepath, mode='w', encoding='utf-8'),
                                  logging.StreamHandler()]) # Log to file and console

    logger.info(f"Starting {SCRIPT_VERSION}")
    logger.info(f"Full command: {' '.join(sys.argv)}") # Log the command used
    logger.info(f"Using config file: {args.config}")
    logger.info(f"Taxonomy directory: {args.taxonomy_dir}")
    logger.info(f"Input concepts file: {args.input_concepts_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Output file: {output_filepath}")
    logger.info(f"Rebuild cache: {args.rebuild_cache}")
    logger.info(f"Limit concepts: {args.limit if args.limit else 'None'}")
    logger.info(f"Debug mode: {args.debug}")

    # --- Load Configuration ---
    config = load_affinity_config(args.config)
    if config is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # Apply command-line LLM overrides if provided
    if args.llm_provider:
        logger.warning(f"Overriding LLM provider from command line: '{args.llm_provider}'")
        config['LLM_PROVIDER'] = args.llm_provider
    if args.llm_model:
        logger.warning(f"Overriding LLM model from command line: '{args.llm_model}'")
        config['LLM_MODEL'] = args.llm_model

    # Set effective provider/model in args for consistent use in functions
    args.llm_provider = config.get('LLM_PROVIDER', 'none') # Default to none if not in config
    args.llm_model = config.get('LLM_MODEL', None)
    logger.info(f"Effective LLM Provider: {args.llm_provider}")
    logger.info(f"Effective LLM Model: {args.llm_model}")

    # --- Load Input Concepts (MODIFIED FOR TEXT FILE) ---
    try:
        logger.info(f"Reading concepts as text lines from: {args.input_concepts_file}")
        with open(args.input_concepts_file, 'r', encoding='utf-8') as f:
            # Read each line, strip leading/trailing whitespace,
            # and keep only non-empty lines.
            input_concepts = [line.strip() for line in f if line.strip()]

        if not input_concepts:
            # Handle case where file exists but is empty or only contains whitespace
            raise ValueError("Input concepts file is empty or contains only whitespace after stripping.")

        logger.info(f"Loaded {len(input_concepts)} concepts from {args.input_concepts_file} (read as text lines).")

    except FileNotFoundError:
        logger.critical(f"Input concepts file not found: '{args.input_concepts_file}'")
        sys.exit(1)
    except ValueError as ve: # Catch the explicit ValueError raised above
        logger.critical(f"Error processing input concepts file '{args.input_concepts_file}': {ve}")
        sys.exit(1)
    except Exception as e:
        # Catch other potential errors like permission issues during file read
        logger.critical(f"Failed to read input concepts file '{args.input_concepts_file}': {e}")
        sys.exit(1)
    # --- END OF MODIFIED SECTION ---

    # --- Load/Build Taxonomy Concepts Cache ---
    concepts_cache_file = os.path.join(args.cache_dir, f"concepts_cache_{CACHE_VERSION}.json")
    taxonomy_concepts = load_taxonomy_concepts(args.taxonomy_dir, concepts_cache_file, args.rebuild_cache, CACHE_VERSION, args.debug)
    if taxonomy_concepts is None:
        logger.critical("Failed to load taxonomy concepts. Exiting.")
        sys.exit(1)
    logger.info(f"Taxonomy concepts ready ({len(taxonomy_concepts)} concepts).")


    # --- Build Keyword Index ---
    if build_keyword_label_index(taxonomy_concepts) is None:
         logger.warning("Failed to build keyword index. Keyword boost will be disabled.")
         # Note: prepare_evidence already handles the case where _keyword_label_index is None

    # --- Load SBERT Model ---
    logger.info("Loading SBERT model...")
    try:
        sbert_model = get_sbert_model() # Assumes utils.py handles model loading
        if sbert_model is None: raise RuntimeError("get_sbert_model() returned None")
        logger.info(f"SBERT model loaded successfully. Dimension: {sbert_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.critical(f"Failed to load SBERT model via utils.py: {e}", exc_info=True)
        sys.exit(1)

    # --- Load/Build Embeddings Cache ---
    embeddings_cache_file = os.path.join(args.cache_dir, f"embeddings_cache_{CACHE_VERSION}.pkl")
    embeddings_data = precompute_taxonomy_embeddings(taxonomy_concepts, sbert_model, embeddings_cache_file, args)
    if embeddings_data is None:
        logger.critical("Failed to load or compute embeddings. Exiting.")
        sys.exit(1)
    primary_embeddings_map, uris_with_embeddings = embeddings_data
    logger.info(f"Taxonomy embeddings ready ({len(uris_with_embeddings)} concepts with embeddings).")

    # --- Generate Definitions ---
    logger.info("Starting affinity definition generation loop...")
    start_time = time.time()
    all_results = generate_affinity_definitions_loop(
        input_concepts,
        config,
        args,
        sbert_model,
        primary_embeddings_map
    )
    end_time = time.time()
    logger.info(f"Affinity definition generation loop finished in {end_time - start_time:.2f} seconds.")

    # --- Save Results ---
    logger.info(f"Saving {len(all_results)} results to {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save results to {output_filepath}: {e}")

    logger.info(f"Script finished. Log file at: {log_filepath}")

if __name__ == "__main__":
    # Check for required environment variables early if needed
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
         print("Warning: Neither OPENAI_API_KEY nor GOOGLE_API_KEY environment variables are set. LLM functionality may be limited.", file=sys.stderr)
         # Depending on strictness, you might exit here if an LLM provider is chosen later that needs a key
    main()