#!/usr/bin/env python3
"""
Generate affinity definitions for travel concepts (v30.1r3 - Args Passing Corrected) - Config-Driven Rule Engine + LLM Assist:
1. Loads detailed configuration (Themes, Rules, Weights, Exclusions) from external JSON. Validates subscores.
2. Stage 1: Identifies anchor concepts, gathers candidate evidence, uses LLM for enrichment suggestions.
3. Stage 2: Python engine applies rules, weights, normalizes, handles VR Sentiment rule,
   selects theme attributes (Keyword+Semantic), deduplicates attributes, adds attribute types,
   populates themes/subscores, validates/finalizes must_not_have, sources exclusionary_concepts from config,
   generates top_defining_attributes list, records failed fallbacks, and assembles final JSON.

Version: 2025-04-17-affinity-rule-engine-v30.1r3 (Args Passing Corrected)
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
# Standard Typing Imports
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from copy import deepcopy
# Compatibility Import for TypedDict (Python 3.7)
try: from typing import TypedDict
except ImportError:
    try: from typing_extensions import TypedDict; print("Info: Using TypedDict from typing_extensions.")
    except ImportError: print("CRITICAL ERROR: TypedDict not found. Install 'typing-extensions'.", file=sys.stderr); sys.exit(1)
# Required ML/NLP Imports
try: from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, util
except ImportError: print("CRITICAL ERROR: rdflib library not found.", file=sys.stderr); sys.exit(1)
try: from sentence_transformers import SentenceTransformer
except ImportError: print("CRITICAL ERROR: sentence-transformers library not found.", file=sys.stderr); sys.exit(1)
try: from sklearn.metrics.pairwise import cosine_similarity
except ImportError: print("CRITICAL ERROR: scikit-learn library not found.", file=sys.stderr); sys.exit(1)
# Optional Imports (NLTK, tqdm)
NLTK_AVAILABLE = False; STOP_WORDS = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'to', 'and', 'or', 'is', 'are', 'was', 'were'])
try: from nltk.corpus import stopwords; STOP_WORDS = set(stopwords.words('english')); NLTK_AVAILABLE = True; print("Info: NLTK stopwords loaded.")
except (ImportError, LookupError) as e: print(f"Warning: NLTK stopwords not found ({type(e).__name__}). Using basic list.")
def tqdm_dummy(iterable, *args, **kwargs): return iterable
tqdm = tqdm_dummy
try: from tqdm import tqdm as real_tqdm; tqdm = real_tqdm; print("Info: tqdm loaded.")
except ImportError: print("Warning: tqdm not found, progress bars disabled.")
# OpenAI Import
try: from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
except ImportError: print("CRITICAL ERROR: openai library not found.", file=sys.stderr); sys.exit(1)


# --- Logging Setup ---
log_filename = "affinity_generation_v30.1.log" # Keep consistent log name for v30.1
if os.path.exists(log_filename):
    try: os.remove(log_filename); print(f"Removed old log file: {log_filename}")
    except OSError as e: print(f"Warning: Could not remove old log file {log_filename}: {e}")
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", handlers=[file_handler, stream_handler])
logger = logging.getLogger("affinity_generator_v30_1") # Consistent logger name

# --- Namespaces ---
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#"); RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/"); OWL = Namespace("http://www.w3.org/2002/07/owl#")

# --- Configuration & Constants ---
CACHE_VERSION = "v20250417.affinity.30.1r3" # <-- v30.1r3 Cache version
DEFAULT_ASSOCIATIONS_FILE = "./datasources/concept_associations.json"
DEFAULT_CORPUS_FILE = "./datasources/travel-terms-corpus.txt"
DEFAULT_CONFIG_FILE = "./affinity_config.json"

KG_RETRIEVAL_TOP_K = 150; KG_RETRIEVAL_SIMILARITY_THRESHOLD = 0.35
ANCHOR_MATCH_MIN_RELEVANCE = 0.30; MAX_ANCHOR_MATCHES = 5
OPENAI_MODEL = "gpt-4o"; OPENAI_TIMEOUT = 180; OPENAI_MAX_RETRIES = 2
LLM_TEMPERATURE = 0.1;
THEME_ATTRIBUTE_SEMANTIC_THRESHOLD = 0.40
THEME_ATTRIBUTE_KEYWORD_MIN_MATCH = 1
TOP_DEFINING_ATTRIBUTES_COUNT = 7
VR_SENTIMENT_BASELINE_WEIGHT = 0.01

TRAVEL_CONTEXT = "travel "; PROPERTY_WEIGHTS_INTERNAL = {"prefLabel": 1.0, "altLabel": 0.95, "rdfsLabel": 0.9, "definition": 0.80, "dctermsDescription": 0.80, "scopeNote": 0.75, "hiddenLabel": 0.7}
KEYWORD_WEIGHT_INTERNAL = 0.50; SEMANTIC_WEIGHT_INTERNAL = 0.50
EXACT_MATCH_BONUS_INTERNAL = 0.01; ASSOCIATION_BOOST_INTERNAL = 0.40

# --- Caches & Globals ---
_model_instance: Optional[SentenceTransformer] = None; _taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict, Dict, List]] = None; _corpus_data_cache: Optional[Dict[str, np.ndarray]] = None
_concept_associations_cache: Optional[Dict[str, List[str]]] = None; _theme_embeddings_cache: Optional[Dict[str, Dict]] = None
_openai_client = None; _config_data: Optional[Dict[str, Any]] = None

# --- Type Hinting for Config Structure ---
class ThemeConfig(TypedDict): type: str; rule: str; weight: float; subScore: Optional[str]; hints: List[str]; opposite_hints: Optional[List[str]]; opposite_uris: Optional[List[str]]; fallback_logic: Optional[Dict[str, Any]]
class ConceptOverrideConfig(TypedDict): theme_overrides: Optional[Dict[str, Dict[str, Any]]]; must_not_have_uris: Optional[List[str]]; category_type: Optional[str]; exclusionary_concepts: Optional[List[str]]; lodging_type: Optional[str]; additional_subscores: Optional[Dict[str, float]]
class AffinityConfig(TypedDict): base_themes: Dict[str, ThemeConfig]; concept_overrides: Dict[str, ConceptOverrideConfig]; master_subscore_list: List[str]

# --- Utility Functions ---
def get_sbert_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        model_name = 'all-MiniLM-L6-v2'; logger.info(f"Loading SBERT model ('{model_name}')...")
        start_time = time.time()
        try: _model_instance = SentenceTransformer(model_name)
        except Exception as e: logger.error(f"SBERT load failed: {e}", exc_info=True); raise RuntimeError("SBERT Model loading failed") from e
        logger.info("Loaded SBERT model in %.2f s", time.time() - start_time)
    return _model_instance

def normalize_concept(concept: Optional[str]) -> str:
    if not isinstance(concept, str) or not concept: return ""
    try: norm = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept); norm = norm.replace("-", " ").replace("_", " "); norm = re.sub(r'[^\w\s]|(\'s\b)', '', norm); norm = ' '.join(norm.lower().split()); return norm
    except Exception: return concept.lower().strip() if isinstance(concept, str) else ""

def calculate_semantic_similarity(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray], term1: str = "term1", term2: str = "term2") -> float:
    if embedding1 is None or embedding2 is None or embedding1.ndim == 0 or embedding2.ndim == 0: return 0.0
    try:
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray): return 0.0
        emb1 = embedding1.reshape(1, -1) if embedding1.ndim == 1 else embedding1; emb2 = embedding2.reshape(1, -1) if embedding2.ndim == 1 else embedding2
        if emb1.shape[0] == 0 or emb1.shape[1] == 0 or emb2.shape[0] == 0 or emb2.shape[1] == 0: return 0.0
        if emb1.shape[1] != emb2.shape[1]: logger.error(f"Embedding dimension mismatch: {term1}{emb1.shape} vs {term2}{emb2.shape}"); return 0.0
        sim = cosine_similarity(emb1, emb2)[0][0]; return max(0.0, min(1.0, float(sim)))
    except Exception as e: logger.warning(f"Similarity calculation error '{term1[:50]}' vs '{term2[:50]}': {e}"); return 0.0

def get_primary_label(uri: str, fallback: Optional[str] = None) -> str:
    label = fallback
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        details = _taxonomy_concepts_cache[uri]
        if details.get("prefLabel"): return details["prefLabel"][0];
        if details.get("altLabel"): return details["altLabel"][0];
        if details.get("rdfsLabel"): return details["rdfsLabel"][0]
        if label is None and details.get("definition"): label = details["definition"][0][:60] + "..."
    if label is None or label == fallback:
        try:
            if '#' in uri: parsed_label = uri.split('#')[-1]
            elif '/' in uri: parsed_label = uri.split('/')[-1]
            else: parsed_label = uri
            if parsed_label != uri: label = parsed_label
        except Exception: pass
    return label if label is not None else uri

def get_concept_type(uri: str) -> Optional[str]:
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        type_uris = _taxonomy_concepts_cache[uri].get("type", [])
        if type_uris: return get_primary_label(type_uris[0])
    return None

# --- Loading Functions ---
def load_affinity_config(config_file: str) -> Optional[AffinityConfig]:
    global _config_data
    if _config_data is not None: return _config_data
    logger.info(f"Loading affinity configuration from: {config_file}")
    if not os.path.exists(config_file): logger.critical(f"FATAL: Config file not found: {config_file}"); return None
    try:
        with open(config_file, 'r', encoding='utf-8') as f: config = json.load(f)
        required_theme_keys = ["type", "rule", "weight", "hints"]; master_subscore_list = config.get("master_subscore_list", [])
        assert "base_themes" in config and isinstance(config["base_themes"], dict), "Config missing 'base_themes'"
        for theme, theme_data in config["base_themes"].items():
            if not all(k in theme_data for k in required_theme_keys): raise ValueError(f"Theme '{theme}' missing required keys: {required_theme_keys}")
            subscore = theme_data.get("subScore")
            if subscore:
                if not isinstance(subscore, str): raise ValueError(f"Theme '{theme}' subScore must be a string.")
                if master_subscore_list and subscore not in master_subscore_list: logger.warning(f"Theme '{theme}' subScore '{subscore}' not in master_subscore_list.")
        config["concept_overrides"] = config.get("concept_overrides", {}); assert isinstance(config["concept_overrides"], dict), "'concept_overrides' invalid"
        config["master_subscore_list"] = master_subscore_list; assert isinstance(config["master_subscore_list"], list), "'master_subscore_list' invalid"
        config["concept_overrides"] = {normalize_concept(k): v for k, v in config["concept_overrides"].items()}
        _config_data = config; logger.info(f"Successfully loaded config for {len(config['base_themes'])} base themes.")
        return _config_data
    except Exception as e: logger.critical(f"FATAL: Error loading config file {config_file}: {e}", exc_info=True); return None















# Replace the ENTIRE load_taxonomy_concepts function with this:
def load_taxonomy_concepts(taxonomy_dir: str, cache_file: str, args: argparse.Namespace) -> Optional[Dict[str, Dict]]:
    """Loads taxonomy concepts from RDF files or cache."""
    global _taxonomy_concepts_cache
    if _taxonomy_concepts_cache is not None: return _taxonomy_concepts_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Loading concepts cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f: cached_data = json.load(f)
            if cached_data.get("cache_version") == CACHE_VERSION and isinstance(cached_data.get("data"), dict):
                _taxonomy_concepts_cache = cached_data["data"]; cache_valid = True; logger.info(f"Loaded {len(_taxonomy_concepts_cache)} concepts from cache.")
            else: logger.info("Cache version/data invalid. Rebuilding.")
        except Exception as e: logger.warning(f"Cache load failed: {e}. Rebuilding.")

    if not cache_valid:
        logger.info(f"Loading concepts from RDF: {taxonomy_dir}"); start_time = time.time(); g = Graph(); files_ok = 0; total_err = 0
        try:
            if not os.path.isdir(taxonomy_dir): raise FileNotFoundError(f"Taxonomy dir not found: {taxonomy_dir}")
            rdf_files = [f for f in os.listdir(taxonomy_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.xml', '.jsonld', '.nt', '.n3'))]; assert rdf_files, f"No RDF files found in {taxonomy_dir}"
            logger.info(f"Parsing {len(rdf_files)} files..."); disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug

            # --- CORRECTED PARSING LOOP ---
            for fn in tqdm(rdf_files, desc="Parsing RDF", disable=disable_tqdm):
                fp = os.path.join(taxonomy_dir, fn)
                try:
                    fmt = util.guess_format(fp)
                    if fmt:
                        logger.debug(f"Parsing {fn} with format {fmt}")
                        g.parse(fp, format=fmt)
                        files_ok += 1
                    else:
                        logger.warning(f"Could not guess format for {fn}, skipping.")
                except Exception as e:
                    total_err += 1
                    logger.error(f"Error parsing file {fn}: {e}", exc_info=args.debug)
            # --- END CORRECTED PARSING LOOP ---

            logger.info(f"Parsed {files_ok}/{len(rdf_files)} files ({total_err} errors)."); assert files_ok > 0, "No RDF files parsed."
            logger.info("Extracting concepts..."); kept_concepts_data = defaultdict(lambda: defaultdict(list)); pot_uris = set(s for s, p, o in g if isinstance(s, URIRef)) | set(o for s, p, o in g if isinstance(o, URIRef)); logger.info(f"Found {len(pot_uris)} potential URIs. Processing...")
            lbl_props = {SKOS.prefLabel: "prefLabel", SKOS.altLabel: "altLabel", RDFS.label: "rdfsLabel", SKOS.hiddenLabel: "hiddenLabel"}; txt_props = {SKOS.definition: "definition", DCTERMS.description: "dctermsDescription", SKOS.scopeNote: "scopeNote"}; rel_props = {SKOS.broader: "broader", SKOS.narrower: "narrower", SKOS.related: "related", OWL.sameAs: "sameAs"}; type_prop = {RDF.type: "type"}; all_props = {**lbl_props, **txt_props, **rel_props, **type_prop}; skip_dep = 0; rem_empty = 0
            for uri in tqdm(pot_uris, desc="Processing URIs", disable=disable_tqdm):
                 if g.value(uri, OWL.deprecated) == Literal(True): skip_dep += 1; continue
                 uri_s = str(uri); current_uri_data = defaultdict(list); has_properties = False
                 for prop, key in all_props.items():
                      for obj in g.objects(uri, prop): val = str(obj).strip() if isinstance(obj, Literal) else str(obj) if isinstance(obj, URIRef) else None;
                      if val: current_uri_data[key].append(val); has_properties = True
                 if has_properties:
                      processed_data = {k: list(set(v)) for k, v in current_uri_data.items() if v};
                      if processed_data: kept_concepts_data[uri_s] = processed_data;
                      else: rem_empty += 1
                 else: rem_empty += 1
            logger.info(f"Extracted {len(kept_concepts_data)} concepts. Skipped {skip_dep} deprecated, removed {rem_empty} non-concepts/empty.")
            _taxonomy_concepts_cache = dict(kept_concepts_data)
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True); cache_data = {"cache_version": CACHE_VERSION, "data": _taxonomy_concepts_cache}
                with open(cache_file, 'w', encoding='utf-8') as f: json.dump(cache_data, f, indent=2); logger.info(f"Saved concepts cache ({len(_taxonomy_concepts_cache)} concepts).")
            except Exception as e: logger.error(f"Failed writing concepts cache: {e}")
            logger.info(f"Taxonomy loading took {time.time() - start_time:.2f}s.")
        except Exception as e: logger.error(f"Taxonomy load error: {e}", exc_info=args.debug); return None
    if not _taxonomy_concepts_cache: logger.error("Failed concept load."); return None
    return _taxonomy_concepts_cache




















# UPDATED: Added args parameter
def precompute_taxonomy_embeddings(taxonomy_concepts: Dict[str, Dict], sbert_model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Optional[Tuple[Dict, Dict, List]]:
    global _taxonomy_embeddings_cache
    if _taxonomy_embeddings_cache is not None: return _taxonomy_embeddings_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file): # Use args here
        logger.info(f"Attempting to load embeddings from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached_data = pickle.load(f)
            if cached_data.get("cache_version") == CACHE_VERSION:
                uri_map = cached_data.get("uri_embeddings_map"); p_embs = cached_data.get("primary_embeddings"); u_list = cached_data.get("uris_list")
                if isinstance(uri_map, dict) and isinstance(p_embs, dict) and isinstance(u_list, list): _taxonomy_embeddings_cache = (uri_map, p_embs, u_list); cache_valid = True; logger.info(f"Loaded {len(u_list)} embeddings from cache.")
                else: logger.warning("Cache structure invalid. Rebuilding.")
            else: logger.info(f"Cache version mismatch. Rebuilding.")
        except Exception as e: logger.warning(f"Cache load failed: {e}. Rebuilding.")
    if not cache_valid:
        logger.info("Pre-computing embeddings..."); start_time = time.time(); uri_embeddings_map = defaultdict(list); primary_embeddings: Dict[str, Optional[np.ndarray]] = {}; uris_list_all: List[str] = []; texts_to_embed_map = defaultdict(list); all_valid_uris = list(taxonomy_concepts.keys()); text_properties = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel", "definition", "dctermsDescription", "scopeNote"]; disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug # Use args here
        for uri in tqdm(all_valid_uris, desc="Gathering Texts", disable=disable_tqdm):
            uris_list_all.append(uri); concept_data = taxonomy_concepts.get(uri, {}); found_texts: Set[Tuple[str, str, str]] = set()
            for prop_key in text_properties:
                for text_value in concept_data.get(prop_key, []):
                    if text_value and isinstance(text_value, str): original_text = text_value.strip(); normalized_text = normalize_concept(original_text);
                    if normalized_text: found_texts.add((prop_key, original_text, normalized_text))
            for prop_key, original_text, normalized_text in found_texts: texts_to_embed_map[normalized_text].append((uri, prop_key, original_text))
        unique_normalized_texts = list(texts_to_embed_map.keys()); embedding_map: Dict[str, Optional[np.ndarray]] = {}
        if unique_normalized_texts:
            logger.info(f"Generating embeddings for {len(unique_normalized_texts)} unique texts..."); batch_size = 128
            try: embeddings_list = sbert_model.encode(unique_normalized_texts, batch_size=batch_size, show_progress_bar=logger.isEnabledFor(logging.INFO)); embedding_map = {text: emb for text, emb in zip(unique_normalized_texts, embeddings_list) if emb is not None}
            except Exception as e: logger.error(f"SBERT encoding failed: {e}", exc_info=True); raise RuntimeError("SBERT Encoding Failed") from e
        primary_embedding_candidates = defaultdict(dict); sbert_dim = sbert_model.get_sentence_embedding_dimension()
        for normalized_text, associated_infos in texts_to_embed_map.items():
            embedding = embedding_map.get(normalized_text);
            if embedding is None or embedding.shape != (sbert_dim,): continue
            for uri, prop_key, original_text in associated_infos: uri_embeddings_map[uri].append((prop_key, original_text, embedding, normalized_text)); primary_embedding_candidates[uri][prop_key] = embedding
        primary_property_priority = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel", "definition", "dctermsDescription", "scopeNote"]; num_with_primary = 0
        for uri in tqdm(uris_list_all, desc="Selecting Primary", disable=disable_tqdm):
            candidates = primary_embedding_candidates.get(uri, {}); chosen_embedding = None
            for prop in primary_property_priority:
                if prop in candidates and candidates[prop] is not None: chosen_embedding = candidates[prop]; break
            if chosen_embedding is not None and isinstance(chosen_embedding, np.ndarray) and chosen_embedding.ndim == 1: primary_embeddings[uri] = chosen_embedding; num_with_primary += 1
            else: primary_embeddings[uri] = None
        final_uris_list = [uri for uri in uris_list_all if primary_embeddings.get(uri) is not None]; final_primary_embeddings = {uri: emb for uri, emb in primary_embeddings.items() if emb is not None}; final_uri_embeddings_map = {uri: data for uri, data in uri_embeddings_map.items() if uri in final_primary_embeddings}
        _taxonomy_embeddings_cache = (final_uri_embeddings_map, final_primary_embeddings, final_uris_list)
        logger.info(f"Finished embedding ({num_with_primary}/{len(uris_list_all)} with primary) in {time.time() - start_time:.2f}s.")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True); cache_data = {"cache_version": CACHE_VERSION, "uri_embeddings_map": _taxonomy_embeddings_cache[0], "primary_embeddings": _taxonomy_embeddings_cache[1], "uris_list": _taxonomy_embeddings_cache[2]}
            with open(cache_file, 'wb') as f: pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL); logger.info(f"Saved embeddings cache.")
        except Exception as e:
            logger.error(f"Failed writing embeddings cache: {e}")
    if not _taxonomy_embeddings_cache or not _taxonomy_embeddings_cache[1]: logger.error("Embedding process failed."); return None
    return _taxonomy_embeddings_cache

# UPDATED: Added args parameter
def load_concept_associations(associations_file: Optional[str], args: argparse.Namespace) -> Dict[str, List[str]]:
    global _concept_associations_cache
    if _concept_associations_cache is not None: return _concept_associations_cache
    if not associations_file: logger.info("No associations file."); _concept_associations_cache = {}; return {}
    if not os.path.exists(associations_file): logger.warning(f"Associations file not found: {associations_file}"); _concept_associations_cache = {}; return {}
    logger.info(f"Loading associations from {associations_file}"); start_time = time.time(); normalized_associations: Dict[str, List[str]] = {}
    try:
        with open(associations_file, 'r', encoding='utf-8') as f: data = json.load(f)
        for key, value_in in data.items():
            normalized_key = normalize_concept(key).lower();
            if not normalized_key: continue
            associated_values: List[str] = []
            if isinstance(value_in, list): associated_values = [normalize_concept(v).lower() for v in value_in if isinstance(v, str) and normalize_concept(v)]
            elif isinstance(value_in, str): normalized_value = normalize_concept(value_in).lower(); associated_values.append(normalized_value)
            if associated_values: normalized_associations[normalized_key] = associated_values
    except Exception as e: logger.error(f"Error loading associations: {e}", exc_info=args.debug); _concept_associations_cache = {}; return {} # Use args here
    _concept_associations_cache = normalized_associations; logger.info(f"Loaded {len(normalized_associations)} associations in {time.time() - start_time:.2f}s.")
    return _concept_associations_cache

# UPDATED: Added args parameter
def load_corpus_data(corpus_file: Optional[str], sbert_model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    global _corpus_data_cache
    if _corpus_data_cache is not None: return _corpus_data_cache
    if not corpus_file: logger.info("No corpus file."); _corpus_data_cache = {}; return {}
    cache_valid = False; corpus_abs_path = os.path.abspath(corpus_file) if corpus_file else None; rebuild = args.rebuild_cache # Use args here
    if not rebuild and os.path.exists(cache_file):
        logger.info(f"Loading corpus cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached = pickle.load(f)
            if cached.get("cache_version") == CACHE_VERSION and cached.get("corpus_file_path") == corpus_abs_path and isinstance(cached.get("data"), dict):
                _corpus_data_cache = cached["data"]; logger.info(f"Loaded {len(_corpus_data_cache)} corpus terms."); cache_valid = True
            else: logger.info("Corpus cache invalid. Rebuilding."); rebuild = True
        except Exception as e: logger.warning(f"Failed corpus cache load: {e}. Rebuilding."); rebuild = True
    if not cache_valid or rebuild:
        logger.info(f"Loading/embedding corpus: {corpus_file}"); start = time.time()
        if not os.path.exists(corpus_file): logger.error(f"Corpus file not found: {corpus_file}"); _corpus_data_cache = {}; return {}
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f: terms = [normalize_concept(l.strip()) for l in f if l.strip()]
            unique = sorted(list(set(filter(None, terms)))); logger.info(f"Found {len(unique)} unique terms.")
            if not unique: logger.warning("No valid terms in corpus."); _corpus_data_cache = {}; return {}
            logger.info("Generating corpus embeddings...")
            embeds = sbert_model.encode(unique, batch_size=128, show_progress_bar=logger.isEnabledFor(logging.INFO))
            _corpus_data_cache = {t: e for t, e in zip(unique, embeds) if e is not None}
            logger.info(f"Finished corpus embedding ({len(_corpus_data_cache)}) in {time.time()-start:.2f}s.")
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True); cache_data = {"cache_version": CACHE_VERSION, "corpus_file_path": corpus_abs_path, "data": _corpus_data_cache}
                with open(cache_file, 'wb') as f: pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL); logger.info(f"Saved corpus cache: {cache_file}")
            except Exception as e:
                 logger.error(f"Failed writing corpus cache: {e}")
        except Exception as e:
             logger.error(f"Error processing corpus: {e}", exc_info=args.debug) # Use args here
             _corpus_data_cache = {}
    return _corpus_data_cache

# UPDATED: Added args parameter
def get_theme_embeddings(sbert_model: SentenceTransformer, themes_def: Dict, args: argparse.Namespace) -> Dict[str, Dict]:
    global _theme_embeddings_cache
    cache_dir = args.cache_dir # Access from passed args
    theme_cache_file = os.path.join(cache_dir, f"themes_{CACHE_VERSION}.pkl")
    hints_structure_hash = hash(json.dumps(themes_def, sort_keys=True))
    if _theme_embeddings_cache is not None: return _theme_embeddings_cache
    rebuild = args.rebuild_cache # Use args here
    if not rebuild and os.path.exists(theme_cache_file):
        logger.info(f"Loading theme embeddings cache: {theme_cache_file}")
        try:
            with open(theme_cache_file, 'rb') as f: cached_theme_data = pickle.load(f)
            if cached_theme_data.get("cache_version") == CACHE_VERSION and cached_theme_data.get("hints_hash") == hints_structure_hash and isinstance(cached_theme_data.get("data"), dict):
                _theme_embeddings_cache = cached_theme_data["data"]; logger.info(f"Loaded theme embeddings for {len(_theme_embeddings_cache)} themes from cache."); return _theme_embeddings_cache
            else: logger.info("Theme cache version/structure mismatch. Recomputing.")
        except Exception as e: logger.warning(f"Failed loading theme embeddings cache: {e}. Recomputing.")
    logger.info("Pre-computing theme embeddings..."); start_time = time.time(); theme_embeddings = {}; all_texts = []; text_map = {}
    for theme_name, data in themes_def.items():
        name_norm = normalize_concept(theme_name);
        if name_norm: all_texts.append(name_norm); text_map[name_norm] = (theme_name, 'name')
        for hint in data.get('hints', []): hint_norm = normalize_concept(hint);
        if hint_norm and hint_norm not in text_map: all_texts.append(hint_norm); text_map[hint_norm] = (theme_name, 'hint')
    if not all_texts: logger.warning("No theme texts to embed."); return {}
    try: embeddings = sbert_model.encode(all_texts, batch_size=32, show_progress_bar=logger.isEnabledFor(logging.DEBUG))
    except Exception as e: logger.error(f"Failed to embed theme texts: {e}", exc_info=True); return {}
    for text, embedding in zip(all_texts, embeddings):
        if embedding is None: continue
        theme_name, type = text_map[text]
        if theme_name not in theme_embeddings: theme_embeddings[theme_name] = {'name_embedding': None, 'hint_embeddings': []}
        if type == 'name': theme_embeddings[theme_name]['name_embedding'] = embedding
        else: theme_embeddings[theme_name]['hint_embeddings'].append(embedding)
    _theme_embeddings_cache = theme_embeddings; logger.info(f"Computed embeddings for {len(theme_embeddings)} themes in {time.time() - start_time:.2f}s.")
    try:
        os.makedirs(os.path.dirname(theme_cache_file), exist_ok=True); cache_to_save = {"cache_version": CACHE_VERSION, "hints_hash": hints_structure_hash, "data": _theme_embeddings_cache}
        with open(theme_cache_file, 'wb') as f: pickle.dump(cache_to_save, f, protocol=pickle.HIGHEST_PROTOCOL); logger.info(f"Saved theme embeddings cache.")
    except Exception as e: logger.error(f"Failed writing theme embedding cache: {e}")
    return _theme_embeddings_cache


# --- === Core Logic Functions for v30.1 === ---

# === Stage 1 Helper: Find Anchors ===
def find_best_taxonomy_matches(
    concept: str, sbert_model: SentenceTransformer, uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]],
    primary_embeddings: Dict[str, np.ndarray], uris_list: List[str], associations: Dict[str, List[str]],
    corpus_data: Dict[str, np.ndarray], taxonomy_concepts: Dict[str, Dict], min_relevance: float, max_matches: int,
    args: argparse.Namespace
) -> List[Tuple[str, float, str, Dict[str, Any]]]:
    logger.info(f"--- Finding best taxonomy matches for: '{concept}' (Min Relevance: {min_relevance}) ---")
    start_time = time.time(); norm_concept_lower = normalize_concept(concept).lower().strip()
    if not norm_concept_lower: logger.warning("Input concept normalized empty."); return []
    try: input_text = (TRAVEL_CONTEXT + norm_concept_lower); concept_embedding = sbert_model.encode([input_text])[0]; assert concept_embedding is not None
    except Exception as e: logger.error(f"SBERT embedding error for '{concept}': {e}", exc_info=True); return []
    concept_scores: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"max_score": 0.0}); disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    keywords = [kw for kw in norm_concept_lower.split() if kw not in STOP_WORDS and len(kw) > 1]; kw_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords]
    for uri in tqdm(uris_list, desc="Scoring Anchors", disable=disable_tqdm):
        uri_data = uri_embeddings_map.get(uri, []); primary_emb = primary_embeddings.get(uri)
        if not uri_data or primary_emb is None: continue
        max_score_for_uri = calculate_semantic_similarity(concept_embedding, primary_emb, norm_concept_lower, f"PrimaryEmb({uri})")
        for prop, orig, txt_emb, norm in uri_data:
            prop_w = PROPERTY_WEIGHTS_INTERNAL.get(prop, 0.5); kw_score = 0.0; sem_score = 0.0
            kw_count = sum(1 for p in kw_patterns if p.search(orig) or p.search(norm)) if keywords else 0
            if kw_count > 0: kw_score = min(1.0, (kw_count / len(keywords) if keywords else 0) + (EXACT_MATCH_BONUS_INTERNAL if norm_concept_lower == norm else 0))
            sem_score = calculate_semantic_similarity(concept_embedding, txt_emb, norm_concept_lower, f"{prop}({norm[:30]}...)")
            comb_score = (kw_score * KEYWORD_WEIGHT_INTERNAL + sem_score * SEMANTIC_WEIGHT_INTERNAL) * prop_w
            if norm_concept_lower in associations and any(a in norm for a in associations[norm_concept_lower]): comb_score = min(1.0, comb_score + ASSOCIATION_BOOST_INTERNAL)
            max_score_for_uri = max(max_score_for_uri, comb_score)
        if max_score_for_uri >= min_relevance: concept_scores[uri]["max_score"] = max_score_for_uri
    sorted_concepts = sorted(concept_scores.items(), key=lambda item: item[1]["max_score"], reverse=True); results_with_data = []
    for uri, data in sorted_concepts:
        concept_kg_data = taxonomy_concepts.get(uri, {}); label = get_primary_label(uri)
        enriched_result = (uri, data["max_score"], label, concept_kg_data); results_with_data.append(enriched_result)
        if len(results_with_data) >= max_matches: break
    logger.info(f"Found {len(results_with_data)} anchor matches >= {min_relevance:.2f} in {time.time() - start_time:.2f}s.")
    return results_with_data

# === Stage 1 Helper: Gather Evidence ===
def gather_candidate_evidence_concepts(
    input_embedding: np.ndarray, anchor_concepts: List[Tuple[str, float, str, Dict[str, Any]]],
    primary_embeddings: Dict[str, np.ndarray], uris_list: List[str],
    min_similarity: float, top_k: int, args: argparse.Namespace
) -> Dict[str, float]:
    logger.info(f"--- Gathering candidate evidence (Top K={top_k}, Min Similarity={min_similarity}) ---")
    start_time = time.time(); candidates: Dict[str, float] = defaultdict(lambda: -1.0)
    if not anchor_concepts: logger.warning("No anchor concepts provided."); return {}
    valid_uris = [uri for uri in uris_list if uri in primary_embeddings]
    if not valid_uris: logger.warning("No URIs with primary embeddings found."); return {}
    all_embeddings = np.array([primary_embeddings[uri] for uri in valid_uris])
    if input_embedding is not None and all_embeddings.shape[0] > 0:
        input_sims = cosine_similarity(input_embedding.reshape(1, -1), all_embeddings)[0]
        for i, uri in enumerate(valid_uris): score = float(input_sims[i]); candidates[uri] = max(candidates[uri], score)
    for uri, score, _, _ in anchor_concepts: candidates[uri] = max(candidates[uri], score)
    final_candidates = {uri: score for uri, score in candidates.items() if score >= min_similarity}
    sorted_uris = sorted(final_candidates, key=final_candidates.get, reverse=True)
    top_k_candidates = {uri: final_candidates[uri] for uri in sorted_uris[:top_k]}
    for uri, score, _, _ in anchor_concepts: # Ensure anchors meeting threshold are included
        if score >= min_similarity and uri not in top_k_candidates: top_k_candidates[uri] = score
    logger.info(f"Gathered {len(top_k_candidates)} candidate evidence concepts in {time.time() - start_time:.2f}s.")
    return top_k_candidates

# === Stage 1 Helper: LLM Prompt ===
def construct_llm_enrichment_prompt(
    input_concept: str, anchor_info: Dict[str, Any], top_candidates: List[Tuple[str, float]],
    config: AffinityConfig, args: argparse.Namespace
) -> Tuple[str, str]:
    """Constructs the prompt for the first LLM call (enrichment & suggestions). V30.1"""
    system_prompt = f"""You are an expert travel taxonomist assisting in defining travel concepts. Analyze provided info. Provide suggestions for classification and exclusion. Output ONLY a valid JSON object."""
    anchor_label = anchor_info.get("label", "N/A"); anchor_def = (anchor_info.get("kg_data", {}).get("definition") or ["N/A"])[0]
    candidate_list_str = "\n".join([f"- {get_primary_label(uri)} ({score:.3f})" for uri, score in top_candidates[:20]])
    output_schema = """```json
{
  "core_definition": "string (Concise 1-2 sentence definition summarizing the input concept)",
  "applicable_lodging_types": "string ('VR Only', 'CL Only', or 'Both')",
  "travel_category_type": "string (Suggest broad category: 'Demographic-Specific', 'Amenity-Based', 'Location-Based', etc.)",
  "potential_must_not_have_uris": ["string (Suggest specific URIs from 'Potentially Relevant Concepts' list that contradict input)"],
  "relevant_subscore_suggestions": ["string (Suggest relevant sub-score names from master list provided)"],
  "theme_summaries": {
     "Theme Name 1": "string (1-sentence summary explaining relevance of this theme to the input concept)",
     "Theme Name 2": "string (Summary...)",
     "...": "..."
  }
}
```"""
    master_subscore_list = config.get("master_subscore_list", [])
    theme_names = list(config.get("base_themes", {}).keys())

    user_prompt = f"""Analyze travel concept '{input_concept}'.
Anchor Concept: {anchor_label} (URI: {anchor_info.get('uri', 'N/A')}, Def: {anchor_def})
Potentially Relevant Concepts (Sample from KG):
{candidate_list_str}
Master List of Available Sub-Scores: {', '.join(master_subscore_list)}
Available Theme Names: {', '.join(theme_names)}

Task: Generate JSON suggestions per schema based on travel knowledge and data.
Output JSON Schema: {output_schema}
Instructions:
- core_definition: Write summary.
- applicable_lodging_types: Choose 'VR Only', 'CL Only', or 'Both'.
- travel_category_type: Suggest high-level type.
- potential_must_not_have_uris: List specific URIs *from the sample list* that contradict the input. Use [] if none.
- relevant_subscore_suggestions: Suggest applicable sub-scores *from master list*.
- theme_summaries: For **each** theme in 'Available Theme Names', provide a 1-sentence summary explaining **if and why** that theme is relevant to the **input concept**. If a theme is not relevant, state that briefly. Generate as a dictionary.

Generate ONLY the JSON object.
"""
    return system_prompt.strip(), user_prompt.strip()

# === OpenAI Helper ===
def get_openai_client() -> Optional[OpenAI]:
    global _openai_client
    if _openai_client is None:
        try: api_key = os.environ.get("OPENAI_API_KEY"); assert api_key, "OPENAI_API_KEY not set"; _openai_client = OpenAI(api_key=api_key); logger.info("OpenAI client initialized.")
        except Exception as e: logger.error(f"Failed to initialize OpenAI client: {e}"); return None
    return _openai_client

def call_openai_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float) -> Optional[Dict[str, Any]]:
    client = get_openai_client();
    if not client: return None
    logger.info(f"Sending request to OpenAI model: {model_name}")
    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"--- System Prompt ---\n{system_prompt}\n--- User Prompt (first 500 chars) ---\n{user_prompt[:500]}...")
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=temperature, timeout=timeout)
            content = response.choices[0].message.content
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"--- Raw OpenAI Response ---\n{content}\n--- END ---")
            if content:
                try:
                    cleaned = content.strip();
                    if cleaned.startswith("```json"): cleaned = cleaned[7:]
                    if cleaned.endswith("```"): cleaned = cleaned[:-3]
                    llm_output = json.loads(cleaned.strip()); assert isinstance(llm_output, dict)
                    logger.info("Successfully parsed JSON response.")
                    return llm_output
                except (json.JSONDecodeError, AssertionError) as json_e:
                     logger.error(f"Failed parsing LLM JSON: {json_e}\nRaw: {content}")
            else:
                 logger.warning("LLM response content empty.")
            return None
        except (APITimeoutError, APIConnectionError, RateLimitError) as api_e:
            logger.warning(f"OpenAI API Error (Try {attempt+1}/{max_retries+1}): {type(api_e).__name__}")
            wait = 5 if isinstance(api_e, RateLimitError) else 2**attempt; time.sleep(wait)
        except Exception as e: logger.error(f"OpenAI Call Unexpected Error (Try {attempt+1}/{max_retries+1}): {e}", exc_info=True); return None
    logger.error("Max retries reached for OpenAI call."); return None

# === Stage 2 Helpers ===
def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: AffinityConfig) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    """Gets the rule, weight, subscore (optional), and fallback logic for a theme, considering concept overrides."""
    base_themes = config["base_themes"]; overrides = config.get("concept_overrides", {}).get(normalized_concept, {}); theme_overrides = overrides.get("theme_overrides", {}).get(theme_name, {});
    base_config = base_themes.get(theme_name)
    if base_config is None: raise ValueError(f"Theme '{theme_name}' not found in base_themes config.")
    rule = theme_overrides.get("rule", base_config.get("rule", "Optional")); weight = theme_overrides.get("weight", base_config.get("weight", 0.0))
    subScore = theme_overrides.get("subScore", base_config.get("subScore")); fallback_logic = theme_overrides.get("fallback_logic", base_config.get("fallback_logic"))
    if subScore is None: logger.log(logging.DEBUG if not args.debug else logging.WARNING, f"Theme '{theme_name}' has no 'subScore' defined.")
    return rule, float(weight), subScore, fallback_logic

def select_theme_attributes(
    theme_name: str, theme_hints: List[str], candidate_evidence: Dict[str, float],
    primary_embeddings: Dict[str, np.ndarray], uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]],
    sbert_model: SentenceTransformer, semantic_threshold: float, keyword_min_match: int, # Use specific thresholds
    args: argparse.Namespace
) -> List[Tuple[str, float]]:
    """Selects candidate URIs relevant to theme using Keyword + Semantic matching."""
    global _theme_embeddings_cache
    if _theme_embeddings_cache is None: _theme_embeddings_cache = get_theme_embeddings(sbert_model, _config_data["base_themes"], args)
    theme_data = _theme_embeddings_cache.get(theme_name)
    if not theme_data: logger.warning(f"No theme data for '{theme_name}'."); return []
    theme_repr_embeddings = []; name_emb = theme_data.get('name_embedding'); hint_embs = theme_data.get('hint_embeddings', [])
    if name_emb is not None: theme_repr_embeddings.append(name_emb); theme_repr_embeddings.extend(hint_embs)
    if not theme_repr_embeddings: logger.warning(f"No embeddings for theme '{theme_name}'."); return []
    hint_patterns = [re.compile(r'\b' + re.escape(normalize_concept(h)) + r'\b', re.IGNORECASE) for h in theme_hints if normalize_concept(h)]
    if not hint_patterns and keyword_min_match > 0: logger.warning(f"No valid hint patterns for theme '{theme_name}'. Keyword match skipped."); keyword_min_match = 0 # Skip keyword check

    relevant_attributes = []
    for uri, score in candidate_evidence.items():
        uri_emb = primary_embeddings.get(uri);
        if uri_emb is None: continue

        # 1. Semantic Check
        max_sim = max((calculate_semantic_similarity(uri_emb, theme_emb, uri, f"Theme({theme_name})") for theme_emb in theme_repr_embeddings), default=0.0)
        if max_sim < semantic_threshold: continue

        # 2. Keyword Check (only if semantic check passed and required)
        kw_match_count = 0
        if keyword_min_match > 0:
            uri_texts = uri_embeddings_map.get(uri, [])
            combined_text = " ".join(normalize_concept(orig) for _, orig, _, norm in uri_texts)
            if not combined_text: combined_text = normalize_concept(get_primary_label(uri))
            if combined_text:
                for pattern in hint_patterns:
                    if pattern.search(combined_text): kw_match_count += 1;
                    if kw_match_count >= keyword_min_match: break
            if kw_match_count < keyword_min_match: continue # Failed keyword check

        # 3. Add if BOTH checks passed (or if keyword check skipped)
        relevant_attributes.append((uri, score))
        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"  SELECT '{get_primary_label(uri)}' for '{theme_name}' (SemSim: {max_sim:.3f}, KW Matches: {kw_match_count})")

    return sorted(relevant_attributes, key=lambda x: x[1], reverse=True)

def apply_theme_rule(theme_name: str, rule: str, selected_attributes: List[Tuple[str, float]], fallback_logic: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Applies the theme rule and returns if passed, and any fallback action."""
    if rule == "Must have 1":
        if len(selected_attributes) >= 1: return True, None
        else: logger.warning(f"Rule '{rule}' failed for theme '{theme_name}'. Applying fallback."); return False, fallback_logic
    return True, None # Optional or unknown rules pass

def normalize_weights(theme_weights: Dict[str, float], additional_subscore_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Normalizes theme and additional subscore weights to sum to 1.0."""
    all_weights_map = {**{f"theme_{k}": v for k, v in theme_weights.items()}, **additional_subscore_weights}; total_weight = sum(all_weights_map.values())
    norm_themes = {k: 0.0 for k in theme_weights}; norm_add_scores = {k: 0.0 for k in additional_subscore_weights}
    if total_weight <= 1e-9: logger.warning("Total weight is zero. Cannot normalize."); return norm_themes, norm_add_scores
    norm_factor = 1.0 / total_weight; normalized_theme_weights = {k: v * norm_factor for k, v in theme_weights.items()}; normalized_additional_subscores = {k: v * norm_factor for k, v in additional_subscore_weights.items()}
    final_total = sum(normalized_theme_weights.values()) + sum(normalized_additional_subscores.values()); logger.info(f"Normalized weights. Initial Sum: {total_weight:.4f}, Final Sum: {final_total:.4f}")
    if abs(final_total - 1.0) > 1e-6: logger.error(f"Weight normalization failed! Final sum is {final_total}")
    return normalized_theme_weights, normalized_additional_subscores

def normalize_concept_weights(attributes: List[Tuple[str, float]], target_theme_weight: float) -> List[Dict[str, Any]]:
    """Normalizes concept scores to sum up to the target_theme_weight. Includes attribute type."""
    normalized_attributes = []
    num_attributes = len(attributes)
    if not attributes or target_theme_weight <= 1e-9:
        return [{"uri": uri, "skos:prefLabel": get_primary_label(uri), "concept_weight": 0.0, "type": get_concept_type(uri)} for uri, _ in attributes]
    total_initial_score = sum(score for _, score in attributes)
    if total_initial_score <= 1e-9:
        equal_weight = target_theme_weight / num_attributes if num_attributes > 0 else 0.0; logger.debug(f"Assigning equal weight: {equal_weight:.6f} for {num_attributes} attributes in theme.")
        for uri, score in attributes: normalized_attributes.append({"uri": uri, "skos:prefLabel": get_primary_label(uri), "concept_weight": round(equal_weight, 6), "type": get_concept_type(uri)})
        return normalized_attributes
    else:
        norm_factor = target_theme_weight / total_initial_score;
        for uri, score in attributes: normalized_weight = score * norm_factor; normalized_attributes.append({"uri": uri, "skos:prefLabel": get_primary_label(uri), "concept_weight": round(max(0.0, normalized_weight), 6), "type": get_concept_type(uri)})
        current_sum = sum(a['concept_weight'] for a in normalized_attributes)
        if abs(current_sum - target_theme_weight) > 1e-5 and normalized_attributes:
            diff = target_theme_weight - current_sum; adjusted_weight = max(0.0, normalized_attributes[-1]['concept_weight'] + diff); logger.warning(f"Concept weight sum mismatch (target: {target_theme_weight:.6f}, sum: {current_sum:.6f}). Adjusting last element by {diff:.6f} to {adjusted_weight:.6f}.")
            normalized_attributes[-1]['concept_weight'] = round(adjusted_weight, 6)
        return normalized_attributes


# --- Main Execution Logic (v30.1) ---
def generate_affinity_definitions(input_concepts: List[str], config: AffinityConfig, args: argparse.Namespace):
    """Generates affinity definitions for the given list of input concepts."""
    logger.info(f"=== Starting Affinity Definition Generation Process (v30.1r3 - Rule Engine) ===")
    start_process_time = time.time()
    # Load resources ONCE
    sbert_model = get_sbert_model()
    taxonomy_concepts = load_taxonomy_concepts(args.taxonomy_dir, os.path.join(args.cache_dir, f"concepts_{CACHE_VERSION}.json"), args)
    if not taxonomy_concepts: logger.critical("Taxonomy concepts failed."); return
    embedding_data = precompute_taxonomy_embeddings(taxonomy_concepts, sbert_model, os.path.join(args.cache_dir, f"embeddings_{CACHE_VERSION}.pkl"), args)
    if not embedding_data: logger.critical("Embeddings failed."); return
    concept_associations = load_concept_associations(args.associations_file, args) # Pass args
    corpus_data = load_corpus_data(args.corpus_file, sbert_model, os.path.join(args.cache_dir, f"corpus_{CACHE_VERSION}.pkl"), args) # Pass args
    uri_embeddings_map, primary_embeddings, uris_list = embedding_data
    get_theme_embeddings(sbert_model, config["base_themes"], args) # Pass args

    all_definitions = []
    logger.info(f"Processing {len(input_concepts)} input concepts...")
    for i, concept_input_string in enumerate(input_concepts):
        logger.info(f"\n--- Processing Concept {i+1}/{len(input_concepts)}: '{concept_input_string}' ---")
        concept_start_time = time.time(); norm_concept_lc = normalize_concept(concept_input_string).lower().strip()
        affinity_definition = { "input_concept": concept_input_string, "normalized_concept": norm_concept_lc, "applicable_lodging_types": "Both", "travel_category": {"uri": None, "name": None, "skos:prefLabel": None, "skos:altLabel": [], "skos:definition": None, "skos:broader": None, "skos:narrower": [], "skos:related": [], "type": "Unknown", "exclusionary_concepts": []}, "core_definition": "N/A", "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [], "must_not_have": [], "processing_metadata": {"version": f"affinity-rule-engine-v30.1r3", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "Processing Failed", "duration_seconds": 0.0, "cache_version": CACHE_VERSION, "llm_model": args.openai_model}, "diagnostics": defaultdict(lambda: defaultdict(dict))}
        llm_suggestions = {}; candidate_evidence = {}
        failed_theme_fallbacks_info = {} # Initialize here per concept

        try:
            # === Stage 1: Concept ID & LLM Enrichment ===
            logger.info("Stage 1: Identifying Concepts & Getting LLM Suggestions...")
            anchor_matches = find_best_taxonomy_matches(concept_input_string, sbert_model, uri_embeddings_map, primary_embeddings, uris_list, concept_associations, corpus_data, taxonomy_concepts, ANCHOR_MATCH_MIN_RELEVANCE, MAX_ANCHOR_MATCHES, args)
            if not anchor_matches: logger.error(f"No anchors found for '{concept_input_string}'. Skipping."); affinity_definition["processing_metadata"]["status"] = "Failed - No Anchors"; raise StopIteration()
            anchor_uri, anchor_score, anchor_label, anchor_kg_data = anchor_matches[0]; affinity_definition["diagnostics"]["stage1"]["anchor"] = {"uri": anchor_uri, "label": anchor_label, "score": anchor_score}
            travel_cat = affinity_definition["travel_category"]; travel_cat["uri"] = anchor_uri; travel_cat["name"] = anchor_label; travel_cat["skos:prefLabel"] = (anchor_kg_data.get("prefLabel") or [anchor_label])[0]; travel_cat["skos:altLabel"] = anchor_kg_data.get("altLabel", []); travel_cat["skos:definition"] = (anchor_kg_data.get("definition") or anchor_kg_data.get("dctermsDescription") or [None])[0]; travel_cat["skos:broader"] = (anchor_kg_data.get("broader") or [None])[0]; travel_cat["skos:narrower"] = anchor_kg_data.get("narrower", []); travel_cat["skos:related"] = anchor_kg_data.get("related", []); logger.info(f"Anchor concept: {travel_cat['skos:prefLabel']} ({anchor_uri})")
            try: input_embedding = sbert_model.encode([("travel " + norm_concept_lc)])[0]; assert input_embedding is not None
            except Exception as e: logger.error(f"Failed input embedding: {e}", exc_info=True); raise StopIteration("Failed Input Embedding")
            candidate_evidence = gather_candidate_evidence_concepts(input_embedding, anchor_matches, primary_embeddings, uris_list, KG_RETRIEVAL_SIMILARITY_THRESHOLD, KG_RETRIEVAL_TOP_K, args)
            affinity_definition["diagnostics"]["stage1"]["candidate_evidence_count"] = len(candidate_evidence)
            if not candidate_evidence: logger.warning(f"No candidate evidence found for '{concept_input_string}'.")
            logger.info("Calling LLM for enrichment suggestions...")
            prompt_sys, prompt_user = construct_llm_enrichment_prompt(concept_input_string, {"uri": anchor_uri, "label": anchor_label, "score": anchor_score, "kg_data": anchor_kg_data}, list(candidate_evidence.items()), config, args)
            llm_enrichment_output = call_openai_llm(prompt_sys, prompt_user, args.openai_model, OPENAI_TIMEOUT, OPENAI_MAX_RETRIES, LLM_TEMPERATURE); affinity_definition["diagnostics"]["stage1"]["llm_call_attempted"] = True
            if llm_enrichment_output: llm_suggestions = llm_enrichment_output; affinity_definition["diagnostics"]["stage1"]["llm_call_success"] = True; logger.info("Received suggestions from LLM."); affinity_definition["diagnostics"]["stage1"]["llm_raw_suggestions"] = llm_suggestions
            else: affinity_definition["diagnostics"]["stage1"]["llm_call_success"] = False; logger.error("LLM call for enrichment failed.")

            # === Stage 2: Rule-Based Assembly ===
            logger.info("Stage 2: Assembling definition using rules and configuration...")
            concept_overrides = config.get("concept_overrides", {}).get(norm_concept_lc, {})
            affinity_definition["core_definition"] = llm_suggestions.get("core_definition", travel_cat["skos:definition"] or "N/A")
            affinity_definition["applicable_lodging_types"] = llm_suggestions.get("applicable_lodging_types") or concept_overrides.get("lodging_type", "Both")
            travel_cat["type"] = llm_suggestions.get("travel_category_type") or concept_overrides.get("category_type", "Unknown")
            # Use config ONLY for exclusionary concepts
            travel_cat["exclusionary_concepts"] = concept_overrides.get("exclusionary_concepts", []) # Get from config directly
            # Assemble and VALIDATE must_not_have list
            must_not_have_set = set(concept_overrides.get("must_not_have_uris", [])) # Start with config
            llm_must_not_uris = llm_suggestions.get("potential_must_not_have_uris", [])
            valid_llm_must_not = set()
            if llm_must_not_uris and isinstance(llm_must_not_uris, list):
                for uri in llm_must_not_uris:
                     if isinstance(uri, str) and ':' in uri and uri in taxonomy_concepts: valid_llm_must_not.add(uri); logger.debug(f"  LLM must_not_have URI '{uri}' validated.")
                     elif isinstance(uri, str): logger.warning(f"  LLM suggested must_not_have URI '{uri}' not found/invalid. Discarding.")
            must_not_have_set.update(valid_llm_must_not) # Add validated LLM suggestions
            affinity_definition["must_not_have"] = [{"uri": uri, "skos:prefLabel": get_primary_label(uri), "scope": None} for uri in sorted(list(must_not_have_set))]
            logger.info(f"Final must_not_have count: {len(affinity_definition['must_not_have'])}")

            # Determine initial theme & additional subscore weights & rules
            initial_theme_weights = {}; theme_rules = {}; theme_subscores = {}; theme_fallback_logic = {}; theme_types = {}; base_themes_config: Dict[str, ThemeConfig] = config["base_themes"]
            for theme_name, theme_config_item in base_themes_config.items(): rule, weight, subscore, fallback = get_dynamic_theme_config(norm_concept_lc, theme_name, config); initial_theme_weights[theme_name] = weight; theme_rules[theme_name] = rule; theme_subscores[theme_name] = subscore; theme_fallback_logic[theme_name] = fallback; theme_types[theme_name] = theme_config_item["type"]
            initial_additional_weights: Dict[str, float] = deepcopy(concept_overrides.get("additional_subscores", {})); suggested_subscores = set(llm_suggestions.get("relevant_subscore_suggestions", [])); master_subscore_list = config.get("master_subscore_list", []); theme_subscore_values = set(filter(None, theme_subscores.values()))
            for subscore_name in suggested_subscores:
                 if subscore_name in master_subscore_list and subscore_name not in theme_subscore_values and subscore_name not in initial_additional_weights: initial_additional_weights[subscore_name] = VR_SENTIMENT_BASELINE_WEIGHT
                 elif subscore_name not in master_subscore_list: logger.warning(f"LLM suggested invalid subscore '{subscore_name}', ignoring.")

            # Apply VR Sentiment Rule before normalization
            lodging_type = affinity_definition["applicable_lodging_types"]; sentiment_theme_name = next((name for name, sub in theme_subscores.items() if sub == "SentimentAffinity"), None); sentiment_subscore = "SentimentAffinity"
            if ("VR" in lodging_type or "Both" in lodging_type): current_sentiment_weight = initial_theme_weights.get(sentiment_theme_name, 0.0) if sentiment_theme_name else 0.0; current_additional_sentiment_weight = initial_additional_weights.get(sentiment_subscore, 0.0);
            if current_sentiment_weight <= 1e-9 and current_additional_sentiment_weight <= 1e-9: logger.warning(f"VR requires '{sentiment_subscore}'. Applying baseline weight {VR_SENTIMENT_BASELINE_WEIGHT}."); initial_additional_weights[sentiment_subscore] = max(initial_additional_weights.get(sentiment_subscore, 0.0), VR_SENTIMENT_BASELINE_WEIGHT)

            # Normalize all weights
            final_theme_weights, final_additional_weights = normalize_weights(initial_theme_weights, initial_additional_weights); affinity_definition["diagnostics"]["stage2"]["normalized_weights"] = {"themes": final_theme_weights, "additional": final_additional_weights}

            # Process each theme
            processed_themes = []
            # failed_theme_fallbacks_info = {} # Re-initialize per concept (done above)
            all_weighted_attributes = [] # Collect all attributes for global top list

            for theme_name, theme_config_item in base_themes_config.items():
                if theme_name not in final_theme_weights or final_theme_weights[theme_name] <= 1e-9: continue
                logger.debug(f"Processing theme: {theme_name}")
                dynamic_rule = theme_rules[theme_name]; final_weight = final_theme_weights[theme_name]; subscore = theme_subscores.get(theme_name); fallback = theme_fallback_logic[theme_name]; theme_type = theme_types[theme_name]

                # Select attributes using Keyword + Semantic matching
                selected_attrs_with_scores = select_theme_attributes(theme_name, theme_config_item["hints"], candidate_evidence, primary_embeddings, uri_embeddings_map, sbert_model, THEME_ATTRIBUTE_SEMANTIC_THRESHOLD, THEME_ATTRIBUTE_KEYWORD_MIN_MATCH, args)
                affinity_definition["diagnostics"]["theme_processing"][theme_name]["candidates_after_slotting"] = len(selected_attrs_with_scores)

                rule_passed, fallback_action = apply_theme_rule(theme_name, dynamic_rule, selected_attrs_with_scores, fallback)

                if rule_passed:
                    # Deduplicate Attributes based on label, keep highest score
                    deduped_attributes: Dict[str, Tuple[str, float]] = {}
                    unique_selected_attributes_with_scores: List[Tuple[str, float]] = []
                    if selected_attrs_with_scores:
                        for uri, score in selected_attrs_with_scores: label = get_primary_label(uri);
                        if label not in deduped_attributes or score > deduped_attributes[label][1]: deduped_attributes[label] = (uri, score)
                        unique_selected_attributes_with_scores = sorted(list(deduped_attributes.values()), key=lambda x: x[1], reverse=True)
                    affinity_definition["diagnostics"]["theme_processing"][theme_name]["attributes_after_dedupe"] = len(unique_selected_attributes_with_scores)

                    # Normalize using the unique list
                    norm_attrs = normalize_concept_weights(unique_selected_attributes_with_scores, final_weight)
                    affinity_definition["diagnostics"]["theme_processing"][theme_name]["normalized_attributes_count"] = len(norm_attrs)

                    if norm_attrs or dynamic_rule == "Optional":
                        theme_entry = {"name": theme_name, "type": theme_type, "rule": dynamic_rule, "theme_weight": round(final_weight, 6), "attributes": []}
                        if subscore: theme_entry["subScore"] = subscore
                        if fallback: theme_entry["fallback_logic"] = fallback
                        theme_summary = llm_suggestions.get("theme_summaries", {}).get(theme_name) # Get LLM summary
                        if theme_summary: theme_entry["theme_summary"] = theme_summary # Add summary if exists
                        for attr_data in norm_attrs:
                             # Add type info if available
                             attr_data["type"] = get_concept_type(attr_data["uri"])
                             theme_entry["attributes"].append(attr_data)
                             # Collect for global top list
                             all_weighted_attributes.append({"uri": attr_data["uri"], "skos:prefLabel": attr_data["skos:prefLabel"], "concept_weight": attr_data["concept_weight"], "source_theme": theme_name})
                        processed_themes.append(theme_entry)
                elif fallback_action:
                    failed_theme_fallbacks_info[theme_name] = fallback_action
                    affinity_definition["diagnostics"]["theme_processing"][theme_name]["rule_failed"] = True

            affinity_definition["themes"] = processed_themes
            # Store info about themes that failed required rules
            if failed_theme_fallbacks_info:
                affinity_definition["failed_fallback_themes"] = failed_theme_fallbacks_info # Add this field

            # Add the additional relevant subscores
            affinity_definition["additional_relevant_subscores"] = [{"subScore": name, "weight": round(weight, 6)} for name, weight in final_additional_weights.items() if weight > 1e-9]

            # Generate Top Defining Attributes list
            logger.info("Generating top defining attributes list...")
            sorted_all_attributes = sorted(all_weighted_attributes, key=lambda x: x["concept_weight"], reverse=True)
            affinity_definition["top_defining_attributes"] = sorted_all_attributes[:TOP_DEFINING_ATTRIBUTES_COUNT]
            affinity_definition["diagnostics"]["stage2"]["top_defining_attributes_count"] = len(affinity_definition["top_defining_attributes"])

            affinity_definition["processing_metadata"]["status"] = "Success"

        except StopIteration: logger.warning(f"Processing stopped early for '{concept_input_string}'. Status: {affinity_definition['processing_metadata']['status']}")
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing '{concept_input_string}': {e}", exc_info=True); affinity_definition["processing_metadata"]["status"] = f"Failed - Exception: {type(e).__name__}";
            if args.debug and 'error_details' not in affinity_definition["diagnostics"]: affinity_definition["diagnostics"]["error_details"] = traceback.format_exc()
        finally:
            affinity_definition["processing_metadata"]["duration_seconds"] = round(time.time() - concept_start_time, 2)
            affinity_definition["diagnostics"]["final_output"]["themes_count"] = len(affinity_definition.get("themes", [])); affinity_definition["diagnostics"]["final_output"]["additional_subscores_count"] = len(affinity_definition.get("additional_relevant_subscores", [])); affinity_definition["diagnostics"]["final_output"]["must_not_have_count"] = len(affinity_definition.get("must_not_have", [])); affinity_definition["diagnostics"]["final_output"]["top_defining_attributes_count"] = len(affinity_definition.get("top_defining_attributes", []))
            try: affinity_definition["diagnostics"] = json.loads(json.dumps(affinity_definition["diagnostics"], default=str))
            except Exception as json_e: logger.error(f"Error converting diagnostics to JSON: {json_e}")
            all_definitions.append(affinity_definition)
            logger.info(f"--- Finished processing '{concept_input_string}' in {affinity_definition['processing_metadata']['duration_seconds']:.2f}s. Status: {affinity_definition['processing_metadata']['status']} ---")

    # --- Save Results ---
    logger.info(f"\n=== Finished processing all {len(all_definitions)} concepts in {time.time() - start_process_time:.2f}s ===")
    output_file = os.path.join(args.output_dir, f"affinity_definitions_{CACHE_VERSION}.json")
    logger.info(f"Saving results to {output_file}...")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_definitions, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}", exc_info=True)


# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Affinity Definitions (v30.1 - Robust Features & Slotting)") # Version update
    parser.add_argument("--concepts", required=True, help="Input concepts file.")
    parser.add_argument("--taxonomy-dir", required=True, help="Taxonomy RDF directory.")
    parser.add_argument("--output-dir", default="./output_v30", help="Output directory.")
    parser.add_argument("--cache-dir", default="./cache_v30", help="Cache directory.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--associations-file", default=DEFAULT_ASSOCIATIONS_FILE, help="Associations JSON.")
    parser.add_argument("--corpus-file", default=DEFAULT_CORPUS_FILE, help="Corpus text file.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild caches.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of concepts.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--openai-model", default=OPENAI_MODEL, help=f"OpenAI model to use (default: {OPENAI_MODEL})")

    # Define args globally within main guard
    args = parser.parse_args()

    # Update global OPENAI_MODEL from parsed args
    OPENAI_MODEL = args.openai_model

    # Setup logging level based on debug flag
    if args.debug: logger.setLevel(logging.DEBUG); [h.setLevel(logging.DEBUG) for h in logging.getLogger().handlers]; logger.info("DEBUG logging enabled.")
    else: logger.setLevel(logging.INFO); [h.setLevel(logging.INFO) for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]

    # Critical Setup
    config = load_affinity_config(args.config_file)
    if config is None: sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"): logger.critical("FATAL: OPENAI_API_KEY env var not set."); sys.exit(1)
    os.makedirs(args.cache_dir, exist_ok=True); os.makedirs(args.output_dir, exist_ok=True)

    # Read input concepts
    input_concepts: List[str] = []
    try:
        with open(args.concepts, 'r', encoding='utf-8') as f: input_concepts = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(input_concepts)} concepts from {args.concepts}")
        if args.limit and args.limit < len(input_concepts): input_concepts = input_concepts[:args.limit]; logger.info(f"Limiting to first {args.limit}.")
        if not input_concepts: logger.error("No concepts found."); sys.exit(1)
    except FileNotFoundError: logger.critical(f"Input concepts file not found: {args.concepts}"); sys.exit(1)
    except Exception as e: logger.critical(f"Error reading input concepts file: {e}"); sys.exit(1)

    logger.info(f"+++ Running Affinity Definition Engine v30.1r3 +++") # Version update
    logger.info(f"  Config File: {args.config_file}"); logger.info(f"  Mode: Hybrid Rule-Based Engine + LLM Assist"); logger.info(f"  LLM Model: {OPENAI_MODEL}"); logger.info(f"  Base Themes Defined: {len(config.get('base_themes', {}))}")

    # --- Execute Main Logic ---
    generate_affinity_definitions(input_concepts, config, args)

    logger.info("=== Affinity Generation Script Finished ===")