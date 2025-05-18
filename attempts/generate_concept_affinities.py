#!/usr/bin/env python3
"""
Generate affinity definitions for travel concepts (v29.4 Refined) - LLM Full Gen + Hints + Robust Python Essential Check:
1. Retrieves relevant KG facts (K=150).
2. Constructs prompt asking GPT-4o for the full structured JSON, including BASE THEME HINTS.
3. Calls OpenAI API (gpt-4o) to generate the main definition structure.
4. Parses the LLM response.
5. Python performs a Post-LLM check: Iterates through *all* relevant identified archetypes. If any archetype's key matches POST_LLM_ESSENTIALS_MAP, ensures its critical predefined essential amenities are present, adding them to supplementary if missing.
6. Saves the potentially augmented result.

Version: 2025-04-16-affinity-kg-openai-v29.4 (LLM Full Gen + Hints + Robust Essential Check)
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
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

# --- Handle Critical Dependency Imports / Optional Imports / Logging Setup ---
try: from rdflib import Graph, Namespace, URIRef, Literal, RDF, util
except ImportError: print("CRITICAL ERROR: rdflib library not found.", file=sys.stderr); sys.exit(1)
try: from sentence_transformers import SentenceTransformer
except ImportError: print("CRITICAL ERROR: sentence-transformers library not found.", file=sys.stderr); sys.exit(1)
try: from sklearn.metrics.pairwise import cosine_similarity
except ImportError: print("CRITICAL ERROR: scikit-learn library not found.", file=sys.stderr); sys.exit(1)
NLTK_AVAILABLE = False; STOP_WORDS = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'to', 'and', 'or', 'is', 'are', 'was', 'were'])
try: from nltk.corpus import stopwords; STOP_WORDS = set(stopwords.words('english')); NLTK_AVAILABLE = True; print("Info: NLTK stopwords loaded.")
except (ImportError, LookupError) as e: print(f"Warning: NLTK stopwords not found ({type(e).__name__}). Using basic list.")
def tqdm_dummy(iterable, *args, **kwargs): return iterable
tqdm = tqdm_dummy
try: from tqdm import tqdm as real_tqdm; tqdm = real_tqdm; print("Info: tqdm loaded.")
except ImportError: print("Warning: tqdm not found, progress bars disabled.")
log_filename = "affinity_generation_v29.4.log" # <-- v29.4 Log filename
if os.path.exists(log_filename):
    try: os.remove(log_filename); print(f"Removed old log file: {log_filename}")
    except OSError as e: print(f"Warning: Could not remove old log file {log_filename}: {e}")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", handlers=[logging.FileHandler(log_filename, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("affinity_generator_v29_4") # <-- v29.4 Logger name

# --- Namespaces ---
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#"); RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/"); OWL = Namespace("http://www.w3.org/2002/07/owl#")

# --- Configuration & Constants ---
CACHE_VERSION = "v20250416.affinity.29.4" # <-- v29.4 Cache version
DEFAULT_ASSOCIATIONS_FILE = "./datasources/concept_associations.json" # Needed by find_best_taxonomy_matches
DEFAULT_CORPUS_FILE = "./datasources/travel-terms-corpus.txt"       # Needed by find_best_taxonomy_matches
MIN_RELEVANCE_THRESHOLD = 0.30; MAX_ANCHOR_MATCHES = 5
PROPERTY_WEIGHTS = {"prefLabel": 1.0, "altLabel": 0.95, "rdfsLabel": 0.9, "definition": 0.80, "dctermsDescription": 0.80, "scopeNote": 0.75, "hiddenLabel": 0.7}
TRAVEL_CONTEXT = "travel "
SYNONYM_SIMILARITY_THRESHOLD = 0.55; MAX_SYNONYMS = 20;
ASSOCIATION_BOOST = 0.40; EXACT_MATCH_BONUS = 0.01; KEYWORD_WEIGHT = 0.50; SEMANTIC_WEIGHT = 0.50;
# KG Retrieval Config
KG_RETRIEVAL_TOP_K = 150
KG_RETRIEVAL_SIMILARITY_THRESHOLD = 0.35
# Archetype Config
CORE_ARCHETYPES = ["luxury travel", "budget travel", "family travel", "business travel", "romantic travel", "adventure travel", "beach vacation", "ski vacation", "city break", "nature retreat", "spa and wellness", "cultural tourism", "historic tourism", "eco tourism"]
ARCHETYPE_SIMILARITY_THRESHOLD = 0.5
MAX_RELEVANT_ARCHETYPES = 3 # Keep checking top 3
# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"
OPENAI_TIMEOUT = 180
OPENAI_MAX_RETRIES = 2
# Theme/Context Names & Base Themes (for prompting) - Unchanged from v29.3
THEME_NAMES = ["Location", "Technology", "Sentiment", "Indoor Amenities", "Outdoor Amenities", "Activities", "Spaces", "Events", "Seasonality", "Group Relevance", "Privacy", "Accessibility", "Sustainability"]
CONTEXT_SECTION_NAMES = ["technological_integration", "sustainability", "accessibility", "temporal_considerations", "economic_context", "activity_level", "cultural_context", "accommodation_type", "user_experience"]
BASE_THEMES = { # Provide hints to LLM
    "Location": {"hints": ["location", "prime location", "exclusive neighborhood", "city center", "downtown", "waterfront", "oceanfront", "beachfront", "convenient", "parking", "transportation", "nearby", "view", "secluded location"]},
    "Technology": {"hints": ["wifi", "internet", "high speed", "charging", "smart tv", "streaming", "media hub", "smart room", "tablet", "sound system", "digital key", "mobile app", "contactless", "business center", "computer"]},
    "Sentiment": {"hints": ["luxury", "opulent", "exclusive", "sophisticated", "elegant", "upscale", "relaxing", "vibrant", "charming", "boutique", "unique", "atmosphere", "ambiance", "quiet", "peaceful", "calm"]},
    "Indoor Amenities": {"hints": ["spa", "sauna", "hot tub", "indoor pool", "bar", "lounge", "restaurant", "michelin", "chef", "wine cellar", "gym", "fitness", "library", "casino", "cinema", "concierge", "butler service", "valet", "turndown"]},
    "Outdoor Amenities": {"hints": ["pool", "outdoor pool", "private pool", "rooftop pool", "poolside bar", "garden", "patio", "terrace", "balcony", "bbq", "fire pit", "view", "beach access", "private beach", "helipad", "yacht", "dock"]},
    "Activities": {"hints": ["hiking", "skiing", "golf", "water sports", "yoga", "cycling", "nightlife", "entertainment", "classes", "tours", "shopping", "wine tasting", "cultural", "historic site", "spa treatments"]},
    "Spaces": {"hints": ["suite", "penthouse", "villa", "balcony", "jacuzzi", "kitchen", "kitchenette", "living room", "dining area", "workspace", "multiple bedrooms"]},
    "Events": {"hints": ["festival", "wedding", "concert", "conference", "gala", "party", "event space", "meeting facilities", "historic building", "cultural event"]},
    "Seasonality": {"hints": ["seasonal", "winter", "summer", "autumn", "spring", "peak season", "off season", "monsoon", "holiday season"]},
    "Group Relevance": {"hints": ["family friendly", "kid friendly", "group travel", "solo traveler", "couples", "adults only", "business traveler", "romantic getaway"]},
    "Privacy": {"hints": ["private", "secluded", "intimate", "quiet", "exclusive access", "gated", "soundproof", "discreet", "personal space", "private entrance"]},
    "Accessibility": {"hints": ["accessible", "wheelchair", "ramp", "elevator", "lift", "disabled", "step free", "roll in shower", "braille", "hearing loop", "ada"]},
    "Sustainability": {"hints": ["sustainable", "eco friendly", "green", "responsible", "carbon neutral", "conservation", "local source", "fair trade", "certified", "renewable energy", "low impact", "community", "geotourism"]}
}
MAX_ATTRIBUTES_PER_THEME_LLM = 10 # Instruction for LLM prompt

# --- Post-LLM Essential Amenity Check Config ---
ENABLE_POST_LLM_ESSENTIAL_CHECK = True # <<< ENABLED BY DEFAULT
POST_LLM_ESSENTIALS_MAP: Dict[str, Dict[str, str]] = {
    # Map primary archetype key (lowercase) -> Dict[URI -> Fallback Label]
    "luxury": {
         "urn:expediagroup:taxonomies:core:#efb0858e-9bf5-3650-af78-43028139b0f8": "WiFi",
         "urn:expediagroup:taxonomies:lcm:#397e8e23-1f70-39fe-ba3f-b4995f06c0ea": "Private pool",
         "urn:expediagroup:taxonomies:acsBaseAttribute:#ButlerService": "ButlerService",
         "urn:expediagroup:taxonomies:acs:#3adeb20e-b4f9-4d2c-bb57-86f6b6c3b0e8": "Smart TV",
    },
    "budget": {
        # "urn:expediagroup:taxonomies:lcm:#7f27512a-acc9-3157-bba7-94b8b3af79f2": "Wireless Internet access-free",
    },
    "family": {
        # Example: Playground, Kids Club
        # "urn:expediagroup:taxonomies:lcm:#4210a09a-1bf5-30f1-bc2c-c53e0130cebe": "Playground on site",
        # "urn:expediagroup:taxonomies:lcm:#efcb861d-c61e-36f0-858b-f9e0491cc4f6": "Kids club"
    },
    "business": {
         # Example: Business Center
         # "urn:expediagroup:taxonomies:lcm:#7720239d-ea4d-3469-a6d4-b6485b7d96a7": "Business center"
    }
}

# --- Caches ---
_model_instance: Optional[SentenceTransformer] = None; _taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict, Dict, List]] = None;
_corpus_data_cache: Optional[Dict[str, np.ndarray]] = None;
_concept_associations_cache: Optional[Dict[str, List[str]]] = None;
_archetype_embeddings_cache: Optional[Dict[str, np.ndarray]] = None
_openai_client = None

# --- Utility Functions ---
# (Keep get_sbert_model, normalize_concept, calculate_semantic_similarity)
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
    try:
        norm = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept); norm = norm.replace("-", " ").replace("_", " ")
        norm = re.sub(r'[^\w\s]|(\'s\b)', '', norm); norm = ' '.join(norm.lower().split())
        return norm
    except Exception: return concept.lower().strip() if isinstance(concept, str) else ""

def calculate_semantic_similarity(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray], term1: str = "term1", term2: str = "term2") -> float:
    if embedding1 is None or embedding2 is None: return 0.0
    try:
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray): return 0.0
        if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
        if embedding1.shape[1] == 0 or embedding2.shape[1] == 0: return 0.0
        if embedding1.shape[1] != embedding2.shape[1]: logger.error(f"Dim mismatch: {term1}{embedding1.shape} vs {term2}{embedding2.shape}"); return 0.0
        sim = cosine_similarity(embedding1, embedding2)[0][0]; similarity = max(0.0, min(1.0, float(sim)))
        return similarity
    except Exception as e: logger.warning(f"Similarity error '{term1[:50]}' vs '{term2[:50]}': {e}"); return 0.0

def get_primary_label(uri: str, fallback: Optional[str] = None) -> str: # Added fallback parameter
    """Gets the best label for a URI from the cache, falling back if needed."""
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        details = _taxonomy_concepts_cache[uri]
        # Prioritize labels
        if details.get("prefLabel"): return details["prefLabel"][0]
        if details.get("altLabel"): return details["altLabel"][0]
        if details.get("rdfsLabel"): return details["rdfsLabel"][0]
        # Use fallback if provided and no labels found
        if fallback: return fallback
        # Fallback to definition snippet or URI fragment
        if details.get("definition"): return details["definition"][0][:60] + "..."
        if details.get("hiddenLabel"): return details["hiddenLabel"][0] # Last resort label
    # Fallback to URI parsing if not in cache or no suitable property found
    try:
        if '#' in uri: return uri.split('#')[-1]
        if '/' in uri: return uri.split('/')[-1]
    except: pass
    # Final fallback
    return fallback if fallback else uri


# --- Loading Functions ---
# (Keep load_taxonomy_concepts, precompute_taxonomy_embeddings, get_archetype_embeddings)
# (Keep load_concept_associations, load_corpus_data)
def load_concept_associations(associations_file: Optional[str]) -> Dict[str, List[str]]:
    global _concept_associations_cache
    if _concept_associations_cache is not None: return _concept_associations_cache
    if not associations_file: logger.info("No associations file."); _concept_associations_cache = {}; return {}
    if not os.path.exists(associations_file): logger.warning(f"Associations file not found: {associations_file}"); _concept_associations_cache = {}; return {}
    logger.info(f"Loading associations from {associations_file}"); start_time = time.time()
    try:
        with open(associations_file, 'r', encoding='utf-8') as f: data = json.load(f)
        normalized_associations: Dict[str, List[str]] = {}
    except Exception as e: logger.error(f"Error loading associations: {e}", exc_info=True); _concept_associations_cache = {}; return {}
    for key, value_in in data.items():
        normalized_key = normalize_concept(key).lower()
        if not normalized_key: continue
        associated_values: List[str] = []
        if isinstance(value_in, list): associated_values = [normalize_concept(v).lower() for v in value_in if isinstance(v, str) and normalize_concept(v)]
        elif isinstance(value_in, str):
            normalized_value = normalize_concept(value_in).lower()
            if normalized_value: associated_values.append(normalized_value)
        if associated_values: normalized_associations[normalized_key] = associated_values
    _concept_associations_cache = normalized_associations; logger.info(f"Loaded {len(normalized_associations)} associations in {time.time() - start_time:.2f}s.")
    return _concept_associations_cache

def load_taxonomy_concepts(taxonomy_dir: str, cache_file: str, args: argparse.Namespace) -> Optional[Dict[str, Dict]]:
    global _taxonomy_concepts_cache
    if _taxonomy_concepts_cache is not None: return _taxonomy_concepts_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Loading concepts cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f: cached_data = json.load(f)
            if cached_data.get("cache_version") == CACHE_VERSION:
                concepts_data = cached_data.get("data")
                if isinstance(concepts_data, dict): _taxonomy_concepts_cache = concepts_data; cache_valid = True; logger.info(f"Loaded {len(_taxonomy_concepts_cache)} concepts from cache.")
                else: logger.warning("Concept cache invalid. Rebuilding.")
            else: logger.info(f"Concept cache version mismatch ({cached_data.get('cache_version')} vs {CACHE_VERSION}). Rebuilding.")
        except Exception as e: logger.warning(f"Failed concept cache load: {e}. Rebuilding.")
    if not cache_valid:
        logger.info(f"Loading concepts from RDF: {taxonomy_dir}"); start_time = time.time()
        concepts = defaultdict(lambda: defaultdict(list)); files_ok = 0; total_err = 0
        try:
            if not os.path.isdir(taxonomy_dir): raise FileNotFoundError(f"Taxonomy dir not found: {taxonomy_dir}")
            rdf_files = [f for f in os.listdir(taxonomy_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.xml', '.jsonld', '.nt', '.n3'))]
            if not rdf_files: logger.error(f"No RDF files: {taxonomy_dir}"); return None
            g = Graph(); logger.info(f"Parsing {len(rdf_files)} files...")
            disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
            for fn in tqdm(rdf_files, desc="Parsing RDF", disable=disable_tqdm):
                fp = os.path.join(taxonomy_dir, fn)
                try:
                    fmt = util.guess_format(fp)
                    if fmt: g.parse(fp, format=fmt); files_ok += 1
                    else: logger.warning(f"Unknown format: {fn}")
                except Exception as e: total_err += 1; logger.error(f"Error parsing {fn}: {e}", exc_info=args.debug)
            logger.info(f"Parsed {files_ok}/{len(rdf_files)} files ({total_err} errors).")
            if files_ok == 0: logger.error("No RDF parsed."); return None
            logger.info("Extracting concepts..."); pot_uris = set(s for s, p, o in g if isinstance(s, URIRef)) | set(o for s, p, o in g if isinstance(o, URIRef))
            logger.info(f"Found {len(pot_uris)} URIs. Processing..."); skip_dep, rem_empty = 0, 0
            lbl_props = {SKOS.prefLabel: "prefLabel", SKOS.altLabel: "altLabel", RDFS.label: "rdfsLabel", SKOS.hiddenLabel: "hiddenLabel"}
            txt_props = {SKOS.definition: "definition", DCTERMS.description: "dctermsDescription", SKOS.scopeNote: "scopeNote"}
            rel_props = {SKOS.broader: "broader", SKOS.narrower: "narrower", SKOS.related: "related", OWL.sameAs: "sameAs"}
            kept_concepts_data = defaultdict(lambda: defaultdict(list))
            for uri in tqdm(pot_uris, desc="Processing URIs", disable=disable_tqdm):
                uri_s = str(uri)
                if g.value(uri, OWL.deprecated) == Literal(True): skip_dep += 1; continue
                is_concept = (uri, RDF.type, SKOS.Concept) in g or (uri, RDF.type, RDFS.Class) in g
                has_properties = False; current_uri_data = defaultdict(list)
                for prop, key in {**lbl_props, **txt_props, **rel_props}.items():
                    for obj in g.objects(uri, prop):
                        val = None
                        if isinstance(obj, Literal): val = str(obj).strip()
                        elif isinstance(obj, URIRef) and key in rel_props: val = str(obj)
                        if val: current_uri_data[key].append(val); has_properties = True
                if is_concept or has_properties:
                    processed_data = {k: list(set(v)) for k, v in current_uri_data.items() if v}
                    if processed_data: kept_concepts_data[uri_s] = processed_data
                    else: rem_empty += 1
                else: rem_empty += 1
            logger.info(f"Extracted {len(kept_concepts_data)} concepts. Skipped {skip_dep}, removed {rem_empty}.")
            _taxonomy_concepts_cache = dict(kept_concepts_data)
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_data = {"cache_version": CACHE_VERSION, "data": _taxonomy_concepts_cache}
                with open(cache_file, 'w', encoding='utf-8') as f: json.dump(cache_data, f, indent=2)
                logger.info(f"Saved concepts cache: {cache_file}")
            except Exception as e: logger.error(f"Failed writing concepts cache: {e}")
            logger.info(f"Taxonomy loading took {time.time() - start_time:.2f}s.")
        except FileNotFoundError as e: logger.error(f"Config error: {e}"); return None
        except Exception as e: logger.error(f"Taxonomy load error: {e}", exc_info=args.debug); return None
    if not _taxonomy_concepts_cache: logger.error("Failed concept load."); return None
    return _taxonomy_concepts_cache

def precompute_taxonomy_embeddings(
    taxonomy_concepts: Dict[str, Dict],
    sbert_model: SentenceTransformer,
    cache_file: str,
    args: argparse.Namespace
) -> Optional[Tuple[Dict[str, List[Tuple[str, str, np.ndarray, str]]], Dict[str, np.ndarray], List[str]]]:
    global _taxonomy_embeddings_cache
    if _taxonomy_embeddings_cache is not None: return _taxonomy_embeddings_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Loading embeddings cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached_data = pickle.load(f)
            if cached_data.get("cache_version") == CACHE_VERSION:
                uri_embeddings_map = cached_data.get("uri_embeddings_map"); primary_embeddings = cached_data.get("primary_embeddings"); uris_list = cached_data.get("uris_list")
                if isinstance(uri_embeddings_map, dict) and isinstance(primary_embeddings, dict) and isinstance(uris_list, list):
                    try:
                        sbert_dim = sbert_model.get_sentence_embedding_dimension()
                        is_valid = all(isinstance(v, np.ndarray) and v.ndim == 1 and v.shape == (sbert_dim,) for v in primary_embeddings.values() if v is not None)
                        if is_valid and set(primary_embeddings.keys()) == set(uris_list): _taxonomy_embeddings_cache = (uri_embeddings_map, primary_embeddings, uris_list); cache_valid = True; logger.info(f"Loaded {len(uris_list)} embeddings from cache.")
                        else: logger.warning(f"Embeddings cache invalid structure/dimensions. Rebuilding.")
                    except Exception as dim_e: logger.warning(f"Error validating embedding dimensions: {dim_e}. Rebuilding.")
                else: logger.warning("Embedding cache structure invalid. Rebuilding.")
            else: logger.info(f"Embedding cache version mismatch ({cached_data.get('cache_version')} vs {CACHE_VERSION}). Rebuilding.")
        except Exception as e: logger.warning(f"Failed embedding cache load: {e}. Rebuilding.")
    if not cache_valid:
        logger.info("Pre-computing embeddings..."); start_time = time.time(); uri_embeddings_map = defaultdict(list); primary_embeddings: Dict[str, Optional[np.ndarray]] = {}
        uris_list_all: List[str] = []; texts_to_embed_map = defaultdict(list); all_valid_uris = list(taxonomy_concepts.keys())
        logger.info(f"Processing {len(all_valid_uris)} concepts for text."); disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
        text_properties = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel", "definition", "dctermsDescription", "scopeNote"]
        for uri in tqdm(all_valid_uris, desc="Gathering Texts", disable=disable_tqdm):
            concept_data = taxonomy_concepts.get(uri, {}); uris_list_all.append(uri); found_texts_for_uri: Set[Tuple[str, str, str]] = set()
            for prop_key in text_properties:
                values = concept_data.get(prop_key, []); values = values if isinstance(values, list) else [values] if isinstance(values, str) else []
                for text_value in values:
                    if text_value and isinstance(text_value, str):
                        original_text = text_value.strip(); normalized_text = normalize_concept(original_text)
                        if normalized_text: found_texts_for_uri.add((prop_key, original_text, normalized_text))
            for prop_key, original_text, normalized_text in found_texts_for_uri: texts_to_embed_map[normalized_text].append((uri, prop_key, original_text))
        unique_normalized_texts = list(texts_to_embed_map.keys()); logger.info(f"Collected {len(unique_normalized_texts)} unique texts.")
        embeddings_list: Optional[List[np.ndarray]] = None; texts_encoded: List[str] = []; embedding_map: Dict[str, Optional[np.ndarray]] = {}
        if unique_normalized_texts:
            logger.info("Generating embeddings..."); batch_size = 128
            try:
                texts_encoded = [text for text in unique_normalized_texts if isinstance(text, str)]
                if texts_encoded:
                    embeddings_list = sbert_model.encode(texts_encoded, batch_size=batch_size, show_progress_bar=logger.isEnabledFor(logging.INFO))
                    embedding_map = {text: emb for text, emb in zip(texts_encoded, embeddings_list) if emb is not None}
                    logger.info(f"Embedded {len(embedding_map)} texts.")
                else: logger.warning("No valid texts to encode."); embeddings_list = []
            except Exception as e: logger.error(f"SBERT encoding failed: {e}", exc_info=True); raise RuntimeError("SBERT Encoding Failed") from e
        else: logger.warning("No texts to embed."); embeddings_list = []
        logger.info("Mapping & selecting primary embeddings..."); primary_embedding_candidates = defaultdict(dict)
        sbert_dim = sbert_model.get_sentence_embedding_dimension()
        for normalized_text, associated_infos in texts_to_embed_map.items():
            embedding = embedding_map.get(normalized_text)
            if embedding is None or not isinstance(embedding, np.ndarray) or embedding.shape != (sbert_dim,): continue
            for uri, prop_key, original_text in associated_infos:
                uri_embeddings_map[uri].append((prop_key, original_text, embedding, normalized_text))
                primary_embedding_candidates[uri][prop_key] = embedding
        primary_property_priority = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel", "definition", "dctermsDescription", "scopeNote"]
        num_with_primary = 0; num_without_primary = 0
        for uri in tqdm(uris_list_all, desc="Selecting Primary", disable=disable_tqdm):
            candidates = primary_embedding_candidates.get(uri, {}); chosen_embedding = None
            for prop in primary_property_priority:
                if prop in candidates and candidates[prop] is not None:
                    candidate_emb = candidates[prop]
                    if isinstance(candidate_emb, np.ndarray) and candidate_emb.ndim == 1 and candidate_emb.shape == (sbert_dim,): chosen_embedding = candidate_emb; break
            if chosen_embedding is not None:
                if isinstance(chosen_embedding, np.ndarray) and chosen_embedding.ndim == 1 and chosen_embedding.shape == (sbert_dim,): primary_embeddings[uri] = chosen_embedding; num_with_primary += 1
                else: logger.warning(f"Chosen embed for {uri} invalid {type(chosen_embedding)}. None."); primary_embeddings[uri] = None; num_without_primary += 1
            else:
                primary_embeddings[uri] = None; num_without_primary += 1
        final_uris_list = [uri for uri in uris_list_all if primary_embeddings.get(uri) is not None]
        final_primary_embeddings = {uri: emb for uri, emb in primary_embeddings.items() if emb is not None}
        final_uri_embeddings_map = {uri: data for uri, data in uri_embeddings_map.items() if uri in final_primary_embeddings}
        _taxonomy_embeddings_cache = (final_uri_embeddings_map, final_primary_embeddings, final_uris_list)
        logger.info(f"Finished embedding in {time.time() - start_time:.2f}s. Concepts: {len(uris_list_all)}, With primary: {num_with_primary}, Without: {num_without_primary}.")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data = {"cache_version": CACHE_VERSION, "uri_embeddings_map": _taxonomy_embeddings_cache[0], "primary_embeddings": _taxonomy_embeddings_cache[1], "uris_list": _taxonomy_embeddings_cache[2]}
            with open(cache_file, 'wb') as f: pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved embeddings cache: {cache_file}")
        except Exception as e: logger.error(f"Failed writing embeddings cache: {e}")
    if not _taxonomy_embeddings_cache or not _taxonomy_embeddings_cache[1]: logger.error("Embedding process failed."); return None
    logger.info(f"Using {len(_taxonomy_embeddings_cache[1])} concepts with valid primary embeddings.")
    return _taxonomy_embeddings_cache

def load_corpus_data(corpus_file: Optional[str], sbert_model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    global _corpus_data_cache
    if _corpus_data_cache is not None: return _corpus_data_cache
    if not corpus_file: logger.info("No corpus file."); _corpus_data_cache = {}; return {}
    cache_valid = False; corpus_abs_path = os.path.abspath(corpus_file) if corpus_file else None; rebuild = args.rebuild_cache
    if not rebuild and os.path.exists(cache_file):
        logger.info(f"Loading corpus cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached = pickle.load(f)
            if cached.get("cache_version") == CACHE_VERSION and cached.get("corpus_file_path") == corpus_abs_path:
                data = cached.get("data"); _corpus_data_cache = data if isinstance(data, dict) else {}; logger.info(f"Loaded {len(_corpus_data_cache)} corpus terms."); cache_valid = True
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
            _corpus_data_cache = {t: e for t, e in zip(unique, embeds) if e is not None and isinstance(e, np.ndarray)}
            logger.info(f"Finished corpus embedding ({len(_corpus_data_cache)}) in {time.time()-start:.2f}s.")
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_data = {"cache_version": CACHE_VERSION, "corpus_file_path": corpus_abs_path, "data": _corpus_data_cache}
                with open(cache_file, 'wb') as f: pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved corpus cache: {cache_file}")
            except Exception as e: logger.error(f"Failed writing corpus cache: {e}")
        except Exception as e: logger.error(f"Error processing corpus: {e}", exc_info=args.debug); _corpus_data_cache = {}
    return _corpus_data_cache

def get_archetype_embeddings(sbert_model: SentenceTransformer) -> Dict[str, np.ndarray]:
    global _archetype_embeddings_cache
    if _archetype_embeddings_cache is None:
        logger.info(f"Embedding {len(CORE_ARCHETYPES)} core archetypes...")
        try:
            embeddings = sbert_model.encode(CORE_ARCHETYPES, batch_size=32, show_progress_bar=False)
            _archetype_embeddings_cache = {text: emb for text, emb in zip(CORE_ARCHETYPES, embeddings) if emb is not None}
            logger.info(f"Successfully embedded {len(_archetype_embeddings_cache)} archetypes.")
        except Exception as e: logger.error(f"Failed archetype embedding: {e}", exc_info=True); _archetype_embeddings_cache = {}
    return _archetype_embeddings_cache

# --- Core Logic Functions ---
def get_dynamic_synonyms(concept: str, concept_embedding: Optional[np.ndarray], uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]], corpus_data: Dict[str, np.ndarray], args: argparse.Namespace) -> List[Tuple[str, float]]:
    synonyms_dict: Dict[str, float] = {}; normalized_input_lower = normalize_concept(concept).lower().strip()
    if concept_embedding is None or not isinstance(concept_embedding, np.ndarray) or concept_embedding.ndim == 0: logger.warning(f"Invalid embedding for '{concept}'."); return []
    if concept_embedding.ndim == 1: concept_embedding = concept_embedding.reshape(1, -1)
    start_time = time.time(); processed_texts: Set[str] = {normalized_input_lower}
    tax_syn_count, corp_syn_count = 0, 0; disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    if uri_embeddings_map:
        all_taxonomy_snippets = [snippet for snippets in uri_embeddings_map.values() for snippet in snippets]
        for prop, orig_text, emb, norm_text in tqdm(all_taxonomy_snippets, desc="Synonyms (Taxonomy)", disable=disable_tqdm, leave=False):
            if not norm_text or norm_text in processed_texts: continue
            similarity = calculate_semantic_similarity(concept_embedding, emb, normalized_input_lower, norm_text)
            if similarity >= SYNONYM_SIMILARITY_THRESHOLD:
                synonyms_dict[norm_text] = max(similarity, synonyms_dict.get(norm_text, 0.0)); tax_syn_count += 1
            processed_texts.add(norm_text)
    if corpus_data:
        for term, emb in tqdm(corpus_data.items(), desc="Synonyms (Corpus)", disable=disable_tqdm, leave=False):
            if not term or term in processed_texts: continue
            similarity = calculate_semantic_similarity(concept_embedding, emb, normalized_input_lower, term)
            if similarity >= SYNONYM_SIMILARITY_THRESHOLD:
                synonyms_dict[term] = max(similarity, synonyms_dict.get(term, 0.0)); corp_syn_count += 1
            processed_texts.add(term)
    synonym_list = sorted([(k,v) for k,v in synonyms_dict.items() if v > 0], key=lambda item: item[1], reverse=True)
    final_synonyms = synonym_list[:MAX_SYNONYMS]
    logger.info(f"Found {len(synonym_list)} synonyms ({tax_syn_count} tax, {corp_syn_count} corpus). Ret {len(final_synonyms)} in {time.time() - start_time:.2f}s.")
    return final_synonyms

def find_best_taxonomy_matches(
    concept: str, sbert_model: SentenceTransformer, uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]],
    primary_embeddings: Dict[str, np.ndarray], uris_list: List[str], associations: Dict[str, List[str]],
    corpus_data: Dict[str, np.ndarray], taxonomy_concepts: Dict[str, Dict], args: argparse.Namespace
) -> Tuple[Optional[np.ndarray], List[Tuple[str, float, str, Dict[str, Any]]]]:
    logger.info(f"--- Finding best taxonomy matches for: '{concept}' ---"); start_time = time.time()
    norm_concept_lower = normalize_concept(concept).lower().strip(); concept_embedding = None
    if not norm_concept_lower: logger.warning("Input concept normalized empty."); return None, []
    try:
        input_text = (TRAVEL_CONTEXT + norm_concept_lower) if TRAVEL_CONTEXT else norm_concept_lower
        concept_embedding = sbert_model.encode([input_text])[0]
        if concept_embedding is None or not isinstance(concept_embedding, np.ndarray): raise ValueError("Embedding failed")
    except Exception as e: logger.error(f"SBERT embedding error for '{concept}': {e}", exc_info=True); return None, []
    dynamic_synonyms = get_dynamic_synonyms(concept, concept_embedding, uri_embeddings_map, corpus_data, args)
    synonym_texts = {syn for syn, score in dynamic_synonyms}; concept_scores: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"max_score": 0.0, "matched_texts": set(), "details": []})
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    keywords = [kw for kw in norm_concept_lower.split() if kw not in STOP_WORDS and len(kw) > 1]
    kw_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords]
    for uri in tqdm(uris_list, desc="Scoring Concepts", disable=disable_tqdm, leave=False):
        uri_data = uri_embeddings_map.get(uri, []); primary_emb = primary_embeddings.get(uri); max_score = 0.0; details = []; matched = set()
        if primary_emb is not None:
            pri_sem_score = calculate_semantic_similarity(concept_embedding, primary_emb, norm_concept_lower, f"PrimaryEmb({uri})")
            max_score = pri_sem_score; details.append(f"PrimarySem({pri_sem_score:.3f})")
            pref_labels = [orig for prop, orig, emb, norm in uri_data if prop == 'prefLabel'];
            if pref_labels: matched.add(pref_labels[0])
        for prop, orig, txt_emb, norm in uri_data:
            prop_w = PROPERTY_WEIGHTS.get(prop, 0.5); kw_score = 0.0; sem_score = 0.0
            kw_count = sum(1 for p in kw_patterns if p.search(orig) or p.search(norm)) if keywords else 0
            if kw_count > 0: kw_score = min((kw_count/len(keywords) if keywords else 0)+(EXACT_MATCH_BONUS if norm_concept_lower == norm else 0), 1.0)
            sem_score = calculate_semantic_similarity(concept_embedding, txt_emb, norm_concept_lower, f"{prop}({norm[:30]}...)")
            comb_score = (kw_score * KEYWORD_WEIGHT + sem_score * SEMANTIC_WEIGHT) * prop_w
            if norm_concept_lower in associations and any(a in norm for a in associations[norm_concept_lower]): comb_score = min(1.0, comb_score + ASSOCIATION_BOOST)
            if norm in synonym_texts: syn_s = next((s for syn, s in dynamic_synonyms if syn == norm), 0.0); comb_score = min(1.0, comb_score + (syn_s * 0.2))
            comb_score = max(0.0, min(1.0, comb_score))
            if comb_score > max_score: max_score = comb_score; details.append(f"{prop}(KW:{kw_score:.2f}|Sem:{sem_score:.2f}|Comb:{comb_score:.3f})"); matched.add(orig)
        if max_score >= MIN_RELEVANCE_THRESHOLD:
            concept_scores[uri]["max_score"] = max_score; concept_scores[uri]["matched_texts"].update(matched)
            concept_scores[uri]["details"] = details; concept_scores[uri]["primary_label"] = get_primary_label(uri)
    sorted_concepts = sorted(concept_scores.items(), key=lambda item: item[1]["max_score"], reverse=True); results_with_data = []
    for uri, data in sorted_concepts:
        if data["max_score"] >= MIN_RELEVANCE_THRESHOLD:
            concept_data_from_taxonomy = taxonomy_concepts.get(uri, {})
            enriched_result = (uri, data["max_score"], data.get("primary_label", "Unknown"), concept_data_from_taxonomy)
            results_with_data.append(enriched_result)
    results = results_with_data; logger.info(f"Found {len(results)} matches >= {MIN_RELEVANCE_THRESHOLD:.2f} in {time.time() - start_time:.2f}s.")
    top_results = results[:MAX_ANCHOR_MATCHES]; logger.info(f"Returning top {len(top_results)} matches.")
    return concept_embedding, top_results

def identify_relevant_archetypes( # Unchanged
    input_embedding: np.ndarray,
    sbert_model: SentenceTransformer,
    args: argparse.Namespace
) -> List[str]:
    archetype_embeddings = get_archetype_embeddings(sbert_model)
    if not archetype_embeddings: return []
    scores = []
    for archetype, arch_emb in archetype_embeddings.items():
        similarity = calculate_semantic_similarity(input_embedding, arch_emb, "input", archetype)
        if similarity >= ARCHETYPE_SIMILARITY_THRESHOLD: scores.append((archetype, similarity))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # Return up to MAX_RELEVANT_ARCHETYPES
    relevant_archetypes = [arch for arch, score in sorted_scores[:MAX_RELEVANT_ARCHETYPES]]
    logger.info(f"Identified relevant archetypes: {relevant_archetypes}")
    return relevant_archetypes

def retrieve_kg_slice( # Unchanged
    input_embedding: np.ndarray,
    relevant_archetypes: List[str],
    primary_embeddings: Dict[str, np.ndarray],
    taxonomy_concepts: Dict[str, Dict],
    sbert_model: SentenceTransformer,
    args: argparse.Namespace
) -> List[Dict[str, Any]]:
    logger.info(f"Retrieving KG slice (Top {KG_RETRIEVAL_TOP_K}, Threshold {KG_RETRIEVAL_SIMILARITY_THRESHOLD})...")
    start_time = time.time()
    candidate_scores: Dict[str, float] = defaultdict(float)
    archetype_embeddings = get_archetype_embeddings(sbert_model)
    all_uris = list(primary_embeddings.keys())
    if not all_uris: logger.warning("No primary embeddings for KG retrieval."); return []
    all_embeddings = np.array([primary_embeddings[uri] for uri in all_uris])
    input_sims = cosine_similarity(input_embedding.reshape(1, -1), all_embeddings)[0]
    for i, uri in enumerate(all_uris): candidate_scores[uri] = max(candidate_scores[uri], float(input_sims[i]))
    for archetype in relevant_archetypes:
        arch_emb = archetype_embeddings.get(archetype)
        if arch_emb is not None:
            arch_sims = cosine_similarity(arch_emb.reshape(1, -1), all_embeddings)[0]
            for i, uri in enumerate(all_uris): candidate_scores[uri] = max(candidate_scores[uri], float(arch_sims[i]) * 0.5) # Weight archetype score lower
    filtered_candidates = {uri: score for uri, score in candidate_scores.items() if score >= KG_RETRIEVAL_SIMILARITY_THRESHOLD}
    sorted_uris = sorted(filtered_candidates, key=filtered_candidates.get, reverse=True)
    top_uris = sorted_uris[:KG_RETRIEVAL_TOP_K]
    kg_slice = []
    for uri in top_uris:
        concept_data = taxonomy_concepts.get(uri, {})
        definition = (concept_data.get("definition", [None])[0] or concept_data.get("dctermsDescription", [None])[0] or "").strip()
        fact = {"uri": uri, "label": get_primary_label(uri), "definition": definition if definition else None, "score": round(filtered_candidates[uri], 4)}
        if fact["label"] != uri: kg_slice.append(fact)

    logger.info(f"Retrieved {len(kg_slice)} relevant KG facts in {time.time() - start_time:.2f}s.")
    if args.debug and kg_slice: logger.debug(f"--- KG Slice Sample ---\n" + "\n".join([f"  URI: {f['uri']}, Label: {f['label']}, Score: {f['score']}, Def: {str(f['definition'])[:100]}..." for f in kg_slice[:min(10, len(kg_slice))]]) + "\n--- End KG Slice Sample ---")
    elif not kg_slice: logger.warning("KG Slice retrieval resulted in 0 facts.")
    return kg_slice

def construct_openai_prompt_v29_4(input_concept: str, relevant_archetypes: List[str], kg_slice: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Constructs the system and user prompts for OpenAI API (v29.4 - Full Gen + Theme Hints)."""
    # This prompt is identical to v29.3 as the essential check is post-LLM
    facts_string = "\n".join([f"- URI: {fact['uri']}\n  Label: {fact['label']}\n  Relevance Score: {fact['score']:.4f}\n  Definition: {fact.get('definition', 'N/A')}" for fact in kg_slice])
    themes_with_hints_string = "\n".join([f"  - Theme: {theme_name}\n    Hints: {', '.join(details.get('hints', []))}" for theme_name, details in BASE_THEMES.items()])
    # Schema includes the context sections LLM should generate
    output_schema_description = """
    ```json
    {
      "core_definition": "string (1-2 sentence definition summarizing the Input Concept based ONLY on the provided facts)",
      "taxonomical_structure": {
        "category": "string (Primary travel category based on Input Concept/Archetypes)",
        "broader": ["string (URIs)"], "narrower": ["string (URIs)"], "related": ["string (URIs)"]
      },
      "attributes": {
        "essential": [ {"uri": "string", "label": "string", "definition": "string|null"} ],
        "supplementary": [ {"uri": "string", "label": "string"} ]
      },
      "themes": [ { "theme": "string", "attributes": [ {"uri": "string", "label": "string"} ] } ],
      "negative_constraints": [ {"label": "string", "reason": "string"} ],
      "examples": ["string (Labels)"],
      "technological_integration": {"attributes": ["string (Labels)"]},
      "sustainability": {"attributes": ["string (Labels)"]},
      "accessibility": {"attributes": ["string (Labels)"]},
      "temporal_considerations": {"attributes": ["string (Labels)"]},
      "economic_context": {"attributes": ["string (Labels)"]},
      "activity_level": {"attributes": ["string (Labels)"]},
      "cultural_context": {"attributes": ["string (Labels)"]},
      "accommodation_type": {"attributes": ["string (Labels)"]},
      "user_experience": {"attributes": ["string (Labels)"]}
    }
    ```"""
    context_list_str = ", ".join(CONTEXT_SECTION_NAMES)
    system_prompt = "You are an expert travel taxonomist analyzing knowledge graph facts to generate structured affinity definitions. You follow instructions precisely and base your output strictly on the provided facts. DO NOT include any information not present in the facts."
    user_prompt = f"""
Analyze the provided Knowledge Graph Facts for the Input Concept and generate a comprehensive affinity definition in the specified JSON format.

**Input Concept:** "{input_concept}"
**Identified Core Archetypes:** {', '.join(relevant_archetypes) if relevant_archetypes else 'None'}

**Knowledge Graph Facts (Use ONLY these facts, paying attention to Relevance Scores):**
--- START FACTS ---
{facts_string}
--- END FACTS ---

**Base Theme Definitions (Use hints for guidance):**
--- START THEMES ---
{themes_with_hints_string}
--- END THEMES ---

**Task:**
Generate a JSON object strictly following the schema below. Populate ALL fields using information found *only* within the provided Knowledge Graph Facts. Ensure all URIs, labels, and definitions in your output originate from these facts. If facts are insufficient for a section, provide an empty list `[]` or appropriate default value. Limit theme attributes to {MAX_ATTRIBUTES_PER_THEME_LLM} per theme.

**Output JSON Schema:**
{output_schema_description}

**Instructions per Section (VERY IMPORTANT: Adhere STRICTLY to provided facts):**
- **core_definition:** Write a 1-2 sentence definition summarizing the Input Concept using the most relevant facts provided.
- **taxonomical_structure.category:** Determine the most appropriate travel category based on the Input Concept and Archetypes.
- **taxonomical_structure.broader/narrower/related:** List URIs ONLY if relations are explicitly mentioned for relevant concepts within the provided facts. Otherwise, use empty lists `[]`.
- **attributes.essential:** Select the top 5-10 facts (using their URI, Label, and Definition) that are most central to the Input Concept and Archetypes, prioritizing facts with higher Relevance Scores found in the provided list.
- **attributes.supplementary:** Select the next 10 most relevant facts (using their URI and Label), prioritizing higher scores, excluding those already in 'essential'. These must also come ONLY from the provided facts.
- **themes:** For each theme defined in the Base Theme Definitions, list the corresponding URI and Label for any fact *from the provided Knowledge Graph Facts* whose label or definition semantically relates to that theme's purpose, using the theme's hints for guidance. Use the exact theme names as keys. Omit theme or use empty attributes list `[]` if no *provided* facts apply. Limit attributes to {MAX_ATTRIBUTES_PER_THEME_LLM} per theme.
- **negative_constraints:** Analyze the 'Input Concept' and 'Identified Core Archetypes'. Now review the provided 'Knowledge Graph Facts'. Identify any facts whose label or definition represents a concept that is semantically *opposite* or *highly contradictory* to the core meaning of the Input Concept '{input_concept}'. Examples: If input is 'budget travel', exclude facts labeled 'butler service', 'private jet access'. If input is 'luxury travel', exclude 'hostel', 'dorm bed'. List the contradictory label and a brief reason. If no strictly contradictory facts are found *within the provided list*, output an empty list `[]`.
- **examples:** List the labels of 3-5 concepts *you selected* for your 'attributes.essential' list.
- **Context Sections ({', '.join(CONTEXT_SECTION_NAMES)}):** For each context section, list the *labels* of facts *from the provided list* that clearly relate to that specific context based on their label or definition, using the hints from the corresponding Base Themes (e.g., Technology hints for technological_integration, Sustainability hints for sustainability, etc.) for guidance. Use exact context names as keys. Omit or use empty list `[]` if no *provided* facts apply. Add a 'criteria' field set to e.g., "Tech identified by Hints".

**Generate ONLY the JSON object as the response.**
"""
    return system_prompt.strip(), user_prompt.strip()

def get_openai_client() -> Optional[OpenAI]:
    global _openai_client
    if _openai_client is None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: logger.error("OPENAI_API_KEY environment variable not set."); return None
            _openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized.")
        except Exception as e: logger.error(f"Failed to initialize OpenAI client: {e}"); return None
    return _openai_client

def call_openai_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int) -> Optional[Dict[str, Any]]:
    client = get_openai_client()
    if not client: return None
    logger.info(f"Sending request to OpenAI model: {model_name}")
    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"--- System Prompt ---\n{system_prompt}\n--- User Prompt (first 500 chars) ---\n{user_prompt[:500]}...")
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=0.1, # Low temp for consistency
             timeout=timeout)
            response_content = response.choices[0].message.content
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"--- Raw OpenAI Response Content ---\n{response_content}\n--- END ---")
            if response_content:
                try:
                    if response_content.strip().startswith("```json"): response_content = response_content.strip()[7:]
                    if response_content.strip().endswith("```"): response_content = response_content.strip()[:-3]
                    llm_output = json.loads(response_content.strip())
                    if isinstance(llm_output, dict): logger.info(f"Successfully parsed JSON response from OpenAI."); return llm_output
                    else: logger.warning(f"OpenAI response is JSON but not a dictionary: {type(llm_output)}")
                except json.JSONDecodeError as json_err: logger.error(f"Failed to parse OpenAI JSON: {json_err}\nRaw content: {response_content}")
            else: logger.warning("OpenAI response content empty.")
            return None
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            logger.error(f"OpenAI API Error (Attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}")
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Error details: {e}")
            wait_time = 5 if isinstance(e, RateLimitError) else 2 ** attempt
            if attempt >= max_retries: logger.error("Max retries reached."); return None
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e: logger.error(f"OpenAI API unexpected error (Attempt {attempt + 1}/{max_retries + 1}): {e}", exc_info=True); return None
    return None

# --- Main Execution ---
def generate_affinity_definitions(input_concepts: List[str], args: argparse.Namespace):
    logger.info("=== Starting Affinity Definition Generation (v29.4 - LLM Full Gen + Hints + Robust Essential Check) ===") # <-- v29.4
    start_process_time = time.time()
    logger.info("--- Loading Shared Resources ---"); load_start_time = time.time()
    sbert_model = get_sbert_model()
    global _taxonomy_concepts_cache
    taxonomy_concepts = load_taxonomy_concepts(args.taxonomy_dir, os.path.join(args.cache_dir, f"concepts_{CACHE_VERSION}.json"), args)
    if not taxonomy_concepts: logger.critical("Taxonomy concepts failed."); return
    embedding_data = precompute_taxonomy_embeddings(taxonomy_concepts, sbert_model, os.path.join(args.cache_dir, f"embeddings_{CACHE_VERSION}.pkl"), args)
    if not embedding_data: logger.critical("Embeddings failed."); return
    concept_associations = load_concept_associations(args.associations_file)
    corpus_data = load_corpus_data(args.corpus_file, sbert_model, os.path.join(args.cache_dir, f"corpus_{CACHE_VERSION}.pkl"), args)
    uri_embeddings_map, primary_embeddings, uris_list = embedding_data
    logger.info(f"--- Resources Loaded (Concepts: {len(taxonomy_concepts)}, Embeddings: {len(primary_embeddings)}, Corpus: {len(corpus_data)}) in {time.time() - load_start_time:.2f}s ---")

    all_definitions = []; total_concepts = len(input_concepts); logger.info(f"Processing {total_concepts} concepts...")
    for i, concept_input_string in enumerate(input_concepts):
        logger.info(f"\n--- Concept {i+1}/{total_concepts}: '{concept_input_string}' ---"); concept_start_time = time.time()
        norm_concept_lc = normalize_concept(concept_input_string).lower().strip()
        diagnostics = defaultdict(lambda: defaultdict(dict))
        # Base structure - LLM populates most, but keep placeholders for metadata/diagnostics
        affinity_definition = {
            "input_concept": concept_input_string, "normalized_concept": norm_concept_lc,
            # --- Fields to be populated primarily by LLM ---
            "core_definition": "N/A",
            "taxonomical_structure": {"category": "Unknown", "broader": [], "narrower": [], "related": []},
            "attributes": {"essential": [], "supplementary": []},
            "themes": [],
            "negative_constraints": [],
            "examples": [],
            # Context sections will be populated by LLM, ensure keys exist
            **{key: {"attributes": [], "criteria": "N/A"} for key in CONTEXT_SECTION_NAMES},
            # --- Fields populated by script ---
            "diagnostics": diagnostics,
            "processing_metadata": {"version": f"affinity-kg-openai-v29.4", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "Processing Failed", "duration_seconds": 0.0, "cache_version": CACHE_VERSION, "llm_model": OPENAI_MODEL} # <-- v29.4
        }
        relevant_archetypes = [] # Keep track of archetypes for essential check

        try:
            # Step 1: Get Input Embedding & Anchors
            logger.info("Step 1: Embedding input & finding anchors...");
            input_embedding, anchor_candidates = find_best_taxonomy_matches(
                concept_input_string, sbert_model, uri_embeddings_map, primary_embeddings, uris_list,
                concept_associations, corpus_data, taxonomy_concepts, args
            )
            if input_embedding is None: raise ValueError("Input embedding failed")
            diagnostics["input_processing"]["embedding_success"] = True
            if anchor_candidates:
                 diagnostics["anchor_selection"]["top_candidates"] = [{"uri": uri, "label": label, "score": round(score, 4)} for uri, score, label, _ in anchor_candidates]
            else: logger.warning("No anchor candidates found.")

            # Step 2: Identify Archetypes
            logger.info("Step 2: Identifying archetypes...");
            relevant_archetypes = identify_relevant_archetypes(input_embedding, sbert_model, args) # Store identified archetypes
            diagnostics["archetype_identification"]["identified_archetypes"] = relevant_archetypes

            # Step 3: Retrieve KG Slice
            logger.info("Step 3: Retrieving KG slice..."); kg_slice = retrieve_kg_slice(input_embedding, relevant_archetypes, primary_embeddings, taxonomy_concepts, sbert_model, args)
            diagnostics["kg_retrieval"]["retrieved_fact_count"] = len(kg_slice)
            if not kg_slice: logger.warning("No KG facts retrieved. LLM generation might be poor."); affinity_definition["processing_metadata"]["status"] = "Failed - No KG Facts"; raise StopIteration()

            # Step 4: Construct FULL Prompt for LLM (with Theme Hints)
            logger.info("Step 4: Constructing OpenAI FULL prompt with hints...");
            system_prompt, user_prompt = construct_openai_prompt_v29_4(concept_input_string, relevant_archetypes, kg_slice) # Use v29.4 prompt func (same as 29.3)
            diagnostics["llm_prompt"]["prompt_char_length"] = len(user_prompt) + len(system_prompt); logger.info(f"Prompt length: {diagnostics['llm_prompt']['prompt_char_length']} chars.")

            # Step 5: Call LLM for FULL definition
            logger.info(f"Step 5: Calling OpenAI LLM ({OPENAI_MODEL}) for full definition...");
            llm_full_output = call_openai_llm(system_prompt, user_prompt, OPENAI_MODEL, OPENAI_TIMEOUT, OPENAI_MAX_RETRIES)
            diagnostics["llm_call"]["attempted"] = True

            # Step 6: Process LLM Full Output
            if llm_full_output and isinstance(llm_full_output, dict):
                logger.info("Step 6: Processing successful LLM response.")
                diagnostics["llm_call"]["success"] = True
                diagnostics["llm_raw_output"] = llm_full_output

                # Merge LLM generated structure, preferring LLM values but keeping base keys
                llm_keys = llm_full_output.keys()
                for key in affinity_definition.keys():
                     if key in llm_keys and key not in ["input_concept", "normalized_concept", "diagnostics", "processing_metadata"]:
                         # Deep merge for attributes dict
                         if key == "attributes" and isinstance(affinity_definition[key], dict) and isinstance(llm_full_output[key], dict):
                             affinity_definition[key]["essential"] = llm_full_output[key].get("essential", [])
                             affinity_definition[key]["supplementary"] = llm_full_output[key].get("supplementary", [])
                         # Deep merge for context sections
                         elif key in CONTEXT_SECTION_NAMES and isinstance(affinity_definition[key], dict) and isinstance(llm_full_output.get(key), dict):
                             affinity_definition[key]["attributes"] = llm_full_output[key].get("attributes", [])
                             affinity_definition[key]["criteria"] = llm_full_output[key].get("criteria", f"{key.replace('_', ' ').title()} identified by LLM")
                         # Overwrite for other keys
                         else:
                              affinity_definition[key] = llm_full_output[key]

                # Basic validation
                required_keys = ["core_definition", "attributes", "themes", "negative_constraints", "examples"]
                missing_keys = [k for k in required_keys if k not in affinity_definition or affinity_definition[k] == "N/A" or affinity_definition[k] == []] # Check if key exists and has value
                if not missing_keys:
                     diagnostics["output_validation"]["schema_basic_check"] = "Pass"

                     # --- << STEP 7: Python Post-LLM Essential Check (Refined Logic) >> ---
                     if ENABLE_POST_LLM_ESSENTIAL_CHECK:
                         logger.info("Step 7: Performing post-LLM essential amenity check (Robust - checks all relevant archetypes)...")
                         essentials_added_count_total = 0
                         archetypes_checked = []
                         checked_uris_for_archetype = defaultdict(list)
                         uris_added_by_check = set() # Track URIs added in this step to avoid duplicates

                         # Ensure supplementary list exists and is a list
                         if not isinstance(affinity_definition.get("attributes", {}).get("supplementary"), list):
                             if "attributes" not in affinity_definition: affinity_definition["attributes"] = {}
                             affinity_definition["attributes"]["supplementary"] = []

                         # Get current URIs from LLM output once
                         current_essential_uris = {a['uri'] for a in affinity_definition["attributes"].get("essential", []) if isinstance(a, dict) and "uri" in a}
                         current_supplementary_uris = {a['uri'] for a in affinity_definition["attributes"].get("supplementary", []) if isinstance(a, dict) and "uri" in a}
                         all_current_uris = current_essential_uris.union(current_supplementary_uris)

                         for arch_string in relevant_archetypes:
                             if not arch_string: continue
                             try:
                                 arch_key = arch_string.split()[0].lower()
                             except IndexError:
                                 continue # Skip if archetype string is weird

                             archetypes_checked.append(arch_string) # Log which archetype is being checked

                             if arch_key in POST_LLM_ESSENTIALS_MAP:
                                 logger.info(f"  Checking essentials for matched archetype key: '{arch_key}' (from '{arch_string}')")
                                 target_essentials = POST_LLM_ESSENTIALS_MAP[arch_key]
                                 checked_uris_for_archetype[arch_key] = list(target_essentials.keys())
                                 essentials_added_for_this_arch = 0

                                 for essential_uri, fallback_label in target_essentials.items():
                                     # Check if URI is already present (from LLM or previous check iteration) OR if already added by this check step
                                     if essential_uri not in all_current_uris and essential_uri not in uris_added_by_check:
                                         label = get_primary_label(essential_uri, fallback=fallback_label) # Use fallback if cache fails
                                         if label != essential_uri: # Check if we got a meaningful label
                                             logger.warning(f"    Essential '{label}' ({essential_uri}) missing for '{arch_key}'. Adding to supplementary.")
                                             affinity_definition["attributes"]["supplementary"].append({"uri": essential_uri, "label": label})
                                             uris_added_by_check.add(essential_uri) # Track URI added
                                             essentials_added_for_this_arch += 1
                                         else:
                                             logger.warning(f"    Essential URI {essential_uri} (for '{arch_key}') not found in cache and no fallback label, cannot add.")

                                 if essentials_added_for_this_arch > 0:
                                     logger.info(f"    Added {essentials_added_for_this_arch} missing essentials for '{arch_key}'.")
                                     essentials_added_count_total += essentials_added_for_this_arch
                                 # Update all_current_uris to include newly added ones for subsequent archetype checks within the same concept
                                 all_current_uris.update(uris_added_by_check)

                             # else: logger.debug(f"  Archetype key '{arch_key}' (from '{arch_string}') not found in POST_LLM_ESSENTIALS_MAP.")

                         logger.info(f"Completed essential check. Total essentials added: {essentials_added_count_total}.")
                         diagnostics["essential_check"] = {
                             "enabled": True,
                             "archetypes_checked": archetypes_checked,
                             "uris_checked_by_archetype": dict(checked_uris_for_archetype),
                             "total_added_count": essentials_added_count_total
                         }

                     else: # Check disabled
                         logger.info("Step 7: Post-LLM essential amenity check disabled.")
                         diagnostics["essential_check"] = {"enabled": False}
                     # --- << End Step 7 >> ---

                     affinity_definition["processing_metadata"]["status"] = "Success"

                else: # Basic validation failed
                     affinity_definition["processing_metadata"]["status"] = "Failed - LLM Output Missing Keys"; diagnostics["output_validation"]["schema_basic_check"] = f"Fail (Missing: {', '.join(missing_keys)})"; logger.error(f"LLM output missing required keys: {missing_keys}")
                     logger.error(f"Problematic LLM Output: {json.dumps(llm_full_output, indent=2)}")

            else: # LLM call failed
                logger.error("LLM call failed or returned invalid JSON."); affinity_definition["processing_metadata"]["status"] = "Failed - LLM Call Error/Invalid JSON"; diagnostics["llm_call"]["success"] = False

        except StopIteration: pass # Handle case where KG slice was empty and processing stopped
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing '{concept_input_string}': {e}", exc_info=True); affinity_definition["processing_metadata"]["status"] = f"Failed - Exception: {type(e).__name__}"
            if args.debug: affinity_definition["processing_metadata"]["error_details"] = traceback.format_exc()
        finally:
            affinity_definition["processing_metadata"]["duration_seconds"] = round(time.time() - concept_start_time, 2)
            # Add final counts based on potentially modified structure
            diagnostics["final_output"]["essential_attr_count"] = len(affinity_definition.get("attributes", {}).get("essential", []))
            diagnostics["final_output"]["supplementary_attr_count"] = len(affinity_definition.get("attributes", {}).get("supplementary", []))
            diagnostics["final_output"]["negative_constraints_count"] = len(affinity_definition.get("negative_constraints", []))
            diagnostics["final_output"]["themes_count"] = len(affinity_definition.get("themes", []))
            for section_name in CONTEXT_SECTION_NAMES:
                 diagnostics["final_output"][f"{section_name}_count"] = len(affinity_definition.get(section_name, {}).get("attributes", []))

            affinity_definition["diagnostics"] = dict(diagnostics) # Finalize diagnostics dict
            all_definitions.append(affinity_definition)
            logger.info(f"--- Finished '{concept_input_string}' in {affinity_definition['processing_metadata']['duration_seconds']:.2f}s. Status: {affinity_definition['processing_metadata']['status']} ---")

    # --- Save Results ---
    logger.info(f"\n=== Finished all {total_concepts} concepts in {time.time() - start_process_time:.2f}s ===")
    output_file = os.path.join(args.output_dir, f"affinity_definitions_{CACHE_VERSION}.json") # v29.4
    logger.info(f"Saving results to {output_file}...")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f: json.dump(all_definitions, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully.")
    except Exception as e: logger.error(f"Failed to save results: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Affinity Definitions (v29.4 - LLM Full Gen + Hints + Robust Essential Check)") # <-- v29.4
    parser.add_argument("--concepts", required=True, help="Input concepts file.")
    parser.add_argument("--taxonomy-dir", required=True, help="Taxonomy RDF directory.")
    parser.add_argument("--output-dir", default="./output", help="Output directory.")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory.")
    parser.add_argument("--associations-file", default=DEFAULT_ASSOCIATIONS_FILE, help="Associations JSON.")
    parser.add_argument("--corpus-file", default=DEFAULT_CORPUS_FILE, help="Corpus text file.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild caches.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of concepts.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--openai-model", default=OPENAI_MODEL, help=f"OpenAI model (default: {OPENAI_MODEL})")
    # Add argument to disable the check if needed, default is True via constant
    parser.add_argument("--disable-essential-check", action="store_true", help="Disable the post-LLM essential amenity check.")
    args = parser.parse_args()
    OPENAI_MODEL = args.openai_model
    if args.disable_essential_check:
        ENABLE_POST_LLM_ESSENTIAL_CHECK = False # Override default if flag is set

    if args.debug: logger.setLevel(logging.DEBUG); [h.setLevel(logging.DEBUG) for h in logging.getLogger().handlers]; logger.info("DEBUG logging enabled.")
    else: logger.setLevel(logging.INFO); [h.setLevel(logging.INFO) for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]

    if not os.environ.get("OPENAI_API_KEY"): logger.critical("FATAL: OPENAI_API_KEY env var not set."); sys.exit(1)
    os.makedirs(args.cache_dir, exist_ok=True); os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.concepts, 'r', encoding='utf-8') as f: input_concepts = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(input_concepts)} concepts from {args.concepts}")
        if args.limit and args.limit < len(input_concepts): input_concepts = input_concepts[:args.limit]; logger.info(f"Limiting to first {args.limit}.")
        if not input_concepts: logger.error("No concepts found."); sys.exit(1)
    except FileNotFoundError: logger.critical(f"Input file not found: {args.concepts}"); sys.exit(1)
    except Exception as e: logger.critical(f"Error reading input file: {e}"); sys.exit(1)

    # Log config for v29.4 (Refined)
    logger.info(f"+++ Using LLM Full Generation + Hints + Robust Python Essential Check Pipeline v29.4 +++")
    logger.info(f"    OpenAI Model: {OPENAI_MODEL}")
    logger.info(f"    KG Retrieval Top K: {KG_RETRIEVAL_TOP_K}")
    logger.info(f"    Prompt Includes: KG Facts + Theme/Context Hints")
    logger.info(f"    LLM Role: Generate FULL definition JSON based on facts and hints")
    logger.info(f"    Python Post-LLM Essential Check Enabled: {ENABLE_POST_LLM_ESSENTIAL_CHECK}")
    if ENABLE_POST_LLM_ESSENTIAL_CHECK:
         logger.info(f"      Check Logic: Iterates through ALL relevant archetypes ({MAX_RELEVANT_ARCHETYPES} max).")
         logger.info(f"      Essential Map Keys: {list(POST_LLM_ESSENTIALS_MAP.keys())}")

    generate_affinity_definitions(input_concepts, args)
    logger.info("=== Affinity Generation Script Finished ===")