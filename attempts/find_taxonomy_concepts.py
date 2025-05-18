#!/usr/bin/env python3
"""
Generate affinity definitions for travel concepts by matching them to taxonomy
concepts and associating those concepts with predefined themes.

Version: 2025-04-13-affinity-consolidated-v14 (Reverted tqdm handling to cleaner style)
Combines robust concept matching (v5 logic) with theme association.
Replaces affinity-generator.py, affinity_taxonomy.py, affinity_scoring.py.
"""

import os
import json
import logging
import argparse
import re
import pickle
import numpy as np
import sys # Import sys for robust error handling early on
from collections import defaultdict
import time
import traceback
from typing import Dict, List, Set, Tuple, Optional

# --- Handle Critical Dependency Imports ---
try: from rdflib import Graph, Namespace, URIRef, Literal, RDF, util
except ImportError: print("CRITICAL ERROR: rdflib library not found. Install: pip install rdflib", file=sys.stderr); sys.exit(1)
try: from sentence_transformers import SentenceTransformer
except ImportError: print("CRITICAL ERROR: sentence-transformers library not found. Install: pip install sentence-transformers", file=sys.stderr); sys.exit(1)
try: from sklearn.metrics.pairwise import cosine_similarity
except ImportError: print("CRITICAL ERROR: scikit-learn library not found. Install: pip install scikit-learn", file=sys.stderr); sys.exit(1)

# --- Handle Optional tqdm Import ---

# Define the dummy function first as the default
def tqdm_dummy(iterable, *args, **kwargs):
    """Dummy tqdm function for when the library isn't installed."""
    # Optionally print a message, or just pass through
    # print("Processing (tqdm not installed)...")
    return iterable

# Assign the dummy function to the name 'tqdm' initially
tqdm = tqdm_dummy

# Now, try to import the real tqdm and overwrite the dummy
try:
    from tqdm import tqdm as real_tqdm # Import with a different name
    tqdm = real_tqdm # If import succeeds, overwrite 'tqdm' with the real function
    print("Info: tqdm library found, progress bars enabled.") # Optional info
except ImportError:
    # If import fails, 'tqdm' still points to tqdm_dummy
    print("Warning: tqdm not found, progress bars will be disabled. Install with: pip install tqdm")
    pass # tqdm already assigned to the dummy function

# --- Setup Logging ---
log_filename = "affinity_generation.log"
if os.path.exists(log_filename):
    try: os.remove(log_filename); print(f"Removed old log file: {log_filename}")
    except OSError as e: print(f"Warning: Could not remove old log file {log_filename}: {e}")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", handlers=[logging.FileHandler(log_filename, mode='w', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger("affinity_generator")

# --- Namespaces ---
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# --- Configuration & Constants ---
# (Constants remain the same as v13)
MIN_RELEVANCE_THRESHOLD = 0.3; OUTPUT_RELEVANCE_THRESHOLD = 0.0; DEBUG_SEMANTIC_THRESHOLD = 0.3; MAX_MATCHES_PER_CONCEPT = 5
KEYWORD_WEIGHT = 0.55; SEMANTIC_WEIGHT = 0.45; RELATIONSHIP_BOOST = 0.35; REL_BOOST_SCORE_THRESHOLD = 0.60
ASSOCIATION_BOOST = 0.40; ANTONYM_PENALTY = -0.5; EXACT_MATCH_BONUS = 0.01
PROPERTY_WEIGHTS = {"prefLabel": 1.0, "altLabel": 0.95, "rdfsLabel": 0.9, "definition": 0.80, "dctermsDescription": 0.80}
SYNONYM_SIMILARITY_THRESHOLD = 0.55; EXCLUSION_SIMILARITY_THRESHOLD = 0.85; MAX_SYNONYMS = 20; TRAVEL_CONTEXT = "travel "
THEME_HINT_MATCH_THRESHOLD = 1; THEME_SEMANTIC_THRESHOLD = 0.50; MAX_ATTRIBUTES_PER_THEME = 5
HINT_CONFIDENCE_WEIGHT = 0.3; SEMANTIC_CONFIDENCE_WEIGHT = 0.7
FAMILY_SEEDS_ANTYNOM_LIST = ["family", "families", "children", "child", "childcare", "kid", "kids", "baby", "babies", "toddler", "toddlers", "youth", "teen", "teens", "infant", "infants", "minor", "minors", "children stay free", "kid suite", "crib", "cribs", "playpen", "playpens", "stroller", "strollers", "high chair", "highchair"]
GENERAL_ANTONYM_SEEDS = list(set(FAMILY_SEEDS_ANTYNOM_LIST + ["min age", "budget", "economy", "basic", "cheap", "noisy", "loud", "shared", "not allowed", "prohibited", "restricted", "no"]))
CACHE_VERSION = "v20250413.affinity.14" # Incremented version
DEFAULT_ASSOCIATIONS_FILE = "./datasources/concept_associations.json"
_model_instance = None; _taxonomy_concepts_cache = None; _taxonomy_embeddings_cache = None; _corpus_data_cache = None; _concept_associations_cache = None

# --- BASE THEMES Definition ---
BASE_THEMES = { "Cleanliness & Hygiene Standards": {"type": "structural", "rule": "Must have 1", "weight": 0.085, "subScore": "CleanlinessAffinity", "hints": ["clean", "hygiene", "sanitized", "disinfected", "housekeeping", "spotless"]}, "Safety & Security Measures": {"type": "structural", "rule": "Must have 1", "weight": 0.076, "subScore": "SafetyAffinity", "hints": ["safe", "secure", "locks", "safety", "security", "guard", "well lit", "smoke detector", "carbon monoxide detector", "first aid"]}, "Comfortable Bedding & Sleep Environment": {"type": "structural", "rule": "Must have 1", "weight": 0.068, "subScore": "BeddingAffinity", "hints": ["bed", "bedding", "comfortable bed", "pillow", "mattress", "quiet room", "sleep quality", "blackout curtains"]}, "Functional, Clean Bathroom Facilities": {"type": "structural", "rule": "Must have 1", "weight": 0.068, "subScore": "BathroomAffinity", "hints": ["bathroom", "shower", "toilet", "clean bathroom", "hot water", "water pressure", "towels", "toiletries"]}, "Fast, Reliable Wi-Fi and Device Charging": {"type": "technological", "rule": "Must have 1", "weight": 0.059, "subScore": "WiFiAffinity", "hints": ["wifi", "wi fi", "internet", "connection", "charging", "outlets", "usb port"]}, "In-Room Climate Control": {"type": "structural", "rule": "Must have 1", "weight": 0.059, "subScore": "ClimateAffinity", "hints": ["heating", "cooling", "air conditioning", "ac", "a c", "thermostat", "temperature control"]}, "Friendly and Attentive Staff or Hosts": {"type": "service", "rule": "Optional", "weight": 0.051, "subScore": "StaffAffinity", "hints": ["friendly", "staff", "host", "helpful", "service", "attentive", "welcoming", "concierge"]}, "Seamless Digital Experience": {"type": "technological", "rule": "Optional", "weight": 0.051, "subScore": "DigitalAffinity", "hints": ["check in", "check out", "digital key", "mobile app", "contactless", "online"]}, "Accurate Listings with Up-to-Date Imagery": {"type": "booking", "rule": "Must have 1", "weight": 0.051, "subScore": "ListingAffinity", "hints": ["accurate listing", "photos", "pictures", "images", "description match", "up to date"]}, "Competitive Pricing & Perceived Value": {"type": "decision", "rule": "Optional", "weight": 0.042, "subScore": "PricingAffinity", "hints": ["price", "value", "affordable", "cost", "deal", "worth it", "reasonable price"]}, "Location Convenience and Safety": {"type": "decision", "rule": "Must have 1", "weight": 0.042, "subScore": "LocationAffinity", "hints": ["location", "convenient", "safe location", "neighborhood", "walkable", "close to", "transportation", "parking"]}, "Quiet & Peaceful Environment": {"type": "comfort", "rule": "Optional", "weight": 0.042, "subScore": "QuietAffinity", "hints": ["quiet", "peaceful", "calm", "noise level", "relaxing"]}, "Complimentary or Easy Meal Access": {"type": "preference", "rule": "Optional", "weight": 0.034, "subScore": "MealAffinity", "hints": ["breakfast", "food", "dining", "restaurant", "meal", "room service", "complimentary breakfast"]}, "Flexibility in Booking and Cancellation Policies": {"type": "preference", "rule": "Optional", "weight": 0.034, "subScore": "FlexibilityAffinity", "hints": ["cancellation", "flexible booking", "policy", "refund", "change dates"]}, "In-Room Tech": {"type": "technological", "rule": "Optional", "weight": 0.034, "subScore": "TechAffinity", "hints": ["tv", "smart tv", "streaming", "sound system", "smart lighting", "technology"]}, "Basic Room Storage & Organization": {"type": "structural", "rule": "Optional", "weight": 0.034, "subScore": "StorageAffinity", "hints": ["storage", "closet", "drawers", "luggage space", "room size"]}, "Pet- and Family-Friendly Accommodations": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "InclusiveAffinity", "hints": ["pet friendly", "family friendly", "kids allowed", "children welcome", "dog friendly", "cribs", "play area"]}, "Sustainability Signals or Options": {"type": "trend", "rule": "Optional", "weight": 0.025, "subScore": "SustainabilityAffinity", "hints": ["sustainable", "eco friendly", "recycling", "green", "ev charging", "local sourcing"]}, "Transparent Communication & Pre-Arrival Info": {"type": "booking", "rule": "Optional", "weight": 0.025, "subScore": "CommunicationAffinity", "hints": ["communication", "information", "pre arrival", "responsive host", "clear instructions"]}, "Personalization or Recognition for Return Guests": {"type": "trend", "rule": "Optional", "weight": 0.017, "subScore": "PersonalizationAffinity", "hints": ["personalized", "loyalty program", "recognition", "return guest", "special touch", "vip"]}, "Sentiment": {"type": "comfort", "rule": "Optional", "weight": 0.025, "subScore": "SentimentAffinity", "hints": ["luxury", "relaxing", "vibrant", "charming", "boutique", "unique", "atmosphere", "ambiance"]}, "Indoor Amenities": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "IndoorAmenityAffinity", "hints": ["spa", "bar", "gym", "fitness center", "sauna", "hot tub", "indoor pool", "lounge", "library", "business center"]}, "Outdoor Amenities": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "OutdoorAmenityAffinity", "hints": ["pool", "outdoor pool", "garden", "patio", "terrace", "balcony", "bbq", "fire pit", "view"]}, "Activities": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "ActivityAffinity", "hints": ["hiking", "nightlife", "tours", "skiing", "golf", "water sports", "entertainment", "classes", "yoga", "cycling"]}, "Spaces": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "SpacesAffinity", "hints": ["suite", "balcony", "kitchen", "kitchenette", "living room", "dining area", "workspace", "multiple bedrooms"]}, "Events": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "EventAffinity", "hints": ["festival", "wedding", "concert", "conference", "event space", "meeting facilities"]} }

# --- Utility Functions ---
# (Functions get_sbert_model, normalize_concept, calculate_semantic_similarity, load_concept_associations,
#  load_taxonomy_concepts, load_corpus_data, precompute_taxonomy_embeddings,
#  get_dynamic_antonyms_synonyms, find_best_taxonomy_matches, get_concept_text, map_uris_to_themes
#  remain the same as the correctly formatted v11, including the specific fixes you pointed out)
def get_sbert_model():
    """Load or return cached Sentence-BERT model."""
    global _model_instance
    if _model_instance is None:
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"Loading Sentence-BERT model ('{model_name}')...")
        start_time = time.time()
        try:
            _model_instance = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Fatal: Failed to load Sentence-BERT model: {e}\n{traceback.format_exc()}")
            raise RuntimeError("SBERT Model loading failed") from e
        logger.info("Loaded Sentence-BERT model in %.2f seconds", time.time() - start_time)
    return _model_instance

def normalize_concept(concept: Optional[str]) -> str:
    """Normalize concept string more robustly."""
    if not isinstance(concept, str) or not concept:
        return ""
    try:
        normalized = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept)
        normalized = normalized.replace("-", " ").replace("_", " ")
        normalized = re.sub(r'[^\w\s]|(\'s\b)', '', normalized)
        normalized = ' '.join(normalized.lower().split())
        return normalized
    except Exception as e:
        logger.warning(f"Normalization failed for concept '{concept}': {e}. Falling back to lower().strip().")
        return concept.lower().strip()

def calculate_semantic_similarity(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray], term1: str = "term1", term2: str = "term2") -> float:
    """Calculate cosine similarity between two SBERT embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    try:
        if not isinstance(embedding1, np.ndarray): embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray): embedding2 = np.array(embedding2)
        if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
        if embedding1.shape[1] != embedding2.shape[1]:
            logger.warning(f"Embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape} for '{term1}' vs '{term2}'. Returning 0.")
            return 0.0
        sim = cosine_similarity(embedding1, embedding2)[0][0]
        similarity = max(0.0, min(1.0, float(sim)))
        if logger.isEnabledFor(logging.DEBUG) and similarity >= DEBUG_SEMANTIC_THRESHOLD:
             term1_short = term1[:50] + '...' if len(term1) > 50 else term1
             term2_short = term2[:50] + '...' if len(term2) > 50 else term2
             logger.debug(f"Similarity between '{term1_short}' and '{term2_short}': {similarity:.4f}")
        return similarity
    except ValueError as ve:
        logger.warning(f"Similarity calculation ValueError for '{term1[:50]}' vs '{term2[:50]}': {ve}. Check embedding shapes/content.")
        return 0.0
    except Exception as e:
        logger.warning(f"Unexpected similarity calculation error for '{term1[:50]}' vs '{term2[:50]}': {e}")
        return 0.0

def load_concept_associations(filepath: Optional[str]) -> Dict[str, List[str]]:
    """Loads concept associations from a JSON file."""
    global _concept_associations_cache
    if _concept_associations_cache is not None:
        return _concept_associations_cache
    associations = {}
    final_associations = {}
    if not filepath:
        logger.warning("Associations file path not specified. No boost applied.")
        _concept_associations_cache = {}
        return associations
    if not os.path.exists(filepath):
        logger.warning(f"Associations file not found: {filepath}. No boost applied.")
        _concept_associations_cache = {}
        return associations
    try:
        logger.info(f"Loading concept associations from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error("Invalid associations format: Top level must be dict.")
            _concept_associations_cache = {}
            return {}
        valid_count = 0
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, list) and all(isinstance(item, str) for item in value):
                 norm_key = normalize_concept(key)
                 norm_values = [normalize_concept(v) for v in value if normalize_concept(v)]
                 if norm_key and norm_values:
                     final_associations[norm_key] = norm_values
                     valid_count += 1
                 else:
                     logger.warning(f"Skipping association pair ('{key}': {value}) empty after normalization.")
            else:
                logger.warning(f"Invalid format for pair ('{key}': {value}). Skipping.")
        if final_associations:
            logger.info(f"Loaded and normalized {valid_count} associations.")
        else:
            logger.warning(f"No valid associations loaded from {filepath}.")
        _concept_associations_cache = final_associations
        return final_associations
    except json.JSONDecodeError as jde:
        logger.error(f"Error decoding JSON from {filepath}: {jde}.")
        _concept_associations_cache = {}
        return {}
    except Exception as e:
        logger.error(f"Error loading associations file {filepath}: {e}\n{traceback.format_exc()}")
        _concept_associations_cache = {}
        return {}

def load_taxonomy_concepts(taxonomy_dir: str, cache_file: str, args: argparse.Namespace) -> Dict:
    """Loads taxonomy concepts, parsing RDFs or using cache file (JSON)."""
    global _taxonomy_concepts_cache
    if _taxonomy_concepts_cache is not None:
        return _taxonomy_concepts_cache
    cache_file = os.path.abspath(cache_file)
    logger.info("Attempting to load taxonomy concepts cache: %s", cache_file)
    if os.path.exists(cache_file) and not args.rebuild_cache:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            if (isinstance(cached_data, dict) and cached_data.get('_cache_info', {}).get('version') == CACHE_VERSION and
                'concepts' in cached_data and isinstance(cached_data['concepts'], dict) and cached_data['concepts']):
                _taxonomy_concepts_cache = cached_data['concepts']
                logger.info(f"Loaded {len(_taxonomy_concepts_cache)} cached concepts (v{CACHE_VERSION})")
                return _taxonomy_concepts_cache
            elif isinstance(cached_data, dict) and cached_data.get('_cache_info', {}).get('version') != CACHE_VERSION:
                 logger.warning("Cache file %s version mismatch. Rebuilding.", cache_file)
            else:
                 logger.warning("Invalid/empty cache %s. Rebuilding.", cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}. Rebuilding.")
    else:
        logger.info("Cache not found or rebuild requested. Building new cache.")

    logger.info("Loading RDFs from: %s", taxonomy_dir)
    if not os.path.isdir(taxonomy_dir):
        logger.error(f"Taxonomy dir not found: {taxonomy_dir}")
        return {}
    try:
        rdf_files = [os.path.join(taxonomy_dir, f) for f in os.listdir(taxonomy_dir) if f.endswith(('.rdf', '.owl', '.ttl', '.nt', '.jsonld'))]
    except OSError as e:
        logger.error(f"Cannot list files in {taxonomy_dir}: {e}")
        return {}
    if not rdf_files:
        logger.error(f"No RDF files found in {taxonomy_dir}")
        return {}
    logger.info(f"Found {len(rdf_files)} RDF files.")

    graph = Graph()
    loaded_files_count = 0
    load_errors = 0
    parsing_start_time = time.time()
    for f in rdf_files:
        try:
            fmt = util.guess_format(f) # Use imported util directly
            if not fmt:
                 fmt_map = {'.rdf': 'xml', '.owl': 'xml', '.ttl': 'turtle', '.nt': 'nt', '.jsonld': 'json-ld'}
                 _, ext = os.path.splitext(f)
                 fmt = fmt_map.get(ext.lower())
            if not fmt:
                logger.debug(f"Skipping unrecognized file: {f}")
                continue
            logger.debug(f"Parsing: {os.path.basename(f)} (format: {fmt})")
            graph.parse(f, format=fmt)
            loaded_files_count += 1
        except Exception as e:
            logger.warning(f"Failed to parse {f}: {e}")
            load_errors += 1
    if not graph:
        logger.error("No valid RDF data loaded.")
        return {}
    logger.info(f"Parsed {loaded_files_count} files ({load_errors} errors), {len(graph)} triples in {time.time() - parsing_start_time:.2f}s.")

    relevant_properties = {SKOS.prefLabel: "prefLabel", SKOS.altLabel: "altLabel", SKOS.definition: "definition", RDFS.label: "rdfsLabel", DCTERMS.description: "dctermsDescription", SKOS.related: "related", SKOS.broader: "broader", SKOS.narrower: "narrower"}
    taxonomy_concepts_build = defaultdict(lambda: defaultdict(list))
    concepts_processed = set()
    triples_processed = 0
    extraction_start_time = time.time()
    for s, p, o in graph:
        triples_processed += 1
        if p in relevant_properties and isinstance(s, URIRef):
            if isinstance(o, (Literal, URIRef)):
                uri = str(s)
                prop_key = relevant_properties[p]
                value = str(o).strip()
                if value and value not in taxonomy_concepts_build[uri][prop_key]:
                    taxonomy_concepts_build[uri][prop_key].append(value)
                    concepts_processed.add(uri)
    logger.info(f"Processed {triples_processed} triples, extracted data for {len(concepts_processed)} URIs in {time.time() - extraction_start_time:.2f}s.")

    taxonomy_concepts_dict = {uri: dict(data) for uri, data in taxonomy_concepts_build.items() if data}
    cache_content = {'_cache_info': {'version': CACHE_VERSION, 'created_at': time.strftime("%Y-%m-%d %H:%M:%S %Z"), 'source_dir': taxonomy_dir, 'num_concepts': len(taxonomy_concepts_dict)}, 'concepts': taxonomy_concepts_dict}
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True) # Corrected Formatting
        if not os.access(os.path.dirname(cache_file), os.W_OK):
            logger.error(f"Cache dir {os.path.dirname(cache_file)} not writable.")
            _taxonomy_concepts_cache = taxonomy_concepts_dict
            return taxonomy_concepts_dict
        logger.info(f"Writing concept cache to {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_content, f, indent=2, ensure_ascii=False)
        if os.path.exists(cache_file):
            logger.info(f"Cached {len(taxonomy_concepts_dict)} concepts (v{CACHE_VERSION})")
        else:
            logger.error(f"Cache file {cache_file} not created.")
    except Exception as e:
        logger.error(f"Failed to write concept cache: {e}")
    _taxonomy_concepts_cache = taxonomy_concepts_dict
    return taxonomy_concepts_dict

def load_corpus_data(corpus_file: Optional[str], model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """Loads terms from a corpus file and computes/caches their embeddings using Pickle."""
    global _corpus_data_cache
    if _corpus_data_cache is not None:
        return _corpus_data_cache
    if not corpus_file:
        logger.info("No corpus file provided.")
        _corpus_data_cache = {}
        return {}
    cache_file = os.path.abspath(cache_file)
    logger.info(f"Attempting corpus cache: {cache_file}")
    if os.path.exists(cache_file) and not args.rebuild_cache:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if (isinstance(cached_data, dict) and cached_data.get('_cache_info', {}).get('version') == CACHE_VERSION and
                'embeddings' in cached_data and isinstance(cached_data['embeddings'], dict)):
                _corpus_data_cache = cached_data['embeddings']
                logger.info(f"Loaded {len(_corpus_data_cache)} cached corpus embeddings (v{CACHE_VERSION})")
                return _corpus_data_cache
            else:
                logger.warning("Corpus cache invalid/version mismatch. Rebuilding.")
        except Exception as e:
            logger.warning(f"Failed load corpus cache: {e}. Rebuilding.")
    else:
        logger.info("Corpus cache not found or rebuild requested.")

    if not os.path.exists(corpus_file): logger.error(f"Corpus file not found: {corpus_file}"); return {}
    if not os.access(corpus_file, os.R_OK): logger.error(f"Corpus file not readable: {corpus_file}"); return {}
    logger.info(f"Loading corpus terms from: {corpus_file}")
    terms = set()
    line_count = 0
    empty_lines = 0
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_count += 1
                term = line.strip().split(',', 1)[0].strip() if ',' in line else line.strip()
                # Corrected Style
                if term:
                    terms.add(term)
                else:
                    empty_lines += 1
        if empty_lines > 0:
            logger.debug(f"Skipped {empty_lines} empty lines.")
    except Exception as e:
        logger.error(f"Error reading {corpus_file}: {e}")
        return {}

    # Corrected Logic Flow
    if not terms:
        logger.warning(f"No terms extracted from {corpus_file}.")
        _corpus_data_cache = {} # Cache empty result
        return {}

    logger.info(f"Extracted {len(terms)} unique terms.")
    terms_to_embed = {t: norm for t, norm in ((t, normalize_concept(t)) for t in sorted(list(terms))) if norm}

    if not terms_to_embed: # Corrected Check
        logger.warning("No valid terms after normalization.")
        _corpus_data_cache = {} # Cache empty result
        return {}

    # Only proceed if terms_to_embed is NOT empty
    logger.info(f"Embedding {len(terms_to_embed)} corpus terms...")
    texts_to_encode = [TRAVEL_CONTEXT + norm for norm in terms_to_embed.values()]

    try:
        embeddings = model.encode(texts_to_encode, batch_size=128, show_progress_bar=True)
    except Exception as e:
        logger.error(f"SBERT encoding failed for corpus: {e}\n{traceback.format_exc()}")
        return {} # Return empty dict on failure

    corpus_data_build = {norm: emb for norm, emb in zip(terms_to_embed.values(), embeddings)}
    cache_content = {'_cache_info': {'version': CACHE_VERSION, 'created_at': time.strftime("%Y-%m-%d %H:%M:%S %Z"), 'source_file': corpus_file, 'num_embeddings': len(corpus_data_build)}, 'embeddings': corpus_data_build}
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if not os.access(os.path.dirname(cache_file), os.W_OK):
            logger.error(f"Cache dir {os.path.dirname(cache_file)} not writable.")
            _corpus_data_cache = corpus_data_build # Still keep in memory
            return corpus_data_build # Return data even if cache write fails
        # Corrected Style
        logger.info(f"Attempting to write corpus cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_content, f)

        if os.path.exists(cache_file):
            cache_size_kb = os.path.getsize(cache_file) / 1024
            logger.info(f"Cached {len(corpus_data_build)} corpus embeddings (v{CACHE_VERSION}) to {cache_file} ({cache_size_kb:.2f} KB)")
        else:
            logger.error(f"Corpus cache {cache_file} not created.")
    except Exception as e:
        logger.error(f"Failed to write corpus cache {cache_file}: {e}")

    _corpus_data_cache = corpus_data_build
    return corpus_data_build


def precompute_taxonomy_embeddings(taxonomy_concepts: Dict, model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Optional[Tuple[Dict, Dict, List]]:
    global _taxonomy_embeddings_cache
    if _taxonomy_embeddings_cache is not None:
        return _taxonomy_embeddings_cache
    cache_file = os.path.abspath(cache_file)
    logger.info(f"Attempting embeddings cache: {cache_file}")
    if os.path.exists(cache_file) and not args.rebuild_cache:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if (isinstance(cached_data, dict) and cached_data.get('_cache_info', {}).get('version') == CACHE_VERSION and
                all(k in cached_data for k in ['uri_embeddings', 'uri_to_index', 'uris_list']) and
                isinstance(cached_data['uri_embeddings'], dict) and isinstance(cached_data['uri_to_index'], dict) and isinstance(cached_data['uris_list'], list) and cached_data['uris_list']):
                 logger.info(f"Loaded cached embeddings for {len(cached_data['uris_list'])} URIs (v{CACHE_VERSION})")
                 _taxonomy_embeddings_cache = (cached_data['uri_embeddings'], cached_data['uri_to_index'], cached_data['uris_list'])
                 return _taxonomy_embeddings_cache
            else:
                logger.warning("Embeddings cache invalid/version mismatch. Rebuilding.")
        except Exception as e:
            logger.warning(f"Failed load embeddings cache: {e}. Rebuilding.")
    else:
        logger.info("Embeddings cache not found or rebuild requested.")

    logger.info("Precomputing taxonomy embeddings...")
    labels_to_embed = []
    uris_list = sorted(list(taxonomy_concepts.keys()))
    embedding_properties = ["prefLabel", "altLabel", "definition", "rdfsLabel", "dctermsDescription"]
    processed_texts = set()
    uri_label_map = defaultdict(list)
    for uri in uris_list:
        data = taxonomy_concepts.get(uri, {})
        for prop in embedding_properties:
            for lbl in data.get(prop, []):
                if lbl and isinstance(lbl, str):
                    normalized = normalize_concept(lbl)
                    if normalized:
                        uri_label_map[uri].append((prop, lbl, normalized))
                        if normalized not in processed_texts:
                            processed_texts.add(normalized)
                            labels_to_embed.append(normalized)
    if not labels_to_embed: logger.error("No labels found. Cannot compute embeddings."); return None
    logger.info(f"Found {len(labels_to_embed)} unique texts from {len(uris_list)} concepts.")
    texts_to_encode = [TRAVEL_CONTEXT + norm_text for norm_text in labels_to_embed]
    logger.info(f"Embedding {len(texts_to_encode)} unique texts...")
    embedding_start_time = time.time()
    # Corrected Style
    try:
        embeddings_array = model.encode(texts_to_encode, batch_size=128, show_progress_bar=True)
    except Exception as e:
        logger.error(f"SBERT encoding failed: {e}\n{traceback.format_exc()}")
        return None
    logger.info(f"Finished embedding {len(texts_to_encode)} texts in {time.time() - embedding_start_time:.2f}s.")

    text_to_embedding = {text: emb for text, emb in zip(labels_to_embed, embeddings_array)}
    uri_embeddings_build = defaultdict(list)
    map_start_time = time.time()
    for uri, label_data_list in uri_label_map.items():
        for prop, original_label, normalized_label in label_data_list:
            embedding = text_to_embedding.get(normalized_label)
            # Corrected Style
            if embedding is not None:
                 uri_embeddings_build[uri].append((prop, original_label, embedding))
            else:
                 logger.warning(f"Consistency issue: {normalized_label} for {uri} not in map.")
    uri_embeddings_dict = dict(uri_embeddings_build)
    uri_to_index_build = {uri: i for i, uri in enumerate(uris_list)}
    logger.info(f"Constructed embeddings map for {len(uri_embeddings_dict)} URIs in {time.time() - map_start_time:.2f}s.")
    cache_content = {'_cache_info': {'version': CACHE_VERSION, 'created_at': time.strftime("%Y-%m-%d %H:%M:%S %Z"), 'num_uris': len(uris_list)}, 'uri_embeddings': uri_embeddings_dict, 'uri_to_index': uri_to_index_build, 'uris_list': uris_list}
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if not os.access(os.path.dirname(cache_file), os.W_OK):
            logger.error(f"Cache dir {os.path.dirname(cache_file)} not writable.")
            _taxonomy_embeddings_cache = (uri_embeddings_dict, uri_to_index_build, uris_list)
            return _taxonomy_embeddings_cache
        logger.info(f"Writing embeddings cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_content, f)
        if os.path.exists(cache_file):
            logger.info(f"Cached embeddings for {len(uris_list)} URIs (v{CACHE_VERSION}).")
        else:
            logger.error(f"Embeddings cache {cache_file} not created.")
    except Exception as e:
        logger.error(f"Failed to write embeddings cache {cache_file}: {e}")
    _taxonomy_embeddings_cache = (uri_embeddings_dict, uri_to_index_build, uris_list)
    return _taxonomy_embeddings_cache


# --- Core Concept Matching Logic ---

def get_dynamic_antonyms_synonyms(concept: str, concept_embedding: np.ndarray, taxonomy_concepts: Dict, uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray]]], corpus_data: Dict[str, np.ndarray]) -> Tuple[Set[str], List[Tuple[str, float]], Set[str]]:
    synonyms_dict = {}
    exclusions_set = set()
    normalized_input_lower = normalize_concept(concept).lower().strip()
    logger.info("Generating dynamic lists for normalized concept: '%s'", normalized_input_lower)
    antonym_seeds_to_check = set(GENERAL_ANTONYM_SEEDS)
    logger.debug("Using %d antonym seeds.", len(antonym_seeds_to_check))
    logger.debug("Scanning taxonomy (%d URIs) for dynamic lists...", len(uri_embeddings_map))
    processed_tax_labels = set()
    scan_start_time = time.time()
    tax_syn_count, tax_exc_count = 0, 0
    for uri, emb_data_list in uri_embeddings_map.items():
        for prop, lbl, emb in emb_data_list:
            lbl_norm = normalize_concept(lbl).lower()
            if not lbl_norm or lbl_norm == normalized_input_lower or lbl_norm in processed_tax_labels:
                continue
            processed_tax_labels.add(lbl_norm)
            sim = calculate_semantic_similarity(concept_embedding, emb, normalized_input_lower, lbl_norm)
            if sim > SYNONYM_SIMILARITY_THRESHOLD and prop in ["prefLabel", "altLabel", "rdfsLabel"]:
                score = (0.4 * PROPERTY_WEIGHTS.get(prop, 0.9)) + (sim * 0.6)
                if lbl_norm not in synonyms_dict or score > synonyms_dict[lbl_norm][1]:
                    synonyms_dict[lbl_norm] = (lbl, score)
                    tax_syn_count +=1
            elif sim > EXCLUSION_SIMILARITY_THRESHOLD:
                exclusions_set.add(lbl_norm)
                tax_exc_count += 1
    logger.debug("Taxonomy scan done in %.2f s (Syn: %d, Exc: %d)", time.time() - scan_start_time, tax_syn_count, tax_exc_count)
    corpus_syn_count, corpus_exc_count = 0, 0
    corpus_scan_start_time = time.time()
    if corpus_data:
        logger.debug("Scanning corpus (%d terms)...", len(corpus_data))
        corpus_labels_norm = list(corpus_data.keys())
        corpus_embeddings_array = np.array(list(corpus_data.values()))
        if len(corpus_embeddings_array) > 0 and corpus_embeddings_array.ndim == 2:
            try:
                similarities = cosine_similarity(concept_embedding.reshape(1,-1), corpus_embeddings_array)[0]
                for i, sim_score in enumerate(similarities):
                    term_norm = corpus_labels_norm[i]
                    if term_norm == normalized_input_lower: continue
                    if sim_score > SYNONYM_SIMILARITY_THRESHOLD:
                        score = sim_score * 0.85
                        if term_norm not in synonyms_dict or score > synonyms_dict[term_norm][1]:
                             synonyms_dict[term_norm] = (term_norm, score)
                             corpus_syn_count += 1
                    elif sim_score > EXCLUSION_SIMILARITY_THRESHOLD:
                        exclusions_set.add(term_norm)
                        corpus_exc_count += 1
            except Exception as e: logger.error("Error calculating corpus similarities: %s", e)
        logger.debug("Corpus scan done in %.2f s (Syn: %d, Exc: %d)", time.time() - corpus_scan_start_time, corpus_syn_count, corpus_exc_count)
    final_synonyms_sorted = sorted(synonyms_dict.items(), key=lambda item: item[1][1], reverse=True)[:MAX_SYNONYMS]
    final_synonyms_list = [(data[0], data[1]) for norm_label, data in final_synonyms_sorted]
    logger.info("--- Dynamic Discovery Summary ---")
    logger.info("  Antonym Seeds: %d", len(antonym_seeds_to_check))
    logger.info("  Synonyms (Top %d): %d -> %s...", MAX_SYNONYMS, len(final_synonyms_list), [(s[:30]+'...', round(w, 3)) for s, w in final_synonyms_list][:5])
    logger.info("  Exclusions: %d -> %s...", len(exclusions_set), sorted(list(exclusions_set))[:5])
    return antonym_seeds_to_check, final_synonyms_list, exclusions_set

def find_best_taxonomy_matches(
    concept_string: str,
    taxonomy_concepts: Dict,
    sbert_model: SentenceTransformer,
    taxonomy_embeddings_data: Tuple[Dict, Dict, List],
    associations_data: Dict,
    corpus_data: Dict,
    args: argparse.Namespace
) -> List[Dict]:
    """Finds best matches, incorporating fix for UnboundLocalError."""
    try:
        overall_start_time = time.time()
        uri_embeddings_map, _, uris_list = taxonomy_embeddings_data
        if not uri_embeddings_map or not uris_list:
            logger.error("Embeddings data empty.")
            return []
        normalized_input_lower = normalize_concept(concept_string).lower().strip()
        logger.info("Finding matches for: '%s'", normalized_input_lower)
        if not normalized_input_lower:
            return []
        try:
            concept_embedding = sbert_model.encode([TRAVEL_CONTEXT + normalized_input_lower])[0]
        except Exception as e:
            logger.error(f"Failed to embed input '{normalized_input_lower}': {e}")
            return []
        antonym_check_list, synonyms_list, exclusion_list = get_dynamic_antonyms_synonyms(
            concept_string, concept_embedding, taxonomy_concepts, uri_embeddings_map, corpus_data
        )
        synonym_scores = {normalize_concept(s_orig).lower(): score for s_orig, score in synonyms_list}
        concept_words_set = set(normalized_input_lower.split())
        target_associated_labels = associations_data.get(normalized_input_lower, [])
        if target_associated_labels:
            logger.info(f"Input '{normalized_input_lower}' triggered associations: {target_associated_labels}")
        is_family_input = any(seed in concept_words_set for seed in FAMILY_SEEDS_ANTYNOM_LIST)
        potential_matches = {}
        normalized_label_cache = {}
        calculation_start_time = time.time()
        logger.debug("Calculating scores for %d URIs...", len(uris_list))
        uris_processed, uris_excluded, uris_penalized, uris_assoc_boosted = 0, 0, 0, 0

        for uri in uris_list:
            emb_data_list = [] # Initialize
            if uri not in uri_embeddings_map:
                logger.debug(f"URI {uri} not in embeddings map.")
                continue
            uris_processed += 1
            emb_data_list = uri_embeddings_map.get(uri)
            if emb_data_list is None:
                logger.warning(f"Got None embeddings for {uri}.")
                continue
            if not isinstance(emb_data_list, list):
                logger.warning(f"Embeddings for {uri} not list: {type(emb_data_list)}")
                continue

            concept_details = taxonomy_concepts.get(uri, {})
            best_uri_score = -float('inf'); best_uri_reason = ""; best_uri_keyword = 0.0; best_uri_semantic = 0.0; best_uri_label = ""; best_uri_prop = ""
            uri_is_exact_match = False; uri_penalty_triggered = False; uri_exclusion_triggered = False; penalty_trigger_label = ""; association_boost_applied_to_uri = False

            try: # Pre-check Pass
                for prop, lbl, embedding in emb_data_list:
                    lbl_lower = normalized_label_cache.get(lbl)
                    if lbl_lower is None:
                        lbl_lower = normalize_concept(lbl).lower()
                        normalized_label_cache[lbl] = lbl_lower
                    if not lbl_lower: continue
                    if lbl_lower == normalized_input_lower:
                        uri_is_exact_match = True; best_uri_score = 1.0; best_uri_reason = "exact"; best_uri_keyword = 1.0; best_uri_semantic = calculate_semantic_similarity(concept_embedding, embedding, normalized_input_lower, lbl_lower); best_uri_label = lbl; best_uri_prop = prop; uri_penalty_triggered = False; uri_exclusion_triggered = False; logger.debug(f"Exact match: {uri}"); break
                    if any(excl_word in lbl_lower.split() for excl_word in exclusion_list):
                        uri_exclusion_triggered = True; logger.debug(f"{uri} excluded via '{lbl_lower}'."); break
                    if not uri_penalty_triggered:
                        for ant in antonym_check_list:
                            pattern = r'\b' + re.escape(ant) + r'\b'; is_family_seed = ant in FAMILY_SEEDS_ANTYNOM_LIST; apply_penalty = (is_family_seed and not is_family_input) or (not is_family_seed)
                            if apply_penalty and re.search(pattern, lbl_lower, re.IGNORECASE):
                                 uri_penalty_triggered = True; penalty_trigger_label = lbl_lower
            except Exception as inner_loop_e:
                logger.error(f"Error processing labels for {uri}: {inner_loop_e}", exc_info=True)
                continue

            if uri_exclusion_triggered: uris_excluded += 1; continue
            if uri_is_exact_match: uri_penalty_triggered = False

            if not uri_is_exact_match: # Scoring Pass
                 best_uri_score_non_exact = -float('inf')
                 try:
                     for prop, label, embedding in emb_data_list:
                         label_lower_norm = normalized_label_cache.get(label)
                         if not label_lower_norm: continue
                         keyword_score_synonym = synonym_scores.get(label_lower_norm, 0.0); keyword_score_overlap = 0.0
                         if concept_words_set:
                             label_words = set(label_lower_norm.split()); common_words = concept_words_set.intersection(label_words)
                             if common_words:
                                 union_size = len(concept_words_set.union(label_words))
                                 if union_size > 0: overlap_score = len(common_words) / union_size; keyword_score_overlap = overlap_score * 0.7
                         keyword_score = max(keyword_score_synonym, keyword_score_overlap) * PROPERTY_WEIGHTS.get(prop, 0.9)
                         semantic_score = calculate_semantic_similarity(concept_embedding, embedding, normalized_input_lower, label_lower_norm) * PROPERTY_WEIGHTS.get(prop, 0.9)
                         current_label_score = (KEYWORD_WEIGHT * keyword_score) + (SEMANTIC_WEIGHT * semantic_score)
                         if current_label_score > best_uri_score_non_exact:
                            best_uri_score_non_exact = current_label_score
                            kw_comp = KEYWORD_WEIGHT * keyword_score
                            sem_comp = SEMANTIC_WEIGHT * semantic_score
                            # Corrected Formatting
                            if kw_comp > 0.05 and sem_comp > 0.05:
                                reason = "keyword+semantic" if abs(kw_comp - sem_comp) < 0.1 else ("keyword_dominant" if kw_comp > sem_comp else "semantic_dominant")
                            elif kw_comp > 0.05:
                                reason = "keyword_only"
                            elif sem_comp > 0.05:
                                reason = "semantic_only"
                            else:
                                reason = "low_signal"
                            best_uri_reason = reason; best_uri_keyword = keyword_score; best_uri_semantic = semantic_score; best_uri_label = label; best_uri_prop = prop
                 except Exception as scoring_loop_e:
                     logger.error(f"Error in scoring loop for {uri}: {scoring_loop_e}", exc_info=True)
                     continue
                 if best_uri_score_non_exact > -float('inf'):
                     best_uri_score = best_uri_score_non_exact

            if best_uri_score == -float('inf'): logger.debug(f"No score calculated for {uri}."); continue # Final Adjustments
            final_score = best_uri_score; final_keyword = best_uri_keyword; final_semantic = best_uri_semantic
            if uri_is_exact_match: final_score = max(final_score, 1.0) + EXACT_MATCH_BONUS; best_uri_reason = "exact_match"
            if target_associated_labels and not association_boost_applied_to_uri and not uri_is_exact_match:
                candidate_labels_normalized = set()
                for prop_key in ["prefLabel", "altLabel", "rdfsLabel"]:
                    for label_val in concept_details.get(prop_key, []):
                        norm_label = normalized_label_cache.get(label_val)
                        if not norm_label: norm_label = normalize_concept(label_val).lower(); normalized_label_cache[label_val] = norm_label
                        if norm_label: candidate_labels_normalized.add(norm_label)
                matched_target = next((t for t in target_associated_labels if t in candidate_labels_normalized), None)
                if matched_target: original_score = final_score; final_score = min(final_score + ASSOCIATION_BOOST, 1.0 + EXACT_MATCH_BONUS); best_uri_reason += "+assoc_boost"; association_boost_applied_to_uri = True; uris_assoc_boosted += 1; logger.info(f"Applied Assoc. Boost (+{ASSOCIATION_BOOST:.2f}) to {uri} via '{matched_target}'")
            if uri_penalty_triggered and not uri_is_exact_match:
                original_score_before_penalty = final_score; final_score = max(final_score + ANTONYM_PENALTY, final_score * 0.2)
                if original_score_before_penalty >= MIN_RELEVANCE_THRESHOLD: logger.info(f"Applied Antonym Penalty ({ANTONYM_PENALTY:.2f}) to {uri} ('{penalty_trigger_label[:60]}...', Score: {original_score_before_penalty:.4f} -> {final_score:.4f})"); uris_penalized += 1
                else: logger.debug(f"Antonym penalty triggered for {uri} but original score {original_score_before_penalty:.4f} below threshold.")
            if final_score >= MIN_RELEVANCE_THRESHOLD: potential_matches[uri] = {"score": final_score, "reason": best_uri_reason, "keyword": final_keyword, "semantic": final_semantic, "label": best_uri_label, "prop": best_uri_prop, "relationship_boost": False, "association_boost": association_boost_applied_to_uri}

        logger.info("Initial scores done (%d URIs, %d excluded, %d penalized, %d assoc boosted) in %.2f s.", uris_processed, uris_excluded, uris_penalized, uris_assoc_boosted, time.time() - calculation_start_time)
        boost_start_time = time.time(); boosted_matches = potential_matches.copy(); booster_uris = {uri for uri, data in potential_matches.items() if data["score"] >= REL_BOOST_SCORE_THRESHOLD}; boost_applied_count = 0
        if booster_uris: # Relationship Boost
            logger.info("Applying Relationship Boost: %d boosters found.", len(booster_uris)); uris_to_check_for_boost = potential_matches.keys() - booster_uris
            for uri_to_boost in uris_to_check_for_boost:
                data = boosted_matches[uri_to_boost];
                if data.get("relationship_boost", False) or data["reason"] == "exact_match": continue
                concept_details_to_boost = taxonomy_concepts.get(uri_to_boost, {}); related_uris = set(concept_details_to_boost.get("related", []) + concept_details_to_boost.get("broader", []) + concept_details_to_boost.get("narrower", []))
                if not related_uris: continue; intersecting_boosters = booster_uris.intersection(related_uris)
                if intersecting_boosters:
                     strongest_booster_uri = max(intersecting_boosters, key=lambda b_uri: potential_matches[b_uri]['score']); booster_score = potential_matches[strongest_booster_uri]['score']
                     if not data.get("relationship_boost"): original_score = data["score"]; boosted_score = min(original_score + RELATIONSHIP_BOOST, 1.0 + EXACT_MATCH_BONUS); boosted_matches[uri_to_boost]["score"] = boosted_score; boosted_matches[uri_to_boost]["relationship_boost"] = True; boosted_matches[uri_to_boost]["reason"] += "+rel_boost"; boost_applied_count += 1; logger.info(f"Applied rel boost (+{RELATIONSHIP_BOOST:.2f}) to {uri_to_boost} via {strongest_booster_uri}")
            logger.info("Applied relationship boost to %d concepts.", boost_applied_count)
        else: logger.info("No concepts met boost threshold (%.2f).", REL_BOOST_SCORE_THRESHOLD)
        logger.debug("Relationship boost done in %.2f s.", time.time() - boost_start_time)
        formatting_start_time = time.time(); pre_dedup_matches = [] # Format/Filter/Deduplicate
        for uri, data in boosted_matches.items():
            if data["score"] >= OUTPUT_RELEVANCE_THRESHOLD:
                pref_labels = taxonomy_concepts.get(uri, {}).get("prefLabel", []); display_label = pref_labels[0] if pref_labels else data["label"]
                if not display_label: display_label = f"Concept <...{uri[-20:]}>"
                pre_dedup_matches.append({"skos:prefLabel": display_label, "uri": uri, "relevance_score": round(data["score"], 4), "match_reason": data["reason"], "keyword_score": round(data["keyword"], 4), "semantic_score": round(data["semantic"], 4), "matched_property": data["prop"], "matched_label": data["label"], "relationship_boost": data["relationship_boost"], "association_boost": data.get("association_boost", False)})
        pre_dedup_matches.sort(key=lambda x: x["relevance_score"], reverse=True); final_matches = []; seen_normalized_labels = set(); deduplication_skipped_count = 0
        for match in pre_dedup_matches:
            norm_label = normalize_concept(match["skos:prefLabel"]).lower()
            if norm_label and norm_label not in seen_normalized_labels: seen_normalized_labels.add(norm_label); final_matches.append(match)
            elif not norm_label: final_matches.append(match)
            else: deduplication_skipped_count += 1; logger.debug(f"Skipping duplicate label '{norm_label}' from {match['uri']}")
        final_matches = final_matches[:MAX_MATCHES_PER_CONCEPT]; logger.debug("Formatted/deduplicated matches (%d skipped) in %.2f s.", deduplication_skipped_count, time.time() - formatting_start_time)
        total_time = time.time() - overall_start_time; logger.info("Found %d relevant concepts for '%s' in %.2f s.", len(final_matches), concept_string, total_time)
        return final_matches
    except Exception as e: logger.error(f"Unexpected error in find_best_taxonomy_matches for '{concept_string}': {e}\n{traceback.format_exc()}"); return []


# --- Theme Association Logic ---

def get_concept_text(uri: str, taxonomy_concepts: Dict) -> str:
    concept_data = taxonomy_concepts.get(uri, {}); texts = []
    for prop in ["prefLabel", "altLabel", "definition", "rdfsLabel", "dctermsDescription"]: texts.extend(concept_data.get(prop, []))
    return " ".join(filter(None, texts)).lower()

def map_uris_to_themes(matched_uris: List[Tuple[str, float]], taxonomy_concepts: Dict, sbert_model: SentenceTransformer) -> Dict[str, List[Dict]]:
    """Associates matched URIs with BASE_THEMES using hints (word boundary) and semantic similarity."""
    theme_assignments = defaultdict(list)
    if not matched_uris: logger.warning("map_uris_to_themes called with empty list."); return theme_assignments
    theme_embeddings = {}; theme_texts = {}; theme_hints_normalized = {}
    logger.debug("Pre-computing theme representations for %d themes...", len(BASE_THEMES)); prep_start_time = time.time()
    for theme_name, config in BASE_THEMES.items():
        hints = config.get("hints", []); theme_text_parts = [theme_name.lower()] + hints; theme_text = " ".join(filter(None, theme_text_parts))
        theme_texts[theme_name] = theme_text
        try: theme_embeddings[theme_name] = sbert_model.encode([theme_text])[0]
        except Exception as e: logger.error(f"Failed to embed theme '{theme_name}': {e}"); continue
        _normalized_hints_set = set() # Compatible with Python < 3.8
        for h in hints:
            h_norm = normalize_concept(h)
            if h_norm:
                 _normalized_hints_set.add(h_norm)
        theme_hints_normalized[theme_name] = _normalized_hints_set
    logger.debug("Theme prep done in %.2f s.", time.time() - prep_start_time)
    logger.info("Mapping %d matched URIs to themes...", len(matched_uris)); processed_uri_count = 0
    for uri, concept_match_score in matched_uris:
        processed_uri_count += 1; concept_text = get_concept_text(uri, taxonomy_concepts)
        if not concept_text: logger.debug(f"Skipping URI {uri}: No text."); continue
        try: concept_embedding = sbert_model.encode([concept_text])[0]
        except Exception as e: logger.error(f"Failed to embed concept URI {uri}: {e}"); continue
        pref_label = taxonomy_concepts.get(uri, {}).get("prefLabel", ["Unknown Label"])[0]; assigned_to_at_least_one_theme = False
        for theme_name, config in BASE_THEMES.items():
            if theme_name not in theme_embeddings or theme_name not in theme_hints_normalized: continue
            hint_match_found = False; num_hints_matched = 0; matched_hints = []
            for norm_hint in theme_hints_normalized[theme_name]:
                 if re.search(r'\b' + re.escape(norm_hint) + r'\b', concept_text, re.IGNORECASE):
                     hint_match_found = True; num_hints_matched += 1; matched_hints.append(norm_hint)
            hint_match_triggered = num_hints_matched >= THEME_HINT_MATCH_THRESHOLD
            semantic_sim = calculate_semantic_similarity(concept_embedding, theme_embeddings[theme_name], f"Concept:{uri}", f"Theme:{theme_name}")
            semantic_match_triggered = semantic_sim >= THEME_SEMANTIC_THRESHOLD
            if hint_match_triggered or semantic_match_triggered:
                 hint_score = 1.0 if hint_match_found else 0.0; semantic_score = semantic_sim
                 theme_confidence = (HINT_CONFIDENCE_WEIGHT * hint_score) + (SEMANTIC_CONFIDENCE_WEIGHT * semantic_score); theme_confidence = max(0.0, min(1.0, theme_confidence))
                 attribute_data = {"skos:prefLabel": pref_label, "uri": uri, "concept_weight": round(theme_confidence, 4), "original_match_score": round(concept_match_score, 4)}
                 theme_assignments[theme_name].append(attribute_data); assigned_to_at_least_one_theme = True
                 log_details = f"Hints: {num_hints_matched} ({matched_hints})" if hint_match_found else "Hints: No"; log_details += f", Sim: {semantic_sim:.3f}"; log_details += f" -> Confidence: {theme_confidence:.3f} (Wtd: H={HINT_CONFIDENCE_WEIGHT}, S={SEMANTIC_CONFIDENCE_WEIGHT})"
                 trigger = [];
                 if hint_match_triggered: trigger.append("Hint")
                 if semantic_match_triggered: trigger.append("Semantic")
                 log_details += f" [Trigger: {' & '.join(trigger)}]"; logger.debug(f"Associated URI {uri} ({pref_label}) with Theme '{theme_name}' ({log_details})")
        if not assigned_to_at_least_one_theme: logger.debug(f"URI {uri} ({pref_label}) not associated with any theme.")
    logger.info("Finished mapping %d URIs.", processed_uri_count); final_assignments = {}; post_proc_start_time = time.time(); logger.debug("Post-processing assignments...")
    for theme_name, attributes in theme_assignments.items():
        sorted_attributes = sorted(attributes, key=lambda x: x["concept_weight"], reverse=True)
        final_assignments[theme_name] = sorted_attributes[:MAX_ATTRIBUTES_PER_THEME]
        if len(attributes) > MAX_ATTRIBUTES_PER_THEME: logger.debug(f"Limited attributes for '{theme_name}' from {len(attributes)} to {MAX_ATTRIBUTES_PER_THEME}.")
    logger.debug("Post-processing done in %.2f s.", time.time() - post_proc_start_time); return final_assignments

# --- Main Orchestration ---

def generate_affinity_definitions(concepts_file: str, taxonomy_dir: str, output_dir: str, associations_file: Optional[str], corpus_file: Optional[str], cache_dir: str, rebuild_cache: bool, debug: bool):
    log_level = logging.DEBUG if debug else logging.INFO; logger.setLevel(log_level)
    for handler in logger.handlers: handler.setLevel(log_level)
    logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")
    try: os.makedirs(output_dir, exist_ok=True); logger.info(f"Output directory: {os.path.abspath(output_dir)}"); os.makedirs(cache_dir, exist_ok=True); logger.info(f"Cache directory: {os.path.abspath(cache_dir)}")
    except OSError as e: logger.critical(f"Fatal: Could not create directories: {e}"); return

    args_for_cache = argparse.Namespace(rebuild_cache=rebuild_cache)
    concept_cache_file = os.path.join(os.path.abspath(cache_dir), "taxonomy_concepts.json")
    embedding_cache_file = os.path.join(os.path.abspath(cache_dir), "taxonomy_embeddings.pkl")
    corpus_cache_file = os.path.join(os.path.abspath(cache_dir), "corpus_embeddings.pkl")
    overall_start_time = time.time(); logger.info("--- Starting Affinity Definition Generation (Version: %s) ---", CACHE_VERSION)

    try:
        resource_start_time = time.time()
        taxonomy_concepts = load_taxonomy_concepts(taxonomy_dir, concept_cache_file, args_for_cache)
        if not taxonomy_concepts: raise RuntimeError("Failed to load taxonomy concepts.")
        sbert_model = get_sbert_model()
        associations_data = load_concept_associations(associations_file)
        corpus_data = load_corpus_data(corpus_file, sbert_model, corpus_cache_file, args_for_cache)
        # Corpus data load now returns {} on failure/empty, check explicitly? Or let downstream handle? Let's assume downstream handles {}
        taxonomy_embeddings_data = precompute_taxonomy_embeddings(taxonomy_concepts, sbert_model, embedding_cache_file, args_for_cache)
        if not taxonomy_embeddings_data: raise RuntimeError("Failed to load or compute taxonomy embeddings.")
        logger.info("--- Finished Loading Resources in %.2f seconds ---", time.time() - resource_start_time)

        try:
            if not os.path.exists(concepts_file): raise FileNotFoundError(f"Input file not found: {concepts_file}")
            with open(concepts_file, 'r', encoding='utf-8') as f: input_concepts = [line.strip() for line in f if line.strip()]
            if not input_concepts: logger.warning(f"No concepts found in {concepts_file}")
            logger.info(f"Loaded {len(input_concepts)} concepts from {concepts_file}")
        except Exception as e: raise RuntimeError(f"Failed to read {concepts_file}: {e}") from e

        affinity_definitions = {"travel_concepts": []}; processed_count = 0; skipped_no_match = 0; skipped_no_themes = 0; processing_start_time = time.time()
        concept_iterator = tqdm(input_concepts, desc="Processing Concepts", unit="concept") if 'tqdm' in globals() and callable(tqdm) else input_concepts

        for concept_str in concept_iterator:
            logger.info(f"--- Processing Input Concept: '{concept_str}' ---"); concept_proc_start_time = time.time()
            matches = find_best_taxonomy_matches(concept_str, taxonomy_concepts, sbert_model, taxonomy_embeddings_data, associations_data, corpus_data, args_for_cache)
            if not matches: logger.warning(f"Skipping '{concept_str}': No suitable match."); skipped_no_match += 1; continue
            top_match = matches[0]; primary_label = top_match["skos:prefLabel"]; primary_uri = top_match["uri"]
            logger.info(f"Top match for '{concept_str}': '{primary_label}' ({primary_uri}) - Score: {top_match['relevance_score']:.4f}")
            uris_to_consider = [(m["uri"], m["relevance_score"]) for m in matches]
            theme_assignments = map_uris_to_themes(uris_to_consider, taxonomy_concepts, sbert_model)
            # TODO: Implement Exclusion Logic Here
            exclusions = [] # Placeholder
            # TODO: Implement Baseline Theme Check Here
            # check_and_add_baseline_themes(theme_assignments, primary_uri, ...)
            if not theme_assignments: logger.warning(f"Skipping '{concept_str}': No themes assigned."); skipped_no_themes += 1; continue

            concept_definition = {"applicable_lodging_types": "Both", "concept_type": "other", "travel_category": {"name": concept_str.title(), "skos:prefLabel": primary_label, "uri": primary_uri, "exclusionary_concepts": []}, "themes": [], "must_not_have": exclusions, "negative_overrides": [], "fallback_logic": {"condition": "At least one theme matched", "action": "Proceed with scoring", "threshold": 0.0}}
            # TODO: Add CONTEXT_SETTINGS here
            themes_added_count = 0
            for theme_name, attributes in theme_assignments.items():
                if theme_name in BASE_THEMES:
                     theme_config = BASE_THEMES[theme_name]
                     # TODO: Implement Dynamic Rule Description Generation Here
                     theme_summary = f"Rule based on '{theme_config.get('rule', 'Optional')}'" # Placeholder
                     concept_definition["themes"].append({"name": theme_name, "type": theme_config.get("type", "unknown"), "rule": theme_config.get("rule", "Optional"), "theme_summary": theme_summary, "theme_weight": theme_config.get("weight", 0.0), "subScore": theme_config.get("subScore", f"{theme_name.replace(' ', '')}Affinity"), "attributes": attributes})
                     themes_added_count += 1
                else: logger.warning(f"Theme '{theme_name}' assigned but not in BASE_THEMES.")
            if concept_definition["themes"]: affinity_definitions["travel_concepts"].append(concept_definition); processed_count += 1; logger.info(f"Processed '{concept_str}' -> '{primary_label}', assigned {themes_added_count} themes.")
            else: logger.warning(f"Skipping '{concept_str}': No BASE_THEMES assigned after checks."); skipped_no_themes += 1
            logger.debug("Processing concept '%s' took %.2f s.", concept_str, time.time() - concept_proc_start_time)

        processing_duration = time.time() - processing_start_time
        logger.info(f"--- Finished Processing in {processing_duration:.2f} seconds ---")
        logger.info(f"Summary: Processed={processed_count}, Skipped (no match)={skipped_no_match}, Skipped (no themes)={skipped_no_themes}")
        output_filename = "affinity_definitions.json"; json_output_file = os.path.join(output_dir, output_filename)
        logger.info(f"Writing {len(affinity_definitions['travel_concepts'])} definitions to {json_output_file}")
        try:
            with open(json_output_file, 'w', encoding='utf-8') as f: json.dump(affinity_definitions, f, indent=2, ensure_ascii=False)
            logger.info("Successfully wrote JSON output.")
        except Exception as e: logger.error(f"Failed to write JSON output: {e}")
    except RuntimeError as rte: logger.critical(f"Runtime Error: {rte}"); print(f"\nCRITICAL ERROR: Check log '{log_filename}'.")
    except Exception as e: logger.critical("Unexpected critical error: %s\n%s", e, traceback.format_exc()); print(f"\nCRITICAL ERROR: Check log '{log_filename}'.")
    finally: total_duration = time.time() - overall_start_time; logger.info("--- Finished in %.2f seconds ---", total_duration); print(f"\nTotal execution time: {total_duration:.2f} s. Log: {log_filename}")

# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate affinity definitions.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concepts", required=True, help="Input file with travel concepts (one per line).")
    parser.add_argument("--taxonomy-dir", required=True, help="Directory with RDF taxonomy files.")
    parser.add_argument("--output-dir", default="./output", help="Directory for output JSON.")
    parser.add_argument("--associations-file", default=DEFAULT_ASSOCIATIONS_FILE, help=f"Concept associations JSON file. Set to '' to disable.")
    parser.add_argument("--corpus-file", default="./datasources/travel-terms-corpus.txt", help="Travel corpus file (optional). Set to '' to disable.")
    parser.add_argument("--cache-dir", default="./cache", help="Directory for cache files.")
    parser.add_argument("--rebuild-cache", action="store_true", default=False, help="Force rebuild of all caches.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging.")
    args = parser.parse_args()
    associations_file_path = args.associations_file if args.associations_file else None; corpus_file_path = args.corpus_file if args.corpus_file else None
    print(f"--- Running Affinity Generator (Version: {CACHE_VERSION}) ---")
    print(f" Config: Concepts='{args.concepts}', Taxonomy='{args.taxonomy_dir}', Output='{args.output_dir}', Assoc='{associations_file_path if associations_file_path else 'N/A'}', Corpus='{corpus_file_path if corpus_file_path else 'N/A'}', Cache='{args.cache_dir}', Rebuild={args.rebuild_cache}, Debug={args.debug}")
    print(f" Log File: {log_filename}"); print("-" * 30)
    generate_affinity_definitions(args.concepts, args.taxonomy_dir, args.output_dir, associations_file_path, corpus_file_path, args.cache_dir, args.rebuild_cache, args.debug)
    print("-" * 30); print("Affinity generation complete."); print("-" * 30)