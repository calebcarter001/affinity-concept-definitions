#!/usr/bin/env python3
"""
Generate affinity definitions for travel concepts (v27.32) to fix WiFi failure by:
1. Finding initial anchor concepts semantically similar to the input.
2. Broadly gathering candidate evidence concepts (using v27.29 semantic threshold: 0.30, with boost for Technology hints).
3. Filtering contradictory evidence (**SEMANTIC FILTER DISABLED FOR 'luxury' input**).
4. Slotting evidence into 13 themes using **ENHANCED hints (v27.24)** and **v27.29 slotting threshold (0.31)** with v27.11 baseline parameters (0.4/0.6 weights).
5. Assembling the final affinity definition JSON.

**Changes from v27.29-diagnostic**:
- Added 0.2 evidence score boost for Technology-hinted URIs.

Version: 2025-04-15-affinity-evidence-v27.32 (Evid Thresh=0.30, Slot Thresh=0.31, Enhanced Hints, Filter Disabled for Lux)
"""

import os
import json
import logging
import argparse
import re
import pickle
import numpy as np
import sys
from collections import defaultdict, Counter
import time
import traceback
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# --- Handle Critical Dependency Imports ---
try:
    from rdflib import Graph, Namespace, URIRef, Literal, RDF, util
except ImportError:
    print("CRITICAL ERROR: rdflib library not found. Install: pip install rdflib", file=sys.stderr)
    sys.exit(1)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("CRITICAL ERROR: sentence-transformers library not found. Install: pip install sentence-transformers", file=sys.stderr)
    sys.exit(1)
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("CRITICAL ERROR: scikit-learn library not found. Install: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

# --- Handle Optional Dependency Imports ---
NLTK_AVAILABLE = False
STOP_WORDS = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'for', 'to', 'and', 'or', 'is', 'are', 'was', 'were'])
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    print("Info: NLTK stopwords loaded.")
except (ImportError, LookupError) as e:
    print(f"Warning: NLTK stopwords not found ({type(e).__name__}). Using basic list.")

def tqdm_dummy(iterable, *args, **kwargs): return iterable
tqdm = tqdm_dummy
try: from tqdm import tqdm as real_tqdm; tqdm = real_tqdm; print("Info: tqdm loaded.")
except ImportError: print("Warning: tqdm not found, progress bars disabled.")

# --- Setup Logging ---
log_filename = "affinity_generation_v27.32.log"
if os.path.exists(log_filename):
    try: os.remove(log_filename); print(f"Removed old log file: {log_filename}")
    except OSError as e: print(f"Warning: Could not remove old log file {log_filename}: {e}")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("affinity_generator_v27_32")

# --- Namespaces ---
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

# --- Configuration & Constants ---
CACHE_VERSION = "v20250415.affinity.27.32"
DEFAULT_ASSOCIATIONS_FILE = "./datasources/concept_associations.json"
DEFAULT_CORPUS_FILE = "./datasources/travel-terms-corpus.txt"
MIN_RELEVANCE_THRESHOLD = 0.30
OUTPUT_RELEVANCE_THRESHOLD = 0.0
MAX_ANCHOR_MATCHES = 10
KEYWORD_WEIGHT = 0.50
SEMANTIC_WEIGHT = 0.50
RELATIONSHIP_BOOST = 0.35
REL_BOOST_SCORE_THRESHOLD = 0.60
ASSOCIATION_BOOST = 0.40
EXACT_MATCH_BONUS = 0.01
PROPERTY_WEIGHTS = {"prefLabel": 1.0, "altLabel": 0.95, "rdfsLabel": 0.9, "definition": 0.80, "dctermsDescription": 0.80, "scopeNote": 0.75, "hiddenLabel": 0.7}
DEBUG_SEMANTIC_THRESHOLD = 0.2
SYNONYM_SIMILARITY_THRESHOLD = 0.55
MAX_SYNONYMS = 20
TRAVEL_CONTEXT = "travel "
RELATION_EXPANSION_DEPTH = 1
CONTEXT_TERM_MIN_LEN = 4
MAX_CONTEXT_TERMS = 30
KEYWORD_EVIDENCE_THRESHOLD = 1
# v27.29 Parameters
SEMANTIC_EVIDENCE_THRESHOLD = 0.30
TOP_N_SEMANTIC_EVIDENCE = 150
THEME_ASSIGNMENT_THRESHOLD = 0.31
MAX_ATTRIBUTES_PER_THEME = 10
THEME_HINT_MATCH_THRESHOLD = 1
# v27.11 Slotting Weights
HINT_CONFIDENCE_WEIGHT = 0.4
SEMANTIC_CONFIDENCE_WEIGHT = 0.6
SEMANTIC_CONTRADICTION_THRESHOLD = 0.50

# Semantic Contradiction Mapping (Disabled for 'luxury')
INPUT_TO_OPPOSITE_URI_SET_MAP: Dict[str, List[str]] = {
    "luxury": [
        "urn:expediagroup:taxonomies:acs:#dfcd2b66-79dd-4684-bb9e-f505c471e11d",
        "urn:expediagroup:taxonomies:acs:#0aaf7bc6-81ea-40b1-b284-4438c689ad42",
        "urn:expediagroup:taxonomies:acs:#f113fb8e-8d6f-4840-a3a6-4c1b42580889",
        "urn:expediagroup:taxonomies:acs:#f44ebe43-3153-4cb3-b5ea-d8cb97c416d1",
        "urn:expediagroup:taxonomies:acs:#9f2ad890-321f-4466-ad19-65b3634d88de",
        "urn:expediagroup:taxonomies:acs:#f7226e81-fbcd-4e80-ae09-63c5aafc80e1",
        "urn:expediagroup:taxonomies:acs:#af80bccf-ed5e-465a-ade9-7573fde64f42",
        "urn:expediagroup:taxonomies:acs:#663ec633-4ff4-42e2-9661-2b02ce176080",
        "urn:expediagroup:taxonomies:acs:#36c4e5df-72f1-47f3-80ad-60921d16c5c9",
        "urn:expediagroup:taxonomies:lcm:#2a71ae22-803c-3682-a187-8de9e52e6401"
    ],
}
INPUT_TO_MUST_NOT_HAVE_URIS: Dict[str, List[str]] = {
    "luxury": [
        "urn:expediagroup:taxonomies:lcm:#8b00a57b-3bcd-31c0-af01-f0545a7abf74",
        "urn:expediagroup:taxonomies:core:#b1492355-8b12-4336-a664-afc4d316326b",
        "urn:expediagroup:taxonomies:acs:#a359487f-fbfa-4758-a79c-7f427f3b0748",
    ]
}

CONTEXT_SETTINGS = [{"context": "PAAS", "multiplier": 1.0}, {"context": "SORT", "multiplier": 1.0}, {"context": "TAAP", "multiplier": 1.0}, {"context": "ADS", "multiplier": 1.0}, {"context": "CHAT", "multiplier": 1.0}]
_model_instance: Optional[SentenceTransformer] = None
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None
_taxonomy_embeddings_cache: Optional[Tuple[Dict, Dict, List]] = None
_corpus_data_cache: Optional[Dict[str, np.ndarray]] = None
_concept_associations_cache: Optional[Dict[str, List[str]]] = None
_theme_embeddings_cache: Optional[Dict[str, Dict]] = None

# Base Themes with Enhanced Hints
BASE_THEMES = {
    "Location": {"type": "decision", "rule": "Must have 1", "weight": 0.042, "subScore": "LocationAffinity",
                 "hints": ["location", "prime location", "exclusive neighborhood", "exclusive district", "city center", "downtown",
                           "waterfront", "oceanfront", "beachfront", "beachfront access",
                           "convenient", "safe location", "neighborhood", "walkable", "close to", "transportation", "parking",
                           "accessible parking", "valet parking",
                           "distance", "proximity", "nearby", "onsite", "ski-in",
                           "city view", "ocean view", "mountain view", "panoramic view", "secluded location", "private island"
                           ]},
    "Technology": {"type": "technological", "rule": "Must have 1", "weight": 0.059, "subScore": "TechAffinity",
                   "hints": [
                       "wifi", "wi fi", "internet", "high speed internet", "free wifi", "reliable connection",
                       "charging", "ev charging", "ev charging station", "outlets", "usb port", "usb c port",
                       "smart tv", "flat screen tv", "large screen tv", "streaming services", "premium channels", "media hub",
                       "smart room", "room controls", "tablet controls", "smart lighting", "smart climate control",
                       "sound system", "bluetooth speaker", "premium sound system",
                       "digital key", "mobile app", "contactless check-in",
                       "business center", "in-room computer"
                   ]},
    "Sentiment": {"type": "comfort", "rule": "Optional", "weight": 0.025, "subScore": "SentimentAffinity",
                  "hints": ["luxury", "opulent", "exclusive", "sophisticated", "elegant", "upscale",
                            "relaxing", "vibrant", "charming", "boutique", "unique", "atmosphere", "ambiance",
                            "quiet", "peaceful", "calm", "noise level"]},
    "Indoor Amenities": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "IndoorAmenityAffinity",
                         "hints": ["spa", "sauna", "hot tub", "indoor pool", "private spa suite",
                                   "bar", "lounge", "fine dining restaurant", "michelin star", "private chef", "wine cellar", "cigar lounge",
                                   "gym", "fitness center", "business center", "library", "casino", "private cinema",
                                   "concierge", "butler service", "personal butler", "valet service", "turndown service"
                                   ]},
    "Outdoor Amenities": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "OutdoorAmenityAffinity",
                          "hints": ["pool", "outdoor pool", "private pool", "infinity pool", "rooftop pool", "poolside bar",
                                    "garden", "manicured gardens", "patio", "terrace", "rooftop terrace", "balcony", "ocean view terrace",
                                    "bbq", "fire pit", "view", "waterfront view",
                                    "beach access", "private beach", "private beach access", "beach bar", "beach",
                                    "helipad", "yacht docking", "private dock"
                                    ]},
    "Activities": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "ActivityAffinity",
                   "hints": ["hiking", "skiing", "golf", "water sports", "yoga", "cycling",
                             "nightlife", "bar hopping", "entertainment", "classes",
                             "tours", "private tours", "custom tours", "yacht tours", "helicopter tours",
                             "shopping", "fashion", "wine tasting", "vineyard", "distillery", "brewery",
                             "cultural", "heritage", "indigenous", "local", "history", "traditional", "native",
                             "abandoned", "ruins", "historic site", "spa treatments"]},
    "Spaces": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "SpacesAffinity",
               "hints": ["suite", "penthouse", "villa", "private balcony", "jacuzzi suite",
                         "kitchen", "kitchenette", "living room", "dining area", "workspace", "dedicated workspace",
                         "multiple bedrooms"]},
    "Events": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "EventAffinity",
               "hints": ["festival", "wedding", "concert", "conference", "gala", "private party",
                         "event space", "meeting facilities",
                         "abbey", "monastery", "historic building",
                         "cultural", "heritage", "indigenous", "local", "history", "traditional", "native", "alcohol"]},
    "Seasonality": {"type": "preference", "rule": "Optional", "weight": 0.020, "subScore": "SeasonalityAffinity",
                    "hints": ["seasonal", "winter", "summer", "autumn", "spring", "peak season", "off season",
                              "monsoon", "dry season", "holiday season", "christmas", "new year", "easter"]},
    "Group Relevance": {"type": "preference", "rule": "Optional", "weight": 0.020, "subScore": "GroupRelevanceAffinity",
                        "hints": ["family friendly", "kid friendly", "child friendly", "group travel", "large group",
                                  "small group", "solo traveler", "couples", "adults only", "business traveler",
                                  "romantic getaway", "girls trip", "guys trip", "stag do", "hen party"]},
    "Privacy": {"type": "comfort", "rule": "Optional", "weight": 0.020, "subScore": "PrivacyAffinity",
                "hints": ["private", "secluded", "intimate", "quiet", "exclusive access", "gated access",
                          "soundproof", "soundproof rooms",
                          "not overlooked", "discreet", "discrete service",
                          "personal space", "seclusion", "private entrance", "secluded villa"
                          ]},
    "Accessibility": {"type": "structural", "rule": "Optional", "weight": 0.020, "subScore": "AccessibilityAffinity",
                      "hints": ["accessible", "wheelchair accessible", "ramp access", "elevator", "lift",
                                "disabled access", "step free access", "roll in shower", "accessible parking",
                                "braille signage", "hearing loop", "ada compliant"]},
    "Sustainability": {"type": "preference", "rule": "Optional", "weight": 0.020, "subScore": "SustainabilityAffinity",
                       "hints": ["sustainable", "eco friendly", "green", "responsible travel", "carbon neutral",
                                 "conservation", "locally sourced", "fair trade", "eco certified",
                                 "renewable energy", "low impact", "community tourism", "geotourism"]}
}

# Utility Functions
def get_sbert_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"Loading Sentence-BERT model ('{model_name}')...")
        start_time = time.time()
        try: _model_instance = SentenceTransformer(model_name)
        except Exception as e: logger.error(f"Fatal: SBERT model load failed: {e}", exc_info=True); raise RuntimeError("SBERT Model loading failed") from e
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
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            logger.debug(f"Non-ndarray input for similarity: {type(embedding1)} vs {type(embedding2)}")
            return 0.0
        if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
        if embedding1.shape[1] == 0 or embedding2.shape[1] == 0:
            logger.debug(f"Zero dimension embedding: {embedding1.shape} vs {embedding2.shape}")
            return 0.0
        if embedding1.shape[1] != embedding2.shape[1]:
            logger.warning(f"Dimension mismatch: {embedding1.shape[1]} vs {embedding2.shape[1]} for '{term1[:50]}' vs '{term2[:50]}'. Returning 0.")
            return 0.0
        sim = cosine_similarity(embedding1, embedding2)[0][0]
        similarity = max(0.0, min(1.0, float(sim)))
        if logger.isEnabledFor(logging.DEBUG) and (similarity >= DEBUG_SEMANTIC_THRESHOLD):
            t1 = term1[:50] + ('...' if len(term1) > 50 else ''); t2 = term2[:50] + ('...' if len(term2) > 50 else '')
            logger.debug(f"Similarity '{t1}' vs '{t2}': {similarity:.4f}")
        return similarity
    except Exception as e:
        logger.warning(f"Error calculating similarity for '{term1[:50]}' vs '{term2[:50]}': {e}")
        return 0.0

def get_primary_label(uri: str, uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]]) -> str:
    if not uri: return "INVALID_URI"
    uri_data = uri_embeddings_map.get(uri)
    if uri_data:
        labels = {'prefLabel': None, 'altLabel': None, 'rdfsLabel': None}
        for prop, orig_text, _, _ in uri_data:
            if prop == 'prefLabel' and labels['prefLabel'] is None: labels['prefLabel'] = orig_text
            elif prop == 'altLabel' and labels['altLabel'] is None: labels['altLabel'] = orig_text
            elif prop == 'rdfsLabel' and labels['rdfsLabel'] is None: labels['rdfsLabel'] = orig_text
        if labels['prefLabel']: return labels['prefLabel']
        if labels['altLabel']: return labels['altLabel']
        if labels['rdfsLabel']: return labels['rdfsLabel']
    if _taxonomy_concepts_cache:
        concept_details = _taxonomy_concepts_cache.get(uri, {})
        if concept_details.get("prefLabel"): return concept_details["prefLabel"][0]
        if concept_details.get("altLabel"): return concept_details["altLabel"][0]
        if concept_details.get("rdfsLabel"): return concept_details["rdfsLabel"][0]
    try:
        if '#' in uri: return uri.split('#')[-1]
        if '/' in uri: return uri.split('/')[-1]
    except: pass
    return uri

# Loading Functions
def load_concept_associations(associations_file: Optional[str]) -> Dict[str, List[str]]:
    global _concept_associations_cache
    if _concept_associations_cache is not None: return _concept_associations_cache
    if not associations_file:
        logger.info("No concept associations file provided.")
        _concept_associations_cache = {}
        return {}
    if not os.path.exists(associations_file):
        logger.warning(f"Concept associations file not found: {associations_file}")
        _concept_associations_cache = {}
        return {}
    logger.info(f"Loading concept associations from {associations_file}")
    start_time = time.time()
    try:
        with open(associations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        normalized_associations: Dict[str, List[str]] = {}
        for key, value_in in data.items():
            normalized_key = normalize_concept(key).lower()
            if not normalized_key:
                continue
            associated_values: List[str] = []
            if isinstance(value_in, list):
                associated_values = [normalize_concept(v).lower() for v in value_in if isinstance(v, str) and normalize_concept(v)]
            elif isinstance(value_in, str):
                normalized_value = normalize_concept(value_in).lower()
                if normalized_value:
                    associated_values.append(normalized_value)
            if associated_values:
                normalized_associations[normalized_key] = associated_values
        _concept_associations_cache = normalized_associations
        logger.info(f"Loaded {len(normalized_associations)} normalized associations in {time.time() - start_time:.2f}s.")
        return _concept_associations_cache
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from associations file {associations_file}: {e}")
        _concept_associations_cache = {}
        return {}
    except Exception as e:
        logger.error(f"Error loading concept associations: {e}", exc_info=True)
        _concept_associations_cache = {}
        return {}

def load_taxonomy_concepts(taxonomy_dir: str, cache_file: str, args: argparse.Namespace) -> Optional[Dict[str, Dict]]:
    global _taxonomy_concepts_cache
    if _taxonomy_concepts_cache is not None: return _taxonomy_concepts_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Attempting to load concepts from cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f: cached_data = json.load(f)
            if cached_data.get("cache_version") == CACHE_VERSION:
                concepts_data = cached_data.get("data")
                if isinstance(concepts_data, dict):
                    _taxonomy_concepts_cache = concepts_data
                    cache_valid = True
                    logger.info(f"Successfully loaded {len(_taxonomy_concepts_cache)} concepts from cache.")
                else: logger.warning("Concept cache data format is invalid. Rebuilding.")
            else:
                logger.info(f"Concept cache version mismatch. Rebuilding.")
        except Exception as e: logger.warning(f"Failed to load concepts from cache: {e}. Rebuilding.")
    if not cache_valid:
        logger.info(f"Loading taxonomy concepts from RDF files in: {taxonomy_dir}")
        start_time = time.time()
        concepts = defaultdict(lambda: defaultdict(list))
        files_ok = 0
        total_err = 0
        try:
            if not os.path.isdir(taxonomy_dir): raise FileNotFoundError(f"Taxonomy directory not found: {taxonomy_dir}")
            rdf_files = [f for f in os.listdir(taxonomy_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.xml', '.jsonld', '.nt', '.n3'))]
            if not rdf_files: logger.error(f"No RDF files found in directory: {taxonomy_dir}"); return None
            g = Graph()
            logger.info(f"Parsing {len(rdf_files)} potential RDF files...")
            disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
            for fn in tqdm(rdf_files, desc="Parsing RDF Files", disable=disable_tqdm):
                fp = os.path.join(taxonomy_dir, fn)
                try:
                    fmt = util.guess_format(fp)
                    if fmt: g.parse(fp, format=fmt); files_ok += 1
                    else: logger.warning(f"Could not determine format for {fn}, skipping.")
                except Exception as e: total_err += 1; logger.error(f"Error parsing file {fn}: {e}", exc_info=args.debug)
            logger.info(f"Parsed {files_ok}/{len(rdf_files)} files with {total_err} errors.")
            if files_ok == 0: logger.error("No RDF files parsed. Cannot proceed."); return None
            logger.info("Extracting concepts and properties...")
            pot_uris = set(s for s, p, o in g if isinstance(s, URIRef)) | set(o for s, p, o in g if isinstance(o, URIRef))
            logger.info(f"Found {len(pot_uris)} potential URIs. Processing...")
            skip_dep, rem_empty = 0, 0
            lbl_props = {SKOS.prefLabel: "prefLabel", SKOS.altLabel: "altLabel", RDFS.label: "rdfsLabel", SKOS.hiddenLabel: "hiddenLabel"}
            txt_props = {SKOS.definition: "definition", DCTERMS.description: "dctermsDescription", SKOS.scopeNote: "scopeNote"}
            rel_props = {SKOS.broader: "broader", SKOS.narrower: "narrower", SKOS.related: "related", SKOS.broadMatch: "broadMatch", SKOS.narrowMatch: "narrowMatch", SKOS.relatedMatch: "relatedMatch", OWL.sameAs: "sameAs"}
            kept_concepts_data = defaultdict(lambda: defaultdict(list))
            for uri in tqdm(pot_uris, desc="Processing URIs", disable=disable_tqdm):
                uri_s = str(uri)
                if g.value(uri, OWL.deprecated) == Literal(True): skip_dep += 1; continue
                is_concept = (uri, RDF.type, SKOS.Concept) in g or (uri, RDF.type, RDFS.Class) in g
                has_properties = False
                current_uri_data = defaultdict(list)
                for prop, key in lbl_props.items():
                    for obj in g.objects(uri, prop):
                        if isinstance(obj, Literal):
                            val = str(obj).strip(); current_uri_data[key].append(val) if val else None; has_properties = True
                for prop, key in txt_props.items():
                    for obj in g.objects(uri, prop):
                        if isinstance(obj, Literal):
                            val = str(obj).strip(); current_uri_data[key].append(val) if val else None; has_properties = True
                for prop, key in rel_props.items():
                    for obj in g.objects(uri, prop):
                        if isinstance(obj, URIRef):
                            current_uri_data[key].append(str(obj)); has_properties = True
                if is_concept or has_properties:
                    processed_data = {k: list(set(v)) for k, v in current_uri_data.items() if v}
                    if processed_data:
                        kept_concepts_data[uri_s] = processed_data
                    else: rem_empty += 1
                else:
                    rem_empty += 1
            logger.info(f"Extracted {len(kept_concepts_data)} concepts. Skipped {skip_dep} deprecated, removed {rem_empty} empty.")
            _taxonomy_concepts_cache = dict(kept_concepts_data)
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_data = {"cache_version": CACHE_VERSION, "data": _taxonomy_concepts_cache}
                with open(cache_file, 'w', encoding='utf-8') as f: json.dump(cache_data, f, indent=2)
                logger.info(f"Saved concepts to cache: {cache_file}")
            except Exception as e: logger.error(f"Failed to write concepts cache: {e}")
            logger.info(f"Taxonomy loading took {time.time() - start_time:.2f}s.")
        except FileNotFoundError as e: logger.error(f"Configuration error: {e}"); return None
        except Exception as e: logger.error(f"Unexpected error during taxonomy loading: {e}", exc_info=args.debug); return None
    if not _taxonomy_concepts_cache:
        logger.error("Failed to load taxonomy concepts.")
        return None
    return _taxonomy_concepts_cache

def precompute_taxonomy_embeddings(taxonomy_concepts: Dict[str, Dict], sbert_model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Optional[Tuple[Dict[str, List[Tuple[str, str, np.ndarray, str]]], Dict[str, np.ndarray], List[str]]]:
    global _taxonomy_embeddings_cache
    if _taxonomy_embeddings_cache is not None: return _taxonomy_embeddings_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Attempting to load embeddings from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached_data = pickle.load(f)
            if cached_data.get("cache_version") == CACHE_VERSION:
                uri_embeddings_map = cached_data.get("uri_embeddings_map")
                primary_embeddings = cached_data.get("primary_embeddings")
                uris_list = cached_data.get("uris_list")
                if isinstance(uri_embeddings_map, dict) and isinstance(primary_embeddings, dict) and isinstance(uris_list, list):
                    try:
                        sbert_dim = sbert_model.get_sentence_embedding_dimension()
                        is_valid = all(isinstance(v, np.ndarray) and v.ndim == 1 and v.shape == (sbert_dim,) for v in primary_embeddings.values() if v is not None)
                        if is_valid and len(uris_list) == len(primary_embeddings):
                            _taxonomy_embeddings_cache = (uri_embeddings_map, primary_embeddings, uris_list)
                            cache_valid = True
                            logger.info(f"Successfully loaded {len(uris_list)} embeddings from cache.")
                        else: logger.warning(f"Embeddings cache invalid. Rebuilding.")
                    except Exception as dim_e: logger.warning(f"Error validating dimensions: {dim_e}. Rebuilding.")
                else: logger.warning("Embedding cache data invalid. Rebuilding.")
            else: logger.info(f"Embedding cache version mismatch. Rebuilding.")
        except Exception as e: logger.warning(f"Failed to load embeddings: {e}. Rebuilding.")
    if not cache_valid:
        logger.info("Pre-computing taxonomy embeddings...")
        start_time = time.time()
        uri_embeddings_map = defaultdict(list)
        primary_embeddings: Dict[str, Optional[np.ndarray]] = {}
        uris_list: List[str] = []
        texts_to_embed_map = defaultdict(list)
        all_valid_uris = list(taxonomy_concepts.keys())
        logger.info(f"Processing {len(all_valid_uris)} concepts to gather texts.")
        disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
        text_properties = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel", "definition", "dctermsDescription", "scopeNote"]
        for uri in tqdm(all_valid_uris, desc="Gathering Texts", disable=disable_tqdm):
            concept_data = taxonomy_concepts.get(uri, {})
            uris_list.append(uri)
            found_texts_for_uri: Set[Tuple[str, str, str]] = set()
            for prop_key in text_properties:
                values = concept_data.get(prop_key, [])
                values = values if isinstance(values, list) else [values] if isinstance(values, str) else []
                for text_value in values:
                    if text_value and isinstance(text_value, str):
                        original_text = text_value.strip()
                        normalized_text = normalize_concept(original_text)
                        if normalized_text: found_texts_for_uri.add((prop_key, original_text, normalized_text))
            for prop_key, original_text, normalized_text in found_texts_for_uri:
                texts_to_embed_map[normalized_text].append((uri, prop_key, original_text))
        unique_normalized_texts = list(texts_to_embed_map.keys())
        logger.info(f"Collected {len(unique_normalized_texts)} unique text snippets to embed.")
        embeddings_list: Optional[List[np.ndarray]] = None
        texts_encoded: List[str] = []
        if unique_normalized_texts:
            logger.info("Generating embeddings...")
            batch_size = 128
            try:
                texts_encoded = [text for text in unique_normalized_texts if isinstance(text, str)]
                if texts_encoded:
                    embeddings_list = sbert_model.encode(texts_encoded, batch_size=batch_size, show_progress_bar=logger.isEnabledFor(logging.INFO))
                    logger.info("Embedding generation complete.")
                else: logger.warning("No valid texts to encode."); embeddings_list = []
            except Exception as e: logger.error(f"Fatal: SBERT encoding failed: {e}", exc_info=True); raise RuntimeError("SBERT Encoding Failed") from e
        else: logger.warning("No texts found."); embeddings_list = []
        logger.info("Mapping embeddings to URIs...")
        embedding_map: Dict[str, Optional[np.ndarray]] = {text: emb for text, emb in zip(texts_encoded, embeddings_list) if emb is not None}
        primary_embedding_candidates = defaultdict(dict)
        sbert_dim = sbert_model.get_sentence_embedding_dimension()
        for normalized_text, associated_infos in texts_to_embed_map.items():
            embedding = embedding_map.get(normalized_text)
            if embedding is None or not isinstance(embedding, np.ndarray) or embedding.shape != (sbert_dim,):
                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Skipping invalid embedding for: '{normalized_text[:50]}...'")
                continue
            for uri, prop_key, original_text in associated_infos:
                uri_embeddings_map[uri].append((prop_key, original_text, embedding, normalized_text))
                primary_embedding_candidates[uri][prop_key] = embedding
        primary_property_priority = ["prefLabel", "altLabel", "rdfsLabel", "hiddenLabel", "definition", "dctermsDescription", "scopeNote"]
        num_with_primary = 0
        num_without_primary = 0
        for uri in tqdm(uris_list, desc="Selecting Primary Embeddings", disable=disable_tqdm):
            candidates = primary_embedding_candidates.get(uri, {})
            chosen_embedding = None
            for prop in primary_property_priority:
                if prop in candidates and candidates[prop] is not None: chosen_embedding = candidates[prop]; break
            if chosen_embedding is None and candidates: chosen_embedding = next((emb for emb in candidates.values() if emb is not None), None)
            if chosen_embedding is not None and isinstance(chosen_embedding, np.ndarray) and chosen_embedding.ndim == 1 and chosen_embedding.shape == (sbert_dim,):
                primary_embeddings[uri] = chosen_embedding
                num_with_primary += 1
            else:
                primary_embeddings[uri] = None
                num_without_primary += 1
                if logger.isEnabledFor(logging.DEBUG):
                    if uri in primary_embedding_candidates: logger.debug(f"No valid primary embedding for {uri}.")
                    else: logger.debug(f"No text candidates for {uri}.")
        final_uris_list = [uri for uri in uris_list if primary_embeddings.get(uri) is not None]
        final_primary_embeddings = {uri: emb for uri, emb in primary_embeddings.items() if emb is not None}
        final_uri_embeddings_map = {uri: data for uri, data in uri_embeddings_map.items() if uri in final_primary_embeddings}
        _taxonomy_embeddings_cache = (final_uri_embeddings_map, final_primary_embeddings, final_uris_list)
        logger.info(f"Finished embedding process in {time.time() - start_time:.2f}s.")
        logger.info(f"Processed: {len(uris_list)}. With embedding: {num_with_primary}. Without: {num_without_primary}.")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data = {
                "cache_version": CACHE_VERSION,
                "uri_embeddings_map": _taxonomy_embeddings_cache[0],
                "primary_embeddings": _taxonomy_embeddings_cache[1],
                "uris_list": _taxonomy_embeddings_cache[2]
            }
            with open(cache_file, 'wb') as f: pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved embeddings to cache: {cache_file}")
        except Exception as e: logger.error(f"Failed to write embeddings cache: {e}")
    if not _taxonomy_embeddings_cache or not _taxonomy_embeddings_cache[1]:
        logger.error("Embedding process failed.")
        return None
    logger.info(f"Using {len(_taxonomy_embeddings_cache[1])} concepts with valid embeddings.")
    return _taxonomy_embeddings_cache

def load_corpus_data(corpus_file: Optional[str], sbert_model: SentenceTransformer, cache_file: str, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    global _corpus_data_cache
    if _corpus_data_cache is not None: return _corpus_data_cache
    if not corpus_file: logger.info("No corpus file specified."); _corpus_data_cache = {}; return {}
    cache_valid = False
    corpus_abs_path = os.path.abspath(corpus_file) if corpus_file else None
    rebuild = args.rebuild_cache
    if not rebuild and os.path.exists(cache_file):
        logger.info(f"Attempting corpus cache load: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached = pickle.load(f)
            if cached.get("cache_version") == CACHE_VERSION and cached.get("corpus_file_path") == corpus_abs_path:
                data = cached.get("data"); _corpus_data_cache = data if isinstance(data, dict) else {}; logger.info(f"Loaded {len(_corpus_data_cache)} corpus terms."); cache_valid = True
            else: logger.info("Corpus cache invalid. Rebuilding."); rebuild = True
        except Exception as e: logger.warning(f"Failed corpus cache load: {e}. Rebuilding."); rebuild = True
    if not cache_valid or rebuild:
        logger.info(f"Loading/embedding corpus: {corpus_file}")
        start = time.time()
        if not os.path.exists(corpus_file): logger.error(f"Corpus file not found: {corpus_file}"); _corpus_data_cache = {}; return {}
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f: terms = [normalize_concept(l.strip()) for l in f if l.strip()]
            unique = sorted(list(set(filter(None, terms))))
            logger.info(f"Found {len(unique)} unique terms.")
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
        except Exception as e: logger.error(f"Error processing corpus: {e}", exc_info=args.debug); _corpus_data_cache = {}; return {}
    return _corpus_data_cache

# Core Logic Functions
def get_dynamic_synonyms(concept: str, concept_embedding: Optional[np.ndarray], uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]], corpus_data: Dict[str, np.ndarray], args: argparse.Namespace) -> List[Tuple[str, float]]:
    synonyms_dict: Dict[str, float] = {}
    normalized_input_lower = normalize_concept(concept).lower().strip()
    logger.info("Generating dynamic synonyms for: '%s'", normalized_input_lower)
    if concept_embedding is None or not isinstance(concept_embedding, np.ndarray) or concept_embedding.ndim == 0:
        logger.warning(f"Cannot generate synonyms for '{concept}' due to invalid embedding.")
        return []
    if concept_embedding.ndim == 1:
        concept_embedding = concept_embedding.reshape(1, -1)
    start_time = time.time()
    processed_texts: Set[str] = {normalized_input_lower}
    tax_syn_count, corp_syn_count = 0, 0
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    if uri_embeddings_map:
        logger.debug("Searching for synonyms in taxonomy...")
        all_taxonomy_snippets = [snippet for snippets in uri_embeddings_map.values() for snippet in snippets]
        for prop, orig_text, emb, norm_text in tqdm(all_taxonomy_snippets, desc="Synonyms (Taxonomy)", disable=disable_tqdm):
            if not norm_text or norm_text in processed_texts: continue
            similarity = calculate_semantic_similarity(concept_embedding, emb, normalized_input_lower, norm_text)
            if similarity >= SYNONYM_SIMILARITY_THRESHOLD:
                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"  Taxonomy synonym: '{norm_text}' (score: {similarity:.4f})")
                synonyms_dict[norm_text] = max(similarity, synonyms_dict.get(norm_text, 0.0))
                tax_syn_count += 1
            processed_texts.add(norm_text)
    if corpus_data:
        logger.debug("Searching for synonyms in corpus...")
        for term, emb in tqdm(corpus_data.items(), desc="Synonyms (Corpus)", disable=disable_tqdm):
            if not term or term in processed_texts: continue
            similarity = calculate_semantic_similarity(concept_embedding, emb, normalized_input_lower, term)
            if similarity >= SYNONYM_SIMILARITY_THRESHOLD:
                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"  Corpus synonym: '{term}' (score: {similarity:.4f})")
                synonyms_dict[term] = max(similarity, synonyms_dict.get(term, 0.0))
                corp_syn_count += 1
            processed_texts.add(term)
    synonym_list = sorted([(k,v) for k,v in synonyms_dict.items() if v > 0], key=lambda item: item[1], reverse=True)
    final_synonyms = synonym_list[:MAX_SYNONYMS]
    logger.info(f"Found {len(synonym_list)} synonyms ({tax_syn_count} tax, {corp_syn_count} corpus) >= {SYNONYM_SIMILARITY_THRESHOLD:.2f}. Ret {len(final_synonyms)} in {time.time() - start_time:.2f}s.")
    return final_synonyms

def find_best_taxonomy_matches(concept: str, sbert_model: SentenceTransformer, uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]], primary_embeddings: Dict[str, np.ndarray], uris_list: List[str], associations: Dict[str, List[str]], corpus_data: Dict[str, np.ndarray], args: argparse.Namespace) -> List[Tuple[str, float, str, List[str]]]:
    logger.info(f"--- Finding best taxonomy matches for: '{concept}' ---")
    start_time = time.time()
    norm_concept_lower = normalize_concept(concept).lower().strip()
    if not norm_concept_lower: logger.warning("Input concept normalized to empty."); return []
    try:
        input_text = (TRAVEL_CONTEXT + norm_concept_lower) if TRAVEL_CONTEXT else norm_concept_lower
        concept_embedding = sbert_model.encode([input_text])[0]
        if concept_embedding is None or not isinstance(concept_embedding, np.ndarray): raise ValueError("Embedding failed")
    except Exception as e: logger.error(f"Error generating embedding for '{concept}': {e}", exc_info=True); return []
    dynamic_synonyms = get_dynamic_synonyms(concept, concept_embedding, uri_embeddings_map, corpus_data, args)
    synonym_texts = {syn for syn, score in dynamic_synonyms}
    concept_scores: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"max_score": 0.0, "matched_texts": set(), "details": []})
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    keywords = [kw for kw in norm_concept_lower.split() if kw not in STOP_WORDS and len(kw) > 1]
    kw_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keywords]
    for uri in tqdm(uris_list, desc="Scoring Concepts", disable=disable_tqdm):
        uri_data = uri_embeddings_map.get(uri, [])
        primary_emb = primary_embeddings.get(uri)
        max_score = 0.0
        details = []
        matched = set()
        if primary_emb is None:
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Skipping primary embedding score for {uri} - missing.")
        else:
            pri_sem_score = calculate_semantic_similarity(concept_embedding, primary_emb, norm_concept_lower, f"PrimaryEmb({uri})")
            max_score = pri_sem_score
            details.append(f"PrimarySem({pri_sem_score:.3f})")
            pref_labels = [orig for prop, orig, emb, norm in uri_data if prop == 'prefLabel']
            if pref_labels: matched.add(pref_labels[0])
        for prop, orig, txt_emb, norm in uri_data:
            prop_w = PROPERTY_WEIGHTS.get(prop, 0.5)
            kw_score = 0.0
            sem_score = 0.0
            kw_count = sum(1 for p in kw_patterns if p.search(orig) or p.search(norm)) if keywords else 0
            if kw_count > 0: kw_score = min((kw_count/len(keywords) if keywords else 0)+(EXACT_MATCH_BONUS if norm_concept_lower == norm else 0), 1.0)
            sem_score = calculate_semantic_similarity(concept_embedding, txt_emb, norm_concept_lower, f"{prop}({norm[:30]}...)")
            comb_score = (kw_score * KEYWORD_WEIGHT + sem_score * SEMANTIC_WEIGHT) * prop_w
            if norm_concept_lower in associations and any(a in norm for a in associations[norm_concept_lower]): comb_score = min(1.0, comb_score + ASSOCIATION_BOOST)
            if norm in synonym_texts: syn_s = next((s for syn, s in dynamic_synonyms if syn == norm), 0.0); comb_score = min(1.0, comb_score + (syn_s * 0.2))
            comb_score = max(0.0, min(1.0, comb_score))
            if comb_score > max_score:
                max_score = comb_score
                details.append(f"{prop}(KW:{kw_score:.2f}|Sem:{sem_score:.2f}|Comb:{comb_score:.3f})")
                matched.add(orig)
            elif logger.isEnabledFor(logging.DEBUG) and comb_score > 0:
                logger.debug(f"  URI {uri[-20:]}: Sub-max score {comb_score:.3f} from '{norm[:30]}...' (KW:{kw_score:.2f}, Sem:{sem_score:.2f})")
        if max_score >= MIN_RELEVANCE_THRESHOLD:
            concept_scores[uri]["max_score"] = max_score
            concept_scores[uri]["matched_texts"].update(matched)
            concept_scores[uri]["details"] = details
            concept_scores[uri]["primary_label"] = get_primary_label(uri, uri_embeddings_map)
    sorted_concepts = sorted(concept_scores.items(), key=lambda item: item[1]["max_score"], reverse=True)
    results = [(uri, data["max_score"], data.get("primary_label", "Unknown"), list(data["matched_texts"])[:5]) for uri, data in sorted_concepts if data["max_score"] >= MIN_RELEVANCE_THRESHOLD]
    logger.info(f"Found {len(results)} matches >= {MIN_RELEVANCE_THRESHOLD:.2f} in {time.time() - start_time:.2f}s.")
    top_results = results[:MAX_ANCHOR_MATCHES]
    logger.info(f"Returning top {len(top_results)} matches.")
    return top_results













def gather_candidate_evidence_concepts(input_concept: str, anchor_concepts: List, taxonomy_concepts: Dict, uri_embeddings_map: Dict, primary_embeddings: Dict, uris_list: List, sbert_model: SentenceTransformer, concept_associations: Dict[str, List[str]], args: argparse.Namespace) -> Dict[str, float]:
    logger.info(f"--- Gathering evidence for: '{input_concept}' (Threshold: {SEMANTIC_EVIDENCE_THRESHOLD}) ---")
    start = time.time()
    candidates: Dict[str, float] = defaultdict(float)
    processed: Set[str] = set()
    if not anchor_concepts: logger.warning("No anchor concepts provided."); return {}
    top_uri, top_score, _, _ = anchor_concepts[0]
    logger.info(f"Top anchor: {top_uri} (Score: {top_score:.3f})")
    for uri, score, _, _ in anchor_concepts:
        candidates[uri] = max(candidates[uri], score)
        processed.add(uri)
    logger.info(f"Added {len(anchor_concepts)} anchor concepts.")
    explore: Set[Tuple[str, int]] = set((u, 0) for u, _, _, _ in anchor_concepts)
    rels = ["broader", "narrower", "related", "broadMatch", "narrowMatch", "relatedMatch", "sameAs"]
    depth = 0
    while depth < RELATION_EXPANSION_DEPTH and explore:
        logger.debug(f"Exploring relationships at depth {depth}...")
        nxt: Set[Tuple[str, int]] = set()
        uris_now = [u for u, d in explore if d == depth]
        proc_now = 0
        for src_uri in uris_now:
            src_score = candidates.get(src_uri, 0.0)
            if src_score < REL_BOOST_SCORE_THRESHOLD: continue
            data = taxonomy_concepts.get(src_uri, {})
            for r_type in rels:
                for r_uri in data.get(r_type, []):
                    if isinstance(r_uri, str) and r_uri not in processed and r_uri in primary_embeddings:
                        boost = max(0.0, min(1.0, src_score * RELATIONSHIP_BOOST))
                        candidates[r_uri] = max(candidates[r_uri], boost)
                        processed.add(r_uri)
                        nxt.add((r_uri, depth + 1))
                        proc_now += 1
        explore.update(nxt)
        depth += 1
        logger.info(f"Found {proc_now} related concepts at depth {depth-1}. Total: {len(candidates)}")
    logger.info("Performing keyword search...")
    norm_in = normalize_concept(input_concept).lower().strip()
    kws = [kw for kw in norm_in.split() if kw not in STOP_WORDS and len(kw) > 1]
    kw_pats = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in kws]
    kw_add = 0
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    if kws:
        kw_uris_to_check = [uri for uri in uris_list if uri not in processed]
        logger.info(f"Keyword searching against {len(kw_uris_to_check)} concepts...")
        for uri in tqdm(kw_uris_to_check, desc="Keyword Search", disable=disable_tqdm):
            data = uri_embeddings_map.get(uri, [])
            matches = 0
            for _, orig, _, norm in data:
                matches += sum(1 for p in kw_pats if p.search(orig) or p.search(norm))
            if matches >= KEYWORD_EVIDENCE_THRESHOLD:
                kw_score = 0.1 + (0.2 * min(1, matches/len(kws)))
                candidates[uri] = max(candidates[uri], kw_score)
                processed.add(uri)
                kw_add += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Added via KW: {uri[-20:]} (Matches: {matches}, Score: {kw_score:.3f})")
        logger.info(f"Added {kw_add} concepts via keyword search. Total: {len(candidates)}")
    else:
        logger.info("No keywords, skipping keyword search.")
    logger.info(f"Performing semantic search (Threshold: {SEMANTIC_EVIDENCE_THRESHOLD})...")
    wifi_uri = "urn:expediagroup:taxonomies:acs:#d86726ff-4a17-4a28-996e-5b81963254a8"
    sem_add = 0
    try:
        in_emb_txt = (TRAVEL_CONTEXT + norm_in) if TRAVEL_CONTEXT else norm_in
        in_emb = sbert_model.encode([in_emb_txt])[0]
        assert in_emb is not None
    except Exception as e:
        logger.error(f"Input embedding failed for '{input_concept}': {e}")
        in_emb = None
    if in_emb is not None:
        target_uris = [u for u in uris_list if u not in processed]
        if not target_uris:
            logger.info("No remaining concepts for semantic search.")
        else:
            # Wireless URIs
            tech_hint_uris = {
                "urn:expediagroup:taxonomies:acs:#d86726ff-4a17-4a28-996e-5b81963254a8",  # WiFi
                "urn:expediagroup:taxonomies:acs:#5bad798b-0ced-45f0-9493-1386d904bfa0",  # WiFi:InRoom
                "urn:expediagroup:taxonomies:lcm:#2877666a-5f21-37ea-a3e7-786e8c2161c5",  # ResortFeeInclu-WiFi access
                "urn:expediagroup:taxonomies:lcm:#89079c53-6244-4545-8243-0d0978f3b4d9",  # WiFi Speed
                "urn:expediagroup:taxonomies:lcm:#2f723753-475a-3198-b738-f1f7254d4087",  # WiFi-Free-Limited-Time-Duration
                "urn:expediagroup:taxonomies:lcm:#7f27512a-acc9-3157-bba7-94b8b3af79f2",  # Wireless Internet access-free
                "urn:expediagroup:taxonomies:lcm:#73ebf773-a7ea-39d9-9ab2-8d37ab8a7c69",  # Wireless Internet access-surch
                "urn:expediagroup:taxonomies:lcm:#1e09a326-11f0-3646-9d94-3b21932c9b6e"   # WiFi-Free-Limited-Time-Amt
            }
            tech_labels = {"wifi", "wi fi", "wireless internet", "internet", "high speed internet", "free wifi"}
            for uri, data in uri_embeddings_map.items():
                for _, orig_text, _, norm in data:
                    norm_text = normalize_concept(orig_text)
                    if any(label == norm_text or label in norm_text for label in tech_labels):
                        tech_hint_uris.add(uri)
            target_embs = np.array([primary_embeddings[u] for u in target_uris])
            if target_embs.shape[0] > 0:
                logger.info(f"Calculating similarity vs {len(target_uris)} embeddings...")
                sims = cosine_similarity(in_emb.reshape(1, -1), target_embs)[0]
                sem_cands = []
                assoc_terms = [normalize_concept(a) for a in concept_associations.get(norm_in, [])]
                for idx, uri in enumerate(target_uris):
                    score = float(sims[idx])
                    assoc_boost = 0.0
                    for _, orig_text, _, norm in uri_embeddings_map.get(uri, []):
                        norm_text = normalize_concept(orig_text)
                        if any(a == norm_text or a in norm_text for a in assoc_terms):
                            assoc_boost = ASSOCIATION_BOOST
                            score = min(1.0, score + assoc_boost)
                            break
                    if uri in tech_hint_uris:
                        score = min(1.0, score + 0.3)
                        if assoc_boost > 0:
                            logger.debug(f"URI {uri} received association boost: {assoc_boost:.4f}")
                    if uri in tech_hint_uris:
                        logger.debug(f"URI {uri} vs Luxury evidence score: {score:.4f}")
                        if score < SEMANTIC_EVIDENCE_THRESHOLD:
                            logger.debug(f"URI {uri} fails evidence threshold: {score:.4f} < {SEMANTIC_EVIDENCE_THRESHOLD}")
                        else:
                            logger.debug(f"URI {uri} passes evidence threshold: {score:.4f}")
                    if score >= SEMANTIC_EVIDENCE_THRESHOLD and uri not in processed:
                        sem_cands.append((uri, score))
                        processed.add(uri)
                        sem_add += 1
                if sem_cands:
                    top_sem = sorted(sem_cands, key=lambda x: x[1], reverse=True)[:TOP_N_SEMANTIC_EVIDENCE]
                    for uri, score in top_sem:
                        candidates[uri] = max(candidates[uri], score)
                    logger.info(f"Added {sem_add} concepts via semantic search (Top {TOP_N_SEMANTIC_EVIDENCE} >= {SEMANTIC_EVIDENCE_THRESHOLD}). Total: {len(candidates)}")
                else:
                    logger.info(f"No concepts met {SEMANTIC_EVIDENCE_THRESHOLD} threshold.")
            else:
                logger.info("Target embeddings empty.")
    else:
        logger.info("Skipping semantic search due to embedding failure.")
    logger.info(f"--- Finished gathering evidence. Found {len(candidates)} candidates in {time.time() - start:.2f}s. ---")
    return dict(candidates)

















def get_theme_embeddings(sbert_model: SentenceTransformer, themes_def: Dict) -> Dict[str, Dict]:
    global _theme_embeddings_cache
    theme_cache_file = os.path.join(args.cache_dir, f"themes_{CACHE_VERSION}.pkl")
    hints_structure_hash = hash(str(sorted([(k, sorted(v.get('hints',[]))) for k, v in themes_def.items()])))
    if _theme_embeddings_cache is not None: return _theme_embeddings_cache
    if not args.rebuild_cache and os.path.exists(theme_cache_file):
        logger.info(f"Attempting to load theme embeddings: {theme_cache_file}")
        try:
            with open(theme_cache_file, 'rb') as f: cached_theme_data = pickle.load(f)
            if (cached_theme_data.get("cache_version") == CACHE_VERSION and
                cached_theme_data.get("hints_hash") == hints_structure_hash):
                _theme_embeddings_cache = cached_theme_data.get("data")
                if isinstance(_theme_embeddings_cache, dict) and len(_theme_embeddings_cache) == len(themes_def):
                    logger.info(f"Loaded theme embeddings for {len(_theme_embeddings_cache)} themes.")
                    return _theme_embeddings_cache
                else: logger.warning("Theme cache data invalid. Recomputing.")
            else: logger.info("Theme cache version or hint mismatch. Recomputing.")
        except Exception as e: logger.warning(f"Failed to load theme embeddings: {e}. Recomputing.")
    logger.info("Pre-computing theme embeddings...")
    start_time = time.time()
    _theme_embeddings_cache = None
    theme_embeddings = {}
    all_texts_to_embed = []
    text_to_theme_map = {}
    for theme_name, data in themes_def.items():
        theme_name_norm = normalize_concept(theme_name)
        if theme_name_norm:
            all_texts_to_embed.append(theme_name_norm)
            text_to_theme_map[theme_name_norm] = (theme_name, 'name')
        hints = data.get('hints', [])
        for hint in hints:
            hint_norm = normalize_concept(hint)
            if hint_norm and hint_norm not in text_to_theme_map:
                all_texts_to_embed.append(hint_norm)
                text_to_theme_map[hint_norm] = (theme_name, 'hint')
    if not all_texts_to_embed: logger.warning("No theme names or hints to embed."); return {}
    try:
        embeddings = sbert_model.encode(all_texts_to_embed, batch_size=32, show_progress_bar=logger.isEnabledFor(logging.DEBUG))
    except Exception as e:
        logger.error(f"Failed to embed theme texts: {e}", exc_info=True)
        return {}
    for text, embedding in zip(all_texts_to_embed, embeddings):
        if embedding is None: continue
        theme_name, text_type = text_to_theme_map[text]
        if theme_name not in theme_embeddings: theme_embeddings[theme_name] = {'name_embedding': None, 'hint_embeddings': [], 'hints': []}
        if text_type == 'name': theme_embeddings[theme_name]['name_embedding'] = embedding
        else:
            theme_embeddings[theme_name]['hint_embeddings'].append(embedding)
            theme_embeddings[theme_name]['hints'].append(text)
    _theme_embeddings_cache = theme_embeddings
    logger.info(f"Computed embeddings for {len(theme_embeddings)} themes in {time.time() - start_time:.2f}s.")
    try:
        os.makedirs(os.path.dirname(theme_cache_file), exist_ok=True)
        cache_to_save = {"cache_version": CACHE_VERSION, "hints_hash": hints_structure_hash, "data": _theme_embeddings_cache}
        with open(theme_cache_file, 'wb') as f_theme: pickle.dump(cache_to_save, f_theme, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved theme embeddings to cache: {theme_cache_file}")
    except Exception as e:
        logger.error(f"Failed writing theme cache: {e}")
    return _theme_embeddings_cache

def filter_contradictory_evidence(input_concept: str, candidate_evidence: Dict[str, float], primary_embeddings: Dict[str, np.ndarray], uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]], opposite_uri_map: Dict[str, List[str]], specific_exclusions_map: Dict[str, List[str]], args: argparse.Namespace) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    filter_action = "specific exclusions only" if not opposite_uri_map else "contradictions and exclusions"
    logger.info(f"--- Filtering {len(candidate_evidence)} candidates ({filter_action}) for: '{input_concept}' ---")
    start_time = time.time()
    valid_evidence: Dict[str, float] = {}
    must_not_have_dict: Dict[str, Dict[str, Any]] = {}
    normalized_input_lower = normalize_concept(input_concept).lower().strip()
    semantic_check_enabled = bool(opposite_uri_map)
    opposite_embeddings: List[Tuple[str, np.ndarray]] = []
    if semantic_check_enabled:
        opposite_uris_for_input = opposite_uri_map.get(normalized_input_lower, [])
        if opposite_uris_for_input:
            logger.info(f"Checking embeddings for {len(opposite_uris_for_input)} opposite URIs.")
            valid_opposite_count = 0
            missing_opposites = []
            for opp_uri in opposite_uris_for_input:
                opp_emb = primary_embeddings.get(opp_uri)
                if opp_emb is not None and isinstance(opp_emb, np.ndarray):
                    opposite_embeddings.append((opp_uri, opp_emb))
                    valid_opposite_count += 1
                else:
                    missing_opposites.append(opp_uri)
                    logger.warning(f"Opposite URI '{opp_uri}' has no embedding.")
            if valid_opposite_count > 0: logger.info(f"Found {valid_opposite_count} valid opposite embeddings.")
            if not valid_opposite_count: logger.warning(f"No valid embeddings for opposites. Semantic check skipped."); semantic_check_enabled = False
        else: logger.info(f"No opposite URIs mapped. Semantic check skipped."); semantic_check_enabled = False
    else:
        logger.info(f"Semantic contradiction check skipped for '{normalized_input_lower}'.")
    specific_exclusions = set(specific_exclusions_map.get(normalized_input_lower, []))
    if specific_exclusions: logger.info(f"Applying {len(specific_exclusions)} exclusions.")
    sem_checks_performed = 0
    sem_excluded_count = 0
    spec_excluded_count = 0
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    for uri, score in tqdm(candidate_evidence.items(), desc="Filtering Evidence", disable=disable_tqdm):
        if not uri or not isinstance(uri, str): continue
        label = get_primary_label(uri, uri_embeddings_map)
        if uri in specific_exclusions:
            spec_excluded_count += 1
            reason = f"Specifically excluded for '{normalized_input_lower}'"
            must_not_have_dict[uri] = {"uri": uri, "label": label, "reason": reason, "score": 1.0}
            continue
        is_semantically_contradictory = False
        sem_reason = ""
        max_sim = 0.0
        if semantic_check_enabled and opposite_embeddings:
            evidence_embedding = primary_embeddings.get(uri)
            if evidence_embedding is not None and isinstance(evidence_embedding, np.ndarray):
                sem_checks_performed += 1
                for opp_uri, opp_embedding in opposite_embeddings:
                    similarity = calculate_semantic_similarity(evidence_embedding, opp_embedding, uri, opp_uri)
                    if similarity >= SEMANTIC_CONTRADICTION_THRESHOLD:
                        is_semantically_contradictory = True
                        if similarity > max_sim:
                            max_sim = similarity
                            sem_reason = f"High similarity ({similarity:.3f}) to opposite '{get_primary_label(opp_uri, uri_embeddings_map)}' ({opp_uri})"
        if is_semantically_contradictory:
            sem_excluded_count += 1
            must_not_have_dict[uri] = {"uri": uri, "label": label, "reason": sem_reason, "score": max_sim}
        else:
            valid_evidence[uri] = score
    logger.info(f"Filtering complete. Performed {sem_checks_performed} semantic checks.")
    logger.info(f"Excluded {spec_excluded_count} via rules, {sem_excluded_count} via contradictions.")
    logger.info(f"Total excluded: {len(must_not_have_dict)}. Returning {len(valid_evidence)} valid concepts.")
    must_not_have_list = sorted(must_not_have_dict.values(), key=lambda x: x['score'], reverse=True)
    logger.info(f"--- Finished filtering in {time.time() - start_time:.2f}s. ---")
    return valid_evidence, must_not_have_list

def slot_evidence_to_themes(valid_evidence: Dict[str, float], uri_embeddings_map: Dict[str, List[Tuple[str, str, np.ndarray, str]]], primary_embeddings: Dict[str, np.ndarray], sbert_model: SentenceTransformer, themes_def: Dict[str, Dict], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    logger.info(f"--- Slotting {len(valid_evidence)} concepts (Threshold={THEME_ASSIGNMENT_THRESHOLD}, Weights={HINT_CONFIDENCE_WEIGHT}/{SEMANTIC_CONFIDENCE_WEIGHT}) ---")
    start_time = time.time()
    slotted_intermediate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    assigned_count = 0
    unassigned_evidence: List[Tuple[str, float]] = []
    theme_embeddings = get_theme_embeddings(sbert_model, themes_def)
    if not theme_embeddings: logger.error("Cannot slot without theme embeddings."); return {}
    norm_theme_hints: Dict[str, List[str]] = {t: [normalize_concept(h) for h in d.get('hints', []) if normalize_concept(h)] for t, d in themes_def.items()}
    wifi_uri = "urn:expediagroup:taxonomies:acs:#d86726ff-4a17-4a28-996e-5b81963254a8"
    disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
    for uri, score in tqdm(valid_evidence.items(), desc="Slotting Evidence", disable=disable_tqdm):
        best_theme = None
        max_confidence = -1.0
        details = ""
        evidence_primary_emb = primary_embeddings.get(uri)
        if evidence_primary_emb is None:
            unassigned_evidence.append((uri, score))
            continue
        uri_texts = uri_embeddings_map.get(uri, [])
        uri_norm_texts = set(norm for _, _, _, norm in uri_texts if norm)
        for theme_name, theme_data in theme_embeddings.items():
            name_emb = theme_data.get('name_embedding')
            hint_embs = theme_data.get('hint_embeddings', [])
            hints = norm_theme_hints[theme_name]
            hint_score = 0.0
            match_count = 0
            if hints and uri_norm_texts:
                found_match_for_uri_theme = False
                for norm_text in uri_norm_texts:
                    if found_match_for_uri_theme: break
                    for hint in hints:
                        if re.search(r'\b' + re.escape(hint) + r'\b', norm_text, re.IGNORECASE):
                            match_count += 1
                            found_match_for_uri_theme = True
                            break
                if match_count >= THEME_HINT_MATCH_THRESHOLD:
                    hint_score = 0.8
            semantic_score = 0.0
            name_sim = calculate_semantic_similarity(evidence_primary_emb, name_emb, uri, f"ThemeName({theme_name})") if name_emb is not None else 0.0
            hint_sim_max = 0.0
            if hint_embs:
                hint_sim_max = max([calculate_semantic_similarity(evidence_primary_emb, h_emb, uri, f"ThemeHint({theme_name})") for h_emb in hint_embs] + [0.0])
            semantic_score = max(name_sim, hint_sim_max)
            confidence = (hint_score * HINT_CONFIDENCE_WEIGHT + semantic_score * SEMANTIC_CONFIDENCE_WEIGHT)
            if uri == wifi_uri and theme_name == "Technology":
                logger.debug(f"Wifi vs Technology: HintScore={hint_score:.3f}, SemScore={semantic_score:.3f}, Confidence={confidence:.3f}")
                if confidence < THEME_ASSIGNMENT_THRESHOLD:
                    logger.debug(f"Wifi fails slotting threshold: {confidence:.3f} < {THEME_ASSIGNMENT_THRESHOLD}")
                else:
                    logger.debug(f"Wifi passes slotting threshold: {confidence:.3f}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"    URI {uri[-20:]} vs Theme '{theme_name}': HintScore={hint_score:.3f}, SemScore={semantic_score:.3f}, Confidence={confidence:.3f}")
            if confidence > max_confidence and confidence >= THEME_ASSIGNMENT_THRESHOLD:
                max_confidence = confidence
                best_theme = theme_name
                details = f"HintScore={hint_score:.3f}, SemScore={semantic_score:.3f}"
        if best_theme:
            label = get_primary_label(uri, uri_embeddings_map)
            slotted_intermediate[best_theme].append({"uri": uri, "label": label, "concept_weight": score, "confidence": max_confidence, "details": details})
            assigned_count += 1
        else:
            unassigned_evidence.append((uri, score))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Unassigned {uri[-20:]} (Score: {score:.3f}) (Max conf: {max_confidence:.3f})")
    logger.info(f"Slotting complete. Assigned {assigned_count} concepts.")
    unassigned_no_emb = sum(1 for u, s in unassigned_evidence if primary_embeddings.get(u) is None)
    unassigned_low_score = len(unassigned_evidence) - unassigned_no_emb
    if unassigned_evidence:
        logger.info(f"{len(unassigned_evidence)} unassigned ({unassigned_no_emb} missing emb, {unassigned_low_score} below threshold {THEME_ASSIGNMENT_THRESHOLD}).")
        if args.debug:
            sorted_unassigned = sorted([e for e in unassigned_evidence if primary_embeddings.get(e[0]) is not None], key=lambda x: x[1], reverse=True)
            logger.debug(f"Top {min(20, len(sorted_unassigned))} Unassigned Concepts:")
            for u, s in sorted_unassigned[:20]:
                logger.debug(f"  - {get_primary_label(u, uri_embeddings_map)} ({u}): {s:.3f}")
    final_slotted_themes = {}
    for theme_name, attributes in slotted_intermediate.items():
        sorted_attrs = sorted(attributes, key=lambda x: (x['confidence'], x['concept_weight']), reverse=True)
        final_slotted_themes[theme_name] = sorted_attrs[:MAX_ATTRIBUTES_PER_THEME]
        if logger.isEnabledFor(logging.DEBUG) and len(attributes) > MAX_ATTRIBUTES_PER_THEME:
            logger.debug(f"  Limited attributes for '{theme_name}' from {len(attributes)} to {MAX_ATTRIBUTES_PER_THEME}.")
    logger.info(f"--- Finished slotting evidence in {time.time() - start_time:.2f}s. ---")
    return final_slotted_themes

def generate_affinity_definitions(input_concepts: List[str], args: argparse.Namespace):
    logger.info("=== Starting Affinity Definition Generation Process (v27.30) ===")
    start_process_time = time.time()
    logger.info("--- Loading Shared Resources ---")
    load_start_time = time.time()
    sbert_model = get_sbert_model()
    taxonomy_concepts = load_taxonomy_concepts(args.taxonomy_dir, os.path.join(args.cache_dir, f"concepts_{CACHE_VERSION}.json"), args)
    if not taxonomy_concepts: logger.critical("Taxonomy concepts failed to load. Aborting."); return
    embedding_data = precompute_taxonomy_embeddings(taxonomy_concepts, sbert_model, os.path.join(args.cache_dir, f"embeddings_{CACHE_VERSION}.pkl"), args)
    if not embedding_data: logger.critical("Taxonomy embeddings failed to load/compute. Aborting."); return
    wifi_uri = "urn:expediagroup:taxonomies:acs:#d86726ff-4a17-4a28-996e-5b81963254a8"
    uri_embeddings_map, primary_embeddings, uris_list = embedding_data
    if wifi_uri in primary_embeddings:
        logger.info(f"Wifi URI found. Embedding norm: {np.linalg.norm(primary_embeddings[wifi_uri]):.4f}")
        label = get_primary_label(wifi_uri, uri_embeddings_map)
        logger.info(f"Wifi primary label: {label}")
    else:
        logger.error(f"Wifi URI {wifi_uri} missing from primary_embeddings.")
    wifi_related_uris = [
        "urn:expediagroup:taxonomies:acs:#d86726ff-4a17-4a28-996e-5b81963254a8",
        "urn:expediagroup:taxonomies:acs:#5bad798b-0ced-45f0-9493-1386d904bfa0",
        "urn:expediagroup:taxonomies:lcm:#2877666a-5f21-37ea-a3e7-786e8c2161c5",
        "urn:expediagroup:taxonomies:lcm:#89079c53-6244-4545-8243-0d0978f3b4d9",
        "urn:expediagroup:taxonomies:lcm:#2f723753-475a-3198-b738-f1f7254d4087",
        "urn:expediagroup:taxonomies:lcm:#ce2600c8-bae1-32da-b5c1-2f35c5f254f5",
        "urn:expediagroup:taxonomies:lcm:#179fd52d-bca7-35f5-92be-d0b66dcd9fbe",
        "urn:expediagroup:taxonomies:lcm:#7f27512a-acc9-3157-bba7-94b8b3af79f2",
        "urn:expediagroup:taxonomies:lcm:#73ebf773-a7ea-39d9-9ab2-8d37ab8a7c69",
        "urn:expediagroup:taxonomies:lcm:#1e09a326-11f0-3646-9d94-3b21932c9b6e"
    ]
    for uri in wifi_related_uris:
        if uri in primary_embeddings:
            logger.debug(f"URI {uri}: Label={get_primary_label(uri, uri_embeddings_map)}, Norm={np.linalg.norm(primary_embeddings[uri]):.4f}")
        else:
            logger.warning(f"URI {uri} missing from embeddings.")
    concept_associations = load_concept_associations(args.associations_file)
    logger.info(f"Using corpus file: {args.corpus_file}")
    corpus_data = load_corpus_data(args.corpus_file, sbert_model, os.path.join(args.cache_dir, f"corpus_{CACHE_VERSION}.pkl"), args)
    logger.info(f"--- Resources Loaded (Concepts: {len(taxonomy_concepts)}, Embeddings: {len(primary_embeddings)}, Corpus: {len(corpus_data)}) in {time.time() - load_start_time:.2f}s ---")
    logger.info("Checking/Recomputing theme embeddings...")
    get_theme_embeddings(sbert_model, BASE_THEMES)
    all_definitions = []
    total_concepts = len(input_concepts)
    logger.info(f"Processing {total_concepts} input concepts...")
    for i, concept_input_string in enumerate(input_concepts):
        logger.info(f"\n--- Processing Concept {i+1}/{total_concepts}: '{concept_input_string}' ---")
        concept_start_time = time.time()
        norm_concept_lc = normalize_concept(concept_input_string).lower().strip()
        affinity_definition = {
            "input_concept": concept_input_string, "normalized_concept": norm_concept_lc,
            "travel_category": {"uri": None, "label": None, "score": 0.0}, "themes": [], "must_not_have": [],
            "context_settings": CONTEXT_SETTINGS, "negative_overrides": [], "applicable_lodging_types": [],
            "processing_metadata": {
                "orientation": "portrait",
                "version": f"affinity-evidence-v27.30",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "Processing Failed", "duration_seconds": 0.0, "cache_version": CACHE_VERSION
            }
        }
        try:
            logger.info("Step 1: Finding anchor concepts...")
            anchor_matches = find_best_taxonomy_matches(concept_input_string, sbert_model, uri_embeddings_map, primary_embeddings, uris_list, concept_associations, corpus_data, args)
            if not anchor_matches: logger.warning(f"No anchors found for '{concept_input_string}'. Skipping."); affinity_definition["processing_metadata"]["status"] = "Failed - No Anchors"; raise StopIteration()
            top_uri, top_score, top_label, _ = anchor_matches[0]
            affinity_definition["travel_category"] = {"uri": top_uri, "label": top_label, "score": round(top_score, 4)}
            logger.info(f"Top anchor: {top_label} ({top_uri}) Score: {top_score:.4f}")
            logger.info(f"Step 2: Gathering candidate evidence (Threshold={SEMANTIC_EVIDENCE_THRESHOLD})...")
            candidate_evidence = gather_candidate_evidence_concepts(concept_input_string, anchor_matches, taxonomy_concepts, uri_embeddings_map, primary_embeddings, uris_list, sbert_model, args)
            if not candidate_evidence: logger.warning(f"No candidate evidence for '{concept_input_string}'.")
            logger.info("Step 3: Filtering contradictory evidence...")
            current_opposite_map = INPUT_TO_OPPOSITE_URI_SET_MAP
            if norm_concept_lc == "luxury":
                logger.warning(">>> DISABLING SEMANTIC CONTRADICTION check for 'luxury' in v27.30 <<<")
                current_opposite_map = {}
            valid_evidence, must_not_have_list = filter_contradictory_evidence(
                concept_input_string, candidate_evidence, primary_embeddings, uri_embeddings_map,
                current_opposite_map, INPUT_TO_MUST_NOT_HAVE_URIS, args
            )
            affinity_definition["must_not_have"] = [{"uri": item["uri"], "label": item["label"], "reason": item["reason"]} for item in must_not_have_list]
            logger.info(f"Identified {len(must_not_have_list)} must_not_have concepts.")
            logger.info("Step 4: Slotting valid evidence into themes...")
            if not valid_evidence: logger.warning("No valid evidence remaining."); slotted_themes = {}
            else:
                slotted_themes = slot_evidence_to_themes(
                    valid_evidence, uri_embeddings_map, primary_embeddings, sbert_model, BASE_THEMES, args
                )
            final_themes_list = []
            for theme_name, attributes in slotted_themes.items():
                theme_info = BASE_THEMES.get(theme_name, {})
                theme_summary = f"{len(attributes)} relevant concepts found."
                formatted_attributes = [{"uri": attr["uri"], "label": attr["label"], "concept_weight": round(attr["concept_weight"], 4)} for attr in attributes]
                final_themes_list.append({
                    "theme": theme_name, "subScore": theme_info.get("subScore", f"{theme_name}Affinity"), "weight": theme_info.get("weight", 0.0),
                    "rule": theme_info.get("rule", "Optional"), "type": theme_info.get("type", "preference"), "theme_summary": theme_summary,
                    "attributes": formatted_attributes
                })
            affinity_definition["themes"] = sorted(final_themes_list, key=lambda x: x["theme"])
            affinity_definition["processing_metadata"]["status"] = "Success"
        except StopIteration: pass
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing '{concept_input_string}': {e}", exc_info=True)
            affinity_definition["processing_metadata"]["status"] = f"Failed - Exception: {type(e).__name__}"
            if args.debug: affinity_definition["processing_metadata"]["error_details"] = traceback.format_exc()
        finally:
            affinity_definition["processing_metadata"]["duration_seconds"] = round(time.time() - concept_start_time, 2)
            all_definitions.append(affinity_definition)
            logger.info(f"--- Finished processing '{concept_input_string}' in {affinity_definition['processing_metadata']['duration_seconds']:.2f}s. Status: {affinity_definition['processing_metadata']['status']} ---")
    logger.info(f"\n=== Finished processing {total_concepts} concepts in {time.time() - start_process_time:.2f}s ===")
    output_file = os.path.join(args.output_dir, f"affinity_definitions_{CACHE_VERSION}.json")
    logger.info(f"Saving results to {output_file}...")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f: json.dump(all_definitions, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully.")
    except Exception as e: logger.error(f"Failed to save results: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Affinity Definitions (v27.30 - Fix WiFi)")
    parser.add_argument("--concepts", required=True, help="Input concepts file.")
    parser.add_argument("--taxonomy-dir", required=True, help="Taxonomy RDF directory.")
    parser.add_argument("--output-dir", default="./output", help="Output directory.")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory.")
    parser.add_argument("--associations-file", default=DEFAULT_ASSOCIATIONS_FILE, help="Associations JSON.")
    parser.add_argument("--corpus-file", default=DEFAULT_CORPUS_FILE, help="Corpus text file.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild caches.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of concepts.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers: handler.setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled.")
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        with open(args.concepts, 'r', encoding='utf-8') as f: input_concepts = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(input_concepts)} concepts from {args.concepts}")
        if args.limit and args.limit < len(input_concepts): input_concepts = input_concepts[:args.limit]; logger.info(f"Limiting to first {args.limit}.")
        if not input_concepts: logger.error("No concepts found."); sys.exit(1)
    except FileNotFoundError: logger.critical(f"Input file not found: {args.concepts}"); sys.exit(1)
    except Exception as e: logger.critical(f"Error reading input file: {e}"); sys.exit(1)
    logger.info(f"Using corpus file: {args.corpus_file}")
    generate_affinity_definitions(input_concepts, args)
    logger.info("=== Affinity Generation Script Finished ===")