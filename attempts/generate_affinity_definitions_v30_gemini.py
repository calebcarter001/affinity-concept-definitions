#!/usr/bin/env python3
"""
Generate affinity definitions for travel concepts (v32.0 - LLM Slotting Enhanced)
Version: 2025-04-21-affinity-rule-engine-v32.0 (LLM Slotting v2 - Robust)
"""

# --- Imports ---
import argparse
import json
import logging
import math
import os
import pickle
import re
import time
import traceback
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any, Set, TypedDict

import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF
from rdflib import util as rdflib_util
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- LLM Imports ---
# Define placeholder Exception classes first
class APITimeoutError(Exception): pass
class APIConnectionError(Exception): pass
class RateLimitError(Exception): pass
OpenAI = None # Default to None
OPENAI_AVAILABLE = False

# Attempt to import REAL OpenAI classes and overwrite placeholders
try:
    # This import will succeed only if 'openai' is installed
    from openai import OpenAI as RealOpenAI, APITimeoutError as RealAPITimeoutError, APIConnectionError as RealAPIConnectionError, RateLimitError as RealRateLimitError

    # If import succeeds, overwrite the placeholders
    OpenAI = RealOpenAI
    APITimeoutError = RealAPITimeoutError
    APIConnectionError = RealAPIConnectionError
    RateLimitError = RealRateLimitError
    OPENAI_AVAILABLE = True
    logging.debug("Successfully imported OpenAI library.")
except ImportError:
    logging.warning("OpenAI library not found. OpenAI provider will be unavailable.")
    # Keep the dummy Exception classes and OpenAI=None
except Exception as e:
    logging.error(f"An unexpected error occurred during OpenAI import: {e}")
    # Keep the dummy Exception classes and OpenAI=None


try: import google.generativeai as genai; GOOGLE_AI_AVAILABLE = True
except ImportError: GOOGLE_AI_AVAILABLE = False; genai = None

# --- Utility Imports ---
try: from kg_utils import get_kg_data
except ImportError: logging.error("FATAL: Import failed 'get_kg_data' from 'kg_utils.py'."); exit(1)
try: from embedding_utils import get_concept_embedding, get_batch_embedding_similarity
except ImportError: logging.error("FATAL: Import failed from 'embedding_utils.py'."); exit(1)
try: from utils import get_sbert_model # Assumes SBERT model loading is here
except ImportError: logging.error("FATAL: Import failed 'get_sbert_model' from 'utils.py'."); exit(1)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Namespaces ---
EXPE = Namespace("urn:expediagroup:taxonomies:core:"); SCHEMA = Namespace("http://schema.org/")

# --- Config Defaults ---
CACHE_VERSION = "v20250421.affinity.32.0_default"; DEFAULT_TAXONOMY_DIR = "./datasources/taxonomies/"; DEFAULT_CACHE_DIR = "./cache/"; DEFAULT_CONFIG_FILE = "./affinity_config_v32.0.json"; OUTPUT_FILE_TEMPLATE = "affinity_definitions_{cache_version}.json"; LOG_FILE_TEMPLATE = "affinity_generation_{cache_version}.log"; MAX_CANDIDATES_FOR_LLM = 75; INITIAL_CANDIDATE_POOL_SIZE = 150; HINT_BOOST_COUNT = 3; EVIDENCE_MIN_SIMILARITY = 0.30; THEME_ATTRIBUTE_MIN_WEIGHT = 0.001; TRAVEL_CONTEXT = "travel "; LLM_TIMEOUT = 180; LLM_MAX_RETRIES = 5; LLM_TEMPERATURE = 0.2; LLM_RETRY_DELAY_SECONDS = 5

# --- Globals ---
_taxonomy_concepts_cache: Optional[Dict[str, Dict]] = None; _taxonomy_embeddings_cache: Optional[Tuple[Dict, Dict, List]] = None; _config_data: Optional[Dict] = None; _openai_client: Optional['OpenAI'] = None; _google_client: Optional[Any] = None

# --- Type Hinting ---
class ThemeOverride(TypedDict, total=False): weight: float; rule: str; hints: Dict[str, List[str]]; description: str; subScore: str; fallback_logic: Optional[Dict]
class ConceptOverride(TypedDict, total=False): lodging_type: str; category_type: str; exclusionary_concepts: List[str]; must_not_have_uris: List[str]; theme_overrides: Dict[str, ThemeOverride]; additional_subscores: Dict[str, float]
class BaseTheme(TypedDict): type: str; rule: str; weight: float; subScore: Optional[str]; hints: Dict[str, List[str]]; description: Optional[str]; fallback_logic: Optional[Dict]
class AffinityConfig(TypedDict): config_version: str; description: str; base_themes: Dict[str, BaseTheme]; concept_overrides: Dict[str, ConceptOverride]; master_subscore_list: List[str]; LLM_API_CONFIG: Dict; STAGE1_CONFIG: Dict; CACHE_VERSION: str;

# --- Utility Functions ---
def normalize_concept(concept: Optional[str]) -> str:
    if not isinstance(concept, str) or not concept: return ""
    try: norm = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept); norm = norm.replace("-", " ").replace("_", " "); norm = re.sub(r'[^\w\s]|(\'s\b)', '', norm); norm = ' '.join(norm.lower().split()); return norm
    except Exception: return concept.lower().strip() if isinstance(concept, str) else ""

def get_primary_label(uri: str, fallback: Optional[str] = None) -> str:
    label = fallback; details = None
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        details = _taxonomy_concepts_cache[uri]
        if details:
            if details.get("prefLabel"): return details["prefLabel"][0]
            if details.get("altLabel"): return details["altLabel"][0]
            if details.get("rdfsLabel"): return details["rdfsLabel"][0]
            if label is None and details.get("definition"): label = details["definition"][0][:60] + "..."
    if label is None or label == fallback:
        try:
            parsed_label = uri;
            if '#' in uri: parsed_label = uri.split('#')[-1]
            elif '/' in uri: parsed_label = uri.split('/')[-1]
            if parsed_label != uri: parsed_label = parsed_label.replace('_', ' ').replace('-', ' '); label = ' '.join(word.capitalize() for word in parsed_label.split() if word)
        except Exception: pass
    return label if label is not None else uri

def get_concept_type_labels(uri: str) -> List[str]:
    type_labels = []
    if _taxonomy_concepts_cache and uri in _taxonomy_concepts_cache:
        type_uris = _taxonomy_concepts_cache[uri].get("type", [])
        for type_uri in type_uris:
            label = get_primary_label(type_uri); known_base_types = {str(SKOS.Concept), str(OWL.Class), str(RDFS.Class)}
            if label != type_uri or type_uri in known_base_types: type_labels.append(label)
    return sorted(list(set(type_labels)))

def get_theme_definition_for_prompt(theme_name: str, theme_data: BaseTheme) -> str:
    if theme_data.get("description"): return theme_data["description"]
    hints = theme_data.get("hints", {}); keywords = hints.get("keywords", [])
    if keywords: hint_summary = ", ".join(keywords[:10]);
    if len(keywords) > 10: hint_summary += "..."; return f"Represents '{theme_name}'. Focuses on concepts like: {hint_summary}"
    else: return f"Theme related to {theme_name}."

def get_dynamic_theme_config(normalized_concept: str, theme_name: str, config: AffinityConfig) -> Tuple[str, float, Optional[str], Optional[Dict]]:
    base_themes = config.get("base_themes", {}); concept_overrides_all = config.get("concept_overrides", {})
    base_theme_data = base_themes.get(theme_name);
    if not base_theme_data: logger.error(f"Base theme '{theme_name}' not found!"); return "Optional", 0.0, None, None
    overrides = concept_overrides_all.get(normalized_concept, {}); theme_overrides = overrides.get("theme_overrides", {}).get(theme_name, {})
    merged_data = {**base_theme_data, **theme_overrides}; final_rule = merged_data.get("rule", "Optional"); final_weight = merged_data.get("weight", 0.0); final_subscore = merged_data.get("subScore"); final_fallback = merged_data.get("fallback_logic")
    if final_rule not in ["Must have 1", "Optional", "Must Not Have"]: logger.warning(f"Invalid rule '{final_rule}'. Defaulting 'Optional'."); final_rule = "Optional"
    if not isinstance(final_weight, (int, float)) or final_weight < 0: logger.warning(f"Invalid weight '{final_weight}'. Defaulting 0.0."); final_weight = 0.0
    return final_rule, float(final_weight), final_subscore, final_fallback

def normalize_weights(weights_dict: Dict[str, float]) -> Dict[str, float]:
    total_weight = sum(weights_dict.values());
    if total_weight <= 0: return {k: 0.0 for k in weights_dict}
    return {k: v / total_weight for k, v in weights_dict.items()}

def deduplicate_attributes(attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not attributes: return []
    seen_labels: Dict[str, Dict[str, Any]] = {};
    for attr in attributes:
        label = attr.get("skos:prefLabel"); weight = attr.get("concept_weight", 0.0); uri = attr.get("uri")
        if not label or not uri: continue
        if label not in seen_labels or weight > seen_labels[label].get("concept_weight", 0.0): seen_labels[label] = attr
    return list(seen_labels.values())

# --- Loading Functions ---
def load_affinity_config(config_file: str) -> Optional[AffinityConfig]:
    global _config_data, LLM_MAX_RETRIES, LLM_RETRY_DELAY_SECONDS, INITIAL_CANDIDATE_POOL_SIZE, HINT_BOOST_COUNT, CACHE_VERSION, MAX_CANDIDATES_FOR_LLM
    if _config_data is not None: return _config_data
    logger.info(f"Loading config: {config_file}")
    if not os.path.exists(config_file): logger.critical(f"Config file not found: {config_file}"); return None
    try:
        with open(config_file, 'r', encoding='utf-8') as f: config = json.load(f)
        # Validation & Parameter Update
        if "base_themes" not in config: raise ValueError("Config missing 'base_themes'")
        for theme, data in config["base_themes"].items():
            if not data.get("description"): logger.error(f"CRITICAL: Theme '{theme}' MUST have a 'description'.")
        LLM_MAX_RETRIES = config.get('LLM_API_CONFIG', {}).get('MAX_RETRIES', LLM_MAX_RETRIES)
        LLM_RETRY_DELAY_SECONDS = config.get('LLM_API_CONFIG', {}).get('RETRY_DELAY_SECONDS', LLM_RETRY_DELAY_SECONDS)
        stage1_cfg = config.get('STAGE1_CONFIG', {})
        INITIAL_CANDIDATE_POOL_SIZE = stage1_cfg.get('INITIAL_CANDIDATE_POOL_SIZE', INITIAL_CANDIDATE_POOL_SIZE)
        MAX_CANDIDATES_FOR_LLM = stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', MAX_CANDIDATES_FOR_LLM)
        HINT_BOOST_COUNT = stage1_cfg.get('HINT_BOOST_COUNT', HINT_BOOST_COUNT)
        CACHE_VERSION = config.get('CACHE_VERSION', CACHE_VERSION)
        config["concept_overrides"] = {normalize_concept(k): v for k, v in config.get("concept_overrides", {}).items()}
        _config_data = config
        logger.info(f"Config loaded. Version: {config.get('config_version', 'N/A')}")
        logger.info(f"Runtime Params: Retries={LLM_MAX_RETRIES}, Pool={INITIAL_CANDIDATE_POOL_SIZE}, LLM_Cand={MAX_CANDIDATES_FOR_LLM}, Boost={HINT_BOOST_COUNT}, Cache={CACHE_VERSION}")
        return _config_data
    except Exception as e: logger.critical(f"FATAL: Error loading/validating config {config_file}: {e}", exc_info=True); return None

# (load_taxonomy_concepts and precompute_taxonomy_embeddings remain the same as provided in the previous corrected response)
def load_taxonomy_concepts(taxonomy_dir: str, cache_file: str, args: argparse.Namespace, current_cache_version: str) -> Optional[Dict[str, Dict]]:
    global _taxonomy_concepts_cache
    if _taxonomy_concepts_cache is not None: return _taxonomy_concepts_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Attempting to load concepts from cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f: cached_data = json.load(f)
            if cached_data.get("cache_version") == current_cache_version and isinstance(cached_data.get("data"), dict):
                _taxonomy_concepts_cache = cached_data["data"]; cache_valid = True; logger.info(f"Loaded {len(_taxonomy_concepts_cache)} concepts from cache (Version: {current_cache_version}).")
            else: logger.info(f"Concept cache version/data invalid. Rebuilding.")
        except Exception as e: logger.warning(f"Concept cache load failed: {e}. Rebuilding.")
    if not cache_valid:
        logger.info(f"Loading concepts from RDF files in: {taxonomy_dir}")
        start_time = time.time(); g = Graph(); files_ok = 0; total_err = 0
        try:
            if not os.path.isdir(taxonomy_dir): raise FileNotFoundError(f"Taxonomy directory not found: {taxonomy_dir}")
            rdf_files = [f for f in os.listdir(taxonomy_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.xml', '.jsonld', '.nt', '.n3'))]
            if not rdf_files: raise FileNotFoundError(f"No RDF files found in directory: {taxonomy_dir}")
            logger.info(f"Found {len(rdf_files)} files. Parsing...")
            disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
            for fn in tqdm(rdf_files, desc="Parsing RDF", disable=disable_tqdm):
                fp = os.path.join(taxonomy_dir, fn)
                try: fmt = rdflib_util.guess_format(fp); g.parse(fp, format=fmt); files_ok += 1
                except Exception as e_parse: total_err += 1; logger.error(f"Error parsing file {fn}: {e_parse}", exc_info=args.debug)
            logger.info(f"Parsed {files_ok}/{len(rdf_files)} files ({total_err} errors).")
            if files_ok == 0: raise RuntimeError("No RDF files parsed.")
            logger.info("Extracting concepts...")
            kept_concepts_data = defaultdict(lambda: defaultdict(list)); all_uris = set(s for s in g.subjects() if isinstance(s, URIRef))
            label_props = {SKOS.prefLabel: "prefLabel", SKOS.altLabel: "altLabel", RDFS.label: "rdfsLabel"}; text_props = {SKOS.definition: "definition", SKOS.scopeNote: "scopeNote"}; rel_props = {SKOS.broader: "broader"}; type_prop = {RDF.type: "type"}
            all_props = {**label_props, **text_props, **rel_props, **type_prop}
            for uri_ref in tqdm(all_uris, desc="Processing URIs", disable=disable_tqdm):
                 if g.value(uri_ref, OWL.deprecated) == Literal(True): continue
                 uri_str = str(uri_ref); current_data = defaultdict(list); has_props = False
                 for prop_uri, prop_key in all_props.items():
                      for obj in g.objects(uri_ref, prop_uri):
                          value = str(obj).strip() if isinstance(obj, Literal) else str(obj)
                          if value: current_data[prop_key].append(value); has_props = True
                 if has_props:
                      processed = {k: sorted(list(set(v))) for k, v in current_data.items() if v}
                      if processed: kept_concepts_data[uri_str] = processed
            _taxonomy_concepts_cache = dict(kept_concepts_data); logger.info(f"Extracted {len(_taxonomy_concepts_cache)} concepts.")
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_data_to_save = {"cache_version": current_cache_version, "data": _taxonomy_concepts_cache}
                with open(cache_file, 'w', encoding='utf-8') as f_cache: json.dump(cache_data_to_save, f_cache, indent=2)
                logger.info(f"Saved concepts cache to: {cache_file}")
            except Exception as e_cache: logger.error(f"Failed writing concepts cache: {e_cache}")
            logger.info(f"Taxonomy loading took {time.time() - start_time:.2f}s.")
        except Exception as e_load: logger.error(f"Taxonomy load error: {e_load}", exc_info=args.debug); return None
    if not _taxonomy_concepts_cache: logger.error("Failed to load concepts."); return None
    return _taxonomy_concepts_cache

def precompute_taxonomy_embeddings(taxonomy_concepts: Dict[str, Dict], sbert_model: SentenceTransformer, cache_file: str, args: argparse.Namespace, current_cache_version: str) -> Optional[Tuple[Dict, Dict, List]]:
    global _taxonomy_embeddings_cache
    if _taxonomy_embeddings_cache is not None: return _taxonomy_embeddings_cache
    cache_valid = False
    if not args.rebuild_cache and os.path.exists(cache_file):
        logger.info(f"Attempting to load embeddings from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f: cached_data = pickle.load(f)
            if cached_data.get("cache_version") == current_cache_version:
                p_embs = cached_data.get("primary_embeddings"); u_list = cached_data.get("uris_list")
                if isinstance(p_embs, dict) and isinstance(u_list, list):
                    _taxonomy_embeddings_cache = ({}, p_embs, u_list)
                    cache_valid = True; logger.info(f"Loaded {len(u_list)} primary embeddings from cache (Version: {current_cache_version}).")
                else: logger.warning("Embeddings cache invalid. Rebuilding.")
            else: logger.info(f"Embeddings cache version mismatch. Rebuilding.")
        except Exception as e: logger.warning(f"Embeddings cache load failed: {e}. Rebuilding.")
    if not cache_valid:
        logger.info("Pre-computing taxonomy embeddings...")
        start_time = time.time()
        primary_embeddings: Dict[str, Optional[np.ndarray]] = {}; uris_list_all: List[str] = []
        texts_to_embed_map = defaultdict(list); all_valid_uris = list(taxonomy_concepts.keys())
        text_properties = ["prefLabel", "altLabel", "rdfsLabel", "definition"]; disable_tqdm = not logger.isEnabledFor(logging.INFO) or args.debug
        logger.info("Step 1: Gathering text properties...")
        for uri in tqdm(all_valid_uris, desc="Gathering Texts", disable=disable_tqdm):
            uris_list_all.append(uri); concept_data = taxonomy_concepts.get(uri, {}); found_texts_for_uri: Set[Tuple[str, str, str]] = set()
            for prop_key in text_properties:
                for text_value in concept_data.get(prop_key, []):
                    if text_value and isinstance(text_value, str):
                        original_text = text_value.strip(); normalized_text = normalize_concept(original_text)
                        if normalized_text: found_texts_for_uri.add((prop_key, original_text, normalized_text))
            for prop_key, original_text, normalized_text in found_texts_for_uri: texts_to_embed_map[normalized_text].append((uri, prop_key, original_text))
        unique_normalized_texts = list(texts_to_embed_map.keys()); embedding_map: Dict[str, Optional[np.ndarray]] = {}
        if unique_normalized_texts:
            logger.info(f"Step 2: Generating embeddings for {len(unique_normalized_texts)} texts...")
            batch_size = 128
            try:
                embeddings_list = sbert_model.encode(unique_normalized_texts, batch_size=batch_size, show_progress_bar=(not disable_tqdm))
                embedding_map = {text: emb for text, emb in zip(unique_normalized_texts, embeddings_list) if emb is not None}
            except Exception as e: logger.error(f"SBERT encoding failed: {e}", exc_info=True); raise RuntimeError("SBERT Encoding Failed") from e
        else: logger.warning("No texts found to embed.")
        logger.info("Step 3: Selecting primary embedding...")
        primary_embedding_candidates = defaultdict(dict); sbert_dim = sbert_model.get_sentence_embedding_dimension()
        for normalized_text, associated_infos in texts_to_embed_map.items():
            embedding = embedding_map.get(normalized_text)
            if embedding is None or not isinstance(embedding, np.ndarray) or embedding.shape != (sbert_dim,): continue
            for uri, prop_key, _ in associated_infos: primary_embedding_candidates[uri][prop_key] = embedding
        primary_property_priority = ["prefLabel", "altLabel", "rdfsLabel", "definition"]
        num_with_primary = 0
        for uri in tqdm(uris_list_all, desc="Selecting Primary", disable=disable_tqdm):
            candidates = primary_embedding_candidates.get(uri, {}); chosen_embedding = None
            for prop in primary_property_priority:
                if prop in candidates and candidates[prop] is not None: chosen_embedding = candidates[prop]; break
            if chosen_embedding is not None and isinstance(chosen_embedding, np.ndarray) and chosen_embedding.ndim == 1: primary_embeddings[uri] = chosen_embedding; num_with_primary += 1
            else: primary_embeddings[uri] = None
        final_uris_list = [uri for uri in uris_list_all if primary_embeddings.get(uri) is not None]
        final_primary_embeddings = {uri: emb for uri, emb in primary_embeddings.items() if emb is not None}
        if not final_uris_list: logger.warning("No concepts have primary embeddings."); _taxonomy_embeddings_cache = ({}, {}, [])
        else: _taxonomy_embeddings_cache = ({}, final_primary_embeddings, final_uris_list)
        logger.info(f"Finished embedding. {num_with_primary}/{len(uris_list_all)} concepts have primary embedding.")
        logger.info(f"Embedding took {time.time() - start_time:.2f}s.")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data_to_save = {"cache_version": current_cache_version, "primary_embeddings": _taxonomy_embeddings_cache[1], "uris_list": _taxonomy_embeddings_cache[2]}
            with open(cache_file, 'wb') as f_cache: pickle.dump(cache_data_to_save, f_cache, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved embeddings cache to: {cache_file}")
        except Exception as e_cache: logger.error(f"Failed writing embeddings cache: {e_cache}")
    if not _taxonomy_embeddings_cache or not _taxonomy_embeddings_cache[1]: logger.error("Embedding process failed."); return None
    return _taxonomy_embeddings_cache


# --- LLM Interaction Functions ---
# (Include get_openai_client, get_google_client, call_llm, construct_llm_slotting_prompt, build_reprompt_prompt from previous corrected response)
def get_openai_client() -> Optional[OpenAI]:
    global _openai_client
    if _openai_client is None and OPENAI_AVAILABLE:
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key: _openai_client = OpenAI(api_key=api_key); logger.info("OpenAI client initialized.")
            else: logger.warning("OPENAI_API_KEY missing."); return None
        except Exception as e: logger.error(f"OpenAI init failed: {e}"); return None
    return _openai_client if OPENAI_AVAILABLE else None

def get_google_client():
    global _google_client
    if _google_client is None and GOOGLE_AI_AVAILABLE:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key: genai.configure(api_key=api_key); logger.info("Google AI client configured."); _google_client = genai
            else: logger.warning("GOOGLE_API_KEY missing."); return None
        except Exception as e: logger.error(f"Google AI config failed: {e}"); return None
    return _google_client if GOOGLE_AI_AVAILABLE else None

def call_llm(system_prompt: str, user_prompt: str, model_name: str, timeout: int, max_retries: int, temperature: float, provider: str) -> Optional[Dict[str, Any]]:
    logger.info(f"LLM call via {provider}, model: {model_name}")
    actual_max_retries = max_retries # Use value from config
    if provider == "openai":
        client = get_openai_client();
        if not client: return None
        for attempt in range(actual_max_retries + 1):
            logger.info(f"OpenAI Call - Attempt {attempt + 1}/{actual_max_retries + 1}")
            try:
                response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"}, temperature=temperature, timeout=timeout)
                content = response.choices[0].message.content
                if content:
                    try: cleaned = content.strip(); llm_output = json.loads(cleaned.strip('` \n').replace("```json","").replace("```","")); return llm_output
                    except json.JSONDecodeError as json_e: logger.error(f"OpenAI JSON parse error (Attempt {attempt+1}): {json_e}\nRaw: {content[:500]}...");
                else: logger.warning(f"OpenAI response empty (Attempt {attempt+1}).")
            except (APITimeoutError, APIConnectionError, RateLimitError) as api_e: logger.warning(f"OpenAI API Error (Attempt {attempt+1}): {type(api_e).__name__}")
            except Exception as e: logger.error(f"OpenAI Call Error (Attempt {attempt+1}): {e}", exc_info=True); return None
            if attempt >= actual_max_retries: logger.error("Max retries reached for OpenAI."); return None
            wait_time = LLM_RETRY_DELAY_SECONDS * (2**attempt) + np.random.uniform(0, 1); logger.info(f"Retrying in {wait_time:.2f}s..."); time.sleep(wait_time)
        return None
    elif provider == "google":
        g_client_module = get_google_client();
        if not g_client_module: return None
        model_name_full = model_name if model_name.startswith("models/") else f"models/{model_name}"
        generation_config = g_client_module.types.GenerationConfig(candidate_count=1, temperature=temperature, response_mime_type="application/json")
        safety_settings=[{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        try:
            model = g_client_module.GenerativeModel(model_name_full, system_instruction=system_prompt, safety_settings=safety_settings)
            for attempt in range(actual_max_retries + 1):
                logger.info(f"Gemini Call - Attempt {attempt + 1}/{actual_max_retries + 1}")
                try:
                    response = model.generate_content([user_prompt], generation_config=generation_config, request_options={'timeout': timeout})
                    if not response.candidates: logger.warning(f"Gemini blocked/empty (Attempt {attempt+1}). Reason: {getattr(response.prompt_feedback, 'block_reason', 'N/A')}."); return None
                    content = response.text
                    if content:
                        try: cleaned = content.strip(); llm_output = json.loads(cleaned.strip('` \n').replace("```json","").replace("```","")); return llm_output
                        except json.JSONDecodeError as json_e: logger.error(f"Gemini JSON parse error (Attempt {attempt+1}): {json_e}\nRaw: {content[:500]}...")
                    else: logger.warning(f"Gemini response content empty (Attempt {attempt+1}).")
                except Exception as api_e: logger.warning(f"Google API Error (Attempt {attempt+1}): {type(api_e).__name__}")
                if attempt >= actual_max_retries: logger.error("Max retries reached for Google."); return None
                wait_time = LLM_RETRY_DELAY_SECONDS * (2**attempt) + np.random.uniform(0, 1); logger.info(f"Retrying in {wait_time:.2f}s..."); time.sleep(wait_time)
        except Exception as e: logger.error(f"Google Call Setup Error: {e}", exc_info=True); return None
        return None
    else: logger.error(f"Unsupported LLM provider: {provider}"); return None

def construct_llm_slotting_prompt(input_concept: str, theme_definitions: List[Dict[str, Any]], candidate_details: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[str, str]:
    system_prompt = """You are an expert travel taxonomist assisting in defining travel concepts. Your task is to assign relevant KG concepts (evidence) to predefined themes based on their semantic meaning. These definitions are crucial for travelers and online travel agencies. Analyze the input travel concept, theme definitions, and candidate details. Assign each candidate URI to one or more themes it strongly relates to. Output ONLY a valid JSON object matching the specified schema."""
    theme_defs_str = "\n".join([f"- **{t['name']}{' (Must Have)' if t.get('is_must_have') else ''}**: {t['description']}" for t in theme_definitions])
    theme_names_list = [t['name'] for t in theme_definitions]; must_haves = [t['name'] for t in theme_definitions if t.get('is_must_have')]; must_haves_str = ", ".join(must_haves) if must_haves else "None"
    candidate_details_str = "\n".join([f"\n{i+1}. URI: {c['uri']}\n   PrefLabel: {c.get('prefLabel', 'N/A')}\n   AltLabels: {', '.join(c.get('altLabel', []))}\n   Definition: {(c.get('definition', '') or '')[:200]}\n   Types: {', '.join(c.get('type_labels', []))}" for i, c in enumerate(candidate_details)]) or "No candidates."
    output_schema = """```json\n{\n  "theme_assignments": {\n    "URI_1": ["Theme_A", "Theme_C"],\n    "URI_2": ["Theme_B"]\n    // ... include only relevant candidate URIs as keys\n  }\n}\n```"""
    user_prompt = f"""Analyze concept: '{input_concept}'\n\nAvailable Themes:\n{theme_defs_str}\n\nCandidate Evidence:\n{candidate_details_str}\n\nTask:\nAssign each candidate URI to relevant theme(s). Pay close attention to 'Must Have' themes: [{must_haves_str}]. Explain reasoning step-by-step, then output ONLY the JSON according to the schema.\n\nThinking Step-by-step:\n[LLM reasoning]\n\nFinal Answer:\n{output_schema}"""
    return system_prompt.strip(), user_prompt.strip()

def build_reprompt_prompt(input_concept: str, theme_name: str, theme_config: BaseTheme, original_candidates_details_map: Dict[str, Dict]) -> str:
    prompt_template = """Re-evaluating concept '{concept}' for mandatory theme '{theme_name}'. Previous attempt failed to assign candidates.\n\nTheme Description:\n- **{theme_name}:** {theme_description}\n{theme_hints_formatted}\n\nCandidate List:\n{candidates_formatted}\n\nInstructions:\nReview the theme and candidates. Identify ALL relevant candidates for '{theme_name}'. Assign AT LEAST ONE if possible. Explain reasoning. Output ONLY JSON: {{"theme_assignments": {{"URI_X": ["{theme_name}"], ...}} }}. If none, return {{"theme_assignments": {{}} }}.\n\nThinking Step-by-step:\n[LLM reasoning]\n\nFinal Answer:\n```json\n{{ ... }}\n```"""
    hints = theme_config.get("hints", {}); hints_str = f"  Hints: {json.dumps(hints)}" if isinstance(hints, dict) and (hints.get("keywords") or hints.get("uris")) else ""
    cand_list = "\n".join([f"- URI: {uri}\n  Label: {cand.get('prefLabel', 'N/A')}\n  Types: {cand.get('type_labels', [])}" for uri, cand in original_candidates_details_map.items()])
    return prompt_template.format(concept=input_concept, theme_name=theme_name, theme_description=theme_config.get('description', 'N/A'), theme_hints_formatted=hints_str, candidates_formatted=cand_list)


# --- Stage 1: Evidence Prep (Corrected Return Signature) ---
def prepare_evidence(input_concept: str, concept_embedding: np.ndarray,
                     primary_embeddings: Dict[str, np.ndarray], uris_list: List[str],
                     config: AffinityConfig, args: argparse.Namespace
                     ) -> Tuple[List[Dict], Dict[str, Dict], Optional[Dict], Dict[str, float]]: # Added scores dict return
    logger.info(f"Starting evidence preparation for: {input_concept}")
    stage1_cfg = config.get('STAGE1_CONFIG', {})
    initial_pool_size = stage1_cfg.get('INITIAL_CANDIDATE_POOL_SIZE', 150)
    max_candidates_for_llm = stage1_cfg.get('MAX_CANDIDATES_FOR_LLM', 75)
    hint_boost_count = stage1_cfg.get('HINT_BOOST_COUNT', 3)
    min_similarity = EVIDENCE_MIN_SIMILARITY

    # 1. Get initial pool scores
    candidate_evidence_scores = get_batch_embedding_similarity(concept_embedding, primary_embeddings)
    if not candidate_evidence_scores: logger.warning(f"No similarity scores calculated for {input_concept}"); return [], {}, None, {}

    filtered_candidates = {uri: score for uri, score in candidate_evidence_scores.items() if score >= min_similarity}
    sorted_uris = sorted(filtered_candidates, key=filtered_candidates.get, reverse=True)
    initial_pool_uris = sorted_uris[:initial_pool_size]

    if not initial_pool_uris: logger.warning(f"No candidates >= {min_similarity} for '{input_concept}'."); return [], {}, None, {}

    # 2. Get details and build initial list
    initial_pool_details_map = get_kg_data(initial_pool_uris)
    initial_candidates_with_details = []
    for uri in initial_pool_uris:
        if uri in initial_pool_details_map:
             details = initial_pool_details_map[uri]
             details['similarity_score'] = filtered_candidates.get(uri, 0.0) # Add score
             # Add prefLabel from details if missing from KG data somehow (fallback)
             if 'prefLabel' not in details and 'skos:prefLabel' not in details:
                  details['prefLabel'] = [get_primary_label(uri)]
             initial_candidates_with_details.append(details)
    if not initial_candidates_with_details: logger.warning(f"No candidate details retrieved for {input_concept}."); return [], {}, None, candidate_evidence_scores
    initial_candidates_with_details.sort(key=lambda x: x['similarity_score'], reverse=True)
    anchor_candidate = initial_candidates_with_details[0]

    # 3. Hint Boosting
    must_have_themes_with_hints = {}; normalized_concept_local = normalize_concept(input_concept)
    concept_cfg_stage1 = config.get("concept_overrides", {}).get(normalized_concept_local, {})
    base_themes_cfg = config.get("base_themes", {})
    active_theme_configs_stage1 = {}
    for theme_name, base_data in base_themes_cfg.items():
        merged_data = {**base_data, **concept_cfg_stage1.get('themes', {}).get(theme_name, {})}
        active_theme_configs_stage1[theme_name] = merged_data
        if merged_data.get("rule_applied") == "Must have 1": must_have_themes_with_hints[theme_name] = merged_data.get("hints", {})

    boosted_candidates_uris = set()
    if must_have_themes_with_hints:
        logger.info(f"Boosting for Must Have themes: {list(must_have_themes_with_hints.keys())}")
        for theme_name, hints in must_have_themes_with_hints.items():
            if not isinstance(hints, dict): continue
            hint_uris = set(hints.get("uris", [])); hint_keywords = [kw.lower() for kw in hints.get("keywords", []) if kw]
            matches_for_theme = []
            for candidate in initial_candidates_with_details:
                uri = candidate['uri']
                # Combine all potential labels and definition for matching
                labels = candidate.get("prefLabel", []) + candidate.get("altLabel", []) + candidate.get("rdfsLabel", []) + candidate.get("skos:prefLabel", []) + candidate.get("skos:altLabel", [])
                definition = candidate.get("definition", []) + candidate.get("skos:definition", [])
                text_content = " ".join(labels).lower() + " " + " ".join(definition).lower()
                is_match = False
                if uri in hint_uris: is_match = True
                else:
                    for kw in hint_keywords:
                        if re.search(r'\b' + re.escape(kw) + r'\b', text_content): is_match = True; break
                if is_match: matches_for_theme.append(candidate)
            matches_for_theme.sort(key=lambda x: x['similarity_score'], reverse=True)
            boosted_uris_for_theme = {match['uri'] for match in matches_for_theme[:hint_boost_count]}
            if boosted_uris_for_theme: logger.info(f"   Boosting {len(boosted_uris_for_theme)} for '{theme_name}'."); boosted_candidates_uris.update(boosted_uris_for_theme)

    # 4. Select final list for LLM
    final_candidates_for_llm_details = []
    selected_uris_for_llm = set()
    final_candidates_for_llm_details.append(anchor_candidate); selected_uris_for_llm.add(anchor_candidate['uri'])
    boosted_candidates_details = sorted([c for c in initial_candidates_with_details if c['uri'] in boosted_candidates_uris], key=lambda x: x['similarity_score'], reverse=True)
    for candidate in boosted_candidates_details:
         if candidate['uri'] not in selected_uris_for_llm and len(final_candidates_for_llm_details) < max_candidates_for_llm:
              final_candidates_for_llm_details.append(candidate); selected_uris_for_llm.add(candidate['uri'])
    remaining_candidates = [c for c in initial_candidates_with_details if c['uri'] not in selected_uris_for_llm]
    fill_count = max_candidates_for_llm - len(final_candidates_for_llm_details)
    if fill_count > 0: final_candidates_for_llm_details.extend(remaining_candidates[:fill_count])
    logger.info(f"Selected {len(final_candidates_for_llm_details)} candidates for LLM.")

    # Prepare map for re-prompting (contains only those sent to LLM)
    original_candidates_map_for_reprompt = {c['uri']: c for c in final_candidates_for_llm_details}

    return final_candidates_for_llm_details, original_candidates_map_for_reprompt, anchor_candidate, candidate_evidence_scores


# --- Stage 2: Rule Application & Finalization (Corrected) ---
# (Paste the `apply_rules_and_finalize` function from the previous response here)
# **Ensure it uses the passed `candidate_evidence_scores` for weighting**
def apply_rules_and_finalize(input_concept: str, llm_assignments: Dict[str, Any], config: AffinityConfig, travel_category: Optional[Dict], top_metadata_candidate: Optional[Dict], original_candidates_map_for_reprompt: Dict[str, Dict], candidate_evidence_scores: Dict[str, float], args: argparse.Namespace) -> Dict[str, Any]:
    """Applies rules, weights, performs re-prompting fallback, and finalizes."""
    logger.info(f"Starting rule application and finalization for: {input_concept}")
    start_time = time.time()
    normalized_concept_local = normalize_concept(input_concept)
    final_definition = {
        "input_concept": input_concept, "normalized_concept": normalized_concept_local,
        "applicable_lodging_types": "Both", "travel_category": travel_category or {},
        "core_definition": "", "top_defining_attributes": [], "themes": [],
        "additional_relevant_subscores": [], "must_not_have": [], "failed_fallback_themes": {},
        "diagnostics": {"theme_processing": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}}
    }
    base_themes = config.get("base_themes", {}); concept_overrides = config.get("concept_overrides", {}).get(normalized_concept_local, {})
    active_theme_configs = {name: {**base_data, **concept_overrides.get('themes', {}).get(name, {})} for name, base_data in base_themes.items()}
    theme_weights = {name: data.get('weight', 1) for name, data in active_theme_configs.items()}; normalized_theme_weights = normalize_weights(theme_weights)
    all_assigned_attributes = []; theme_to_uris_map = defaultdict(list); parsed_assignments = llm_assignments.get("theme_assignments", {})
    if isinstance(parsed_assignments, dict):
        for uri, themes_list in parsed_assignments.items():
            if uri in original_candidates_map_for_reprompt:
                for theme_name in themes_list:
                    if theme_name in active_theme_configs: theme_to_uris_map[theme_name].append(uri)
    processed_themes = []; failed_initial_must_haves = {}
    for theme_name, theme_data in active_theme_configs.items():
        diag_theme = final_definition["diagnostics"]["theme_processing"][theme_name] = {"llm_assigned_count": 0, "attributes_after_weighting": 0, "status": "Pending", "rule_failed": False}
        final_rule, _, final_subscore, _ = get_dynamic_theme_config(normalized_concept_local, theme_name, config)
        final_weight_for_theme = normalized_theme_weights.get(theme_name, 0.0)
        if final_weight_for_theme <= 0: diag_theme["status"] = "Skipped (zero weight)"; continue
        assigned_uris = theme_to_uris_map.get(theme_name, []); diag_theme["llm_assigned_count"] = len(assigned_uris)
        theme_output = {"theme_name": theme_name, "theme_type": theme_data.get("type", "unknown"), "rule_applied": final_rule, "normalized_theme_weight": round(final_weight_for_theme, 6), "subScore": final_subscore, "llm_summary": None, "attributes": []}
        if final_rule == "Must have 1" and not assigned_uris:
            logger.warning(f"Theme '{theme_name}' failed initial 'Must have 1' rule for '{input_concept}'.")
            failed_initial_must_haves[theme_name] = {"reason": "Failed 'Must have 1' rule (No LLM assignment)"}; diag_theme["status"] = "Failed Rule (Initial)"; diag_theme["rule_failed"] = True
            processed_themes.append(theme_output); continue
        theme_attributes_data = []; theme_total_initial_score = 0.0
        for uri in assigned_uris:
            if uri in original_candidates_map_for_reprompt:
                # *** FIX: Use the PASSED IN candidate_evidence_scores ***
                score = candidate_evidence_scores.get(uri, 0.01) # Get score from the full initial map
                theme_attributes_data.append({"uri": uri, "details": original_candidates_map_for_reprompt[uri], "initial_score": score})
                theme_total_initial_score += score
        for attr_data in theme_attributes_data:
            proportion = (attr_data['initial_score'] / theme_total_initial_score) if theme_total_initial_score > 0 else (1 / len(theme_attributes_data) if theme_attributes_data else 0)
            attr_weight = final_weight_for_theme * proportion
            if attr_weight >= THEME_ATTRIBUTE_MIN_WEIGHT:
                # Ensure using consistent label key ('prefLabel' used in details map)
                pref_label = attr_data['details'].get('prefLabel', [get_primary_label(attr_data['uri'])])[0] if isinstance(attr_data['details'].get('prefLabel'), list) else attr_data['details'].get('prefLabel', get_primary_label(attr_data['uri']))
                final_attr = {"uri": attr_data['uri'], "skos:prefLabel": pref_label, "concept_weight": round(attr_weight, 6), "type": attr_data['details'].get('type_labels', [])}
                theme_output["attributes"].append(final_attr)
                all_assigned_attributes.append(final_attr)
        diag_theme["attributes_after_weighting"] = len(theme_output["attributes"]); theme_output["attributes"].sort(key=lambda x: x['concept_weight'], reverse=True)
        processed_themes.append(theme_output); diag_theme["status"] = "Processed (Initial)"
    final_definition["themes"] = processed_themes

    # --- Re-prompting Fallback ---
    reprompt_diag = final_definition["diagnostics"]["reprompting_fallback"]
    if failed_initial_must_haves and original_candidates_map_for_reprompt:
        logger.info(f"Attempting re-prompting for {len(failed_initial_must_haves)} themes: {list(failed_initial_must_haves.keys())}")
        for theme_name in list(failed_initial_must_haves.keys()):
            reprompt_diag["attempts"] += 1; theme_config_fallback = active_theme_configs.get(theme_name)
            if not theme_config_fallback: logger.error(f"Config missing for '{theme_name}'."); reprompt_diag["failures"] += 1; continue
            logger.info(f"Building re-prompt for: {theme_name}")
            reprompt = build_reprompt_prompt(input_concept, theme_name, theme_config_fallback, original_candidates_map_for_reprompt)
            llm_reprompt_result = call_llm(reprompt, reprompt, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)
            if llm_reprompt_result and llm_reprompt_result["success"]:
                fallback_assignments = llm_reprompt_result["response"].get("theme_assignments", {})
                newly_assigned_uris = set();
                if isinstance(fallback_assignments, dict):
                    for uri, assigned_themes_list in fallback_assignments.items():
                        if theme_name in assigned_themes_list and uri in original_candidates_map_for_reprompt: newly_assigned_uris.add(uri)
                if newly_assigned_uris:
                    logger.info(f"Re-prompt OK for '{theme_name}': {len(newly_assigned_uris)} assignments.")
                    reprompt_diag["successes"] += 1; theme_index = next((i for i, t in enumerate(final_definition["themes"]) if t["theme_name"] == theme_name), -1)
                    if theme_index != -1:
                        current_theme_output = final_definition["themes"][theme_index]; current_uris = {a['uri'] for a in current_theme_output["attributes"]}; added_count = 0
                        for uri in newly_assigned_uris:
                             if uri not in current_uris:
                                 details = original_candidates_map_for_reprompt[uri]
                                 pref_label_fb = details.get('prefLabel', get_primary_label(uri)) # Get label
                                 final_attr = {"uri": uri, "skos:prefLabel": pref_label_fb, "concept_weight": 0.0001, "type": details.get('type_labels', []), "comment": "Added via fallback"}
                                 current_theme_output["attributes"].append(final_attr); all_assigned_attributes.append(final_attr); added_count += 1
                        if added_count > 0: logger.info(f"Added {added_count} fallback attributes to '{theme_name}'."); current_theme_output["attributes"].sort(key=lambda x: x['concept_weight'], reverse=True)
                        if current_theme_output["attributes"]: # Rule now passes
                             final_definition["diagnostics"]["theme_processing"][theme_name]["status"] = "Processed (Fallback)"; final_definition["diagnostics"]["theme_processing"][theme_name]["rule_failed"] = False
                             failed_initial_must_haves.pop(theme_name, None)
                        else: final_definition["diagnostics"]["theme_processing"][theme_name]["status"] = "Failed Rule (Fallback Failed)"; reprompt_diag["failures"] += 1
                    else: logger.error(f"Theme '{theme_name}' not found for fallback merge."); reprompt_diag["failures"] += 1
                else: logger.warning(f"Re-prompt for '{theme_name}' gave no assignments."); final_definition["diagnostics"]["theme_processing"][theme_name]["status"] = "Failed Rule (Fallback No Assign)"; reprompt_diag["failures"] += 1
            else: logger.error(f"Re-prompt LLM call failed for '{theme_name}'."); final_definition["diagnostics"]["theme_processing"][theme_name]["status"] = "Failed Rule (Fallback API Error)"; reprompt_diag["failures"] += 1
    final_definition["failed_fallback_themes"] = failed_initial_must_haves
    # --- Finalize Other Fields ---
    concept_cfg_final = config.get("concept_overrides", {}).get(normalized_concept_local, {})
    final_definition["applicable_lodging_types"] = concept_cfg_final.get("lodging_type", "Both")
    if final_definition["travel_category"]:
         final_definition["travel_category"]["type"] = concept_cfg_final.get("category_type", "Uncategorized")
         final_definition["travel_category"]["exclusionary_concepts"] = concept_cfg_final.get("exclusionary_concepts", [])
    final_definition["must_not_have"] = [{"uri": uri, "skos:prefLabel": get_primary_label(uri), "scope": None} for uri in sorted(list(set(concept_cfg_final.get("must_not_have_uris", []))))]
    final_definition["additional_relevant_subscores"] = concept_cfg_final.get("additional_relevant_subscores", [])
    # Generate top attributes
    attr_weights = {};
    for attr in all_assigned_attributes:
        uri = attr['uri']; weight = attr['concept_weight']
        if uri not in attr_weights or weight > attr_weights[uri]['concept_weight']: attr_weights[uri] = attr
    sorted_top_attributes = sorted(attr_weights.values(), key=lambda x: x['concept_weight'], reverse=True)
    final_definition['top_defining_attributes'] = [{k: v for k, v in attr.items() if k != 'comment'} for attr in sorted_top_attributes[:25]]
    # Final diagnostics counts
    final_diag = final_definition["diagnostics"]["final_output"] = {}; final_diag["must_not_have_count"] = len(final_definition["must_not_have"]); final_diag["additional_subscores_count"] = len(final_definition["additional_relevant_subscores"]); final_diag["themes_count"] = len(final_definition["themes"]); final_diag["failed_fallback_themes_count"] = len(final_definition["failed_fallback_themes"]); final_diag["top_defining_attributes_count"] = len(final_definition['top_defining_attributes'])
    logger.info(f"Finalization completed in {time.time() - start_time:.2f}s.")
    return final_definition

# --- Main Generation Loop ---
def generate_affinity_definitions_loop(concepts: List[str], config: AffinityConfig, args: argparse.Namespace,
                                       sbert_model: SentenceTransformer,
                                       primary_embeddings: Dict[str, np.ndarray], uris_list: List[str]):
    """Main loop to generate definitions for a list of concepts."""
    all_definitions = []
    effective_cache_version = config.get("CACHE_VERSION", "unknown_v32")

    for concept in tqdm(concepts, desc="Processing Concepts"):
        start_concept_time = time.time()
        logger.info(f"=== Processing Concept: '{concept}' ===")
        normalized_concept = normalize_concept(concept)
        affinity_definition: Dict[str, Any] = { # Base structure
            "input_concept": concept, "normalized_concept": normalized_concept,
            "applicable_lodging_types": "Both", "travel_category": {}, "core_definition": "",
            "top_defining_attributes": [], "themes": [], "additional_relevant_subscores": [],
            "must_not_have": [], "failed_fallback_themes": {}, "processing_metadata": {},
            "diagnostics": { "stage1": {}, "llm_slotting": {}, "reprompting_fallback": {"attempts": 0, "successes": 0, "failures": 0}, "stage2": {}, "theme_processing": {}, "final_output": {}, "error_details": None }
        }

        try:
            # Get concept embedding
            concept_embedding = get_concept_embedding(TRAVEL_CONTEXT + normalized_concept, sbert_model)
            if concept_embedding is None: raise ValueError("Failed to generate embedding")

            # Stage 1: Prepare Evidence
            candidates_for_llm_details, original_candidates_map, anchor_candidate, candidate_evidence_scores = prepare_evidence(
                concept, concept_embedding, primary_embeddings, uris_list, config, args
            )
            affinity_definition["diagnostics"]["stage1"] = {
                "status": "Completed",
                # "candidate_evidence_count_initial_pool": len(candidate_evidence_scores), # Maybe add this back if needed
                "candidate_evidence_count_for_llm": len(candidates_for_llm_details)
            }

            # Determine travel category
            travel_category_data = None
            if anchor_candidate and 'uri' in anchor_candidate:
                 kg_lookup = get_kg_data([anchor_candidate['uri']]) # Use kg_utils
                 if anchor_candidate['uri'] in kg_lookup: travel_category_data = kg_lookup[anchor_candidate['uri']]
            affinity_definition["travel_category"] = travel_category_data or {"uri": None, "name": concept, "type": None, "exclusionary_concepts": []}

            # Skip if no candidates for LLM
            if not candidates_for_llm_details:
                 logger.warning(f"No candidates for LLM processing for '{concept}'.")
                 affinity_definition["processing_metadata"]["status"] = "Skipped - No Candidates for LLM"
                 affinity_definition["diagnostics"]["llm_slotting"] = {"status": "Skipped", "llm_call_attempted": False}
                 final_stage2 = apply_rules_and_finalize(concept, {}, config, affinity_definition["travel_category"], anchor_candidate, {}, candidate_evidence_scores, args) # Run finalize with empty assignments
                 affinity_definition.update(final_stage2) # Merge structure
                 affinity_definition["diagnostics"]["stage2"] = {"status": "Skipped (No LLM Input)"}
                 affinity_definition["diagnostics"].update(final_stage2.get("diagnostics", {}))
                 all_definitions.append(affinity_definition); continue

            # Stage 1.5: LLM Slotting Call
            diag_llm = affinity_definition["diagnostics"]["llm_slotting"]
            diag_llm["status"] = "Not Started"; llm_assignments_parsed = {}
            if args.llm_provider != "none":
                 diag_llm["status"] = "Started"; diag_llm["llm_call_attempted"] = True
                 # Determine active themes for prompt
                 concept_cfg_prompt = config.get("concept_overrides", {}).get(normalized_concept, {})
                 active_themes_for_prompt_list = []
                 for tn, bd in config.get("base_themes", {}).items():
                      md = {**bd, **concept_cfg_prompt.get('themes', {}).get(tn, {})}
                      active_themes_for_prompt_list.append({"name": tn, "description": get_theme_definition_for_prompt(tn, md), "is_must_have": md.get("rule_applied") == "Must have 1"})
                 # Get concept type (simple approach for now)
                 concept_type_for_prompt = travel_category_data.get('type', 'Uncategorized') if travel_category_data else 'Uncategorized'

                 system_prompt, user_prompt = construct_llm_slotting_prompt(concept, active_themes_for_prompt_list, candidates_for_llm_details, args)
                 llm_assignments_raw = call_llm(system_prompt, user_prompt, args.llm_model, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_TEMPERATURE, args.llm_provider)
                 diag_llm["llm_raw_response"] = llm_assignments_raw
                 if isinstance(llm_assignments_raw, dict) and isinstance(llm_assignments_raw.get("theme_assignments"), dict):
                      diag_llm["llm_call_success"] = True; raw_assignments = llm_assignments_raw["theme_assignments"]; diag_llm["assignments_parsed_count"] = len(raw_assignments)
                      validated_assignments = {}; valid_theme_names = set(t['name'] for t in active_themes_for_prompt_list); uris_sent = set(original_candidates_map.keys())
                      for uri, themes in raw_assignments.items():
                           if uri in uris_sent:
                                valid_themes = [t for t in themes if isinstance(t, str) and t in valid_theme_names]
                                if valid_themes: validated_assignments[uri] = valid_themes
                           else: logger.warning(f"LLM assigned invalid URI '{uri}' for '{concept}'.")
                      llm_assignments_parsed = validated_assignments; diag_llm["assignments_validated_count"] = sum(len(v) for v in llm_assignments_parsed.values()); diag_llm["status"] = "Completed"
                 else: logger.warning(f"LLM call failed/invalid for '{concept}'."); diag_llm["llm_call_success"] = False; diag_llm["status"] = "Failed"
            else: logger.info("LLM provider 'none'."); diag_llm["status"] = "Skipped"

            # Stage 2: Apply Rules & Finalize (Includes Re-prompting)
            final_stage2 = apply_rules_and_finalize(
                 concept, {"theme_assignments": llm_assignments_parsed}, config,
                 affinity_definition["travel_category"], anchor_candidate,
                 original_candidates_map, candidate_evidence_scores, args
            )
            affinity_definition.update(final_stage2) # Merge results
            affinity_definition["diagnostics"]["stage2"] = {"status": "Completed"}
            affinity_definition["diagnostics"].update(final_stage2.get("diagnostics", {})) # Merge diagnostics

            # Determine final status
            if affinity_definition.get("failed_fallback_themes"): affinity_definition["processing_metadata"]["status"] = "Success with Failed Rules"
            elif diag_llm["status"] == "Failed": affinity_definition["processing_metadata"]["status"] = "Warning - LLM Slotting Failed"
            else: affinity_definition["processing_metadata"]["status"] = "Success"

        except Exception as e:
            logger.error(f"Failed processing concept '{concept}': {e}", exc_info=True)
            affinity_definition["processing_metadata"]["status"] = f"Failed - Exception: {type(e).__name__}"
            affinity_definition["diagnostics"]["error_details"] = traceback.format_exc()

        finally:
            end_concept_time = time.time()
            affinity_definition["processing_metadata"]["version"] = f"affinity-rule-engine-v32.0"
            affinity_definition["processing_metadata"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            affinity_definition["processing_metadata"]["duration_seconds"] = round(end_concept_time - start_concept_time, 2)
            affinity_definition["processing_metadata"]["cache_version"] = effective_cache_version
            affinity_definition["processing_metadata"]["llm_provider"] = args.llm_provider
            affinity_definition["processing_metadata"]["llm_model"] = args.llm_model if args.llm_provider != "none" else None
            all_definitions.append(affinity_definition)
            logger.info(f"=== Finished Concept: '{concept}' ({affinity_definition['processing_metadata']['status']}) ===")

    return all_definitions

# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description=f"Generate Affinity Definitions (v32.0)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept-list-file", default=None, help="Path to concepts file."); parser.add_argument("-t", "--taxonomy-dir", default=DEFAULT_TAXONOMY_DIR, help="Taxonomy RDF dir."); parser.add_argument("-c", "--config-file", default=DEFAULT_CONFIG_FILE, help="Affinity config JSON (v32.0)."); parser.add_argument("-o", "--output-dir", default="./output/", help="Output directory."); parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory."); parser.add_argument("--llm-provider", default="google", choices=["google", "openai", "none"], help="LLM provider."); parser.add_argument("--llm-model", default="gemini-1.5-flash-latest", help="LLM model name."); parser.add_argument("--rebuild-cache", action="store_true", help="Force cache rebuild."); parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    config = load_affinity_config(args.config_file)
    if not config: exit(1)
    effective_cache_version = config.get("CACHE_VERSION", CACHE_VERSION)

    os.makedirs(args.output_dir, exist_ok=True); os.makedirs(args.cache_dir, exist_ok=True)
    log_filename = os.path.join(args.output_dir, LOG_FILE_TEMPLATE.format(cache_version=effective_cache_version))
    file_handler = logging.FileHandler(log_filename, mode='w'); file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'))
    logger.handlers.clear(); logger.addHandler(logging.StreamHandler()); logger.addHandler(file_handler)
    if args.debug: logger.setLevel(logging.DEBUG); logger.info("Debug logging enabled.")
    else: logger.setLevel(logging.INFO)

    logger.info(f"+++ Running Affinity Definition Engine v32.0 +++")
    logger.info(f"  Config: {args.config_file} (v: {config.get('config_version', 'N/A')})")
    logger.info(f"  Cache Version: {effective_cache_version}")
    logger.info(f"  LLM Provider: {args.llm_provider}")
    if args.llm_provider != "none": logger.info(f"  LLM Model: {args.llm_model}")

    # Pre-flight & Load Data
    sbert_model = get_sbert_model()
    concepts_cache_file = os.path.join(args.cache_dir, f"concepts_{effective_cache_version}.json")
    embeddings_cache_file = os.path.join(args.cache_dir, f"embeddings_{effective_cache_version}.pkl")
    taxonomy_concepts = load_taxonomy_concepts(args.taxonomy_dir, concepts_cache_file, args, effective_cache_version)
    if not taxonomy_concepts: exit(1)
    embeddings_data = precompute_taxonomy_embeddings(taxonomy_concepts, sbert_model, embeddings_cache_file, args, effective_cache_version)
    if not embeddings_data: exit(1)
    _, primary_embeddings, uris_list = embeddings_data

    # Get Concepts
    input_concepts = []
    if args.concept_list_file:
        try:
            with open(args.concept_list_file, 'r', encoding='utf-8') as f: input_concepts = [line.strip() for line in f if line.strip()]
            logger.info(f"Read {len(input_concepts)} concepts from {args.concept_list_file}.")
        except Exception as e: logger.error(f"Could not read concept list file: {e}. Exiting."); exit(1)

    # Execute
    if input_concepts:
        logger.info(f"Processing {len(input_concepts)} concepts...")
        all_final_definitions = generate_affinity_definitions_loop(input_concepts, config, args, sbert_model, primary_embeddings, uris_list)
        output_filename = os.path.join(args.output_dir, OUTPUT_FILE_TEMPLATE.format(cache_version=effective_cache_version))
        logger.info(f"Writing {len(all_final_definitions)} definitions to {output_filename}")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f: json.dump(all_final_definitions, f, indent=2, ensure_ascii=False)
        except Exception as e: logger.error(f"Failed to write output: {e}", exc_info=True)
    else: logger.info("No concepts provided.")
    logger.info("=== Affinity Generation Script Finished ===")

if __name__ == "__main__":
    main()