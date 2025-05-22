# -*- coding: utf-8 -*-
"""
Utility functions for the Travel Concept Affinity Generator project (v34.0.2+ compatible).
Extracted and refactored from main script versions.
Includes NaN detection in embedding precomputation and setup for detailed, separate log files.
"""

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

# --- Third-Party Imports & Availability Checks ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: numpy library not found (pip install numpy).", file=sys.stderr)
    np = None; NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    DEFAULT_SBERT_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
except ImportError:
    print("CRITICAL ERROR: sentence-transformers library not found.", file=sys.stderr)
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None # type: ignore
    DEFAULT_SBERT_MODEL_NAME = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.utils.validation import _assert_all_finite
    SKLEARN_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: scikit-learn library not found.", file=sys.stderr)
    SKLEARN_AVAILABLE = False
    cosine_similarity = None # type: ignore
    def _assert_all_finite(*args, **kwargs): pass

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found (pip install tqdm), progress bars disabled.", file=sys.stderr)
    def tqdm(iterable, *args, **kwargs): # type: ignore
        yield from iterable

# --- Optional RDFlib Import ---
RDFLIB_AVAILABLE = False
try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import SKOS, RDFS, DCTERMS, OWL, RDF
    from rdflib import util as rdflib_util
    RDFLIB_AVAILABLE = True
except ImportError:
    print("Warning: rdflib library not found (pip install rdflib). KG-related functions may fail.", file=sys.stderr)
    class Graph: pass # type: ignore
    class URIRef: pass # type: ignore
    class Literal: pass # type: ignore
    class Namespace: pass # type: ignore
    class SKOS: pass # type: ignore
    class RDF: pass # type: ignore
    class RDFS: pass # type: ignore
    class OWL: pass # type: ignore
    class DCTERMS: pass # type: ignore
    class rdflib_util: pass # type: ignore

# --- Logger ---
logger = logging.getLogger(__name__)

# --- Globals for Detailed Loggers ---
_nan_logger: Optional[logging.Logger] = None
_llm_logger: Optional[logging.Logger] = None
_stage1_logger: Optional[logging.Logger] = None

# --- Utility Functions ---

def setup_logging(log_level: int = logging.INFO, log_filepath: Optional[str] = None, debug_mode: bool = False):
    """Configures the ROOT logger based on arguments."""
    log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_filepath:
        try:
             log_dir = os.path.dirname(log_filepath);
             if log_dir: os.makedirs(log_dir, exist_ok=True)
             handlers.append(logging.FileHandler(log_filepath, mode='w', encoding='utf-8'))
        except OSError as e: print(f"Warning: Could not create main log dir/file {log_filepath}: {e}", file=sys.stderr)
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, handlers=handlers, force=True)
    logging.info(f"Root logging setup complete. Level: {logging.getLevelName(log_level)}. File: {log_filepath or 'Console Only'}")

def setup_detailed_loggers(output_dir: str, cache_version: str):
    """Sets up separate log files for detailed diagnostics."""
    global _nan_logger, _llm_logger, _stage1_logger
    log_configs = {
        'nan_logger': f'nan_embeddings_{cache_version}.log',
        'llm_logger': f'llm_interactions_{cache_version}.log',
        'stage1_logger': f'stage1_candidates_{cache_version}.log'
    }
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for logger_name, filename in log_configs.items():
        try:
            logger_instance = logging.getLogger(logger_name)
            if logger_instance.hasHandlers():
                 for handler in logger_instance.handlers[:]: logger_instance.removeHandler(handler)
            log_path = os.path.join(output_dir, filename)
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.setLevel(logging.DEBUG)
            logger_instance.propagate = False
            globals()[f"_{logger_name}"] = logger_instance
            logging.info(f"Detailed logger '{logger_name}' configured. Output: {log_path}")
        except Exception as e:
            logging.error(f"Failed to set up detailed logger '{logger_name}': {e}")
            globals()[f"_{logger_name}"] = None

def normalize_concept(concept: Optional[str]) -> str:
    """Normalizes concept text: lowercase, remove punctuation, split camelCase, handle spaces."""
    if not isinstance(concept, str) or not concept: return ""
    try:
        norm = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', concept); norm = norm.replace("-", " ").replace("_", " ")
        norm = re.sub(r'[^\w\s]|(\'s\b)', '', norm); norm = ' '.join(norm.lower().split())
        return norm
    except Exception as e: logger.debug(f"Normalize failed: {e}"); return concept.lower().strip()

def get_primary_label(uri: str, taxonomy_concepts_cache: Dict[str, Dict], fallback: Optional[str] = None) -> str:
    """Gets the best primary label for a URI from the provided concepts cache."""
    label = fallback;
    if not taxonomy_concepts_cache: return fallback or uri
    details = taxonomy_concepts_cache.get(uri)
    if isinstance(details, dict):
        for prop in ["skos:prefLabel", "rdfs:label"]:
            val_list = details.get(prop, []); values = val_list if isinstance(val_list, list) else [val_list]
            for val in values:
                if isinstance(val, str) and val.strip(): return val.strip()
        alt_labels = details.get("skos:altLabel", []); alt_values = alt_labels if isinstance(alt_labels, list) else [alt_labels]
        for alt_val in alt_values:
            if isinstance(alt_val, str) and alt_val.strip(): return alt_val.strip()
        if fallback is None:
            definitions = details.get("skos:definition", []); def_values = definitions if isinstance(definitions, list) else [definitions]
            for def_val in def_values:
                 if isinstance(def_val, str) and def_val.strip():
                      definition_text = def_val.strip(); label = definition_text[:60] + ("..." if len(definition_text) > 60 else "")
                      return label
    if label is None or label == fallback:
        try:
            parsed_label = uri;
            if '#' in uri: parsed_label = uri.split('#')[-1]
            elif '/' in uri: parsed_label = uri.split('/')[-1]
            if parsed_label and parsed_label != uri:
                parsed_label = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', parsed_label); parsed_label = parsed_label.replace('_', ' ').replace('-', ' ');
                label = ' '.join(word.capitalize() for word in parsed_label.split() if word)
        except Exception as e: logger.debug(f"URI parsing failed for {uri}: {e}")
    return label if label is not None else fallback if fallback is not None else uri

def get_concept_type_labels(uri: str, taxonomy_concepts_cache: Dict[str, Dict]) -> List[str]:
    """Gets the labels of the RDF types associated with a concept URI."""
    type_labels: Set[str] = set();
    if not taxonomy_concepts_cache: return []
    details = taxonomy_concepts_cache.get(uri)
    if isinstance(details, dict):
        type_uris = details.get("type", []); type_uris = type_uris if isinstance(type_uris, list) else [type_uris]
        for type_uri_str in type_uris:
            if isinstance(type_uri_str, str):
                 label = get_primary_label(type_uri_str, taxonomy_concepts_cache, fallback=type_uri_str)
                 if label != type_uri_str or "Concept" in label or "Class" in label: type_labels.add(label)
    return sorted(list(type_labels))

def get_sbert_model(model_name: Optional[str] = None) -> Optional[SentenceTransformer]:
    """Loads and returns a SentenceTransformer model. Uses a default if no name is provided."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None: logger.error("Cannot load SBERT model, library missing."); return None
    model_to_load = model_name if model_name else DEFAULT_SBERT_MODEL_NAME
    if not model_to_load: logger.error("No SBERT model name specified."); return None
    logger.info(f"Loading SentenceTransformer model: {model_to_load}")
    try:
        model = SentenceTransformer(model_to_load); logger.info(f"Successfully loaded SBERT model: {model_to_load}")
        return model
    except Exception as e: logger.error(f"Failed to load SBERT model '{model_to_load}': {e}", exc_info=True); return None

def load_affinity_config(config_file: str) -> Optional[Dict]:
    """Loads, validates slightly, and returns the affinity configuration as dict."""
    logger.info(f"Loading configuration from: {config_file}")
    if not os.path.exists(config_file): logger.critical(f"FATAL: Config file not found: '{config_file}'"); return None
    try:
        with open(config_file, 'r', encoding='utf-8') as f: config = json.load(f)
        if not isinstance(config, dict): raise ValueError("Config is not a valid JSON object")
        if "concept_overrides" in config and isinstance(config["concept_overrides"], dict):
             config["concept_overrides"] = {normalize_concept(k): v for k, v in config["concept_overrides"].items()}
        else: config["concept_overrides"] = {}
        config.setdefault("base_themes", {})
        if isinstance(config["base_themes"], dict):
            for theme, data in config["base_themes"].items():
                if not isinstance(data, dict): logger.warning(f"Theme '{theme}' config is not a dictionary.")
                elif not data.get("description"): logger.warning(f"Theme '{theme}' missing description.")
        else: logger.warning("'base_themes' in config is not a dictionary.")
        logger.info(f"Config loaded. Version: {config.get('cache_version', 'Not Specified')}")
        return config
    except (json.JSONDecodeError, ValueError) as e: logger.critical(f"FATAL Error loading/validating config '{config_file}': {e}", exc_info=True); return None
    except Exception as e: logger.critical(f"FATAL Unexpected error loading config '{config_file}': {e}", exc_info=True); return None

def get_cache_filename(base_name: str, cache_version: str, cache_dir: str, params: Optional[Dict[str, Any]] = None, extension: str = ".pkl") -> str:
    """Generates a standardized cache filename including cache version and parameters."""
    filename_parts = [base_name, cache_version]
    if params:
         param_str = "_".join(f"{k}-{str(v).replace('/', '_').replace(' ', '_')}" for k, v in sorted(params.items()))
         filename_parts.append(param_str)
    filename = "_".join(part for part in filename_parts if part) + extension
    filename = filename.replace(":", "-").replace("\\", "_").replace("#", "")
    return os.path.join(cache_dir, filename)

def load_cache(cache_filename: str, file_format: str = 'pickle') -> Optional[Any]:
    """Loads data from a cache file (pickle or json)."""
    if not os.path.exists(cache_filename): logger.info(f"Cache file not found: {cache_filename}"); return None
    logger.info(f"Attempting to load cache from: {cache_filename} (Format: {file_format})")
    try:
        if file_format == 'pickle':
            with open(cache_filename, 'rb') as f: data = pickle.load(f)
        elif file_format == 'json':
            with open(cache_filename, 'r', encoding='utf-8') as f: data = json.load(f)
        else: raise ValueError(f"Unsupported cache format: {file_format}")
        logger.info(f"Successfully loaded cache from: {cache_filename}"); return data
    except EOFError: logger.warning(f"Cache file {cache_filename} empty/corrupted (EOFError)."); return None
    except pickle.UnpicklingError: logger.warning(f"Cache file {cache_filename} corrupted (UnpicklingError)."); return None
    except Exception as e: logger.warning(f"Failed to load cache file {cache_filename}: {e}.", exc_info=logger.level <= logging.DEBUG); return None

def save_cache(data: Any, cache_filename: str, file_format: str = 'pickle'):
    """Saves data to a cache file (pickle or json)."""
    logger.info(f"Saving cache to: {cache_filename} (Format: {file_format})")
    try:
        cache_dir = os.path.dirname(cache_filename);
        if cache_dir: os.makedirs(cache_dir, exist_ok=True)
        if file_format == 'pickle':
            with open(cache_filename, 'wb') as f: pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif file_format == 'json':
            with open(cache_filename, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        else: raise ValueError(f"Unsupported cache format: {file_format}")
        logger.info(f"Successfully saved cache to: {cache_filename}")
    except Exception as e: logger.error(f"Failed writing cache to {cache_filename}: {e}", exc_info=True)

def load_taxonomy_concepts(
    taxonomy_dir: str, cache_file: str, rebuild_cache: bool, current_cache_version: str, debug_mode: bool
) -> Optional[Dict[str, Dict]]:
    """Loads concepts from RDF taxonomy files or cache, converting to internal structure."""
    if not rebuild_cache and os.path.exists(cache_file):
        cache_data = load_cache(cache_file); 
        if cache_data and isinstance(cache_data, dict) and len(cache_data) > 0:
            logger.info(f"Loaded {len(cache_data)} concepts from cache ({cache_file})")
            return cache_data
    
    if not RDFLIB_AVAILABLE:
        logger.critical("FATAL: RDFlib required to load taxonomy but not available.")
        return None
    
    logger.info(f"Loading RDF taxonomy files from {taxonomy_dir} (rebuild_cache={rebuild_cache})...")
    
    kg_formats = ["ttl", "n3", "rdf", "xml", "nt", "trix", "trig"]
    taxonomy_files = []
    try:
        for root, _, files in os.walk(taxonomy_dir):
            for filename in files:
                if any(filename.lower().endswith(f".{fmt}") for fmt in kg_formats):
                    taxonomy_files.append(os.path.join(root, filename))
    except Exception as e:
        logger.error(f"Error scanning taxonomy directory: {e}", exc_info=True)
        return None
    
    if not taxonomy_files:
        logger.critical(f"FATAL: No RDF files found in {taxonomy_dir}")
        return None
    
    g = Graph()
    concepts_cache: Dict[str, Dict] = {}
    total_files = len(taxonomy_files)
    
    logger.info(f"Found {total_files} taxonomy files to parse.")
    for idx, taxonomy_file in enumerate(tqdm(taxonomy_files, desc="Loading taxonomy files")):
        if debug_mode:
            logger.debug(f"Parsing file {idx+1}/{total_files}: {taxonomy_file}")
        try:
            g.parse(taxonomy_file, format=rdflib_util.guess_format(taxonomy_file))
        except Exception as e:
            logger.error(f"Failed to parse {taxonomy_file}: {e}", exc_info=debug_mode)
    
    logger.info(f"RDF Graph parsing complete. Graph has {len(g)} triples.")

    # Extract all URIs in the graph as potential concepts
    all_subjects = set(s for s in g.subjects() if isinstance(s, URIRef))
    logger.info(f"Found {len(all_subjects)} unique URIs as potential concepts.")

    # Process each URI to create the concepts cache
    for uri in tqdm(all_subjects, desc="Processing concepts"):
        uri_str = str(uri)
        concept_data: Dict[str, Any] = {}
        
        # Get all predicate-object pairs for this URI
        for pred, obj in g.predicate_objects(uri):
            pred_str = str(pred)
            
            # Get the last part of the predicate URI as the property name
            prop_name = pred_str.split("/")[-1].split("#")[-1]
            
            # Use namespace prefixes for common properties
            if str(pred) == str(RDF.type):
                prop_name = "type"
            elif str(pred) == str(RDFS.label):
                prop_name = "rdfs:label"
            elif str(pred) == str(SKOS.prefLabel):
                prop_name = "skos:prefLabel"
            elif str(pred) == str(SKOS.altLabel):
                prop_name = "skos:altLabel"
            elif str(pred) == str(SKOS.definition):
                prop_name = "skos:definition"
            
            # Format the object value based on its type
            if isinstance(obj, Literal):
                obj_val = str(obj)
            elif isinstance(obj, URIRef):
                obj_val = str(obj)
            else:
                obj_val = str(obj)
            
            # Add to the concept data, handling multiples as lists
            if prop_name in concept_data:
                if isinstance(concept_data[prop_name], list):
                    concept_data[prop_name].append(obj_val)
                else:
                    concept_data[prop_name] = [concept_data[prop_name], obj_val]
            else:
                concept_data[prop_name] = obj_val
        
        # Only include URIs that have at least one label or definition
        has_label = any(k in concept_data for k in ["rdfs:label", "skos:prefLabel", "skos:altLabel", "skos:definition"])
        if has_label or "type" in concept_data:
            concepts_cache[uri_str] = concept_data
    
    logger.info(f"Processed {len(concepts_cache)} concepts with labels/definitions out of {len(all_subjects)} URIs.")
    
    # Save to cache for future use
    if cache_file:
        save_cache(concepts_cache, cache_file)
    
    return concepts_cache

def precompute_taxonomy_embeddings(
    taxonomy_concepts: Dict[str, Dict], sbert_model: SentenceTransformer, cache_file: str,
    current_cache_version: str, rebuild_cache: bool, debug_mode: bool
) -> Optional[Tuple[Dict[str, np.ndarray], List[str]]]:
    """Precomputes embeddings for all concepts in the taxonomy."""
    global _nan_logger
    
    if not rebuild_cache and os.path.exists(cache_file):
        cache_data = load_cache(cache_file)
        if cache_data and isinstance(cache_data, tuple) and len(cache_data) == 2:
            embeddings, uri_list = cache_data
            if isinstance(embeddings, dict) and isinstance(uri_list, list) and len(embeddings) > 0:
                logger.info(f"Loaded {len(embeddings)} embeddings from cache ({cache_file})")
                return embeddings, uri_list
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not sbert_model:
        logger.error("Cannot precompute embeddings: SentenceTransformer model not available.")
        return None
    
    logger.info(f"Precomputing embeddings for {len(taxonomy_concepts)} concepts...")
    
    # Prepare the texts to embed
    texts_to_embed = []
    uri_list = []
    
    # Helper weight constants for constructing the embedding text
    weights = {
        "skos:prefLabel": 3.0,  # Highest weight to preferred labels
        "rdfs:label": 2.5,      # High weight to RDF labels
        "skos:altLabel": 1.5,   # Medium weight to alternate labels
        "skos:definition": 1.0,  # Base weight for definitions
    }
    
    for uri, data in tqdm(taxonomy_concepts.items(), desc="Preparing for embedding"):
        # Skip concepts without any labels
        if not any(k in data for k in weights.keys()):
            continue
        
        # Create a weighted text by repeating terms based on their importance
        parts = []
        
        for prop, weight in weights.items():
            if prop in data:
                values = data[prop] if isinstance(data[prop], list) else [data[prop]]
                for val in values:
                    if isinstance(val, str) and val.strip():
                        # Add the text weight times (integer)
                        parts.extend([val.strip()] * int(weight))
                        # If there's a fractional part, sometimes add one more
                        if weight % 1 > 0 and np.random.random() < (weight % 1):
                            parts.append(val.strip())
        
        if parts:
            text = " ".join(parts)
            texts_to_embed.append(text)
            uri_list.append(uri)
    
    logger.info(f"Computed texts for {len(texts_to_embed)} concepts. Computing embeddings...")
    
    # Compute embeddings in batches
    batch_size = 32
    embeddings = {}
    nan_detected = 0
    
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Computing embeddings"):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_uris = uri_list[i:i+batch_size]
        
        try:
            batch_embeddings = sbert_model.encode(batch_texts, convert_to_numpy=True)
            
            for j, (uri, embedding) in enumerate(zip(batch_uris, batch_embeddings)):
                # Check for NaN values
                try:
                    if SKLEARN_AVAILABLE:
                        _assert_all_finite(embedding)
                    elif NUMPY_AVAILABLE and np is not None:
                        if np.isnan(embedding).any() or np.isinf(embedding).any():
                            raise ValueError("NaN or Inf detected")
                except Exception as e:
                    nan_detected += 1
                    if _nan_logger:
                        _nan_logger.warning(f"NaN/Inf detected in embedding for URI: {uri}")
                        _nan_logger.warning(f"Text: {texts_to_embed[i+j][:100]}...")
                    if debug_mode:
                        logger.warning(f"NaN/Inf detected in embedding for URI: {uri}")
                    continue
                
                embeddings[uri] = embedding
                
        except Exception as e:
            logger.error(f"Error computing embeddings for batch {i//batch_size}: {e}", exc_info=debug_mode)
    
    if nan_detected > 0:
        logger.warning(f"NaN/Inf values detected in {nan_detected} embeddings. These were skipped.")
    
    logger.info(f"Computed {len(embeddings)} embeddings successfully.")
    
    # Save to cache for future use
    if cache_file:
        save_cache((embeddings, uri_list), cache_file)
    
    return embeddings, uri_list

def get_concept_embedding(text_to_embed: str, model: Optional[SentenceTransformer]) -> Optional[np.ndarray]:
    """Gets the embedding vector for a concept text using the provided model."""
    if not model or not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.debug("Cannot get concept embedding: model not available")
        return None
    
    try:
        embedding = model.encode([text_to_embed], convert_to_numpy=True)[0]
        
        if SKLEARN_AVAILABLE:
            try:
                _assert_all_finite(embedding)
            except:
                logger.warning(f"NaN/Inf detected in embedding for: {text_to_embed[:50]}...")
                return None
        elif NUMPY_AVAILABLE and np is not None and (np.isnan(embedding).any() or np.isinf(embedding).any()):
            logger.warning(f"NaN/Inf detected in embedding for: {text_to_embed[:50]}...")
            return None
        
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text: {e}", exc_info=True)
        return None

def get_batch_embedding_similarity(target_embedding: Optional[np.ndarray], candidate_embeddings_map: Dict[str, Optional[np.ndarray]]) -> Dict[str, float]:
    """Computes cosine similarity between target and all candidates."""
    similarities: Dict[str, float] = {}
    
    if target_embedding is None or not NUMPY_AVAILABLE:
        return similarities
    
    if cosine_similarity is not None and SKLEARN_AVAILABLE:
        # Faster implementation using sklearn
        candidates, uris = [], []
        for uri, embed in candidate_embeddings_map.items():
            if embed is not None:
                candidates.append(embed)
                uris.append(uri)
        
        if candidates:
            try:
                # Convert to numpy array and reshape
                candidates_array = np.vstack(candidates)
                target_reshaped = target_embedding.reshape(1, -1)
                
                # Calculate all similarities at once
                sim_scores = cosine_similarity(target_reshaped, candidates_array)[0]
                
                # Create result dictionary
                for uri, score in zip(uris, sim_scores):
                    if not math.isnan(score):
                        similarities[uri] = float(score)
            except Exception as e:
                logger.error(f"Error in batch similarity calculation: {e}", exc_info=True)
    else:
        # Fallback manual implementation
        for uri, candidate_embedding in candidate_embeddings_map.items():
            if candidate_embedding is not None and np is not None:
                try:
                    # Calculate cosine similarity manually
                    dot_product = np.dot(target_embedding, candidate_embedding)
                    norm_product = np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding)
                    
                    if norm_product > 0:
                        similarity = dot_product / norm_product
                        if not math.isnan(similarity):
                            similarities[uri] = float(similarity)
                except Exception as e:
                    logger.debug(f"Error calculating similarity for {uri}: {e}")
    
    return similarities

def get_kg_data(uris: List[str], taxonomy_concepts_cache: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
    """Retrieves detailed information for a list of URIs from the taxonomy cache."""
    result: Dict[str, Dict[str, Any]] = {}
    
    for uri in uris:
        if uri in taxonomy_concepts_cache:
            concept_data = taxonomy_concepts_cache[uri]
            
            # Get the best label
            label = get_primary_label(uri, taxonomy_concepts_cache)
            
            # Get other details
            concept_types = get_concept_type_labels(uri, taxonomy_concepts_cache)
            
            # Get definitions if available
            definitions = []
            if "skos:definition" in concept_data:
                def_values = concept_data["skos:definition"]
                if isinstance(def_values, list):
                    definitions.extend([d for d in def_values if isinstance(d, str) and d.strip()])
                elif isinstance(def_values, str) and def_values.strip():
                    definitions.append(def_values.strip())
            
            # Store in the result
            result[uri] = {
                "label": label,
                "types": concept_types,
                "definitions": definitions,
                "uri": uri
            }
    
    return result

def build_keyword_label_index(taxonomy_concepts: Dict[str, Dict]) -> Optional[Dict[str, Set[str]]]:
    """Builds an index mapping normalized keywords to their corresponding concept URIs."""
    if not taxonomy_concepts:
        return None
    
    keyword_index: Dict[str, Set[str]] = defaultdict(set)
    
    for uri, concept_data in tqdm(taxonomy_concepts.items(), desc="Building keyword index"):
        # Extract all labels from the concept
        all_labels = []
        
        for label_property in ["skos:prefLabel", "rdfs:label", "skos:altLabel"]:
            if label_property in concept_data:
                values = concept_data[label_property]
                if isinstance(values, list):
                    all_labels.extend([l for l in values if isinstance(l, str)])
                elif isinstance(values, str):
                    all_labels.append(values)
        
        # Process each label
        for label in all_labels:
            if not label or not isinstance(label, str):
                continue
            
            # Normalize and add to index
            normalized_label = normalize_concept(label)
            if normalized_label:
                keyword_index[normalized_label].add(uri)
                
                # Also add each individual word as a keyword
                for word in normalized_label.split():
                    if len(word) > 2:  # Skip very short words
                        keyword_index[word].add(uri)
    
    return keyword_index

def save_results_json(results_data: List[Dict], output_filepath: str):
    """Saves the results to a JSON file with human-readable formatting."""
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving results to {output_filepath}: {e}", exc_info=True)
