import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set, Union, Tuple, Any
from dataclasses import dataclass, field
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, SKOS, XSD
from tqdm import tqdm
import sys
from urllib.parse import quote
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import re # For safe_uri_local_name

# --- Basic Logging Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Define Namespaces (must align with ExpediaAffinityDefinitionModel.ttl) ---
AFFMDL = Namespace("urn:com:expedia:affinitymodel#")
AFFMDL_INST = Namespace("urn:com:expedia:affinitymodel:instance:")

@dataclass
class ValidationIssue:
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    category: str  # 'FORMAT', 'TAXONOMY', 'NAMESPACE', 'DATA_MISSING', 'DATA_TYPE'
    message: str
    location: Optional[str] = None # E.g., path in JSON, or subject URI
    suggestion: Optional[str] = None

class ValidationResult:
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.stats: Dict[str, Any] = {
            'conversion_stats': {
                'total_definitions_in_json': 0,
                'processed_affinity_definitions': 0,
                'total_triples_generated': 0,
                'batches_processed': 0,
            },
            'rdf_graph_stats': {
                'final_triple_count': 0,
                'unique_subjects': set(),
                'unique_predicates': set(),
                'unique_objects_uris': set(),
                'unique_objects_literals': set(),
                'namespaces_used': set(),
            },
            'taxonomy_validation_stats': { # Only populated if taxonomy_dir is provided
                'total_taxonomies_loaded': 0,
                'total_valid_taxonomy_uris': 0,
                'concepts_by_taxonomy_source_file': {},
                'unvalidated_concept_uris_found': set(), # URIs from JSON data not found in loaded taxonomies
                'validated_concept_uris_count': 0,
            }
        }

    def add_issue(self, severity: str, category: str, message: str, location: Optional[str] = None, suggestion: Optional[str] = None):
        self.issues.append(ValidationIssue(severity, category, message, location, suggestion))

    def has_errors(self) -> bool: # Changed from has_critical_errors for general error checking
        return any(issue.severity == 'ERROR' for issue in self.issues)

    def to_dict(self) -> Dict:
        # Convert sets to lists/counts for JSON serialization
        serializable_stats = json.loads(json.dumps(self.stats, default=lambda o: list(o) if isinstance(o, set) else str(o)))
        serializable_stats['rdf_graph_stats']['unique_subjects_count'] = len(self.stats['rdf_graph_stats']['unique_subjects'])
        serializable_stats['rdf_graph_stats']['unique_predicates_count'] = len(self.stats['rdf_graph_stats']['unique_predicates'])
        serializable_stats['rdf_graph_stats']['unique_objects_uris_count'] = len(self.stats['rdf_graph_stats']['unique_objects_uris'])
        serializable_stats['rdf_graph_stats']['unique_objects_literals_count'] = len(self.stats['rdf_graph_stats']['unique_objects_literals'])
        if 'unvalidated_concept_uris_found' in serializable_stats.get('taxonomy_validation_stats', {}):
            serializable_stats['taxonomy_validation_stats']['unvalidated_concept_uris_count'] = len(self.stats['taxonomy_validation_stats']['unvalidated_concept_uris_found'])


        return {
            'issues': [issue.__dict__ for issue in self.issues],
            'stats': serializable_stats
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

class JsonToExpediaAffinityRDF:
    def __init__(self, taxonomy_dir: Optional[str] = None):
        self.taxonomy_dir = Path(taxonomy_dir) if taxonomy_dir else None
        self.all_valid_taxonomy_uris: Set[str] = set()
        self.uri_to_taxonomy_source_file: Dict[str, Set[str]] = defaultdict(set)
        self._taxonomies_loaded: bool = False

    def _add_literal_if_present(self, g: Graph, subject: URIRef, predicate: URIRef, value: Any, datatype: Optional[URIRef] = None, result: ValidationResult = None, location: Optional[str] = None):
        """Adds a literal triple to the graph if the value is not None, with basic type validation."""
        if value is not None:
            try:
                if isinstance(value, bool):
                    lit = Literal(value, datatype=XSD.boolean)
                elif datatype == XSD.integer and not isinstance(value, int):
                    if result: result.add_issue('WARNING', 'DATA_TYPE', f"Expected integer for {predicate}, got {type(value)} ('{value}')", location)
                    lit = Literal(int(value), datatype=XSD.integer) # Attempt cast
                elif datatype == XSD.decimal and not isinstance(value, (int, float)):
                    if result: result.add_issue('WARNING', 'DATA_TYPE', f"Expected decimal/float for {predicate}, got {type(value)} ('{value}')", location)
                    lit = Literal(float(value), datatype=XSD.decimal) # Attempt cast
                elif datatype == XSD.double and not isinstance(value, (int, float)):
                    if result: result.add_issue('WARNING', 'DATA_TYPE', f"Expected double/float for {predicate}, got {type(value)} ('{value}')", location)
                    lit = Literal(float(value), datatype=XSD.double) # Attempt cast
                elif datatype:
                    lit = Literal(str(value), datatype=datatype) # Ensure string conversion for other datatypes
                else:
                    lit = Literal(str(value)) # Default to string if no datatype
                g.add((subject, predicate, lit))
            except ValueError as e:
                if result: result.add_issue('ERROR', 'DATA_TYPE', f"Failed to cast value '{value}' to {datatype} for {predicate}: {e}", location)


    def _get_nested(self, data: Dict, keys: List[str], default: Any = None) -> Any:
        current_level = data
        for i, key in enumerate(keys):
            path_so_far = ".".join(map(str,keys[:i+1]))
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            elif isinstance(current_level, list) and isinstance(key, int) and 0 <= key < len(current_level):
                current_level = current_level[key]
            else:
                # logger.debug(f"Path not found: {path_so_far} in data. Returning default.")
                return default
        return current_level

    def _safe_uri_local_name(self, text: str) -> str:
        if not text: return "unknown_concept"
        s = str(text).strip().replace(" ", "_")
        s = re.sub(r'(?u)[^\w-]', '', s) # Remove non-alphanumeric (unicode-aware), keep underscore and hyphen
        return quote(s[:100]) # URI encode and limit length

    def _get_unique_output_path(self, base_path_str: str) -> str:
        path = Path(base_path_str)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {path.parent}")
        if not path.exists():
            return str(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{path.stem}_{timestamp}{path.suffix}"
        unique_path = path.parent / new_filename
        logger.info(f"Output file {path} already exists. Using unique filename: {unique_path}")
        return str(unique_path)

    def _load_one_taxonomy(self, file_path: Path) -> Dict[str, Any]:
        """Loads a single taxonomy file."""
        try:
            graph = Graph()
            graph.parse(str(file_path), format=rdflib.util.guess_format(str(file_path)))
            concepts = set()
            uri_map = defaultdict(set)
            for s, _, o in graph: # Iterate over all triples
                if isinstance(s, URIRef):
                    concepts.add(str(s))
                    uri_map[str(s)].add(file_path.name)
                if isinstance(o, URIRef):
                    concepts.add(str(o))
                    uri_map[str(o)].add(file_path.name)
            return {'name': file_path.name, 'concepts': concepts, 'uri_map': uri_map, 'success': True, 'error': None}
        except Exception as e:
            return {'name': file_path.name, 'concepts': set(), 'uri_map': defaultdict(set), 'success': False, 'error': str(e)}

    def _load_taxonomies_if_needed(self, result: ValidationResult):
        if not self.taxonomy_dir or self._taxonomies_loaded:
            return

        if not self.taxonomy_dir.exists() or not self.taxonomy_dir.is_dir():
            result.add_issue('WARNING', 'TAXONOMY', f"Taxonomy directory not found or not a directory: {self.taxonomy_dir}")
            self._taxonomies_loaded = True # Mark as "attempted"
            return

        rdf_files = [p for p in self.taxonomy_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.rdf', '.owl', '.ttl', '.n3', '.nt']]
        if not rdf_files:
            result.add_issue('WARNING', 'TAXONOMY', f"No RDF files found in taxonomy directory: {self.taxonomy_dir}")
            self._taxonomies_loaded = True
            return

        logger.info(f"Loading {len(rdf_files)} taxonomy files from {self.taxonomy_dir}...")
        
        # Use ThreadPoolExecutor for potentially faster I/O bound tasks
        # Adjust max_workers based on your system and number of files
        max_workers = min(multiprocessing.cpu_count(), len(rdf_files), 16) 
        
        loaded_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._load_one_taxonomy, f): f for f in rdf_files}
            for future in tqdm(as_completed(future_to_file), total=len(rdf_files), desc="Loading Taxonomies"):
                file_data = future.result()
                if file_data['success']:
                    self.all_valid_taxonomy_uris.update(file_data['concepts'])
                    for uri, sources in file_data['uri_map'].items():
                        self.uri_to_taxonomy_source_file[uri].update(sources)
                    result.stats['taxonomy_validation_stats']['concepts_by_taxonomy_source_file'][file_data['name']] = len(file_data['concepts'])
                    loaded_count += 1
                else:
                    result.add_issue('WARNING', 'TAXONOMY', f"Failed to load/parse {file_data['name']}: {file_data['error']}")
        
        result.stats['taxonomy_validation_stats']['total_taxonomies_loaded'] = loaded_count
        result.stats['taxonomy_validation_stats']['total_valid_taxonomy_uris'] = len(self.all_valid_taxonomy_uris)
        logger.info(f"Finished loading taxonomies. Loaded {loaded_count} files, {len(self.all_valid_taxonomy_uris)} unique URIs.")
        self._taxonomies_loaded = True


    def _validate_and_add_uri_link(self, g: Graph, subject: URIRef, predicate: URIRef, uri_value: Optional[str], result: ValidationResult, location: str, concept_description: str):
        """Validates a URI and adds the triple if valid."""
        if not uri_value:
            result.add_issue('WARNING', 'DATA_MISSING', f"Missing URI for {concept_description}", location)
            return

        # Basic URI format validation
        if not (uri_value.startswith("urn:") or uri_value.startswith("http:") or uri_value.startswith("https:")):
             result.add_issue('WARNING', 'FORMAT', f"Unusual URI scheme for {concept_description}: {uri_value}", location, "Ensure URI uses standard schemes like urn:, http:, https:")
        
        # Taxonomy validation (if taxonomies were loaded)
        if self.taxonomy_dir and self.all_valid_taxonomy_uris and uri_value not in self.all_valid_taxonomy_uris:
            result.add_issue('WARNING', 'TAXONOMY', f"{concept_description} URI not found in loaded taxonomies: {uri_value}", location, "Verify URI exists in specified taxonomies or add taxonomy source.")
            result.stats['taxonomy_validation_stats']['unvalidated_concept_uris_found'].add(uri_value)
        elif self.taxonomy_dir and self.all_valid_taxonomy_uris:
             result.stats['taxonomy_validation_stats']['validated_concept_uris_count'] +=1


        g.add((subject, predicate, URIRef(uri_value)))


    def _process_definition_to_rdf(self, g: Graph, def_data: Dict, result: ValidationResult) -> int:
        """Converts a single JSON affinity definition object to RDF triples."""
        triples_before = len(g)

        input_concept = def_data.get("input_concept")
        if not input_concept: # Should have been caught earlier, but defensive
            result.add_issue('ERROR', 'DATA_MISSING', "Definition missing 'input_concept' field.")
            return 0

        norm_concept = def_data.get("normalized_concept", input_concept)
        inst_local_name = self._safe_uri_local_name(norm_concept)
        aff_def_uri = AFFMDL_INST[f"{inst_local_name}_AffinityDefinition"]

        g.add((aff_def_uri, RDF.type, AFFMDL.AffinityDefinition))
        g.add((aff_def_uri, skos.prefLabel, Literal(f"Affinity Definition for '{input_concept}'")))
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.inputConceptLabel, input_concept, result=result, location=f"{input_concept}/inputConceptLabel")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.normalizedConceptLabel, norm_concept, result=result, location=f"{input_concept}/normalizedConceptLabel")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.applicableLodgingTypes, def_data.get("applicable_lodging_types"), result=result, location=f"{input_concept}/applicableLodgingTypes")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.requiresGeoCheck, def_data.get("requires_geo_check"), result=result, location=f"{input_concept}/requiresGeoCheck")
        if def_data.get("failed_fallback_themes") is not None:
            self._add_literal_if_present(g, aff_def_uri, AFFMDL.failedFallbackThemesAsJson, json.dumps(def_data.get("failed_fallback_themes")), result=result, location=f"{input_concept}/failedFallbackThemesAsJson")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.affinityScoreTotalAllocated, def_data.get("affinity_score_total_allocated"), XSD.decimal, result=result, location=f"{input_concept}/affinityScoreTotalAllocated")

        # Travel Category Link
        tc_json = def_data.get("travel_category", {})
        if tc_json and tc_json.get("uri"): # Ensure tc_json is not None before accessing .get("uri")
            tc_link_bnode = BNode()
            g.add((aff_def_uri, AFFMDL.hasTravelCategoryLink, tc_link_bnode))
            g.add((tc_link_bnode, RDF.type, AFFMDL.TravelCategoryLink))
            self._validate_and_add_uri_link(g, tc_link_bnode, AFFMDL.linkedConceptUri, tc_json.get("uri"), result, f"{input_concept}/travel_category/uri", "Travel Category URI")
            
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.observedLabel, tc_json.get("label"), result=result, location=f"{input_concept}/travel_category/label")
            for idx, type_val in enumerate(tc_json.get("types", [])):
                self._add_literal_if_present(g, tc_link_bnode, AFFMDL.observedType, type_val, result=result, location=f"{input_concept}/travel_category/types[{idx}]")
            for idx, def_val in enumerate(tc_json.get("definitions", [])):
                self._add_literal_if_present(g, tc_link_bnode, AFFMDL.observedDefinition, def_val, result=result, location=f"{input_concept}/travel_category/definitions[{idx}]")
            
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualSbertScore, tc_json.get("sbert_score"), XSD.double, result=result, location=f"{input_concept}/travel_category/sbert_score")
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualKeywordScore, tc_json.get("keyword_score"), XSD.double, result=result, location=f"{input_concept}/travel_category/keyword_score")
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualCombinedScore, tc_json.get("combined_score"), XSD.double, result=result, location=f"{input_concept}/travel_category/combined_score")
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualCombinedScoreUnbiased, tc_json.get("combined_score_unbiased"), XSD.double, result=result, location=f"{input_concept}/travel_category/combined_score_unbiased")
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualIsBiased, tc_json.get("biased"), result=result, location=f"{input_concept}/travel_category/biased")
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualEffectiveAlpha, tc_json.get("effective_alpha"), XSD.double, result=result, location=f"{input_concept}/travel_category/effective_alpha")
            self._add_literal_if_present(g, tc_link_bnode, AFFMDL.contextualBiasReason, tc_json.get("bias_reason"), result=result, location=f"{input_concept}/travel_category/bias_reason")
        elif tc_json: # tc_json exists but no uri
             result.add_issue('WARNING', 'DATA_MISSING', f"Travel category data present but URI is missing for {input_concept}", f"{input_concept}/travel_category")


        # Top Defining Attributes
        for idx, attr_json in enumerate(def_data.get("top_defining_attributes", [])):
            loc = f"{input_concept}/top_defining_attributes[{idx}]"
            if attr_json.get("uri"):
                attr_link_bnode = BNode()
                g.add((aff_def_uri, AFFMDL.hasTopDefiningAttributeLink, attr_link_bnode))
                g.add((attr_link_bnode, RDF.type, AFFMDL.RankedAttributeLink))
                self._validate_and_add_uri_link(g, attr_link_bnode, AFFMDL.attributeUri, attr_json.get("uri"), result, f"{loc}/uri", "Top Defining Attribute URI")
                self._add_literal_if_present(g, attr_link_bnode, AFFMDL.observedSkosPrefLabel, attr_json.get("skos:prefLabel"), result=result, location=f"{loc}/skos:prefLabel")
                self._add_literal_if_present(g, attr_link_bnode, AFFMDL.contextualConceptWeight, attr_json.get("concept_weight"), XSD.decimal, result=result, location=f"{loc}/concept_weight")
                for type_idx, type_val in enumerate(attr_json.get("type", [])):
                    self._add_literal_if_present(g, attr_link_bnode, AFFMDL.observedConceptType, type_val, result=result, location=f"{loc}/type[{type_idx}]")
            else:
                 result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for a top defining attribute in {input_concept}", loc)


        # Themes
        for theme_idx, theme_json in enumerate(def_data.get("themes", [])):
            theme_loc_base = f"{input_concept}/themes[{theme_idx}]"
            theme_name = theme_json.get("theme_name")
            if not theme_name:
                result.add_issue('WARNING', 'DATA_MISSING', f"Theme name missing for theme at index {theme_idx} in {input_concept}", theme_loc_base)
                continue # Skip this theme if it has no name to identify its diagnostics

            theme_inst_bnode = BNode()
            g.add((aff_def_uri, AFFMDL.hasThemeInstance, theme_inst_bnode))
            g.add((theme_inst_bnode, RDF.type, AFFMDL.ThemeInstance))
            g.add((theme_inst_bnode, skos.prefLabel, Literal(f"Theme: {theme_name} for {norm_concept}")))
            
            self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.themeName, theme_name, result=result, location=f"{theme_loc_base}/theme_name")
            self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.themeType, theme_json.get("theme_type"), result=result, location=f"{theme_loc_base}/theme_type")
            self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.ruleApplied, theme_json.get("rule_applied"), result=result, location=f"{theme_loc_base}/rule_applied")
            self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.normalizedThemeWeight, theme_json.get("normalized_theme_weight"), XSD.decimal, result=result, location=f"{theme_loc_base}/normalized_theme_weight")
            self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.subScoreName, theme_json.get("subScore"), result=result, location=f"{theme_loc_base}/subScore")
            self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.llmSummary, theme_json.get("llm_summary"), result=result, location=f"{theme_loc_base}/llm_summary")

            for attr_idx, attr_json in enumerate(theme_json.get("attributes", [])):
                attr_loc = f"{theme_loc_base}/attributes[{attr_idx}]"
                if attr_json.get("uri"):
                    theme_attr_link_bnode = BNode()
                    g.add((theme_inst_bnode, AFFMDL.hasRankedAttributeLink, theme_attr_link_bnode))
                    g.add((theme_attr_link_bnode, RDF.type, AFFMDL.RankedAttributeLink))
                    self._validate_and_add_uri_link(g, theme_attr_link_bnode, AFFMDL.attributeUri, attr_json.get("uri"), result, f"{attr_loc}/uri", "Theme Attribute URI")
                    self._add_literal_if_present(g, theme_attr_link_bnode, AFFMDL.observedSkosPrefLabel, attr_json.get("skos:prefLabel"), result=result, location=f"{attr_loc}/skos:prefLabel")
                    self._add_literal_if_present(g, theme_attr_link_bnode, AFFMDL.contextualConceptWeight, attr_json.get("concept_weight"), XSD.decimal, result=result, location=f"{attr_loc}/concept_weight")
                    for type_idx_t, type_val_t in enumerate(attr_json.get("type", [])):
                        self._add_literal_if_present(g, theme_attr_link_bnode, AFFMDL.observedConceptType, type_val_t, result=result, location=f"{attr_loc}/type[{type_idx_t}]")
                else:
                    result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for an attribute in theme '{theme_name}' for {input_concept}", attr_loc)
            
            # Theme Processing Diagnostics
            theme_diag_data = self._get_nested(def_data, ["diagnostics", "theme_processing", theme_name], {})
            if theme_diag_data:
                self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.diagThemeLLMAssignedCount, theme_diag_data.get("llm_assigned_count"), XSD.integer, result=result, location=f"{theme_loc_base}/diag/llm_assigned_count")
                self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.diagThemeAttributesAfterWeighting, theme_diag_data.get("attributes_after_weighting"), XSD.integer, result=result, location=f"{theme_loc_base}/diag/attributes_after_weighting")
                self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.diagThemeStatus, theme_diag_data.get("status"), result=result, location=f"{theme_loc_base}/diag/status")
                self._add_literal_if_present(g, theme_inst_bnode, AFFMDL.diagThemeRuleFailed, theme_diag_data.get("rule_failed"), result=result, location=f"{theme_loc_base}/diag/rule_failed")

        # Key Associated Concepts Unthemed
        for idx, unthemed_json in enumerate(def_data.get("key_associated_concepts_unthemed", [])):
            loc = f"{input_concept}/key_associated_concepts_unthemed[{idx}]"
            if unthemed_json.get("uri"):
                unthemed_link_bnode = BNode()
                g.add((aff_def_uri, AFFMDL.hasKeyAssociatedConceptUnthemedLink, unthemed_link_bnode))
                g.add((unthemed_link_bnode, RDF.type, AFFMDL.UnthemedConceptLink))
                self._validate_and_add_uri_link(g, unthemed_link_bnode, AFFMDL.conceptUri, unthemed_json.get("uri"), result, f"{loc}/uri", "Unthemed Concept URI")
                self._add_literal_if_present(g, unthemed_link_bnode, AFFMDL.observedSkosPrefLabel, unthemed_json.get("skos:prefLabel"), result=result, location=f"{loc}/skos:prefLabel")
                self._add_literal_if_present(g, unthemed_link_bnode, AFFMDL.scoreValue, unthemed_json.get("combined_score"), XSD.double, result=result, location=f"{loc}/combined_score")
                for type_idx_u, type_val_u in enumerate(unthemed_json.get("type", [])):
                    self._add_literal_if_present(g, unthemed_link_bnode, AFFMDL.observedConceptType, type_val_u, result=result, location=f"{loc}/type[{type_idx_u}]")
            else:
                result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for a key unthemed concept in {input_concept}", loc)

        # Additional Relevant Subscores
        for idx, subscore_json in enumerate(def_data.get("additional_relevant_subscores", [])):
            loc = f"{input_concept}/additional_relevant_subscores[{idx}]"
            subscore_link_bnode = BNode()
            g.add((aff_def_uri, AFFMDL.hasAdditionalRelevantSubscoreLink, subscore_link_bnode))
            g.add((subscore_link_bnode, RDF.type, AFFMDL.SubscoreWeighting))
            self._add_literal_if_present(g, subscore_link_bnode, AFFMDL.subscoreName, subscore_json.get("subscore_name"), result=result, location=f"{loc}/subscore_name")
            self._add_literal_if_present(g, subscore_link_bnode, AFFMDL.subscoreWeight, subscore_json.get("weight"), XSD.double, result=result, location=f"{loc}/weight")


        # Must Not Have
        for idx, mnh_json in enumerate(def_data.get("must_not_have", [])):
            loc = f"{input_concept}/must_not_have[{idx}]"
            if mnh_json.get("uri"):
                mnh_link_bnode = BNode()
                g.add((aff_def_uri, AFFMDL.hasMustNotHaveLink, mnh_link_bnode))
                g.add((mnh_link_bnode, RDF.type, AFFMDL.MustNotHaveLink))
                self._validate_and_add_uri_link(g, mnh_link_bnode, AFFMDL.conceptUri, mnh_json.get("uri"), result, f"{loc}/uri", "Must Not Have URI")
                self._add_literal_if_present(g, mnh_link_bnode, AFFMDL.observedSkosPrefLabel, mnh_json.get("skos:prefLabel"), result=result, location=f"{loc}/skos:prefLabel")
                self._add_literal_if_present(g, mnh_link_bnode, AFFMDL.identificationSourceScope, mnh_json.get("scope"), result=result, location=f"{loc}/scope")
            else:
                result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for a must_not_have entry in {input_concept}", loc)

        # Processing Metadata
        pm_json = def_data.get("processing_metadata", {})
        if pm_json:
            pm_bnode = BNode()
            g.add((aff_def_uri, AFFMDL.hasProcessingMetadata, pm_bnode))
            g.add((pm_bnode, RDF.type, AFFMDL.ProcessingMetadata))
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingOverallStatus, pm_json.get("status"), result=result, location=f"{input_concept}/processing_metadata/status")
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingEngineVersion, pm_json.get("version"), result=result, location=f"{input_concept}/processing_metadata/version")
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingTimestamp, pm_json.get("timestamp"), XSD.dateTime, result=result, location=f"{input_concept}/processing_metadata/timestamp")
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingTotalDurationSeconds, pm_json.get("total_duration_seconds"), XSD.decimal, result=result, location=f"{input_concept}/processing_metadata/total_duration_seconds")
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingCacheVersionUsed, pm_json.get("cache_version"), result=result, location=f"{input_concept}/processing_metadata/cache_version")
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingLlmProviderUsed, pm_json.get("llm_provider"), result=result, location=f"{input_concept}/processing_metadata/llm_provider")
            self._add_literal_if_present(g, pm_bnode, AFFMDL.processingLlmModelUsed, pm_json.get("llm_model"), result=result, location=f"{input_concept}/processing_metadata/llm_model")

        # Diagnostics Bundle
        diag_json = def_data.get("diagnostics", {})
        if diag_json:
            diag_bundle_bnode = BNode()
            g.add((aff_def_uri, AFFMDL.hasDiagnosticsBundle, diag_bundle_bnode))
            g.add((diag_bundle_bnode, RDF.type, AFFMDL.DiagnosticsBundle))

            self._add_literal_if_present(g, diag_bundle_bnode, AFFMDL.diagLodgingTypeDeterminationResult, self._get_nested(diag_json, ["lodging_type_determination", "result"]), result=result, location=f"{input_concept}/diagnostics/lodging_type_determination/result")
            self._add_literal_if_present(g, diag_bundle_bnode, AFFMDL.diagThemeProcessingDetailsAsJson, json.dumps(self._get_nested(diag_json, ["theme_processing"], {})), result=result, location=f"{input_concept}/diagnostics/theme_processing")
            self._add_literal_if_present(g, diag_bundle_bnode, AFFMDL.diagErrorDetailsString, self._get_nested(diag_json, ["error_details"]), result=result, location=f"{input_concept}/diagnostics/error_details")

            llm_neg_json = self._get_nested(diag_json, ["llm_negation"], {})
            if llm_neg_json:
                llm_neg_bnode = BNode()
                g.add((diag_bundle_bnode, AFFMDL.diagHasLLMNegationDiagnostics, llm_neg_bnode))
                g.add((llm_neg_bnode, RDF.type, AFFMDL.LLMNegationDiagnostics))
                self._add_literal_if_present(g, llm_neg_bnode, AFFMDL.diagLlmNegAttempted, llm_neg_json.get("attempted"), result=result, location=f"{input_concept}/diagnostics/llm_negation/attempted")
                self._add_literal_if_present(g, llm_neg_bnode, AFFMDL.diagLlmNegSuccess, llm_neg_json.get("success"), result=result, location=f"{input_concept}/diagnostics/llm_negation/success")
                self._add_literal_if_present(g, llm_neg_bnode, AFFMDL.diagLlmNegUrisFoundCount, llm_neg_json.get("uris_found"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/llm_negation/uris_found")
                self._add_literal_if_present(g, llm_neg_bnode, AFFMDL.diagLlmNegErrorMessage, llm_neg_json.get("error"), result=result, location=f"{input_concept}/diagnostics/llm_negation/error")
                for uri_idx, checked_uri in enumerate(llm_neg_json.get("candidates_checked", [])):
                    if checked_uri: self._validate_and_add_uri_link(g, llm_neg_bnode, AFFMDL.diagLlmNegCandidateCheckedUri, checked_uri, result, f"{input_concept}/diagnostics/llm_negation/candidates_checked[{uri_idx}]", "LLM Negation Candidate URI")
                for uri_idx, neg_uri in enumerate(llm_neg_json.get("identified_negating_uris", [])):
                    if neg_uri: self._validate_and_add_uri_link(g, llm_neg_bnode, AFFMDL.diagLlmNegIdentifiedNegatingUriValue, neg_uri, result, f"{input_concept}/diagnostics/llm_negation/identified_negating_uris[{uri_idx}]", "LLM Identified Negating URI")
            
            s1_json = self._get_nested(diag_json, ["stage1"], {})
            if s1_json:
                s1_bnode = BNode()
                g.add((diag_bundle_bnode, AFFMDL.diagHasStage1Diagnostics, s1_bnode))
                g.add((s1_bnode, RDF.type, AFFMDL.Stage1Diagnostics))
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1Status, s1_json.get("status"), result=result, location=f"{input_concept}/diagnostics/stage1/status")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1ErrorMessage, s1_json.get("error"), result=result, location=f"{input_concept}/diagnostics/stage1/error")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1OverallSelectionMethod, s1_json.get("selection_method"), result=result, location=f"{input_concept}/diagnostics/stage1/selection_method")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1SbertCandidateCountInitial, s1_json.get("sbert_candidate_count_initial"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/stage1/sbert_candidate_count_initial")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1KeywordCandidateCountInitial, s1_json.get("keyword_candidate_count_initial"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/stage1/keyword_candidate_count_initial")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1UniqueCandidatesBeforeRanking, s1_json.get("unique_candidates_before_ranking"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/stage1/unique_candidates_before_ranking")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1LlmCandidateOutputCount, s1_json.get("llm_candidate_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/stage1/llm_candidate_count")
                self._add_literal_if_present(g, s1_bnode, AFFMDL.diagS1TotalDurationSeconds, s1_json.get("duration_seconds"), XSD.decimal, result=result, location=f"{input_concept}/diagnostics/stage1/duration_seconds")

                exp_json = self._get_nested(s1_json, ["expansion"], {})
                if exp_json:
                    exp_bnode = BNode()
                    g.add((s1_bnode, AFFMDL.diagS1HasExpansionDetails, exp_bnode))
                    g.add((exp_bnode, RDF.type, AFFMDL.ExpansionDiagnostics))
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpAttempted, exp_json.get("attempted"), result=result, location=f"{input_concept}/diagnostics/stage1/expansion/attempted")
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpSuccessful, exp_json.get("successful"), result=result, location=f"{input_concept}/diagnostics/stage1/expansion/successful")
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpTermCount, exp_json.get("count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/stage1/expansion/count")
                    for term_idx, term in enumerate(exp_json.get("terms", [])):
                        self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpTerm, term, result=result, location=f"{input_concept}/diagnostics/stage1/expansion/terms[{term_idx}]")
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpKeywordResultCount, exp_json.get("keyword_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/stage1/expansion/keyword_count")
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpErrorMessage, exp_json.get("error"), result=result, location=f"{input_concept}/diagnostics/stage1/expansion/error")
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpSelectionMethodUsed, exp_json.get("selection_method"), result=result, location=f"{input_concept}/diagnostics/stage1/expansion/selection_method")
                    self._add_literal_if_present(g, exp_bnode, AFFMDL.diagExpDuration, exp_json.get("duration_seconds"), XSD.decimal, result=result, location=f"{input_concept}/diagnostics/stage1/expansion/duration_seconds")

            # ... (Similar detailed mapping for LLMSlottingDiagnostics, RepromptingFallbackDiagnostics, Stage2Diagnostics) ...
            llms_json = self._get_nested(diag_json, ["llm_slotting"], {})
            if llms_json:
                llms_bnode = BNode()
                g.add((diag_bundle_bnode, AFFMDL.diagHasLLMSlottingDiagnostics, llms_bnode))
                g.add((llms_bnode, RDF.type, AFFMDL.LLMSlottingDiagnostics))
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsStatus, llms_json.get("status"), result=result, location=f"{input_concept}/diagnostics/llm_slotting/status")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsErrorMessage, llms_json.get("error"), result=result, location=f"{input_concept}/diagnostics/llm_slotting/error")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsProviderUsed, llms_json.get("llm_provider"), result=result, location=f"{input_concept}/diagnostics/llm_slotting/llm_provider")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsModelUsed, llms_json.get("llm_model"), result=result, location=f"{input_concept}/diagnostics/llm_slotting/llm_model")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsCallAttempted, llms_json.get("llm_call_attempted"), result=result, location=f"{input_concept}/diagnostics/llm_slotting/llm_call_attempted")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsAttemptsMade, llms_json.get("attempts_made"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/llm_slotting/attempts_made")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsCallSuccess, llms_json.get("llm_call_success"), result=result, location=f"{input_concept}/diagnostics/llm_slotting/llm_call_success")
                self._add_literal_if_present(g, llms_bnode, AFFMDL.diagLlmsDurationSeconds, llms_json.get("duration_seconds"), XSD.decimal, result=result, location=f"{input_concept}/diagnostics/llm_slotting/duration_seconds")

            rf_json = self._get_nested(diag_json, ["reprompting_fallback"], {})
            if rf_json:
                rf_bnode = BNode()
                g.add((diag_bundle_bnode, AFFMDL.diagHasRepromptingFallbackDiagnostics, rf_bnode))
                g.add((rf_bnode, RDF.type, AFFMDL.RepromptingFallbackDiagnostics))
                self._add_literal_if_present(g, rf_bnode, AFFMDL.diagRfAttemptsCount, rf_json.get("attempts"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/reprompting_fallback/attempts")
                self._add_literal_if_present(g, rf_bnode, AFFMDL.diagRfSuccessesCount, rf_json.get("successes"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/reprompting_fallback/successes")
                self._add_literal_if_present(g, rf_bnode, AFFMDL.diagRfFailuresCount, rf_json.get("failures"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/reprompting_fallback/failures")

            s2_json = self._get_nested(diag_json, ["stage2"], {})
            if s2_json:
                s2_bnode = BNode()
                g.add((diag_bundle_bnode, AFFMDL.diagHasStage2Diagnostics, s2_bnode))
                g.add((s2_bnode, RDF.type, AFFMDL.Stage2Diagnostics))
                self._add_literal_if_present(g, s2_bnode, AFFMDL.diagS2Status, s2_json.get("status"), result=result, location=f"{input_concept}/diagnostics/stage2/status")
                self._add_literal_if_present(g, s2_bnode, AFFMDL.diagS2DurationSeconds, s2_json.get("duration_seconds"), XSD.decimal, result=result, location=f"{input_concept}/diagnostics/stage2/duration_seconds")
                self._add_literal_if_present(g, s2_bnode, AFFMDL.diagS2ErrorMessage, s2_json.get("error"), result=result, location=f"{input_concept}/diagnostics/stage2/error")
            
            fo_json = self._get_nested(diag_json, ["final_output"], {})
            if fo_json:
                fo_bnode = BNode()
                g.add((diag_bundle_bnode, AFFMDL.diagHasFinalOutputCounts, fo_bnode))
                g.add((fo_bnode, RDF.type, AFFMDL.FinalOutputCounts))
                self._add_literal_if_present(g, fo_bnode, AFFMDL.diagFoMustNotHaveCount, fo_json.get("must_not_have_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/final_output/must_not_have_count")
                self._add_literal_if_present(g, fo_bnode, AFFMDL.diagFoAdditionalSubscoresCount, fo_json.get("additional_subscores_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/final_output/additional_subscores_count")
                self._add_literal_if_present(g, fo_bnode, AFFMDL.diagFoThemesCount, fo_json.get("themes_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/final_output/themes_count")
                self._add_literal_if_present(g, fo_bnode, AFFMDL.diagFoUnthemedConceptsCapturedCount, fo_json.get("unthemed_concepts_captured_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/final_output/unthemed_concepts_captured_count")
                self._add_literal_if_present(g, fo_bnode, AFFMDL.diagFoFailedFallbackThemesCount, fo_json.get("failed_fallback_themes_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/final_output/failed_fallback_themes_count")
                self._add_literal_if_present(g, fo_bnode, AFFMDL.diagFoTopDefiningAttributesCount, fo_json.get("top_defining_attributes_count"), XSD.integer, result=result, location=f"{input_concept}/diagnostics/final_output/top_defining_attributes_count")

        return len(g) - triples_before

    def _collect_rdf_graph_stats(self, graph: Graph, result: ValidationResult):
        """Collects final statistics from the generated graph."""
        result.stats['rdf_graph_stats']['final_triple_count'] = len(graph)
        for s, p, o in graph:
            result.stats['rdf_graph_stats']['unique_subjects'].add(str(s))
            result.stats['rdf_graph_stats']['unique_predicates'].add(str(p))
            if isinstance(o, URIRef) or isinstance(o, BNode):
                result.stats['rdf_graph_stats']['unique_objects_uris'].add(str(o))
            else: # Literal
                result.stats['rdf_graph_stats']['unique_objects_literals'].add(str(o))
        for prefix, namespace_uri in graph.namespaces():
            result.stats['rdf_graph_stats']['namespaces_used'].add(f"{prefix}: {namespace_uri}")


    def convert_and_validate_json_to_rdf(self, json_filepath: str, rdf_output_filepath: str, batch_size: int = 100) -> ValidationResult:
        """
        Main method to convert JSON to RDF, including validation steps.
        This method incorporates the structure of JsonToValidatedRDF.
        """
        result = ValidationResult()

        # Load taxonomies if a directory is provided
        self._load_taxonomies_if_needed(result)
        if result.has_errors() and any(issue.category == 'TAXONOMY' for issue in result.issues if issue.severity == 'ERROR'):
             logger.error("Critical error during taxonomy loading. Aborting conversion.")
             return result

        g = Graph()
        self._bind_namespaces(g)

        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                all_definitions_data = json.load(f)
        except FileNotFoundError:
            result.add_issue('ERROR', 'FORMAT', f"Input JSON file not found: {json_filepath}")
            return result
        except json.JSONDecodeError as e:
            result.add_issue('ERROR', 'FORMAT', f"Error decoding JSON from {json_filepath}: {e}")
            return result

        if not isinstance(all_definitions_data, list):
            all_definitions_data = [all_definitions_data]

        total_definitions = len(all_definitions_data)
        result.stats['conversion_stats']['total_definitions_in_json'] = total_definitions
        logger.info(f"Starting conversion of {total_definitions} definition(s) from {json_filepath}")

        # For now, let's process without explicit batching for simplicity of merging graph.
        # Batching primarily helps if memory becomes an issue for very large single graph objects.
        # If individual definitions are large, batching reads might be useful, but here we read all JSON first.

        for def_data in tqdm(all_definitions_data, desc="Processing Definitions"):
            triples_added_for_def = self._process_definition_to_rdf(g, def_data, result)
            result.stats['conversion_stats']['total_triples_generated'] += triples_added_for_def
            if triples_added_for_def > 0 or def_data.get("input_concept"): # Count if attempted
                 result.stats['conversion_stats']['processed_affinity_definitions'] += 1
        
        # Collect final graph statistics
        self._collect_rdf_graph_stats(g, result)
        
        # Final save
        unique_rdf_output_filepath = self._get_unique_output_path(rdf_output_filepath)
        try:
            g.serialize(destination=unique_rdf_output_filepath, format="turtle")
            logger.info(f"Successfully converted JSON to RDF Turtle: {unique_rdf_output_filepath}")
            logger.info(f"Total triples generated: {result.stats['conversion_stats']['total_triples_generated']}")
        except Exception as e:
            result.add_issue('ERROR', 'FORMAT', f"Error serializing RDF graph: {e}")
        
        return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python json_to_expedia_affinity_rdf.py <input_json_file> <output_turtle_file> [taxonomy_directory] [batch_size (not currently used for processing)]")
        print("Example: python json_to_expedia_affinity_rdf.py babyfriendly_output.json babyfriendly_definition.ttl ./path/to/taxonomies")
        sys.exit(1)

    input_json = sys.argv[1]
    output_rdf = sys.argv[2]
    taxonomy_folder = sys.argv[3] if len(sys.argv) > 3 else None # Optional
    # batch_size_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 100 # Batching of JSON read not implemented yet

    if not Path(input_json).exists():
        logger.error(f"Input JSON file does not exist: {input_json}")
        sys.exit(1)

    converter = JsonToExpediaAffinityRDF(taxonomy_dir=taxonomy_folder)
    validation_report = converter.convert_and_validate_json_to_rdf(input_json, output_rdf)

    report_path = Path(output_rdf).parent / f"{Path(output_rdf).stem}_validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f_report:
        f_report.write(validation_report.to_json(indent=2))
    logger.info(f"Validation report saved to: {report_path}")

    if validation_report.has_errors():
        logger.error("Conversion completed with errors. See report for details.")
        sys.exit(1)
    else:
        logger.info("Conversion completed successfully.")
        sys.exit(0)