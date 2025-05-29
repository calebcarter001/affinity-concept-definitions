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

# --- Define Namespaces (must align with ExpediaAffinityDefinitionModel.ttl v1.2.2) ---
AFFMDL = Namespace("urn:com:expedia:affinitymodel#")
AFFMDL_INST = Namespace("urn:com:expedia:affinitymodel:instance:")

# --- Mappings for JSON 'type' strings to Ontology Class URIs ---
JSON_TYPE_TO_ONTOLOGY_CLASS = {
    "Eg Travel Core Concept": Namespace("urn:expediagroup:ontologies:core:#")["EgTravelCoreConcept"],
    "Category": Namespace("urn:expediagroup:ontologies:acsCategories:#")["Category"], # Corrected from acs:#
    "Base Term": Namespace("urn:expediagroup:taxonomies:acsBaseTerms:#")["BaseTerm"],
    "Attribute Template": Namespace("urn:expediagroup:taxonomies:acs:#")["AttributeTemplate"],
    "Attribute Template Detail": Namespace("urn:expediagroup:taxonomies:acs:#")["AttributeTemplateDetail"],
    "Place": Namespace("urn:expediagroup:taxonomies:places:#")["Place"],
    "Enumerations": Namespace("urn:expediagroup:taxonomies:acsEnumerations:#")["Enumeration"],
    "Lcssection": Namespace("urn:expe:taxo:text:")["Lcssection"],
    "Tmptconcept": Namespace("urn:expediagroup:taxonomies:tmpt:#")["TMPTConcept"], # Assuming maps to this
    "Activities Concept": Namespace("urn:expediagroup:ontologies:core:#")["ActivitiesConcept"], # Assuming from core
    "Product": Namespace("urn:expediagroup:ontologies:product_ontology:#")["product"],
    "Checkout Signal Detail Type": Namespace("urn:expe:taxo:checkout:")["SignalDetailType"], # Placeholder, verify
    "Property Image Tag": Namespace("urn:expe:taxo:property-media:concepts:")["PropertyImageTag"], # Placeholder
    "Pcs": Namespace("urn:expediagroup:taxonomies:acsPCS:#")["PCS"], # Placeholder, verify
    "Event": Namespace("urn:expe:taxo:events:")["Event"], # Placeholder
    "Review Question Category": Namespace("urn:expe:taxo:review_categories:")["ReviewQuestionCategory"], # Placeholder
    "Eg Concept": Namespace("urn:expediagroup:ontologies:eg-skos-extensions:#")["EgConcept"]
    # Add more mappings as identified from your data and target ontologies
}


@dataclass
class ValidationIssue:
    severity: str
    category: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

class ValidationResult:
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.stats: Dict[str, Any] = {
            'conversion_stats': {
                'total_definitions_in_json': 0,
                'processed_affinity_definitions': 0,
                'total_triples_generated': 0,
            },
            'rdf_graph_stats': {
                'final_triple_count': 0,
                'unique_subjects': set(),
                'unique_predicates': set(),
                'unique_objects_uris': set(),
                'unique_objects_literals': set(),
                'namespaces_used': set(),
            },
            'taxonomy_validation_stats': {
                'total_taxonomies_loaded': 0,
                'total_valid_taxonomy_uris': 0,
                'concepts_by_taxonomy_source_file': {},
                'unvalidated_concept_uris_found': set(),
                'validated_concept_uris_count': 0,
            }
        }

    def add_issue(self, severity: str, category: str, message: str, location: Optional[str] = None, suggestion: Optional[str] = None):
        self.issues.append(ValidationIssue(severity, category, message, location, suggestion))
        if severity == 'ERROR':
            logger.error(f"VALIDATION ERROR: {message} (Location: {location})")
        elif severity == 'WARNING':
            logger.warning(f"VALIDATION WARNING: {message} (Location: {location})")


    def has_errors(self) -> bool:
        return any(issue.severity == 'ERROR' for issue in self.issues)

    def to_dict(self) -> Dict:
        serializable_stats = json.loads(json.dumps(self.stats, default=lambda o: list(o) if isinstance(o, set) else str(o)))
        for key_group in ['rdf_graph_stats', 'taxonomy_validation_stats']:
            for key, value in list(serializable_stats.get(key_group, {}).items()):
                if isinstance(self.stats[key_group].get(key), set):
                    serializable_stats[key_group][f"{key}_count"] = len(self.stats[key_group][key])
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

    def _add_literal_if_present(self, g: Graph, subject: Union[URIRef, BNode], predicate: URIRef, value: Any, datatype: Optional[URIRef] = None, result: Optional[ValidationResult] = None, location: Optional[str] = None):
        if value is not None:
            try:
                if isinstance(value, bool):
                    lit = Literal(value, datatype=XSD.boolean)
                elif datatype == XSD.integer and isinstance(value, (int, float, str)): # Allow float/str conversion
                    lit = Literal(int(float(value)), datatype=XSD.integer)
                elif datatype == XSD.decimal and isinstance(value, (int, float, str)):
                    lit = Literal(float(value), datatype=XSD.decimal)
                elif datatype == XSD.double and isinstance(value, (int, float, str)):
                    lit = Literal(float(value), datatype=XSD.double)
                elif datatype == XSD.dateTime and isinstance(value, str):
                    try:
                        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
                        lit = Literal(value, datatype=XSD.dateTime)
                    except ValueError:
                        if result: result.add_issue('WARNING', 'DATA_TYPE', f"Invalid dateTime format for {predicate}: '{value}'. Storing as string.", location)
                        lit = Literal(str(value))
                elif datatype:
                    lit = Literal(str(value), datatype=datatype)
                else:
                    lit = Literal(str(value))
                g.add((subject, predicate, lit))
            except (ValueError, TypeError) as e:
                msg = f"Failed to cast value '{value}' (type: {type(value)}) to {str(datatype) if datatype else 'string'} for {predicate}: {e}"
                if result: result.add_issue('WARNING', 'DATA_TYPE', msg, location, suggestion="Check data type in JSON or ontology.")
                else: logger.warning(msg)

    def _get_nested(self, data: Dict, keys: List[str], default: Any = None) -> Any:
        current_level = data
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                return default
        return current_level

    def _safe_uri_local_name(self, text: str, context: str = "") -> str:
        if not text: return f"unknown_{context}" if context else "unknown"
        s = str(text).strip().replace(" ", "_").replace("/", "_").replace(":", "_").replace("#", "_").replace("(", "").replace(")", "")
        s = re.sub(r'(?u)[^\w-]', '', s)
        name = quote(s[:80])
        return f"{name}_{context}" if context else name

    def _get_unique_output_path(self, base_path_str: str) -> str:
        path = Path(base_path_str)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return str(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{path.stem}_{timestamp}{path.suffix}"
        unique_path = path.parent / new_filename
        logger.info(f"Output file {path} already exists. Using unique filename: {unique_path}")
        return str(unique_path)

    def _load_one_taxonomy(self, file_path: Path) -> Dict[str, Any]:
        try:
            graph = Graph()
            fmt = rdflib.util.guess_format(str(file_path))
            if fmt is None and file_path.suffix.lower() in ['.rdf', '.xml']:
                fmt = 'xml'
            graph.parse(str(file_path), format=fmt)
            concepts = set()
            uri_map = defaultdict(set)
            for s, _, o in graph:
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
            self._taxonomies_loaded = True
            return

        rdf_suffixes = ['.rdf', '.owl', '.ttl', '.n3', '.nt', '.xml']
        rdf_files = [p for p in self.taxonomy_dir.rglob('*') if p.is_file() and p.suffix.lower() in rdf_suffixes]
        if not rdf_files:
            result.add_issue('WARNING', 'TAXONOMY', f"No recognized RDF files found in taxonomy directory: {self.taxonomy_dir}")
            self._taxonomies_loaded = True
            return

        logger.info(f"Loading {len(rdf_files)} taxonomy files from {self.taxonomy_dir}...")
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

    def _validate_and_add_uri_link(self, g: Graph, subject: Union[URIRef, BNode], predicate: URIRef, uri_value: Optional[str], result: ValidationResult, location: str, concept_description: str) -> Optional[URIRef]:
        if not uri_value:
            result.add_issue('WARNING', 'DATA_MISSING', f"Missing URI for {concept_description}", location)
            return None

        target_uri_ref = URIRef(uri_value)
        g.add((subject, predicate, target_uri_ref))

        if not (uri_value.startswith("urn:") or uri_value.startswith("http:") or uri_value.startswith("https:")):
             result.add_issue('WARNING', 'FORMAT', f"Unusual URI scheme for {concept_description}: {uri_value}", location, "Ensure URI uses standard schemes.")

        if self.taxonomy_dir and self.all_valid_taxonomy_uris and uri_value not in self.all_valid_taxonomy_uris:
            result.add_issue('WARNING', 'TAXONOMY', f"{concept_description} URI '{uri_value}' not found in loaded taxonomies.", location, "Verify URI or taxonomy sources.")
            result.stats['taxonomy_validation_stats']['unvalidated_concept_uris_found'].add(uri_value)
        elif self.taxonomy_dir and self.all_valid_taxonomy_uris:
             result.stats['taxonomy_validation_stats']['validated_concept_uris_count'] +=1
        return target_uri_ref

    def _add_concept_details_to_external_uri(self, g: Graph, concept_uri_ref: URIRef, concept_json_data: Dict, result: ValidationResult, location_base: str):
        """Adds skos:prefLabel and rdf:type to an external concept URI if present in JSON."""
        # Use skos:prefLabel from the JSON if available for the external concept
        # This assumes the JSON's "skos:prefLabel" is the authoritative one for that URI.
        # If the label is in JSON under a different key (e.g., "label"), adjust accordingly.
        label_key_to_check = "skos:prefLabel" if "skos:prefLabel" in concept_json_data else "label"

        if concept_json_data.get(label_key_to_check):
            self._add_literal_if_present(g, concept_uri_ref, SKOS.prefLabel, concept_json_data[label_key_to_check], result=result, location=f"{location_base}/{label_key_to_check}")

        json_types = concept_json_data.get("type", []) # JSON 'type' for attributes
        if not json_types: # For travel_category, JSON 'types'
            json_types = concept_json_data.get("types", [])

        if not isinstance(json_types, list): json_types = [json_types]

        for type_str in json_types:
            if type_str:
                class_uri = JSON_TYPE_TO_ONTOLOGY_CLASS.get(type_str)
                if class_uri:
                    g.add((concept_uri_ref, RDF.type, class_uri))
                else:
                    result.add_issue('WARNING', 'TAXONOMY', f"No ontology class mapping for JSON type '{type_str}' for URI {concept_uri_ref}", f"{location_base}/type", f"Add mapping for '{type_str}' to JSON_TYPE_TO_ONTOLOGY_CLASS.")


    def _process_definition_to_rdf(self, g: Graph, def_data: Dict, result: ValidationResult) -> int:
        triples_before = len(g)
        input_concept = def_data.get("input_concept")
        norm_concept = def_data.get("normalized_concept", input_concept)
        inst_local_name_base = self._safe_uri_local_name(norm_concept)
        aff_def_uri = AFFMDL_INST[f"{inst_local_name_base}_AffinityDefinition"]

        g.add((aff_def_uri, RDF.type, AFFMDL.AffinityDefinition))
        g.add((aff_def_uri, rdfs.label, Literal(f"Affinity Definition for '{input_concept}'")))
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.inputConceptLabel, input_concept, result=result, location=f"{aff_def_uri}/inputConceptLabel")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.normalizedConceptLabel, norm_concept, result=result, location=f"{aff_def_uri}/normalizedConceptLabel")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.applicableLodgingTypes, def_data.get("applicable_lodging_types"), result=result, location=f"{aff_def_uri}/applicableLodgingTypes")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.requiresGeoCheck, def_data.get("requires_geo_check"), result=result, location=f"{aff_def_uri}/requiresGeoCheck")
        if def_data.get("failed_fallback_themes") is not None:
             self._add_literal_if_present(g, aff_def_uri, AFFMDL.failedFallbackThemesAsJson, json.dumps(def_data.get("failed_fallback_themes")), result=result, location=f"{aff_def_uri}/failedFallbackThemesAsJson")
        self._add_literal_if_present(g, aff_def_uri, AFFMDL.affinityScoreTotalAllocated, def_data.get("affinity_score_total_allocated"), XSD.decimal, result=result, location=f"{aff_def_uri}/affinityScoreTotalAllocated")

        # Travel Category Link
        tc_json = def_data.get("travel_category", {})
        if tc_json and tc_json.get("uri"):
            tc_link_uri = AFFMDL_INST[f"{inst_local_name_base}_TravelCategoryLink"]
            g.add((aff_def_uri, AFFMDL.hasTravelCategoryLink, tc_link_uri))
            g.add((tc_link_uri, RDF.type, AFFMDL.TravelCategoryLink))
            g.add((tc_link_uri, rdfs.label, Literal(f"Anchor Details for '{norm_concept}' Definition")))
            
            linked_tc_uri_ref = self._validate_and_add_uri_link(g, tc_link_uri, AFFMDL.linkedConceptUri, tc_json.get("uri"), result, f"{aff_def_uri}/travel_category/uri", "Travel Category URI")
            if linked_tc_uri_ref: # If URI was validly added
                 self._add_concept_details_to_external_uri(g, linked_tc_uri_ref, tc_json, result, f"{aff_def_uri}/travel_category")
            
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.observedLabel, tc_json.get("label"), result=result, location=f"{tc_link_uri}/observedLabel")
            for idx, type_val in enumerate(tc_json.get("types", [])): # Store observed types on the link node
                self._add_literal_if_present(g, tc_link_uri, AFFMDL.observedType, type_val, result=result, location=f"{tc_link_uri}/observedType[{idx}]")
            for idx, def_val in enumerate(tc_json.get("definitions", [])): # Store observed definitions on the link node
                self._add_literal_if_present(g, tc_link_uri, AFFMDL.observedDefinition, def_val, result=result, location=f"{tc_link_uri}/observedDefinition[{idx}]")
            
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualSbertScore, tc_json.get("sbert_score"), XSD.double)
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualKeywordScore, tc_json.get("keyword_score"), XSD.double)
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualCombinedScore, tc_json.get("combined_score"), XSD.double)
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualCombinedScoreUnbiased, tc_json.get("combined_score_unbiased"), XSD.double)
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualIsBiased, tc_json.get("biased"))
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualEffectiveAlpha, tc_json.get("effective_alpha"), XSD.double)
            self._add_literal_if_present(g, tc_link_uri, AFFMDL.contextualBiasReason, tc_json.get("bias_reason"))
        elif tc_json:
             result.add_issue('WARNING', 'DATA_MISSING', f"Travel category data present but URI is missing for {input_concept}", f"{aff_def_uri}/travel_category")

        # Top Defining Attributes
        for idx, attr_json in enumerate(def_data.get("top_defining_attributes", [])):
            loc_base = f"{aff_def_uri}/topDefiningAttribute[{idx}]"
            if attr_json.get("uri"):
                attr_uri_str = attr_json["uri"]
                attr_label_for_uri = attr_json.get("skos:prefLabel", attr_uri_str.split('#')[-1].split('/')[-1])
                attr_link_local_name = self._safe_uri_local_name(attr_label_for_uri, f"TopAttr{idx}")
                attr_link_uri = AFFMDL_INST[f"{inst_local_name_base}_TopDefiningAttribute_{attr_link_local_name}"]
                
                g.add((aff_def_uri, AFFMDL.hasTopDefiningAttributeLink, attr_link_uri))
                g.add((attr_link_uri, RDF.type, AFFMDL.RankedAttributeLink))
                g.add((attr_link_uri, rdfs.label, Literal(f"Top Attribute Link: {attr_label_for_uri}")))

                linked_attr_uri_ref = self._validate_and_add_uri_link(g, attr_link_uri, AFFMDL.attributeUri, attr_uri_str, result, f"{loc_base}/uri", "Top Defining Attribute URI")
                if linked_attr_uri_ref:
                     self._add_concept_details_to_external_uri(g, linked_attr_uri_ref, attr_json, result, loc_base)

                self._add_literal_if_present(g, attr_link_uri, AFFMDL.observedSkosPrefLabel, attr_json.get("skos:prefLabel"))
                self._add_literal_if_present(g, attr_link_uri, AFFMDL.contextualConceptWeight, attr_json.get("concept_weight"), XSD.decimal)
                for type_idx, type_val in enumerate(attr_json.get("type", [])):
                    self._add_literal_if_present(g, attr_link_uri, AFFMDL.observedConceptType, type_val)
            else:
                 result.add_issue('WARNING', 'DATA_MISSING', "URI missing for a top defining attribute.", loc_base)
        
        # Themes
        for theme_idx, theme_json in enumerate(def_data.get("themes", [])):
            theme_name = theme_json.get("theme_name")
            theme_loc_base = f"{aff_def_uri}/themes[{theme_idx}({theme_name or 'Unnamed'})]"
            if not theme_name:
                result.add_issue('WARNING', 'DATA_MISSING', f"Theme name missing for theme at index {theme_idx}", theme_loc_base)
                # Continue processing other aspects of the theme if possible, or skip
            
            theme_inst_local_name = self._safe_uri_local_name(theme_name or f"UnnamedTheme{theme_idx}", "Theme")
            theme_inst_uri = AFFMDL_INST[f"{inst_local_name_base}_{theme_inst_local_name}"]
            
            g.add((aff_def_uri, AFFMDL.hasThemeInstance, theme_inst_uri))
            g.add((theme_inst_uri, RDF.type, AFFMDL.ThemeInstance))
            g.add((theme_inst_uri, rdfs.label, Literal(f"Theme: {theme_name or 'Unnamed'} for {norm_concept}")))
            
            self._add_literal_if_present(g, theme_inst_uri, AFFMDL.themeName, theme_name)
            self._add_literal_if_present(g, theme_inst_uri, AFFMDL.themeType, theme_json.get("theme_type"))
            self._add_literal_if_present(g, theme_inst_uri, AFFMDL.ruleApplied, theme_json.get("rule_applied"))
            self._add_literal_if_present(g, theme_inst_uri, AFFMDL.normalizedThemeWeight, theme_json.get("normalized_theme_weight"), XSD.decimal)
            self._add_literal_if_present(g, theme_inst_uri, AFFMDL.subScoreName, theme_json.get("subScore"))
            self._add_literal_if_present(g, theme_inst_uri, AFFMDL.llmSummary, theme_json.get("llm_summary"))

            for attr_idx, attr_json in enumerate(theme_json.get("attributes", [])):
                attr_loc_base = f"{theme_inst_uri}/attribute[{attr_idx}]"
                if attr_json.get("uri"):
                    attr_uri_str = attr_json["uri"]
                    attr_theme_local_name = self._safe_uri_local_name(attr_json.get("skos:prefLabel", attr_uri_str.split('#')[-1].split('/')[-1]), f"ThemeAttr{attr_idx}")
                    theme_attr_link_uri = AFFMDL_INST[f"{inst_local_name}_{theme_inst_local_name}_Attr_{attr_theme_local_name}"]

                    g.add((theme_inst_uri, AFFMDL.hasRankedAttributeLink, theme_attr_link_uri))
                    g.add((theme_attr_link_uri, RDF.type, AFFMDL.RankedAttributeLink))
                    g.add((theme_attr_link_uri, rdfs.label, Literal(f"AttrLink: {attr_json.get('skos:prefLabel')} in Theme '{theme_name}'")))
                    
                    linked_attr_uri_ref_theme = self._validate_and_add_uri_link(g, theme_attr_link_uri, AFFMDL.attributeUri, attr_uri_str, result, f"{attr_loc_base}/uri", "Theme Attribute URI")
                    if linked_attr_uri_ref_theme:
                        self._add_concept_details_to_external_uri(g, linked_attr_uri_ref_theme, attr_json, result, attr_loc_base)

                    self._add_literal_if_present(g, theme_attr_link_uri, AFFMDL.observedSkosPrefLabel, attr_json.get("skos:prefLabel"))
                    self._add_literal_if_present(g, theme_attr_link_uri, AFFMDL.contextualConceptWeight, attr_json.get("concept_weight"), XSD.decimal)
                    for type_val in attr_json.get("type", []):
                        self._add_literal_if_present(g, theme_attr_link_uri, AFFMDL.observedConceptType, type_val)
                else:
                    result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for attribute in theme '{theme_name}'", attr_loc_base)

            theme_diag_data = self._get_nested(def_data, ["diagnostics", "theme_processing", theme_name], {})
            if theme_diag_data:
                self._add_literal_if_present(g, theme_inst_uri, AFFMDL.diagThemeLLMAssignedCount, theme_diag_data.get("llm_assigned_count"), XSD.integer)
                self._add_literal_if_present(g, theme_inst_uri, AFFMDL.diagThemeAttributesAfterWeighting, theme_diag_data.get("attributes_after_weighting"), XSD.integer)
                self._add_literal_if_present(g, theme_inst_uri, AFFMDL.diagThemeStatus, theme_diag_data.get("status"))
                self._add_literal_if_present(g, theme_inst_uri, AFFMDL.diagThemeRuleFailed, theme_diag_data.get("rule_failed"))
        
        # Key Associated Concepts Unthemed
        for idx, unthemed_json in enumerate(def_data.get("key_associated_concepts_unthemed", [])):
            loc_base = f"{aff_def_uri}/keyAssociatedConceptUnthemed[{idx}]"
            if unthemed_json.get("uri"):
                unthemed_uri_str = unthemed_json["uri"]
                unthemed_local_name = self._safe_uri_local_name(unthemed_json.get("skos:prefLabel", unthemed_uri_str.split('#')[-1].split('/')[-1]), f"Unthemed{idx}")
                unthemed_link_uri = AFFMDL_INST[f"{inst_local_name_base}_UnthemedConcept_{unthemed_local_name}"]

                g.add((aff_def_uri, AFFMDL.hasKeyAssociatedConceptUnthemedLink, unthemed_link_uri))
                g.add((unthemed_link_uri, RDF.type, AFFMDL.UnthemedConceptLink))
                g.add((unthemed_link_uri, rdfs.label, Literal(f"Unthemed Link: {unthemed_json.get('skos:prefLabel')}")))
                
                linked_unthemed_uri_ref = self._validate_and_add_uri_link(g, unthemed_link_uri, AFFMDL.conceptUri, unthemed_uri_str, result, f"{loc_base}/uri", "Unthemed Concept URI")
                if linked_unthemed_uri_ref:
                    self._add_concept_details_to_external_uri(g, linked_unthemed_uri_ref, unthemed_json, result, loc_base)
                
                self._add_literal_if_present(g, unthemed_link_uri, AFFMDL.observedSkosPrefLabel, unthemed_json.get("skos:prefLabel"))
                self._add_literal_if_present(g, unthemed_link_uri, AFFMDL.scoreValue, unthemed_json.get("combined_score"), XSD.double)
                for type_val in unthemed_json.get("type", []):
                     self._add_literal_if_present(g, unthemed_link_uri, AFFMDL.observedConceptType, type_val)
            else:
                result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for a key unthemed concept.", loc_base)

        # Additional Relevant Subscores
        for idx, subscore_json in enumerate(def_data.get("additional_relevant_subscores", [])):
            subscore_name = subscore_json.get("subscore_name")
            loc_base = f"{aff_def_uri}/additionalRelevantSubscore[{idx}]"
            if subscore_name:
                subscore_local_name = self._safe_uri_local_name(subscore_name, f"Subscore{idx}")
                subscore_link_uri = AFFMDL_INST[f"{inst_local_name_base}_Subscore_{subscore_local_name}"]
                
                g.add((aff_def_uri, AFFMDL.hasAdditionalRelevantSubscoreLink, subscore_link_uri))
                g.add((subscore_link_uri, RDF.type, AFFMDL.SubscoreWeighting))
                g.add((subscore_link_uri, rdfs.label, Literal(f"Subscore Weighting: {subscore_name}")))
                self._add_literal_if_present(g, subscore_link_uri, AFFMDL.subscoreName, subscore_name)
                self._add_literal_if_present(g, subscore_link_uri, AFFMDL.subscoreWeight, subscore_json.get("weight"), XSD.double)
            else:
                result.add_issue('WARNING', 'DATA_MISSING', f"Subscore name missing.", loc_base)

        # Must Not Have
        for idx, mnh_json in enumerate(def_data.get("must_not_have", [])):
            loc_base = f"{aff_def_uri}/mustNotHave[{idx}]"
            if mnh_json.get("uri"):
                mnh_uri_str = mnh_json["uri"]
                mnh_local_name = self._safe_uri_local_name(mnh_json.get("skos:prefLabel", mnh_uri_str.split('#')[-1].split('/')[-1]), f"MustNotHave{idx}")
                mnh_link_uri = AFFMDL_INST[f"{inst_local_name_base}_MustNotHave_{mnh_local_name}"]

                g.add((aff_def_uri, AFFMDL.hasMustNotHaveLink, mnh_link_uri))
                g.add((mnh_link_uri, RDF.type, AFFMDL.MustNotHaveLink))
                g.add((mnh_link_uri, rdfs.label, Literal(f"Must Not Have Link: {mnh_json.get('skos:prefLabel')}")))

                linked_mnh_uri_ref = self._validate_and_add_uri_link(g, mnh_link_uri, AFFMDL.conceptUri, mnh_uri_str, result, f"{loc_base}/uri", "Must Not Have URI")
                if linked_mnh_uri_ref:
                    self._add_concept_details_to_external_uri(g, linked_mnh_uri_ref, mnh_json, result, loc_base)

                self._add_literal_if_present(g, mnh_link_uri, AFFMDL.observedSkosPrefLabel, mnh_json.get("skos:prefLabel"))
                self._add_literal_if_present(g, mnh_link_uri, AFFMDL.identificationSourceScope, mnh_json.get("scope"))
            else:
                result.add_issue('WARNING', 'DATA_MISSING', f"URI missing for a must_not_have entry.", loc_base)
        
        # Processing Metadata
        pm_json = def_data.get("processing_metadata", {})
        if pm_json:
            pm_uri = AFFMDL_INST[f"{inst_local_name_base}_ProcessingMetadata"]
            g.add((aff_def_uri, AFFMDL.hasProcessingMetadata, pm_uri))
            g.add((pm_uri, RDF.type, AFFMDL.ProcessingMetadata))
            g.add((pm_uri, rdfs.label, Literal(f"Processing Metadata for {norm_concept} definition")))
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingOverallStatus, pm_json.get("status"))
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingEngineVersion, pm_json.get("version"))
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingTimestamp, pm_json.get("timestamp"), XSD.dateTime)
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingTotalDurationSeconds, pm_json.get("total_duration_seconds"), XSD.decimal)
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingCacheVersionUsed, pm_json.get("cache_version"))
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingLlmProviderUsed, pm_json.get("llm_provider"))
            self._add_literal_if_present(g, pm_uri, AFFMDL.processingLlmModelUsed, pm_json.get("llm_model"))

        # Diagnostics Bundle
        diag_json = def_data.get("diagnostics", {})
        if diag_json:
            diag_bundle_uri = AFFMDL_INST[f"{inst_local_name_base}_DiagnosticsBundle"]
            g.add((aff_def_uri, AFFMDL.hasDiagnosticsBundle, diag_bundle_uri))
            g.add((diag_bundle_uri, RDF.type, AFFMDL.DiagnosticsBundle))
            g.add((diag_bundle_uri, rdfs.label, Literal(f"Diagnostics for {norm_concept} definition")))

            self._add_literal_if_present(g, diag_bundle_uri, AFFMDL.diagLodgingTypeDeterminationResult, self._get_nested(diag_json, ["lodging_type_determination", "result"]))
            self._add_literal_if_present(g, diag_bundle_uri, AFFMDL.diagThemeProcessingDetailsAsJson, json.dumps(self._get_nested(diag_json, ["theme_processing"], {})))
            self._add_literal_if_present(g, diag_bundle_uri, AFFMDL.diagErrorDetailsString, self._get_nested(diag_json, ["error_details"]))

            # LLM Negation Diagnostics
            llm_neg_json = self._get_nested(diag_json, ["llm_negation"], {})
            if llm_neg_json:
                llm_neg_uri = AFFMDL_INST[f"{inst_local_name_base}_LLMNegationDiagnostics"]
                g.add((diag_bundle_uri, AFFMDL.diagHasLLMNegationDiagnostics, llm_neg_uri))
                g.add((llm_neg_uri, RDF.type, AFFMDL.LLMNegationDiagnostics))
                g.add((llm_neg_uri, rdfs.label, Literal(f"LLM Negation Diagnostics for {norm_concept}")))
                self._add_literal_if_present(g, llm_neg_uri, AFFMDL.diagLlmNegAttempted, llm_neg_json.get("attempted"))
                self._add_literal_if_present(g, llm_neg_uri, AFFMDL.diagLlmNegSuccess, llm_neg_json.get("success"))
                self._add_literal_if_present(g, llm_neg_uri, AFFMDL.diagLlmNegUrisFoundCount, llm_neg_json.get("uris_found"), XSD.integer)
                self._add_literal_if_present(g, llm_neg_uri, AFFMDL.diagLlmNegErrorMessage, llm_neg_json.get("error"))
                for uri_idx, checked_uri in enumerate(llm_neg_json.get("candidates_checked", [])):
                    if checked_uri: self._validate_and_add_uri_link(g, llm_neg_uri, AFFMDL.diagLlmNegCandidateCheckedUri, checked_uri, result, f"{llm_neg_uri}/candidateChecked[{uri_idx}]", "LLM Negation Candidate URI")
                for uri_idx, neg_uri in enumerate(llm_neg_json.get("identified_negating_uris", [])):
                    if neg_uri: self._validate_and_add_uri_link(g, llm_neg_uri, AFFMDL.diagLlmNegIdentifiedNegatingUriValue, neg_uri, result, f"{llm_neg_uri}/identifiedNegating[{uri_idx}]", "LLM Identified Negating URI")
            
            # Stage 1 Diagnostics
            s1_json = self._get_nested(diag_json, ["stage1"], {})
            if s1_json:
                s1_uri = AFFMDL_INST[f"{inst_local_name_base}_Stage1Diagnostics"]
                g.add((diag_bundle_uri, AFFMDL.diagHasStage1Diagnostics, s1_uri))
                g.add((s1_uri, RDF.type, AFFMDL.Stage1Diagnostics))
                g.add((s1_uri, rdfs.label, Literal(f"Stage 1 Diagnostics for {norm_concept}")))
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1Status, s1_json.get("status"))
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1ErrorMessage, s1_json.get("error"))
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1OverallSelectionMethod, s1_json.get("selection_method"))
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1SbertCandidateCountInitial, s1_json.get("sbert_candidate_count_initial"), XSD.integer)
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1KeywordCandidateCountInitial, s1_json.get("keyword_candidate_count_initial"), XSD.integer)
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1UniqueCandidatesBeforeRanking, s1_json.get("unique_candidates_before_ranking"), XSD.integer)
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1LlmCandidateOutputCount, s1_json.get("llm_candidate_count"), XSD.integer)
                self._add_literal_if_present(g, s1_uri, AFFMDL.diagS1TotalDurationSeconds, s1_json.get("duration_seconds"), XSD.decimal)

                exp_json = self._get_nested(s1_json, ["expansion"], {})
                if exp_json:
                    exp_uri = AFFMDL_INST[f"{inst_local_name_base}_ExpansionDiagnostics"]
                    g.add((s1_uri, AFFMDL.diagS1HasExpansionDetails, exp_uri))
                    g.add((exp_uri, RDF.type, AFFMDL.ExpansionDiagnostics))
                    g.add((exp_uri, rdfs.label, Literal(f"Expansion Diagnostics for {norm_concept}")))
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpAttempted, exp_json.get("attempted"))
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpSuccessful, exp_json.get("successful"))
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpTermCount, exp_json.get("count"), XSD.integer)
                    for term_idx, term in enumerate(exp_json.get("terms", [])):
                        if term: self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpTerm, term, location=f"{exp_uri}/term[{term_idx}]")
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpKeywordResultCount, exp_json.get("keyword_count"), XSD.integer)
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpErrorMessage, exp_json.get("error"))
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpSelectionMethodUsed, exp_json.get("selection_method"))
                    self._add_literal_if_present(g, exp_uri, AFFMDL.diagExpDuration, exp_json.get("duration_seconds"), XSD.decimal)

            # LLM Slotting Diagnostics
            llms_json = self._get_nested(diag_json, ["llm_slotting"], {})
            if llms_json:
                llms_uri = AFFMDL_INST[f"{inst_local_name_base}_LLMSlottingDiagnostics"]
                g.add((diag_bundle_uri, AFFMDL.diagHasLLMSlottingDiagnostics, llms_uri))
                g.add((llms_uri, RDF.type, AFFMDL.LLMSlottingDiagnostics))
                g.add((llms_uri, rdfs.label, Literal(f"LLM Slotting Diagnostics for {norm_concept}")))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsStatus, llms_json.get("status"))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsErrorMessage, llms_json.get("error"))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsProviderUsed, llms_json.get("llm_provider"))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsModelUsed, llms_json.get("llm_model"))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsCallAttempted, llms_json.get("llm_call_attempted"))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsAttemptsMade, llms_json.get("attempts_made"), XSD.integer)
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsCallSuccess, llms_json.get("llm_call_success"))
                self._add_literal_if_present(g, llms_uri, AFFMDL.diagLlmsDurationSeconds, llms_json.get("duration_seconds"), XSD.decimal)

            # Reprompting Fallback Diagnostics
            rf_json = self._get_nested(diag_json, ["reprompting_fallback"], {})
            if rf_json:
                rf_uri = AFFMDL_INST[f"{inst_local_name_base}_RepromptingFallbackDiagnostics"]
                g.add((diag_bundle_uri, AFFMDL.diagHasRepromptingFallbackDiagnostics, rf_uri))
                g.add((rf_uri, RDF.type, AFFMDL.RepromptingFallbackDiagnostics))
                g.add((rf_uri, rdfs.label, Literal(f"Reprompting Fallback Diagnostics for {norm_concept}")))
                self._add_literal_if_present(g, rf_uri, AFFMDL.diagRfAttemptsCount, rf_json.get("attempts"), XSD.integer)
                self._add_literal_if_present(g, rf_uri, AFFMDL.diagRfSuccessesCount, rf_json.get("successes"), XSD.integer)
                self._add_literal_if_present(g, rf_uri, AFFMDL.diagRfFailuresCount, rf_json.get("failures"), XSD.integer)

            # Core Definitional URIs Diagnostics (New)
            core_def_uris_json = self._get_nested(diag_json, ["core_definitional_uris"], {})
            if core_def_uris_json:
                core_def_uri_inst = AFFMDL_INST[f"{inst_local_name_base}_CoreDefUrisDiagnostics"]
                g.add((diag_bundle_uri, AFFMDL.diagHasCoreDefinitionalUrisDiagnostics, core_def_uri_inst))
                g.add((core_def_uri_inst, RDF.type, AFFMDL.CoreDefinitionalUrisDiagnostics))
                g.add((core_def_uri_inst, rdfs.label, Literal(f"Core Definitional URIs Diagnostics for {norm_concept}")))
                for uri_str_id in core_def_uris_json.get("identified", []):
                    if uri_str_id: self._validate_and_add_uri_link(g, core_def_uri_inst, AFFMDL.diagCoreDefIdentifiedUri, uri_str_id, result, f"{core_def_uri_inst}/identified", "Identified Core Def URI")
                for uri_str_prom in core_def_uris_json.get("promoted_to_top_attrs", []):
                    if uri_str_prom: self._validate_and_add_uri_link(g, core_def_uri_inst, AFFMDL.diagCoreDefPromotedUri, uri_str_prom, result, f"{core_def_uri_inst}/promoted", "Promoted Core Def URI")


            # Stage 2 Diagnostics
            s2_json = self._get_nested(diag_json, ["stage2"], {})
            if s2_json:
                s2_uri = AFFMDL_INST[f"{inst_local_name_base}_Stage2Diagnostics"]
                g.add((diag_bundle_uri, AFFMDL.diagHasStage2Diagnostics, s2_uri))
                g.add((s2_uri, RDF.type, AFFMDL.Stage2Diagnostics))
                g.add((s2_uri, rdfs.label, Literal(f"Stage 2 Diagnostics for {norm_concept}")))
                self._add_literal_if_present(g, s2_uri, AFFMDL.diagS2Status, s2_json.get("status"))
                self._add_literal_if_present(g, s2_uri, AFFMDL.diagS2DurationSeconds, s2_json.get("duration_seconds"), XSD.decimal)
                self._add_literal_if_present(g, s2_uri, AFFMDL.diagS2ErrorMessage, s2_json.get("error"))
            
            # Final Output Counts
            fo_json = self._get_nested(diag_json, ["final_output"], {})
            if fo_json:
                fo_uri = AFFMDL_INST[f"{inst_local_name_base}_FinalOutputCounts"]
                g.add((diag_bundle_uri, AFFMDL.diagHasFinalOutputCounts, fo_uri))
                g.add((fo_uri, RDF.type, AFFMDL.FinalOutputCounts))
                g.add((fo_uri, rdfs.label, Literal(f"Final Output Counts for {norm_concept}")))
                self._add_literal_if_present(g, fo_uri, AFFMDL.diagFoMustNotHaveCount, fo_json.get("must_not_have_count"), XSD.integer)
                self._add_literal_if_present(g, fo_uri, AFFMDL.diagFoAdditionalSubscoresCount, fo_json.get("additional_subscores_count"), XSD.integer)
                self._add_literal_if_present(g, fo_uri, AFFMDL.diagFoThemesCount, fo_json.get("themes_count"), XSD.integer)
                self._add_literal_if_present(g, fo_uri, AFFMDL.diagFoUnthemedConceptsCapturedCount, fo_json.get("unthemed_concepts_captured_count"), XSD.integer)
                self._add_literal_if_present(g, fo_uri, AFFMDL.diagFoFailedFallbackThemesCount, fo_json.get("failed_fallback_themes_count"), XSD.integer)
                self._add_literal_if_present(g, fo_uri, AFFMDL.diagFoTopDefiningAttributesCount, fo_json.get("top_defining_attributes_count"), XSD.integer)

        return len(g) - triples_before

    def _collect_rdf_graph_stats(self, graph: Graph, result: ValidationResult):
        result.stats['rdf_graph_stats']['final_triple_count'] = len(graph)
        unique_subjects = set()
        unique_predicates = set()
        unique_object_uris = set()
        unique_object_literals = set()

        for s, p, o in graph:
            unique_subjects.add(str(s))
            unique_predicates.add(str(p))
            if isinstance(o, (URIRef, BNode)):
                unique_object_uris.add(str(o))
            else:
                unique_object_literals.add(str(o))
        
        result.stats['rdf_graph_stats']['unique_subjects'] = unique_subjects
        result.stats['rdf_graph_stats']['unique_predicates'] = unique_predicates
        result.stats['rdf_graph_stats']['unique_objects_uris'] = unique_object_uris
        result.stats['rdf_graph_stats']['unique_objects_literals'] = unique_object_literals

        for prefix, namespace_uri in graph.namespaces():
            result.stats['rdf_graph_stats']['namespaces_used'].add(f"{prefix}: {namespace_uri}")

    def convert_and_validate_json_to_rdf(self, json_filepath: str, rdf_output_filepath: str) -> ValidationResult:
        result = ValidationResult()
        self._load_taxonomies_if_needed(result)
        if result.has_errors() and any(issue.category == 'TAXONOMY' and issue.severity == 'ERROR' for issue in result.issues):
             logger.error("Critical error during taxonomy loading. Aborting RDF conversion.")
             return result

        g = Graph()
        g.bind("affmdl", AFFMDL)
        g.bind("affmdl-inst", AFFMDL_INST)
        g.bind("skos", SKOS)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)
        
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
            result.add_issue('WARNING', 'FORMAT', "Input JSON is not a list of definitions. Processing as a single definition.")
            all_definitions_data = [all_definitions_data]

        total_definitions = len(all_definitions_data)
        result.stats['conversion_stats']['total_definitions_in_json'] = total_definitions
        logger.info(f"Starting conversion of {total_definitions} definition(s) from {json_filepath}")

        for def_data in tqdm(all_definitions_data, desc="Converting JSON Definitions"):
            if not isinstance(def_data, dict):
                result.add_issue('ERROR', 'FORMAT', f"Encountered non-dictionary item in definition list: {type(def_data)}", f"Item index: {result.stats['conversion_stats']['processed_affinity_definitions']}")
                continue
            triples_added = self._process_definition_to_rdf(g, def_data, result)
            result.stats['conversion_stats']['total_triples_generated'] += triples_added
            if def_data.get("input_concept"):
                 result.stats['conversion_stats']['processed_affinity_definitions'] += 1
        
        self._collect_rdf_graph_stats(g, result)
        
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
        print("Usage: python json_to_expedia_affinity_rdf_v2_1.py <input_json_file> <output_turtle_file> [taxonomy_directory]")
        print("Example: python json_to_expedia_affinity_rdf_v2_1.py affinity_definitions.json definitions.ttl ./data/taxonomies")
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_rdf_path = sys.argv[2]
    taxonomy_folder = sys.argv[3] if len(sys.argv) > 3 else None

    if not Path(input_json_path).exists():
        logger.error(f"Input JSON file does not exist: {input_json_path}")
        sys.exit(1)

    converter = JsonToExpediaAffinityRDF(taxonomy_dir=taxonomy_folder)
    validation_report = converter.convert_and_validate_json_to_rdf(input_json_path, output_rdf_path)

    report_path = Path(output_rdf_path).parent / f"{Path(output_rdf_path).stem}_validation_report.json"
    try:
        with open(report_path, 'w', encoding='utf-8') as f_report:
            f_report.write(validation_report.to_json(indent=2))
        logger.info(f"Validation report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Could not write validation report: {e}")

    if validation_report.has_errors():
        logger.error("Conversion completed with one or more ERRORS. See report for details.")
        sys.exit(1)
    elif any(issue.severity == 'WARNING' for issue in validation_report.issues):
        logger.warning("Conversion completed with WARNINGS. See report for details.")
        sys.exit(0)
    else:
        logger.info("Conversion completed successfully with no errors or warnings.")
        sys.exit(0)