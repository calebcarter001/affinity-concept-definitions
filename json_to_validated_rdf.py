import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set, Union, Tuple, Any
from dataclasses import dataclass
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, SKOS, XSD
from tqdm import tqdm
import sys
from urllib.parse import quote
from datetime import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Define Namespaces ---
AFFMDL = Namespace("urn:com:expedia:affinitymodel#")
AFFMDL_INST = Namespace("urn:com:expedia:affinitymodel:instance:")

@dataclass
class ValidationIssue:
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    category: str  # 'FORMAT', 'TAXONOMY', 'NAMESPACE'
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

class ValidationResult:
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.stats: Dict = {
            'total_triples': 0,
            'unique_subjects': 0,
            'unique_predicates': 0,
            'unique_objects': 0,
            'unique_uris': set(),
            'namespaces': set(),
            'taxonomy_stats': {
                'total_taxonomies_loaded': 0,
                'total_valid_concepts': 0,
                'concepts_by_taxonomy': {},
                'invalid_concepts': [],
                'concept_sources': {}
            },
            'conversion_stats': {
                'total_definitions': 0,
                'processed_definitions': 0,
                'total_triples_added': 0,
                'batches_processed': 0
            }
        }
    
    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
    
    def has_critical_errors(self) -> bool:
        return any(issue.severity == 'ERROR' for issue in self.issues)
    
    def to_dict(self) -> Dict:
        return {
            'issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'message': issue.message,
                    'location': issue.location,
                    'suggestion': issue.suggestion
                }
                for issue in self.issues
            ],
            'stats': {
                'total_triples': self.stats['total_triples'],
                'unique_subjects': self.stats['unique_subjects'],
                'unique_predicates': self.stats['unique_predicates'],
                'unique_objects': self.stats['unique_objects'],
                'unique_uris': len(self.stats['unique_uris']),
                'namespaces': list(self.stats['namespaces']),
                'taxonomy_stats': {
                    'total_taxonomies_loaded': self.stats['taxonomy_stats']['total_taxonomies_loaded'],
                    'total_valid_concepts': self.stats['taxonomy_stats']['total_valid_concepts'],
                    'concepts_by_taxonomy': self.stats['taxonomy_stats']['concepts_by_taxonomy'],
                    'invalid_concepts': list(self.stats['taxonomy_stats']['invalid_concepts']),
                    'concept_sources': {k: list(v) for k, v in self.stats['taxonomy_stats']['concept_sources'].items()}
                },
                'conversion_stats': self.stats['conversion_stats']
            }
        }
    
    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

class JsonToValidatedRDF:
    def __init__(self, taxonomy_dir: Optional[str] = None):
        self.taxonomy_dir = Path(taxonomy_dir) if taxonomy_dir else None
        self.taxonomy_graphs: Dict[str, Graph] = {}
        self.all_valid_uris: Set[str] = set()
        self.uri_to_taxonomy: Dict[str, Set[str]] = defaultdict(set)
        self._load_status = {'loaded': False, 'error': None}
        
    def _get_nested(self, data: Dict, keys: List[str], default=None) -> Union[Dict, List, str, int, float, bool, None]:
        """Safely access nested dictionary keys."""
        current_level = data
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            elif isinstance(current_level, list) and isinstance(key, int) and 0 <= key < len(current_level):
                current_level = current_level[key]
            else:
                return default
        return current_level

    def convert_and_validate(self, json_filepath: str, rdf_output_filepath: str, batch_size: int = 100) -> ValidationResult:
        """Convert JSON to RDF and validate in a single pass."""
        result = ValidationResult()
        
        # Step 1: Load taxonomies if needed
        if self.taxonomy_dir and not self._load_status['loaded']:
            self._load_taxonomies_parallel(result)
            self._load_status['loaded'] = True
            if result.has_critical_errors():
                return result
        
        # Step 2: Initialize RDF graph
        g = Graph()
        self._bind_namespaces(g)
        
        # Step 3: Load and validate JSON input
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                all_definitions_data = json.load(f)
        except FileNotFoundError:
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='FORMAT',
                message=f"Input JSON file not found: {json_filepath}"
            ))
            return result
        except json.JSONDecodeError as e:
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='FORMAT',
                message=f"Error decoding JSON from {json_filepath}: {e}"
            ))
            return result
        
        if not isinstance(all_definitions_data, list):
            all_definitions_data = [all_definitions_data]
        
        # Step 4: Process in batches with validation
        total_definitions = len(all_definitions_data)
        result.stats['conversion_stats']['total_definitions'] = total_definitions
        logger.info(f"Processing {total_definitions} definition(s)")
        
        batches = [all_definitions_data[i:i + batch_size] 
                  for i in range(0, len(all_definitions_data), batch_size)]
        
        unique_output_path = self._get_unique_output_path(rdf_output_filepath)
        
        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_idx}/{len(batches)} "
                       f"({result.stats['conversion_stats']['processed_definitions']}/{total_definitions} definitions)")
            
            # Process batch
            batch_triples = 0
            for definition_data in tqdm(batch, 
                                      desc=f"Batch {batch_idx}/{len(batches)}", 
                                      unit="def"):
                triples_added = self._process_definition(definition_data, g)
                batch_triples += triples_added
                result.stats['conversion_stats']['total_triples_added'] += triples_added
                result.stats['conversion_stats']['processed_definitions'] += 1
            
            # Validate batch
            self._validate_batch(list(g)[-batch_triples:], result)
            result.stats['conversion_stats']['batches_processed'] += 1
            
            logger.info(f"Batch {batch_idx} complete: {batch_triples} triples added and validated")
            
            # Save intermediate results for large datasets
            if len(batches) > 1 and batch_idx < len(batches):
                temp_file = f"{unique_output_path}.batch{batch_idx}"
                g.serialize(destination=temp_file, format="turtle")
                logger.info(f"Saved intermediate batch {batch_idx} to {temp_file}")
        
        # Final validation of namespaces
        self._validate_required_namespaces(g, result)
        
        # Final save
        logger.info(f"Serializing final RDF graph ({len(g)} triples)")
        try:
            g.serialize(destination=unique_output_path, format="turtle")
            logger.info(f"Successfully converted and validated JSON to RDF: {unique_output_path}")
        except Exception as e:
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='FORMAT',
                message=f"Error serializing RDF graph: {e}"
            ))
        
        return result

    def _bind_namespaces(self, g: Graph) -> None:
        """Bind all required namespaces to the graph."""
        g.bind("affmdl", AFFMDL)
        g.bind("affmdl-inst", AFFMDL_INST)
        g.bind("skos", SKOS)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

    def _get_unique_output_path(self, base_path_str: str) -> str:
        """Generate a unique output path."""
        path = Path(base_path_str)
        if not path.exists():
            return str(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{path.stem}_{timestamp}{path.suffix}"
        unique_path = path.parent / new_filename
        logger.info(f"Output file {path} already exists. Using: {unique_path}")
        return str(unique_path)

    def _safe_uri_name(self, text: str) -> str:
        """Convert text to a safe URI component."""
        if not text:
            return "unknown_concept"
        safe_text = text.replace(" ", "_").replace("/", "_").replace(":", "_").replace("#", "_")
        return quote(safe_text, safe="-_")

    def _load_taxonomies_parallel(self, result: ValidationResult) -> None:
        """Load taxonomies in parallel for better performance."""
        if not self.taxonomy_dir or not self.taxonomy_dir.exists():
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='TAXONOMY',
                message=f"Taxonomy directory not found: {self.taxonomy_dir}"
            ))
            return

        rdf_files = list(self.taxonomy_dir.glob("*.rdf"))
        if not rdf_files:
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='TAXONOMY',
                message=f"No RDF files found in taxonomy directory: {self.taxonomy_dir}"
            ))
            return

        def load_taxonomy(file_path):
            try:
                graph = Graph()
                graph.parse(str(file_path), format="xml")
                concepts = set()
                uri_map = defaultdict(set)
                
                for s, p, o in graph:
                    if isinstance(s, URIRef):
                        concepts.add(str(s))
                        uri_map[str(s)].add(file_path.name)
                    if isinstance(o, URIRef):
                        concepts.add(str(o))
                        uri_map[str(o)].add(file_path.name)
                
                return {
                    'name': file_path.name,
                    'graph': graph,
                    'concepts': concepts,
                    'uri_map': uri_map,
                    'success': True
                }
            except Exception as e:
                return {
                    'name': file_path.name,
                    'error': str(e),
                    'success': False
                }

        max_workers = min(multiprocessing.cpu_count(), len(rdf_files))
        logger.info(f"Loading {len(rdf_files)} taxonomy files using {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(load_taxonomy, f): f for f in rdf_files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    data = future.result()
                    if data['success']:
                        self.taxonomy_graphs[data['name']] = data['graph']
                        self.all_valid_uris.update(data['concepts'])
                        for uri, sources in data['uri_map'].items():
                            self.uri_to_taxonomy[uri].update(sources)
                        
                        result.stats['taxonomy_stats']['total_taxonomies_loaded'] += 1
                        result.stats['taxonomy_stats']['total_valid_concepts'] += len(data['concepts'])
                        result.stats['taxonomy_stats']['concepts_by_taxonomy'][data['name']] = len(data['concepts'])
                    else:
                        result.add_issue(ValidationIssue(
                            severity='ERROR',
                            category='TAXONOMY',
                            message=f"Failed to load taxonomy {file_path}: {data['error']}"
                        ))
                except Exception as e:
                    result.add_issue(ValidationIssue(
                        severity='ERROR',
                        category='TAXONOMY',
                        message=f"Error processing taxonomy {file_path}: {e}"
                    ))

        if not self.taxonomy_graphs:
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='TAXONOMY',
                message="No taxonomies were successfully loaded"
            ))
            return

        logger.info(f"Successfully loaded {len(self.taxonomy_graphs)} taxonomies with "
                   f"{len(self.all_valid_uris)} unique URIs")

    def _process_definition(self, definition_data: Dict, g: Graph) -> int:
        """Process a single definition and add its triples to the graph."""
        input_concept_str = definition_data.get("input_concept")
        if not input_concept_str:
            logger.warning("Skipping definition due to missing 'input_concept'.")
            return 0
        
        triples_before = len(g)
        
        normalized_concept_str = definition_data.get("normalized_concept", input_concept_str)
        instance_local_name = self._safe_uri_name(normalized_concept_str)
        aff_def_inst_uri = AFFMDL_INST[f"{instance_local_name}_AffinityDefinition"]
        
        g.add((aff_def_inst_uri, RDF.type, AFFMDL.AffinityDefinition))
        g.add((aff_def_inst_uri, SKOS.prefLabel, Literal(f"Affinity Definition for '{input_concept_str}'")))
        g.add((aff_def_inst_uri, AFFMDL.inputConcept, Literal(input_concept_str)))
        g.add((aff_def_inst_uri, AFFMDL.normalizedConcept, Literal(normalized_concept_str)))
        
        if definition_data.get("applicable_lodging_types"):
            g.add((aff_def_inst_uri, AFFMDL.applicableLodgingTypes, Literal(definition_data["applicable_lodging_types"])))
        
        if definition_data.get("affinity_score_total_allocated") is not None:
            g.add((aff_def_inst_uri, AFFMDL.affinityScoreTotalAllocated, Literal(definition_data["affinity_score_total_allocated"], datatype=XSD.decimal)))
        
        if definition_data.get("requires_geo_check") is not None:
            g.add((aff_def_inst_uri, AFFMDL.requiresGeoCheck, Literal(definition_data["requires_geo_check"], datatype=XSD.boolean)))
        
        if definition_data.get("failed_fallback_themes"):
            g.add((aff_def_inst_uri, AFFMDL.failedFallbackThemes, Literal(json.dumps(definition_data["failed_fallback_themes"]))))

        # --- Travel Category (Anchor) ---
        tc_data = definition_data.get("travel_category", {})
        if tc_data.get("uri"):
            tc_link_inst = BNode()
            g.add((aff_def_inst_uri, AFFMDL.hasTravelCategory, tc_link_inst))
            g.add((tc_link_inst, RDF.type, AFFMDL.TravelCategoryLink))
            g.add((tc_link_inst, AFFMDL.linkedConceptUri, URIRef(tc_data["uri"])))

            if tc_data.get("label"):
                g.add((tc_link_inst, AFFMDL.observedLabel, Literal(tc_data["label"])))
            if tc_data.get("types"):
                for t_type in tc_data["types"]:
                    g.add((tc_link_inst, AFFMDL.observedType, Literal(t_type)))
            if tc_data.get("definitions"):
                for t_def in tc_data["definitions"]:
                    g.add((tc_link_inst, AFFMDL.observedDefinition, Literal(t_def)))
            
            if tc_data.get("sbert_score") is not None:
                g.add((tc_link_inst, AFFMDL.sbertScore, Literal(tc_data["sbert_score"], datatype=XSD.double)))
            if tc_data.get("keyword_score") is not None:
                g.add((tc_link_inst, AFFMDL.keywordScore, Literal(tc_data["keyword_score"], datatype=XSD.double)))
            if tc_data.get("combined_score") is not None:
                g.add((tc_link_inst, AFFMDL.combinedScore, Literal(tc_data["combined_score"], datatype=XSD.double)))
            if tc_data.get("combined_score_unbiased") is not None:
                g.add((tc_link_inst, AFFMDL.combinedScoreUnbiased, Literal(tc_data["combined_score_unbiased"], datatype=XSD.double)))
            if tc_data.get("biased") is not None:
                g.add((tc_link_inst, AFFMDL.isBiased, Literal(tc_data["biased"], datatype=XSD.boolean)))
            if tc_data.get("effective_alpha") is not None:
                g.add((tc_link_inst, AFFMDL.effectiveAlpha, Literal(tc_data["effective_alpha"], datatype=XSD.double)))
            if tc_data.get("bias_reason"):
                g.add((tc_link_inst, AFFMDL.biasReason, Literal(tc_data["bias_reason"])))

        # --- Top Defining Attributes ---
        for attr_data in definition_data.get("top_defining_attributes", []):
            if attr_data.get("uri"):
                attr_link_inst = BNode()
                g.add((aff_def_inst_uri, AFFMDL.hasTopDefiningAttribute, attr_link_inst))
                g.add((attr_link_inst, RDF.type, AFFMDL.RankedAttributeLink))
                g.add((attr_link_inst, AFFMDL.attributeUri, URIRef(attr_data["uri"])))
                if attr_data.get("skos:prefLabel"):
                    g.add((attr_link_inst, AFFMDL.attributeSkosPrefLabel, Literal(attr_data["skos:prefLabel"])))
                if attr_data.get("concept_weight") is not None:
                    g.add((attr_link_inst, AFFMDL.attributeConceptWeight, Literal(attr_data["concept_weight"], datatype=XSD.decimal)))
                if attr_data.get("type"):
                    for attr_type in attr_data["type"]:
                        g.add((attr_link_inst, AFFMDL.attributeObservedType, Literal(attr_type)))
        
        # --- Themes ---
        for theme_data in definition_data.get("themes", []):
            theme_inst = BNode()
            g.add((aff_def_inst_uri, AFFMDL.hasTheme, theme_inst))
            g.add((theme_inst, RDF.type, AFFMDL.ThemeInstance))
            
            if theme_data.get("theme_name"):
                g.add((theme_inst, AFFMDL.themeName, Literal(theme_data["theme_name"])))
            if theme_data.get("theme_type"):
                g.add((theme_inst, AFFMDL.themeType, Literal(theme_data["theme_type"])))
            if theme_data.get("rule_applied"):
                g.add((theme_inst, AFFMDL.themeRuleApplied, Literal(theme_data["rule_applied"])))
            if theme_data.get("normalized_theme_weight") is not None:
                g.add((theme_inst, AFFMDL.themeNormalizedWeight, Literal(theme_data["normalized_theme_weight"], datatype=XSD.decimal)))
            if theme_data.get("subScore"):
                g.add((theme_inst, AFFMDL.themeSubScoreName, Literal(theme_data["subScore"])))
            if theme_data.get("llm_summary"):
                g.add((theme_inst, AFFMDL.themeLLMSummary, Literal(theme_data["llm_summary"])))

            # Theme Attributes
            for theme_attr_data in theme_data.get("attributes", []):
                if theme_attr_data.get("uri"):
                    theme_attr_link_inst = BNode()
                    g.add((theme_inst, AFFMDL.hasRankedAttribute, theme_attr_link_inst))
                    g.add((theme_attr_link_inst, RDF.type, AFFMDL.RankedAttributeLink))
                    g.add((theme_attr_link_inst, AFFMDL.attributeUri, URIRef(theme_attr_data["uri"])))
                    if theme_attr_data.get("skos:prefLabel"):
                         g.add((theme_attr_link_inst, AFFMDL.attributeSkosPrefLabel, Literal(theme_attr_data["skos:prefLabel"])))
                    if theme_attr_data.get("concept_weight") is not None:
                        g.add((theme_attr_link_inst, AFFMDL.attributeConceptWeight, Literal(theme_attr_data["concept_weight"], datatype=XSD.decimal)))
                    if theme_attr_data.get("type"):
                        for attr_type in theme_attr_data["type"]:
                            g.add((theme_attr_link_inst, AFFMDL.attributeObservedType, Literal(attr_type)))
            
            # Theme Processing Diagnostics
            theme_diag_key = theme_data.get("theme_name")
            theme_proc_data = self._get_nested(definition_data, ["diagnostics", "theme_processing", theme_diag_key], {})
            if theme_proc_data:
                if theme_proc_data.get("llm_assigned_count") is not None:
                    g.add((theme_inst, AFFMDL.diagnosticThemeLLMAssignedCount, Literal(theme_proc_data["llm_assigned_count"], datatype=XSD.integer)))
                if theme_proc_data.get("attributes_after_weighting") is not None:
                    g.add((theme_inst, AFFMDL.diagnosticThemeAttributesAfterWeighting, Literal(theme_proc_data["attributes_after_weighting"], datatype=XSD.integer)))
                if theme_proc_data.get("status"):
                    g.add((theme_inst, AFFMDL.diagnosticThemeStatus, Literal(theme_proc_data["status"])))
                if theme_proc_data.get("rule_failed") is not None:
                    g.add((theme_inst, AFFMDL.diagnosticThemeRuleFailed, Literal(theme_proc_data["rule_failed"], datatype=XSD.boolean)))

        # --- Key Associated Concepts Unthemed ---
        for unthemed_data in definition_data.get("key_associated_concepts_unthemed", []):
            if unthemed_data.get("uri"):
                unthemed_link_inst = BNode()
                g.add((aff_def_inst_uri, AFFMDL.hasKeyAssociatedConceptUnthemed, unthemed_link_inst))
                g.add((unthemed_link_inst, RDF.type, AFFMDL.UnthemedConceptLink))
                g.add((unthemed_link_inst, AFFMDL.conceptUri, URIRef(unthemed_data["uri"])))
                if unthemed_data.get("skos:prefLabel"):
                    g.add((unthemed_link_inst, AFFMDL.conceptSkosPrefLabel, Literal(unthemed_data["skos:prefLabel"])))
                if unthemed_data.get("combined_score") is not None:
                    g.add((unthemed_link_inst, AFFMDL.conceptCombinedScore, Literal(unthemed_data["combined_score"], datatype=XSD.double)))
                if unthemed_data.get("type"):
                     for unthemed_type in unthemed_data["type"]:
                        g.add((unthemed_link_inst, AFFMDL.conceptObservedType, Literal(unthemed_type)))
        
        # --- Additional Relevant Subscores ---
        for subscore_item in definition_data.get("additional_relevant_subscores", []):
            subscore_link_inst = BNode()
            g.add((aff_def_inst_uri, AFFMDL.hasAdditionalRelevantSubscore, subscore_link_inst))
            g.add((subscore_link_inst, RDF.type, AFFMDL.SubscoreWeighting))
            if subscore_item.get("subscore_name"):
                g.add((subscore_link_inst, AFFMDL.subscoreName, Literal(subscore_item["subscore_name"])))
            if subscore_item.get("weight") is not None:
                g.add((subscore_link_inst, AFFMDL.weightValue, Literal(subscore_item["weight"], datatype=XSD.double)))

        # --- Must Not Have ---
        for mnh_data in definition_data.get("must_not_have", []):
            if mnh_data.get("uri"):
                mnh_link_inst = BNode()
                g.add((aff_def_inst_uri, AFFMDL.hasMustNotHaveConcept, mnh_link_inst))
                g.add((mnh_link_inst, RDF.type, AFFMDL.MustNotHaveLink))
                g.add((mnh_link_inst, AFFMDL.conceptUri, URIRef(mnh_data["uri"])))
                if mnh_data.get("skos:prefLabel"):
                    g.add((mnh_link_inst, AFFMDL.conceptSkosPrefLabel, Literal(mnh_data["skos:prefLabel"])))
                if mnh_data.get("scope"):
                    g.add((mnh_link_inst, AFFMDL.identificationScope, Literal(mnh_data["scope"])))
        
        # --- Processing Metadata ---
        pm_data = definition_data.get("processing_metadata", {})
        if pm_data:
            pm_inst = BNode()
            g.add((aff_def_inst_uri, AFFMDL.hasProcessingMetadata, pm_inst))
            g.add((pm_inst, RDF.type, AFFMDL.ProcessingMetadata))
            if pm_data.get("status"):
                g.add((pm_inst, AFFMDL.processingStatus, Literal(pm_data["status"])))
            if pm_data.get("version"):
                g.add((pm_inst, AFFMDL.engineVersion, Literal(pm_data["version"])))
            if pm_data.get("timestamp"):
                g.add((pm_inst, AFFMDL.timestamp, Literal(pm_data["timestamp"], datatype=XSD.dateTime)))
            if pm_data.get("total_duration_seconds") is not None:
                g.add((pm_inst, AFFMDL.totalDurationSeconds, Literal(pm_data["total_duration_seconds"], datatype=XSD.decimal)))
            if pm_data.get("cache_version"):
                g.add((pm_inst, AFFMDL.cacheVersionUsed, Literal(pm_data["cache_version"])))
            if pm_data.get("llm_provider"):
                g.add((pm_inst, AFFMDL.llmProviderUsed, Literal(pm_data["llm_provider"])))
            if pm_data.get("llm_model"):
                g.add((pm_inst, AFFMDL.llmModelUsed, Literal(pm_data["llm_model"])))

        # --- Diagnostics ---
        diag_data = definition_data.get("diagnostics", {})
        if diag_data:
            diag_bundle_inst = BNode()
            g.add((aff_def_inst_uri, AFFMDL.hasDiagnostics, diag_bundle_inst))
            g.add((diag_bundle_inst, RDF.type, AFFMDL.DiagnosticsBundle))

            if self._get_nested(diag_data, ["lodging_type_determination", "result"]):
                g.add((diag_bundle_inst, AFFMDL.diagLodgingTypeDeterminationResult, 
                       Literal(diag_data["lodging_type_determination"]["result"])))

            # LLM Negation Diagnostics
            llm_neg_data = self._get_nested(diag_data, ["llm_negation"], {})
            if llm_neg_data:
                llm_neg_inst = BNode()
                g.add((diag_bundle_inst, AFFMDL.diagHasLLMNegationDiagnostics, llm_neg_inst))
                g.add((llm_neg_inst, RDF.type, AFFMDL.LLMNegationDiagnostics))
                if llm_neg_data.get("attempted") is not None:
                    g.add((llm_neg_inst, AFFMDL.llmNegAttempted, Literal(llm_neg_data["attempted"], datatype=XSD.boolean)))
                if llm_neg_data.get("success") is not None:
                    g.add((llm_neg_inst, AFFMDL.llmNegSuccess, Literal(llm_neg_data["success"], datatype=XSD.boolean)))
                if llm_neg_data.get("uris_found") is not None:
                    g.add((llm_neg_inst, AFFMDL.llmNegUrisFound, Literal(llm_neg_data["uris_found"], datatype=XSD.integer)))
                if llm_neg_data.get("error"):
                    g.add((llm_neg_inst, AFFMDL.llmNegError, Literal(llm_neg_data["error"])))
                for checked_uri in llm_neg_data.get("candidates_checked", []):
                    if checked_uri:
                        g.add((llm_neg_inst, AFFMDL.llmNegCandidateCheckedUri, URIRef(checked_uri)))
                for neg_uri in llm_neg_data.get("identified_negating_uris", []):
                    if neg_uri:
                        g.add((llm_neg_inst, AFFMDL.llmNegIdentifiedNegatingUri, URIRef(neg_uri)))

            # Stage 1 Diagnostics
            s1_data = self._get_nested(diag_data, ["stage1"], {})
            if s1_data:
                s1_inst = BNode()
                g.add((diag_bundle_inst, AFFMDL.diagHasStage1Diagnostics, s1_inst))
                g.add((s1_inst, RDF.type, AFFMDL.Stage1Diagnostics))
                if s1_data.get("status"):
                    g.add((s1_inst, AFFMDL.s1Status, Literal(s1_data["status"])))
                if s1_data.get("error"):
                    g.add((s1_inst, AFFMDL.s1Error, Literal(s1_data["error"])))
                if s1_data.get("selection_method"):
                    g.add((s1_inst, AFFMDL.s1OverallSelectionMethod, Literal(s1_data["selection_method"])))
                if s1_data.get("sbert_candidate_count_initial") is not None:
                    g.add((s1_inst, AFFMDL.s1SbertCandidateCountInitial, 
                           Literal(s1_data["sbert_candidate_count_initial"], datatype=XSD.integer)))
                if s1_data.get("keyword_candidate_count_initial") is not None:
                    g.add((s1_inst, AFFMDL.s1KeywordCandidateCountInitial, 
                           Literal(s1_data["keyword_candidate_count_initial"], datatype=XSD.integer)))
                if s1_data.get("unique_candidates_before_ranking") is not None:
                    g.add((s1_inst, AFFMDL.s1UniqueCandidatesBeforeRanking, 
                           Literal(s1_data["unique_candidates_before_ranking"], datatype=XSD.integer)))
                if s1_data.get("llm_candidate_count") is not None:
                    g.add((s1_inst, AFFMDL.s1LlmCandidateOutputCount, 
                           Literal(s1_data["llm_candidate_count"], datatype=XSD.integer)))
                if s1_data.get("duration_seconds") is not None:
                    g.add((s1_inst, AFFMDL.s1TotalDurationSeconds, 
                           Literal(s1_data["duration_seconds"], datatype=XSD.decimal)))

                exp_data = self._get_nested(s1_data, ["expansion"], {})
                if exp_data:
                    exp_inst = BNode()
                    g.add((s1_inst, AFFMDL.s1HasExpansionDetails, exp_inst))
                    g.add((exp_inst, RDF.type, AFFMDL.ExpansionDiagnostics))
                    if exp_data.get("attempted") is not None:
                        g.add((exp_inst, AFFMDL.expAttempted, Literal(exp_data["attempted"], datatype=XSD.boolean)))
                    if exp_data.get("successful") is not None:
                        g.add((exp_inst, AFFMDL.expSuccessful, Literal(exp_data["successful"], datatype=XSD.boolean)))
                    if exp_data.get("count") is not None:
                        g.add((exp_inst, AFFMDL.expTermCount, Literal(exp_data["count"], datatype=XSD.integer)))
                    for term in exp_data.get("terms", []):
                        if term:
                            g.add((exp_inst, AFFMDL.expTerm, Literal(term)))
                    if exp_data.get("keyword_count") is not None:
                        g.add((exp_inst, AFFMDL.expKeywordResultCount, Literal(exp_data["keyword_count"], datatype=XSD.integer)))
                    if exp_data.get("error"):
                        g.add((exp_inst, AFFMDL.expErrorMessage, Literal(exp_data["error"])))
                    if exp_data.get("selection_method"):
                        g.add((exp_inst, AFFMDL.expSelectionMethodUsed, Literal(exp_data["selection_method"])))
                    if exp_data.get("duration_seconds") is not None:
                        g.add((exp_inst, AFFMDL.expDuration, Literal(exp_data["duration_seconds"], datatype=XSD.decimal)))

            # LLM Slotting Diagnostics
            llms_data = self._get_nested(diag_data, ["llm_slotting"], {})
            if llms_data:
                llms_inst = BNode()
                g.add((diag_bundle_inst, AFFMDL.diagHasLLMSlottingDiagnostics, llms_inst))
                g.add((llms_inst, RDF.type, AFFMDL.LLMSlottingDiagnostics))
                if llms_data.get("status"):
                    g.add((llms_inst, AFFMDL.llmsStatus, Literal(llms_data["status"])))
                if llms_data.get("duration_seconds") is not None:
                    g.add((llms_inst, AFFMDL.llmsDurationSeconds, Literal(llms_data["duration_seconds"], datatype=XSD.decimal)))

            # Reprompting Fallback Diagnostics
            rf_data = self._get_nested(diag_data, ["reprompting_fallback"], {})
            if rf_data:
                rf_inst = BNode()
                g.add((diag_bundle_inst, AFFMDL.diagHasRepromptingFallbackDiagnostics, rf_inst))
                g.add((rf_inst, RDF.type, AFFMDL.RepromptingFallbackDiagnostics))
                if rf_data.get("attempts") is not None:
                    g.add((rf_inst, AFFMDL.rfAttempts, Literal(rf_data["attempts"], datatype=XSD.integer)))

            # Stage 2 Diagnostics
            s2_data = self._get_nested(diag_data, ["stage2"], {})
            if s2_data:
                s2_inst = BNode()
                g.add((diag_bundle_inst, AFFMDL.diagHasStage2Diagnostics, s2_inst))
                g.add((s2_inst, RDF.type, AFFMDL.Stage2Diagnostics))
                if s2_data.get("status"):
                    g.add((s2_inst, AFFMDL.s2Status, Literal(s2_data["status"])))
            
            # Theme Processing Details (as JSON string for simplicity)
            if self._get_nested(diag_data, ["theme_processing"]):
                g.add((diag_bundle_inst, AFFMDL.diagThemeProcessingDetailsJson, 
                       Literal(json.dumps(diag_data["theme_processing"]))))

            # Final Output Counts
            fo_data = self._get_nested(diag_data, ["final_output"], {})
            if fo_data:
                fo_inst = BNode()
                g.add((diag_bundle_inst, AFFMDL.diagHasFinalOutputCounts, fo_inst))
                g.add((fo_inst, RDF.type, AFFMDL.FinalOutputCounts))
                if fo_data.get("must_not_have_count") is not None:
                    g.add((fo_inst, AFFMDL.foMustNotHaveCount, 
                           Literal(fo_data["must_not_have_count"], datatype=XSD.integer)))
                if fo_data.get("additional_subscores_count") is not None:
                    g.add((fo_inst, AFFMDL.foAdditionalSubscoresCount, 
                           Literal(fo_data["additional_subscores_count"], datatype=XSD.integer)))

            if self._get_nested(diag_data, ["error_details"]):
                g.add((diag_bundle_inst, AFFMDL.diagErrorDetails, 
                       Literal(diag_data["error_details"])))

        return len(g) - triples_before

    def _validate_batch(self, batch: List[Tuple], result: ValidationResult) -> None:
        """Validate a batch of triples."""
        subjects = set()
        predicates = set()
        objects = set()
        uris = set()
        concepts_to_validate = set()
        
        concept_predicates = {
            URIRef("urn:com:expedia:affinitymodel#linkedConceptUri"),
            URIRef("urn:com:expedia:affinitymodel#attributeUri"),
            URIRef("urn:com:expedia:affinitymodel#conceptUri")
        }
        
        for s, p, o in batch:
            result.stats['total_triples'] += 1
            
            subjects.add(str(s))
            predicates.add(str(p))
            objects.add(str(o))
            
            if isinstance(s, URIRef):
                uris.add(str(s))
                self._validate_uri(s, 'subject', result)
            if isinstance(p, URIRef):
                uris.add(str(p))
                self._validate_uri(p, 'predicate', result)
            if isinstance(o, URIRef):
                uris.add(str(o))
                self._validate_uri(o, 'object', result)
                
                if p in concept_predicates:
                    concepts_to_validate.add(str(o))
        
        result.stats['unique_subjects'] += len(subjects)
        result.stats['unique_predicates'] += len(predicates)
        result.stats['unique_objects'] += len(objects)
        result.stats['unique_uris'].update(uris)
        
        for concept in concepts_to_validate:
            if concept not in self.all_valid_uris:
                result.add_issue(ValidationIssue(
                    severity='ERROR',
                    category='TAXONOMY',
                    message=f"Concept URI not found in any taxonomy: {concept}",
                    suggestion="Verify the concept URI is correct"
                ))
            else:
                result.stats['taxonomy_stats']['concept_sources'][concept] = self.uri_to_taxonomy[concept]

    def _validate_uri(self, uri: URIRef, position: str, result: ValidationResult) -> None:
        """Validate individual URI format and structure."""
        if not isinstance(uri, URIRef):
            return
            
        uri_str = str(uri)
        
        if not uri_str.startswith(('http://', 'https://', 'urn:')):
            result.add_issue(ValidationIssue(
                severity='WARNING',
                category='FORMAT',
                message=f"URI in {position} position does not start with standard protocol: {uri_str}",
                suggestion="URIs should typically start with 'http://', 'https://', or 'urn:'"
            ))
        
        problematic_chars = ' <>"{}|\\^`'
        found_chars = [c for c in problematic_chars if c in uri_str]
        if found_chars:
            result.add_issue(ValidationIssue(
                severity='ERROR',
                category='FORMAT',
                message=f"URI contains invalid characters {found_chars}: {uri_str}",
                suggestion="Remove or encode special characters in URI"
            ))

    def _validate_required_namespaces(self, graph: Graph, result: ValidationResult) -> None:
        """Validate that all required namespaces are present."""
        required_namespaces = {
            'rdf': str(RDF),
            'rdfs': str(RDFS),
            'skos': str(SKOS),
            'xsd': str(XSD),
            'affmdl': "urn:com:expedia:affinitymodel#",
            'affmdl-inst': "urn:com:expedia:affinitymodel:instance:"
        }
        
        found_namespaces = {prefix: str(uri) for prefix, uri in graph.namespaces()}
        
        for prefix, uri in required_namespaces.items():
            if prefix not in found_namespaces:
                result.add_issue(ValidationIssue(
                    severity='ERROR',
                    category='NAMESPACE',
                    message=f"Required namespace prefix '{prefix}' is missing",
                    suggestion=f"Add namespace binding: {prefix}: {uri}"
                ))
            elif found_namespaces[prefix] != uri:
                result.add_issue(ValidationIssue(
                    severity='WARNING',
                    category='NAMESPACE',
                    message=f"Namespace URI mismatch for prefix '{prefix}'",
                    suggestion=f"Expected {uri}, found {found_namespaces[prefix]}"
                ))

def write_theme_json(theme_data, output_file):
    print("\nDEBUG: Writing theme data to JSON")
    print("Checking attribute groups before writing:")
    for group in theme_data["attribute_groups"]:
        print(f"\nGroup: {group['name']}")
        print(f"Number of attributes: {len(group['attributes'])}")
        if group['name'] == 'ski_related_attributes':
            print("Ski attributes content:")
            for attr in group['attributes']:
                print(f"  {attr}")

    with open(output_file, 'w') as f:
        json.dump(theme_data, f, indent=2)

    print("\nDEBUG: Verifying written data")
    with open(output_file, 'r') as f:
        written_data = json.load(f)
        for group in written_data["attribute_groups"]:
            if group['name'] == 'ski_related_attributes':
                print("\nSki attributes in written file:")
                print(f"Number of attributes: {len(group['attributes'])}")
                for attr in group['attributes']:
                    print(f"  {attr}")

    print("\nDEBUG: Finished writing theme data to JSON")

if __name__ == "__main__":
    if len(sys.argv) < 2:  # Changed from 3 to 2 to allow optional output file
        print("Usage: python json_to_validated_rdf.py <input_json_file> [output_turtle_file] [taxonomy_dir] [batch_size]")
        sys.exit(1)
    
    input_json_path = sys.argv[1]
    
    # Generate default output filename if not provided
    if len(sys.argv) > 2 and not sys.argv[2].startswith("../"): # Check if it's an output file or taxonomy_dir
        output_rdf_path = sys.argv[2]
        taxonomy_dir_index = 3
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Derive output filename from input filename
        input_path_obj = Path(input_json_path)
        output_filename = f"{input_path_obj.stem}_affinity_definitions_{timestamp}.ttl"
        # Place output in a subdirectory 'rdf_outputs' relative to the input JSON file's directory
        # or a default 'rdf_outputs' if the input is in the current directory.
        output_dir = input_path_obj.parent / "rdf_outputs" if input_path_obj.parent != Path(".") else Path("rdf_outputs")
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
        output_rdf_path = str(output_dir / output_filename)
        logger.info(f"Output file not specified. Defaulting to: {output_rdf_path}")
        taxonomy_dir_index = 2 # If output is not specified, taxonomy dir is the next optional arg

    taxonomy_dir = sys.argv[taxonomy_dir_index] if len(sys.argv) > taxonomy_dir_index and not sys.argv[taxonomy_dir_index].isdigit() else "datasources"
    batch_size_index = taxonomy_dir_index + 1
    batch_size = int(sys.argv[batch_size_index]) if len(sys.argv) > batch_size_index and sys.argv[batch_size_index].isdigit() else 100
    
    converter = JsonToValidatedRDF(taxonomy_dir)
    result = converter.convert_and_validate(input_json_path, output_rdf_path, batch_size)
    
    # Print results
    print(result.to_json())
    
    # Exit with status code
    sys.exit(1 if result.has_critical_errors() else 0) 