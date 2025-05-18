#!/usr/bin/env python3
"""
Travel Concepts to Taxonomy Mapping Tool with Enhanced Concept Types and Sub-Themes
"""

import csv
import re
import os
import argparse
import signal
import psutil
import multiprocessing
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, URIRef
import sys
import time
from tqdm import tqdm
import logging
import pickle
from multiprocessing import Pool, cpu_count, Manager
import json
import atexit
import platform
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('taxonomy_mapping.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def cleanup():
    logger.info("Script execution completed. Cleaning up resources.")

atexit.register(cleanup)

CONFIDENCE_SCORES = {'direct': (90, 95), 'partial': (60, 75), 'weak': (1, 39), 'none': (0, 0)}
PARTIAL_CONFIDENCE_THRESHOLD = 40
STATE_FILE = 'mapping_state.json'

class TaxonomyConcept:
    def __init__(self, uri=None, uuid=None, pref_label=None):
        self.uri = uri
        self.uuid = uuid
        self.pref_label = pref_label
        self.alt_labels = []
        self.hidden_labels = []
        self.definition = None
        self.broader = []
        self.narrower = []
        self.related = []
        self.exact_matches = []
        self.close_matches = []
        self.external_id = None
        self.desmet_urn = None
        self.simplified_display_label = None

    def get_all_labels(self):
        all_labels = [self.pref_label] if self.pref_label else []
        all_labels.extend(self.alt_labels)
        all_labels.extend(self.hidden_labels)
        if self.simplified_display_label:
            all_labels.append(self.simplified_display_label)
        return [label.lower() for label in all_labels if label]

    def __str__(self):
        return f"{self.pref_label} ({self.uri})"

class TaxonomyParser:
    def __init__(self):
        self.concepts = {}
        self.label_to_concepts = defaultdict(list)
        self.related_concepts = defaultdict(list)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.concept_embeddings = {}

    def parse_rdf(self, file_path):
        cache_file = file_path + ".pickle"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) > os.path.getmtime(file_path):
            try:
                logger.info(f"Loading cached taxonomy from {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.concepts = cached_data['concepts']
                    self.label_to_concepts = cached_data['label_to_concepts']
                    self.related_concepts = cached_data['related_concepts']
                    self.concept_embeddings = cached_data['concept_embeddings']
                logger.info(f"Successfully loaded {len(self.concepts)} concepts from cache")
                return True
            except Exception as e:
                logger.error(f"Error loading cache: {e}. Will parse original file.")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if file_size_mb > available_memory_mb * 0.8:
            logger.warning(f"WARNING: File size ({file_size_mb:.2f} MB) exceeds 80% of available memory ({available_memory_mb:.2f} MB)")

        g = Graph()
        try:
            logger.info(f"Parsing RDF file: {file_path}")
            g.parse(file_path, format="xml")
            logger.info(f"Successfully parsed RDF file with {len(g)} triples")
        except Exception as e:
            logger.error(f"Error parsing RDF file {file_path}: {e}")
            return False

        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
        concept_count = 0

        for s, p, o in g.triples((None, RDF.type, None)):
            uri = str(s)
            concept = TaxonomyConcept(uri=uri)
            concept.uuid = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1] if '/' in uri else "No UUID"
            for pred, obj in g.predicate_objects(s):
                pred_str = str(pred)
                obj_str = str(obj)
                if pred == SKOS.prefLabel:
                    concept.pref_label = obj_str
                elif pred == SKOS.altLabel:
                    concept.alt_labels.append(obj_str)
                elif pred == SKOS.hiddenLabel:
                    concept.hidden_labels.append(obj_str)
                elif pred == SKOS.definition:
                    concept.definition = obj_str
                elif pred == SKOS.broader:
                    concept.broader.append(obj_str)
                elif pred == SKOS.narrower:
                    concept.narrower.append(obj_str)
                elif pred == SKOS.related:
                    concept.related.append(obj_str)
                elif pred == SKOS.exactMatch:
                    concept.exact_matches.append(obj_str)
                elif pred == SKOS.closeMatch:
                    concept.close_matches.append(obj_str)
                elif pred_str.endswith('externalID'):
                    concept.external_id = obj_str
                elif pred_str.endswith('EgTravelCoreConcept-DesMetURN'):
                    concept.desmet_urn = obj_str
                elif pred_str.endswith('simplifiedDisplayLabel'):
                    concept.simplified_display_label = obj_str

            if concept.pref_label:
                self.add_concept(concept)
                concept_count += 1
                if concept_count % 1000 == 0:
                    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    logger.info(f"Processed {concept_count} concepts. Memory usage: {mem_usage:.2f} MB")

        logger.info(f"Extracted {concept_count} concepts with preferred labels")
        self.link_concepts()
        self.resolve_hierarchy_labels()

        for uri, concept in self.concepts.items():
            concept_label = concept.pref_label if concept.pref_label else ""
            if concept_label:
                embedding = self.semantic_model.encode(concept_label, convert_to_tensor=True)
                self.concept_embeddings[uri] = embedding

        try:
            logger.info(f"Saving parsed taxonomy to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'concepts': self.concepts,
                    'label_to_concepts': self.label_to_concepts,
                    'related_concepts': self.related_concepts,
                    'concept_embeddings': self.concept_embeddings
                }, f)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

        return True

    def add_concept(self, concept):
        self.concepts[concept.uri] = concept
        for label in concept.get_all_labels():
            if label:
                normalized_label = self._normalize_label(label)
                self.label_to_concepts[normalized_label].append(concept)

    def _normalize_label(self, label):
        if not label:
            return ""
        normalized = label.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def link_concepts(self):
        for uri, concept in self.concepts.items():
            for related_uri in concept.related:
                self.related_concepts[uri].append(related_uri)

    def resolve_hierarchy_labels(self):
        for uri, concept in self.concepts.items():
            concept.broader = [self._resolve_uri_to_label(b_uri) for b_uri in concept.broader]
            concept.narrower = [self._resolve_uri_to_label(n_uri) for n_uri in concept.narrower]

    def _resolve_uri_to_label(self, uri):
        if uri in self.concepts and self.concepts[uri].pref_label:
            return self.concepts[uri].pref_label
        return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]

    def find_concepts_by_label(self, label):
        normalized = self._normalize_label(label)
        return self.label_to_concepts.get(normalized, [])

    def find_concepts_by_partial_label(self, partial_label):
        normalized = self._normalize_label(partial_label)
        results = []
        for label, concepts in self.label_to_concepts.items():
            if re.search(r'\b' + re.escape(normalized) + r'\b', label):
                results.extend(concepts)
                for concept in concepts:
                    related = self.find_related_concepts(concept.uri)
                    results.extend(related)
                    broader = [self.concepts.get(b) for b in concept.broader if b in self.concepts and isinstance(b, str)]
                    narrower = [self.concepts.get(n) for n in concept.narrower if n in self.concepts and isinstance(n, str)]
                    results.extend([c for c in broader if c])
                    results.extend([c for c in narrower if c])
        return list(set(results))

    def find_related_concepts(self, uri):
        related_uris = self.related_concepts.get(uri, [])
        return [self.concepts.get(related_uri) for related_uri in related_uris if related_uri in self.concepts]

class ConceptMapper:
    COMPONENT_PATTERNS = [
        (r'24hour|allday|overnight', 'time'),
        (r'desk|service|staff|reception|concierge|checkin|checkout', 'service'),
        (r'beach|mountain|lake|city|airport|downtown|ocean|rural', 'location'),
        (r'yoga|swimming|skiing|tour|spa|fitness|golf|tennis|archery|shooting|climbing|skydiving|surfing|kayaking', 'activity'),
        (r'restaurant|dining|meal|bar|buffet|kitchen|cuisine', 'food'),
        (r'room|suite|bathroom|villa|apartment|cabin|property|balcony|bedroom|kitchen|pool|garage', 'accommodation'),
        (r'access|accessible|mobility|handicap', 'accessibility'),
        (r'shuttle|parking|transport|car|carrental', 'transportation'),
        (r'wifi|internet|smart|tech|digital', 'technology'),
        (r'sanitized|health|safe|wellness|medical', 'health'),
        (r'meeting|conference|wedding|event|venue', 'event'),
        (r'pet|dog|cat', 'pet'),
        (r'luxury|budget|economy|basic', 'luxury'),
        (r'smoking|nonsmoking', 'smoking'),
        (r'private|shared|public', 'privacy'),
        (r'free|paid|charge', 'modifier'),
        (r'solo|family|business|group|context', 'travelercontext'),
        (r'bookingdata|primarycontact|unitoccupant', 'bookingdata'),
        (r'onlinecheckin|skipthecounter|checkin', 'checkinprocess'),
        (r'checkout|signal|financialadjustment|giftcard', 'checkout'),
        (r'payment|creditcard|paypal|paymenterror', 'payment'),
        (r'true|false|yes|no|value', 'datavalues'),
        (r'price|discount|fee|demandprice', 'pricing'),
        (r'timezone|utc|est|pst', 'timezone'),
        (r'cancel|cancellation|penalty|reason', 'cancellation'),
        (r'preference|affinity|beachlover|adventure', 'preference'),
        (r'review|verified|positive|negative|category|state', 'review'),
        (r'policy|terms|cancellationpolicy|paymentpolicy', 'policy'),
        (r'rating|star|5star|3star', 'rating'),
        (r'culture|heritage|traditional', 'cultural'),
        (r'eco|sustainable|green', 'sustainability'),
        (r'wellness|health|retreat', 'wellness'),
        (r'cardio|gym|fitness|treadmill|elliptical', 'fitness'),
        (r'air|ac|centralair|heating|cooling', 'accommodation'),
        (r'theater|theatre|cinema|movies|performance|hall|venue|concert|opera|gallery|museum|park|beach|zoo|aquarium|stadium|arena|mall|market|landmark|monument', 'poi'),
        (r'near|close|by|adjacent|proximity', 'proximity'),
    ]

    def __init__(self, taxonomy_parser):
        self.taxonomy_parser = taxonomy_parser
        self.fallback_categories = {
            'service': ['reception', 'service', 'staff', 'amenity', 'concierge', 'checkin', 'checkout'],
            'location': ['location', 'area', 'place', 'view', 'ocean', 'downtown', 'rural'],
            'food': ['dining', 'restaurant', 'meal', 'cuisine', 'bar', 'buffet', 'kitchen'],
            'activity': ['recreation', 'activity', 'sport', 'leisure', 'spa', 'fitness', 'tour', 'archery', 'climbing', 'skydiving', 'surfing', 'kayaking'],
            'accommodation': ['room', 'suite', 'property', 'unit', 'villa', 'apartment', 'cabin', 'balcony', 'bathroom', 'bedroom', 'kitchen', 'pool', 'garage', 'air', 'centralair', 'heating', 'cooling'],
            'facility': ['facility', 'feature', 'equipment', 'infrastructure', 'pool', 'gym', 'parking'],
            'transportation': ['transportation', 'shuttle', 'parking', 'public transport', 'car rental', 'carrental'],
            'accessibility': ['accessible', 'disability', 'mobility', 'handicap'],
            'technology': ['wifi', 'internet', 'tech', 'smart', 'digital'],
            'health': ['health', 'safety', 'sanitization', 'medical', 'wellness'],
            'event': ['event', 'meeting', 'conference', 'wedding', 'venue'],
            'pet': ['pet', 'dog', 'cat'],
            'luxury': ['luxury', 'budget', 'economy', 'basic'],
            'smoking': ['smoking', 'nonsmoking'],
            'privacy': ['private', 'shared', 'public'],
            'travelercontext': ['solo', 'family', 'business', 'group', 'context'],
            'bookingdata': ['bookingdata', 'primarycontact', 'unitoccupant'],
            'checkinprocess': ['checkin', 'onlinecheckin', 'skipthecounter'],
            'checkout': ['checkout', 'signal', 'financialadjustment', 'giftcard'],
            'payment': ['payment', 'creditcard', 'paypal', 'paymenterror'],
            'datavalues': ['true', 'false', 'yes', 'no', 'value'],
            'pricing': ['price', 'discount', 'fee', 'demandprice'],
            'timezone': ['timezone', 'utc', 'est', 'pst'],
            'cancellation': ['cancel', 'cancellation', 'penalty', 'reason'],
            'preference': ['preference', 'affinity', 'beachlover', 'adventure'],
            'review': ['review', 'verified', 'positive', 'negative', 'category', 'state'],
            'policy': ['policy', 'terms', 'cancellationpolicy', 'paymentpolicy'],
            'rating': ['rating', 'star', '5star', '3star'],
            'fitness': ['cardio', 'gym', 'fitness', 'treadmill', 'elliptical', 'exercise'],
            'poi': ['theater', 'cinema', 'performance', 'hall', 'venue', 'museum', 'gallery', 'park', 'beach', 'zoo', 'aquarium', 'stadium', 'arena', 'mall', 'market', 'landmark', 'monument'],
        }
        self.contradictory_patterns = {
            '24hour': ['limited hours', 'seasonal', 'closure'],
            'accessible': ['inaccessible', 'stairs only'],
            'free': ['fee', 'charge', 'paid'],
            'private': ['shared', 'public'],
            'restaurant': ['closed', 'unavailable'],
            'petfriendly': ['no pets', 'pet-free'],
            'familyfriendly': ['adults only', 'no children'],
            'quiet': ['noisy', 'lively', 'bustling'],
            'luxury': ['budget', 'economy', 'basic'],
            'nonsmoking': ['smoking allowed', 'smoking area'],
            'onsite': ['offsite', 'nearby'],
            'solo': ['group', 'family'],
            'positivereview': ['negative review', 'poor review'],
            '5star': ['1star', '2star', 'low rating'],
            'onlinecheckin': ['manualcheckin', 'inpersoncheckin'],
            'offthegrid': ['wifi', 'internet', 'smart', 'tech', 'digital'],
            'lakeviewroom': ['oceanview', 'mountainview'],
        }
        self.inappropriate_contexts = {
            'activity': ['lobby', 'reception', 'checkin', 'checkout'],
            'fitness': ['lobby', 'reception', 'kitchen'],
            'poi': ['room', 'suite', 'bathroom', 'kitchen'],
        }
        self.sub_themes = {
            "Indoor Amenities": ['room', 'suite', 'bathroom', 'kitchen', 'jacuzzi', 'fireplace', '24hour', 'air', 'centralair', 'heating', 'cooling'],
            "Outdoor Amenities": ['pool', 'garden', 'beach', 'terrace', 'patio', 'balcony'],
            "Activities": ['yoga', 'hiking', 'skiing', 'tour', 'spa', 'cooking', 'art', 'archery', 'climbing', 'skydiving', 'surfing', 'kayaking'],
            "Sentiment": ['luxury', 'quiet', 'romantic', 'budget', 'friendly'],
            "Imagery": ['view', 'ocean', 'mountain', 'historical', 'scenic'],
            "Cultural Immersion": ['culture', 'heritage', 'traditional', 'festival', 'indigenous'],
            "Policy Context": ['policy', 'pet', 'smoking', 'cancellation', 'allinclusive'],
            "Implementation": ['checkin', 'shuttle', 'wifi', 'booking', 'transportation'],
            "Sustainability": ['eco', 'green', 'sustainable', 'carbon', 'organic'],
            "Traveler Demographics": ['family', 'solo', 'couples', 'adult', 'kids'],
            "Wellness & Health": ['wellness', 'health', 'retreat', 'spa', 'meditation', 'cardio', 'gym', 'fitness', 'exercise'],
            "Adventure Sports": ['action', 'extreme', 'adventure', 'skydiving', 'surfing'],
            "Points of Interest": ['theater', 'cinema', 'performance', 'hall', 'venue', 'museum', 'gallery', 'park', 'beach', 'zoo', 'aquarium', 'stadium', 'arena', 'mall', 'market', 'landmark', 'monument'],
        }
        self.manual_mappings = {
            'cardioequipment': {'pref_label': 'Cardio Equipment', 'uri': 'urn:manual:cardioequipment', 'uuid': 'cardioequipment', 'type': 'Wellness-Based', 'sub_theme': 'Wellness & Health'},
            'centralair': {'pref_label': 'Central Air Conditioning', 'uri': 'urn:manual:centralair', 'uuid': 'centralair', 'type': 'Lodging-Based', 'sub_theme': 'Indoor Amenities'},
            'climbingwall': {'pref_label': 'Climbing Wall', 'uri': 'urn:manual:climbingwall', 'uuid': 'climbingwall', 'type': 'Activity-Based', 'sub_theme': 'Activities'},
            'actionsports': {'pref_label': 'Action Sports', 'uri': 'urn:manual:actionsports', 'uuid': 'actionsports', 'type': 'Activity-Based', 'sub_theme': 'Adventure Sports'},
            'archeryrange': {'pref_label': 'Archery Range', 'uri': 'urn:manual:archeryrange', 'uuid': 'archeryrange', 'type': 'Activity-Based', 'sub_theme': 'Activities'},
        }
        self.theme_weights = {
            "Indoor Amenities": 0.25,
            "Outdoor Amenities": 0.15,
            "Activities": 0.15,
            "Sentiment": 0.10,
            "Imagery": 0.10,
            "Cultural Immersion": 0.05,
            "Policy Context": 0.10,
            "Implementation": 0.05,
            "Sustainability": 0.05,
            "Traveler Demographics": 0.05,
            "Wellness & Health": 0.10,
            "Adventure Sports": 0.10,
            "Points of Interest": 0.15
        }

    def map_concept(self, pause_flag, travel_concept):
        while pause_flag.value:
            time.sleep(1)

        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        if current_memory > available_memory * 0.8:
            logger.warning(f"High memory usage: {current_memory:.2f}MB used, {available_memory:.2f}MB available")

        if travel_concept.lower() in self.manual_mappings:
            manual = self.manual_mappings[travel_concept.lower()]
            return [{
                'Travel Concept': travel_concept,
                'Taxonomy Definition': manual['pref_label'],
                'Matched Taxonomy URN': manual['uri'],
                'Matched Taxonomy UUID': manual['uuid'],
                'Concept Type': manual['type'],
                'Matched Sub-Theme': manual['sub_theme'],
                'Theme Weight': self.theme_weights.get(manual['sub_theme'], 0.05),
                'Sub-Score': 'Property Attribute Affinity',
                'Rationale': 'Manual mapping',
                'Confidence (%)': 95,
                'Must_Not_Have Concepts': 'N/A',
                'Negative_Overrides': 'N/A',
                'Fallback_Logic': 'Manual mapping applied',
                'Semantic Similarity': 1.0,
                'Themes': json.dumps([{
                    'name': manual['sub_theme'],
                    'type': 'structural',
                    'rule': 'Optional',
                    'weight': self.theme_weights.get(manual['sub_theme'], 0.05),
                    'sub_score': 'Property Attribute Affinity',
                    'attributes': [{'skos:prefLabel': manual['pref_label'], 'uri': manual['uri'], 'confidence': 95, 'concept_weight': 0.95}]
                }])
            }]

        results = []
        components = self._component_analysis(travel_concept)
        travel_embedding = self.taxonomy_parser.semantic_model.encode(travel_concept, convert_to_tensor=True)

        is_proximity = any('proximity' in comp[1] for comp in components)
        is_poi = any('poi' in comp[1] for comp in components)
        domain = 'poi' if is_proximity or is_poi else self._determine_domain(travel_concept)

        direct_matches = self._find_direct_matches(travel_concept)
        themes = {}
        if direct_matches:
            for concept in direct_matches:
                concept_embedding = self.taxonomy_parser.concept_embeddings.get(concept.uri)
                if concept_embedding is None:
                    concept_label = concept.pref_label if concept.pref_label else ""
                    concept_embedding = self.taxonomy_parser.semantic_model.encode(concept_label, convert_to_tensor=True)
                semantic_similarity = util.cos_sim(travel_embedding, concept_embedding).item()

                if semantic_similarity > 0.4:
                    sub_theme = "Points of Interest" if domain == 'poi' else self._determine_sub_theme(travel_concept)
                    if domain in self.inappropriate_contexts:
                        forbidden_contexts = self.inappropriate_contexts[domain]
                        if any(ctx in concept.pref_label.lower() for ctx in forbidden_contexts):
                            logger.info(f"Skipping match for '{travel_concept}' to '{concept.pref_label}' due to inappropriate context")
                            continue
                    confidence = self._calculate_confidence(travel_concept, concept, 'direct')
                    confidence = self._adjust_confidence_for_semantics(semantic_similarity, confidence)
                    if is_proximity:
                        confidence = min(100, confidence + 10)
                    theme_key = sub_theme.lower().replace(' ', '_')
                    if theme_key not in themes:
                        themes[theme_key] = {
                            'name': sub_theme,
                            'type': 'structural' if sub_theme not in ['Sentiment', 'Imagery'] else sub_theme.lower(),
                            'rule': 'Optional' if sub_theme != 'Indoor Amenities' else 'Must have 1',
                            'base_weight': self.theme_weights.get(sub_theme, 0.05),
                            'attributes': []
                        }
                    themes[theme_key]['attributes'].append({
                        'skos:prefLabel': concept.pref_label,
                        'uri': concept.uri,
                        'confidence': confidence,
                        'concept_weight': confidence / 100,
                        'source': 'TaxonomyParser'
                    })
                    results.append(self._create_mapping_result(travel_concept, concept, confidence, 'Direct match', sub_theme, semantic_similarity))
            if results:
                logger.info(f"Mapped '{travel_concept}' to {len(results)} direct matches")

        if components and not results:
            for component, component_type in components:
                component_matches = self._find_component_matches(component, component_type)
                if component_matches:
                    match_type = 'partial'
                    for concept in component_matches:
                        concept_embedding = self.taxonomy_parser.concept_embeddings.get(concept.uri)
                        if concept_embedding is None:
                            concept_label = concept.pref_label if concept.pref_label else ""
                            concept_embedding = self.taxonomy_parser.semantic_model.encode(concept_label, convert_to_tensor=True)
                        semantic_similarity = util.cos_sim(travel_embedding, concept_embedding).item()

                        if semantic_similarity > 0.4:
                            sub_theme = "Points of Interest" if domain == 'poi' else self._determine_sub_theme(component)
                            if domain in self.inappropriate_contexts:
                                forbidden_contexts = self.inappropriate_contexts[domain]
                                if any(ctx in concept.pref_label.lower() for ctx in forbidden_contexts):
                                    logger.info(f"Skipping match for '{travel_concept}' to '{concept.pref_label}' due to inappropriate context")
                                    continue
                            confidence = self._calculate_confidence(travel_concept, concept, match_type)
                            confidence = self._adjust_confidence_for_relevance(travel_concept, component, concept, confidence)
                            confidence = self._adjust_confidence_for_semantics(semantic_similarity, confidence)
                            if confidence >= PARTIAL_CONFIDENCE_THRESHOLD:
                                rationale = f"Partial: Matched '{component}'"
                                if is_proximity:
                                    confidence = min(100, confidence + 10)
                                theme_key = sub_theme.lower().replace(' ', '_')
                                if theme_key not in themes:
                                    themes[theme_key] = {
                                        'name': sub_theme,
                                        'type': 'structural' if sub_theme not in ['Sentiment', 'Imagery'] else sub_theme.lower(),
                                        'rule': 'Optional' if sub_theme != 'Indoor Amenities' else 'Must have 1',
                                        'base_weight': self.theme_weights.get(sub_theme, 0.05),
                                        'attributes': []
                                    }
                                themes[theme_key]['attributes'].append({
                                    'skos:prefLabel': concept.pref_label,
                                    'uri': concept.uri,
                                    'confidence': confidence,
                                    'concept_weight': confidence / 100,
                                    'source': 'TaxonomyParser'
                                })
                                results.append(self._create_mapping_result(travel_concept, concept, confidence, rationale, sub_theme, semantic_similarity))
            if results:
                logger.info(f"Mapped '{travel_concept}' to {len(results)} partial matches")

        if not results or all(r['Confidence (%)'] < PARTIAL_CONFIDENCE_THRESHOLD for r in results):
            domain = 'poi' if is_proximity or is_poi else self._determine_domain(travel_concept)
            fallback_concepts = self._find_fallback_concepts(domain)
            if fallback_concepts:
                concept = fallback_concepts[0]
                concept_embedding = self.taxonomy_parser.concept_embeddings.get(concept.uri)
                if concept_embedding is None:
                    concept_label = concept.pref_label if concept.pref_label else ""
                    concept_embedding = self.taxonomy_parser.semantic_model.encode(concept_label, convert_to_tensor=True)
                semantic_similarity = util.cos_sim(travel_embedding, concept_embedding).item()

                if semantic_similarity > 0.2:
                    sub_theme = "Points of Interest" if domain == 'poi' else 'N/A'
                    confidence = self._calculate_confidence(travel_concept, concept, 'weak')
                    confidence = self._adjust_confidence_for_semantics(semantic_similarity, confidence)
                    if is_proximity:
                        confidence = min(100, confidence + 10)
                    theme_key = sub_theme.lower().replace(' ', '_')
                    if theme_key not in themes:
                        themes[theme_key] = {
                            'name': sub_theme,
                            'type': 'location' if domain == 'poi' else 'structural',
                            'rule': 'Optional',
                            'base_weight': self.theme_weights.get(sub_theme, 0.05),
                            'attributes': []
                        }
                    themes[theme_key]['attributes'].append({
                        'skos:prefLabel': concept.pref_label,
                        'uri': concept.uri,
                        'confidence': confidence,
                        'concept_weight': confidence / 100,
                        'source': 'TaxonomyParser'
                    })
                    results = [self._create_mapping_result(travel_concept, concept, confidence, f"Fallback: General {domain} concept", sub_theme, semantic_similarity)]
                    logger.info(f"Mapped '{travel_concept}' to fallback concept: {concept.pref_label}")
            else:
                concept_type = 'Location-Based' if domain == 'poi' else 'Generic'
                sub_theme = "Points of Interest" if domain == 'poi' else 'N/A'
                results = [{
                    'Travel Concept': travel_concept,
                    'Taxonomy Definition': 'No direct match in taxonomy',
                    'Matched Taxonomy URN': 'No URN Match',
                    'Matched Taxonomy UUID': 'No UUID',
                    'Concept Type': concept_type,
                    'Matched Sub-Theme': sub_theme,
                    'Theme Weight': self.theme_weights.get(sub_theme, 0.05),
                    'Sub-Score': 'Property Attribute Affinity',
                    'Rationale': 'No match found in the taxonomy',
                    'Confidence (%)': 0,
                    'Must_Not_Have Concepts': 'N/A',
                    'Negative_Overrides': 'N/A',
                    'Fallback_Logic': f"No appropriate {domain} fallback found",
                    'Semantic Similarity': 0.0,
                    'Themes': json.dumps([])
                }]
                logger.info(f"No match found for '{travel_concept}'")

        # Normalize theme weights after all themes are populated
        total_base_weight = sum(theme['base_weight'] for theme in themes.values())
        for theme in themes.values():
            theme['weight'] = theme['base_weight'] / total_base_weight if total_base_weight > 0 else 0
            del theme['base_weight']
            total_concept_weight = sum(attr['concept_weight'] for attr in theme['attributes'])
            for attr in theme['attributes']:
                attr['concept_weight'] = attr['concept_weight'] / total_concept_weight if total_concept_weight > 0 else 1.0
            theme['sub_score'] = {
                'indoor_amenities': 'Property Attribute Affinity',
                'outdoor_amenities': 'Property Attribute Affinity',
                'activities': 'Property Attribute Affinity',
                'sentiment': 'Sentiment Scores',
                'imagery': 'Image Affinity Scores',
                'cultural_immersion': 'Destination Affinity Scores',
                'policy_context': 'Property Attribute Affinity',
                'implementation': 'Property Attribute Affinity',
                'sustainability': 'Property Attribute Affinity',
                'traveler_demographics': 'Group Intelligence Scores',
                'wellness_&_health': 'Property Attribute Affinity',
                'adventure_sports': 'Property Attribute Affinity',
                'points_of_interest': 'Geospatial Affinity Scores'
            }.get(theme['name'].lower().replace(' ', '_'), 'Property Attribute Affinity')

        if results:
            best_result = max(results, key=lambda x: x['Confidence (%)'])
            best_result['Themes'] = json.dumps(list(themes.values()))
            results = [best_result]
            logger.info(f"After deduplication, kept 1 result with {len(themes)} themes for '{travel_concept}'")

        return results

    def _component_analysis(self, travel_concept):
        components = []
        normalized = self.taxonomy_parser._normalize_label(travel_concept)
        if self.taxonomy_parser.find_concepts_by_label(normalized):
            return [(travel_concept, 'phrase')]
        for pattern, c_type in self.COMPONENT_PATTERNS:
            if re.search(pattern, travel_concept, re.IGNORECASE):
                match = re.search(pattern, travel_concept, re.IGNORECASE)
                components.append((match.group(0), c_type))
        if not components:
            words = re.findall(r'[A-Za-z]+', travel_concept)
            components = [(word, 'unknown') for word in words]
        return components

    def _find_direct_matches(self, travel_concept):
        matches = self.taxonomy_parser.find_concepts_by_label(travel_concept)
        if not matches:
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', travel_concept).replace('_', ' ').replace('-', ' ')
            if spaced != travel_concept:
                matches = self.taxonomy_parser.find_concepts_by_label(spaced)
        return matches

    def _find_component_matches(self, component, component_type):
        matches = self.taxonomy_parser.find_concepts_by_label(component)
        if not matches:
            matches = self.taxonomy_parser.find_concepts_by_partial_label(component)
        return matches

    def _calculate_confidence(self, travel_concept, concept, match_type):
        if isinstance(concept, dict):
            return 0 if match_type == 'none' else 1
        min_score, max_score = CONFIDENCE_SCORES[match_type]
        confidence = (min_score + max_score) / 2
        if match_type == 'direct' and concept.pref_label.lower() == travel_concept.lower():
            confidence = max_score
        elif concept.definition:
            tc_words = set(re.findall(r'\w+', travel_concept.lower()))
            def_words = set(re.findall(r'\w+', concept.definition.lower()))
            overlap = len(tc_words.intersection(def_words)) / max(1, len(tc_words))
            confidence += (max_score - min_score) * overlap * 0.5
        return round(max(min_score, min(max_score, confidence)))

    def _adjust_confidence_for_relevance(self, travel_concept, component, concept, base_confidence):
        if len(travel_concept) <= 3 and len(concept.pref_label) > len(travel_concept):
            if concept.definition:
                tc_words = set(re.findall(r'\w+', travel_concept.lower()))
                def_words = set(re.findall(r'\w+', concept.definition.lower()))
                overlap = len(tc_words.intersection(def_words)) / max(1, len(tc_words))
                if overlap < 0.1:
                    min_score, max_score = CONFIDENCE_SCORES['weak']
                    return round(min_score + (max_score - min_score) * 0.1)
        return base_confidence

    def _adjust_confidence_for_semantics(self, semantic_similarity, base_confidence):
        adjusted_confidence = base_confidence * semantic_similarity
        return round(max(0, min(100, adjusted_confidence)))

    def _determine_domain(self, travel_concept):
        for domain, keywords in self.fallback_categories.items():
            if any(kw in travel_concept.lower() for kw in keywords):
                return domain
        return 'service'

    def _determine_sub_theme(self, component):
        for theme, keywords in self.sub_themes.items():
            if any(kw in component.lower() for kw in keywords):
                return theme
        return 'N/A'

    def _find_fallback_concepts(self, domain):
        fallback_labels = self.fallback_categories.get(domain, [])
        for label in fallback_labels:
            concepts = self.taxonomy_parser.find_concepts_by_partial_label(label)
            if concepts:
                return concepts
        for label in ['general', 'miscellaneous', 'other']:
            concepts = self.taxonomy_parser.find_concepts_by_partial_label(label)
            if concepts:
                return concepts
        return list(self.taxonomy_parser.concepts.values())[:1] if self.taxonomy_parser.concepts else []

    def _find_must_not_have_concepts(self, travel_concept, concept):
        must_not = []
        for pattern, contradictions in self.contradictory_patterns.items():
            if pattern in travel_concept.lower():
                for contra in contradictions:
                    contra_concepts = self.taxonomy_parser.find_concepts_by_partial_label(contra)
                    for c in contra_concepts:
                        if c.uri != concept.uri:
                            must_not.append(f"{c.uri} - {c.pref_label}")
        return must_not if must_not else ["N/A"]

    def _find_negative_overrides(self, concept):
        return ["N/A"]

    def _create_mapping_result(self, travel_concept, concept, confidence, rationale, sub_theme, semantic_similarity=0.0):
        must_not = self._find_must_not_have_concepts(travel_concept, concept) if not isinstance(concept, dict) else ["N/A"]
        neg_overrides = self._find_negative_overrides(concept) if not isinstance(concept, dict) else ["N/A"]
        domain = self._determine_domain(travel_concept)
        concept_type_map = {
            'accommodation': 'Lodging-Based',
            'activity': 'Activity-Based',
            'service': 'Service-Based',
            'policy': 'Policy-Based',
            'health': 'Wellness-Based',
            'wellness': 'Wellness-Based',
            'cultural': 'Cultural-Based',
            'sustainability': 'Sustainability-Based',
            'travelercontext': 'Demographic-Based',
            'location': 'Location-Based',
            'event': 'Event-Based',
            'fitness': 'Wellness-Based',
            'poi': 'Location-Based',
            'proximity': 'Location-Based',
        }
        if 'eco' in travel_concept.lower() or 'sustainable' in travel_concept.lower():
            concept_type = 'Sustainability-Based'
        elif 'view' in travel_concept.lower() or 'front' in travel_concept.lower():
            concept_type = 'Location-Based'
        elif 'festival' in travel_concept.lower() or 'wedding' in travel_concept.lower():
            concept_type = 'Event-Based'
        else:
            concept_type = concept_type_map.get(domain, 'Generic')

        fallback = self._find_fallback_concepts(domain)
        fallback_str = f"{fallback[0].uri} - {fallback[0].pref_label}" if fallback else f"No {domain} fallback"
        return {
            'Travel Concept': travel_concept,
            'Taxonomy Definition': concept.pref_label if not isinstance(concept, dict) else concept['Taxonomy Definition'],
            'Matched Taxonomy URN': concept.uri if not isinstance(concept, dict) else concept['Matched Taxonomy URN'],
            'Matched Taxonomy UUID': concept.uuid if not isinstance(concept, dict) else concept['Matched Taxonomy UUID'],
            'Concept Type': concept_type,
            'Matched Sub-Theme': sub_theme,
            'Theme Weight': self.theme_weights.get(sub_theme, 0.05),
            'Sub-Score': {
                'Indoor Amenities': 'Property Attribute Affinity',
                'Outdoor Amenities': 'Property Attribute Affinity',
                'Activities': 'Property Attribute Affinity',
                'Sentiment': 'Sentiment Scores',
                'Imagery': 'Image Affinity Scores',
                'Cultural Immersion': 'Destination Affinity Scores',
                'Policy Context': 'Property Attribute Affinity',
                'Implementation': 'Property Attribute Affinity',
                'Sustainability': 'Property Attribute Affinity',
                'Traveler Demographics': 'Group Intelligence Scores',
                'Wellness & Health': 'Property Attribute Affinity',
                'Adventure Sports': 'Property Attribute Affinity',
                'Points of Interest': 'Geospatial Affinity Scores'
            }.get(sub_theme, 'Property Attribute Affinity'),
            'Rationale': rationale,
            'Confidence (%)': confidence,
            'Must_Not_Have Concepts': "; ".join(must_not),
            'Negative_Overrides': "; ".join(neg_overrides),
            'Fallback_Logic': fallback_str,
            'Semantic Similarity': round(semantic_similarity, 2),
            'Themes': ''  # Will be populated later
        }

def is_travel_related(concept):
    travel_keywords = [
        'room', 'hotel', 'resort', 'villa', 'suite', 'travel', 'tour', 'activity', 'sport',
        'beach', 'mountain', 'city', 'restaurant', 'dining', 'spa', 'fitness', 'pool', 'checkin',
        'checkout', 'booking', 'pet', 'family', 'luxury', 'budget', 'policy', 'review', 'rating',
        'equipment', 'air', 'climbing', 'wall'
    ]
    poi_keywords = [
        'theater', 'cinema', 'performance', 'hall', 'venue', 'museum', 'gallery', 'park', 'beach',
        'zoo', 'aquarium', 'stadium', 'arena', 'mall', 'market', 'landmark', 'monument'
    ]
    proximity_keywords = ['near', 'close', 'by', 'adjacent', 'proximity']
    potentially_irrelevant = ['acting', 'drama']
    if any(keyword in concept.lower() for keyword in potentially_irrelevant):
        if any(keyword in concept.lower() for keyword in travel_keywords + poi_keywords + proximity_keywords + ['workshop', 'tour']):
            logger.info(f"Allowing potentially irrelevant concept '{concept}' due to travel/POI context")
            return True
        logger.info(f"Flagging potentially irrelevant concept '{concept}'")
        return False
    return True

def pause_handler(signum, frame, pause_flag, current_position, total_concepts, taxonomy_file, pool=None):
    pause_flag.value = not pause_flag.value
    status = "PAUSED" if pause_flag.value else "RESUMED"
    logger.info(f"Processing {status}")
    print(f"\nProcessing {status}. Press {'Ctrl+Z' if platform.system() != 'Windows' else 'Ctrl+C'} again to {'resume' if pause_flag.value else 'pause'}.")
    if pause_flag.value:
        save_state(current_position, total_concepts, taxonomy_file)
        print(f"State saved at position {current_position}/{total_concepts}")

def save_state(concepts_processed, total_concepts, taxonomy_file):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'concepts_processed': concepts_processed,
                'total_concepts': total_concepts,
                'taxonomy_file': taxonomy_file,
                'timestamp': time.time()
            }, f)
        logger.info(f"Saved state: {concepts_processed}/{total_concepts} concepts processed")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    return None

def process_taxonomy(args):
    concepts_file, taxonomy_file, output_file, resume, pause_flag, test = args

    if platform.system() != 'Windows':
        signal.signal(signal.SIGTSTP, lambda signum, frame: pause_handler(signum, frame, pause_flag, 0, 0, taxonomy_file))
    else:
        signal.signal(signal.SIGINT, lambda signum, frame: pause_handler(signum, frame, pause_flag, 0, 0, taxonomy_file))

    travel_concepts = []
    if concepts_file and os.path.exists(concepts_file):
        with open(concepts_file, 'r', encoding='utf-8') as f:
            travel_concepts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        travel_concepts = ["24hourfrontdesk", "accessiblebathroom", "yoga", "àlacarterestaurant", "beachaccess"]
        logger.info("Using default travel concepts...")

    seen = set()
    travel_concepts = [x for x in travel_concepts if not (x in seen or seen.add(x))]
    travel_concepts = [concept for concept in travel_concepts if is_travel_related(concept)]
    logger.info(f"Filtered to {len(travel_concepts)} travel-related concepts: {travel_concepts[:5]}...")

    if test:
        travel_concepts = travel_concepts[:5]
        logger.info(f"Test mode enabled: Processing only the first 5 concepts: {travel_concepts}")

    total_concepts = len(travel_concepts)
    logger.info(f"Total unique concepts to process for {taxonomy_file}: {total_concepts}")

    start_index = 0
    all_mappings = []
    if resume and not test:
        state = load_state()
        if state and state['taxonomy_file'] == taxonomy_file:
            start_index = state['concepts_processed']
            logger.info(f"Resuming from concept #{start_index}")
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        all_mappings = list(reader)
                        logger.info(f"Loaded {len(all_mappings)} existing mappings from {output_file}")
                except Exception as e:
                    logger.error(f"Error loading existing mappings: {e}")
                    all_mappings = []

    taxonomy_parser = TaxonomyParser()
    if not taxonomy_parser.parse_rdf(taxonomy_file):
        logger.error(f"Skipping {taxonomy_file} due to parsing error.")
        return

    if not taxonomy_parser.concepts:
        logger.error(f"No concepts loaded from {taxonomy_file}. Skipping.")
        return

    mapper = ConceptMapper(taxonomy_parser)
    start_time = time.time()

    with tqdm(total=total_concepts - start_index, desc=f"Processing concepts ({os.path.basename(taxonomy_file)})", unit="concept") as pbar:
        for i, travel_concept in enumerate(travel_concepts[start_index:], start=start_index + 1):
            mappings = mapper.map_concept(pause_flag, travel_concept)
            all_mappings.extend(mappings)
            pbar.update(1)

            if i % 100 == 0 or i == total_concepts:
                elapsed_time = time.time() - start_time
                percent_complete = (i / total_concepts) * 100
                concepts_remaining = total_concepts - i
                avg_time_per_concept = elapsed_time / (i - start_index) if i > start_index else 0
                estimated_time_remaining = avg_time_per_concept * concepts_remaining
                logger.info(
                    f"Processed {i}/{total_concepts} concepts ({percent_complete:.2f}%) for {taxonomy_file}. "
                    f"Elapsed time: {elapsed_time:.2f}s. "
                    f"Estimated time remaining: {estimated_time_remaining:.2f}s."
                )
                save_state(i, total_concepts, taxonomy_file)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'Travel Concept', 'Taxonomy Definition', 'Matched Taxonomy URN', 'Matched Taxonomy UUID',
            'Concept Type', 'Matched Sub-Theme', 'Theme Weight', 'Sub-Score', 'Rationale', 'Confidence (%)',
            'Must_Not_Have Concepts', 'Negative_Overrides', 'Fallback_Logic', 'Semantic Similarity', 'Themes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_mappings)

    total_time = time.time() - start_time
    logger.info(f"Completed processing {taxonomy_file}. Results written to {output_file}")
    logger.info(f"Processed {total_concepts} unique concepts, resulting in {len(all_mappings)} mappings.")
    logger.info(f"Total processing time for {taxonomy_file}: {total_time:.2f} seconds.")

def process_all_taxonomies(concepts_file, taxonomy_dir, output_dir, resume=False, test=False):
    if not os.path.exists(taxonomy_dir):
        logger.error(f"Taxonomy directory '{taxonomy_dir}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(taxonomy_dir):
        logger.error(f"'{taxonomy_dir}' is not a directory.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    taxonomy_files = [f for f in os.listdir(taxonomy_dir) if f.endswith(('.xml', '.rdf'))]
    if not taxonomy_files:
        logger.error(f"No .xml or .rdf files found in {taxonomy_dir}.")
        sys.exit(1)

    logger.info(f"Found {len(taxonomy_files)} taxonomy files to process: {taxonomy_files}")

    if platform.system() != 'Windows':
        os.setpgrp()

    manager = Manager()
    pause_flag = manager.Value('b', False)
    overall_start_time = time.time()
    tasks = [(concepts_file, os.path.join(taxonomy_dir, tf),
              os.path.join(output_dir, os.path.splitext(tf)[0] + "_mappings.csv"), resume, pause_flag, test) for tf in taxonomy_files]

    def parent_signal_handler(signum, frame):
        pause_flag.value = not pause_flag.value
        status = "PAUSED" if pause_flag.value else "RESUMED"
        logger.info(f"Parent process: Processing {status}")
        print(f"\nParent process: Processing {status}. Press {'Ctrl+Z' if platform.system() != 'Windows' else 'Ctrl+C'} again to {'resume' if pause_flag.value else 'pause'}.")

    if platform.system() != 'Windows':
        signal.signal(signal.SIGTSTP, parent_signal_handler)
    else:
        signal.signal(signal.SIGINT, parent_signal_handler)

    try:
        with Pool(processes=cpu_count()) as pool:
            pool.map(process_taxonomy, tasks)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Terminating child processes.")
        pool.terminate()
        pool.join()
        sys.exit(1)

    overall_time = time.time() - overall_start_time
    logger.info(f"All taxonomies processed. Total time: {overall_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description='Map travel concepts to taxonomy concepts across multiple taxonomies.')
    parser.add_argument('--concepts', help='Path to travel concepts file')
    parser.add_argument('--taxonomy-dir', required=True, help='Directory containing RDF/XML taxonomy files (.xml or .rdf)')
    parser.add_argument('--output-dir', required=True, help='Directory to save output CSV files')
    parser.add_argument('--resume', action='store_true', help='Resume from previous state if available')
    parser.add_argument('--test', action='store_true', help='Run in test mode with only the first 5 concepts')
    args = parser.parse_args()
    process_all_taxonomies(args.concepts, args.taxonomy_dir, args.output_dir, args.resume, args.test)

if __name__ == '__main__':
    main()