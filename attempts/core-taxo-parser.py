#!/usr/bin/env python3
"""
Travel Concepts to Taxonomy Mapping Tool with Enhanced Concept Types and Sub-Themes
"""

import csv
import re
import os
import argparse
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, URIRef
import sys
import time
from tqdm import tqdm
import logging
import pickle
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIDENCE_SCORES = {'direct': (90, 95), 'partial': (60, 75), 'weak': (1, 39), 'none': (0, 0)}
PARTIAL_CONFIDENCE_THRESHOLD = 40

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

    def parse_rdf(self, file_path):
        cache_file = file_path + ".pickle"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) > os.path.getmtime(file_path):
            logging.info(f"Loading cached taxonomy from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.concepts = cached_data['concepts']
                self.label_to_concepts = cached_data['label_to_concepts']
                self.related_concepts = cached_data['related_concepts']
            return True

        g = Graph()
        try:
            g.parse(file_path, format="xml")
        except Exception as e:
            print(f"Error parsing RDF file {file_path}: {e}")
            return False

        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
        RDF_TYPE = RDF.type

        for s, p, o in g.triples((None, RDF_TYPE, None)):
            uri = str(s)
            concept = TaxonomyConcept(uri=uri)
            if '#' in uri:
                concept.uuid = uri.split('#')[-1]
            else:
                concept.uuid = uri.split('/')[-1] if '/' in uri else "No UUID"

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

        self.link_concepts()
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'concepts': self.concepts,
                'label_to_concepts': self.label_to_concepts,
                'related_concepts': self.related_concepts
            }, f)
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
            for broader_uri in concept.broader:
                if broader_uri in self.concepts:
                    self.concepts[broader_uri].narrower.append(uri)
            for related_uri in concept.related:
                self.related_concepts[uri].append(related_uri)

    def find_concepts_by_label(self, label):
        normalized = self._normalize_label(label)
        return self.label_to_concepts.get(normalized, [])

    def find_concepts_by_partial_label(self, partial_label):
        normalized = self._normalize_label(partial_label)
        results = []
        for label, concepts in self.label_to_concepts.items():
            if re.search(r'\b' + re.escape(normalized) + r'\b', label):
                results.extend(concepts)
        return list(set(results))

    def find_related_concepts(self, uri):
        related_uris = self.related_concepts.get(uri, [])
        return [self.concepts.get(related_uri) for related_uri in related_uris if related_uri in self.concepts]

class ConceptMapper:
    def __init__(self, taxonomy_parser):
        self.taxonomy_parser = taxonomy_parser
        self.fallback_categories = {
            'service': ['reception', 'service', 'staff', 'amenity', 'concierge', 'checkin', 'checkout'],
            'location': ['location', 'area', 'place', 'view', 'ocean', 'downtown', 'rural'],
            'food': ['dining', 'restaurant', 'meal', 'cuisine', 'bar', 'buffet', 'kitchen'],
            'activity': ['recreation', 'activity', 'sport', 'leisure', 'spa', 'fitness', 'tour'],
            'accommodation': ['room', 'suite', 'property', 'unit', 'villa', 'apartment', 'cabin', 'balcony', 'bathroom', 'bedroom', 'kitchen', 'pool', 'garage'],
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
            'rating': ['rating', 'star', '5star', '3star']
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
            'onlinecheckin': ['manualcheckin', 'inpersoncheckin']
        }
        self.sub_themes = {
            "Indoor Amenities": ['room', 'suite', 'bathroom', 'kitchen', 'jacuzzi', 'fireplace', '24hour'],
            "Outdoor Amenities": ['pool', 'garden', 'beach', 'terrace', 'patio', 'balcony'],
            "Activities": ['yoga', 'hiking', 'skiing', 'tour', 'spa', 'cooking', 'art'],
            "Sentiment": ['luxury', 'quiet', 'romantic', 'budget', 'friendly'],
            "Imagery": ['view', 'ocean', 'mountain', 'historical', 'scenic'],
            "Cultural Immersion": ['culture', 'heritage', 'traditional', 'festival', 'indigenous'],
            "Policy Context": ['policy', 'pet', 'smoking', 'cancellation', 'allinclusive'],
            "Implementation": ['checkin', 'shuttle', 'wifi', 'booking', 'transportation'],
            "Sustainability": ['eco', 'green', 'sustainable', 'carbon', 'organic'],
            "Traveler Demographics": ['family', 'solo', 'couples', 'adult', 'kids'],
            "Wellness & Health": ['wellness', 'health', 'retreat', 'spa', 'meditation']
        }

    def map_concept(self, travel_concept):
        results = []
        components = self._component_analysis(travel_concept)

        direct_matches = self._find_direct_matches(travel_concept)
        if direct_matches:
            for concept in direct_matches:
                confidence = self._calculate_confidence(travel_concept, concept, 'direct')
                results.append(self._create_mapping_result(travel_concept, concept, confidence, 'Direct match', 'N/A'))
            return results

        if components:
            for component, component_type in components:
                component_matches = self._find_component_matches(component, component_type)
                if component_matches:
                    match_type = 'partial'
                    for concept in component_matches:
                        confidence = self._calculate_confidence(travel_concept, concept, match_type)
                        confidence = self._adjust_confidence_for_relevance(travel_concept, component, concept, confidence)
                        if confidence >= PARTIAL_CONFIDENCE_THRESHOLD:
                            sub_theme = self._determine_sub_theme(component)
                            rationale = f"Partial: Matched '{component}'"
                            results.append(self._create_mapping_result(travel_concept, concept, confidence, rationale, sub_theme))

        if not results or all(r['Confidence (%)'] < PARTIAL_CONFIDENCE_THRESHOLD for r in results):
            domain = self._determine_domain(travel_concept)
            fallback_concepts = self._find_fallback_concepts(domain)
            if fallback_concepts:
                concept = fallback_concepts[0]
                confidence = self._calculate_confidence(travel_concept, concept, 'weak')
                results = [self._create_mapping_result(travel_concept, concept, confidence, f"Fallback: General {domain} concept", 'N/A')]
            else:
                results = [{
                    'Travel Concept': travel_concept,
                    'Taxonomy Definition': 'No direct match in taxonomy',
                    'Matched Taxonomy URN': 'No URN Match',
                    'Matched Taxonomy UUID': 'No UUID',
                    'Concept Type': 'Generic',
                    'Matched Sub-Theme': 'N/A',
                    'Rationale': 'No match found in the taxonomy',
                    'Confidence (%)': 0,
                    'Must_Not_Have Concepts': 'N/A',
                    'Negative_Overrides': 'N/A',
                    'Fallback_Logic': f"No appropriate {domain} fallback found"
                }]

        return results

    def _component_analysis(self, travel_concept):
        patterns = [
            (r'24hour|allday|overnight', 'time'),
            (r'desk|service|staff|reception|concierge|checkin|checkout', 'service'),
            (r'beach|mountain|lake|city|airport|downtown|ocean|rural', 'location'),
            (r'yoga|swimming|skiing|tour|spa|fitness|golf|tennis', 'activity'),
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
            (r'wellness|health|retreat', 'wellness')
        ]
        components = []
        for pattern, c_type in patterns:
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

    def _determine_domain(self, travel_concept):
        domains = {
            'service': ['desk', 'service', 'staff', 'hour', 'concierge', 'checkin', 'checkout'],
            'location': ['beach', 'mountain', 'lake', 'city', 'access', 'downtown', 'ocean', 'rural'],
            'food': ['restaurant', 'dining', 'meal', 'bar', 'buffet', 'kitchen', 'cuisine'],
            'activity': ['yoga', 'swimming', 'skiing', 'spa', 'fitness', 'golf', 'tennis', 'tour'],
            'accommodation': ['room', 'suite', 'bathroom', 'villa', 'apartment', 'cabin', 'property', 'balcony', 'bedroom', 'kitchen', 'pool', 'garage'],
            'facility': ['pool', 'wifi', 'shuttle', 'parking', 'gym', 'facility', 'feature', 'equipment'],
            'transportation': ['shuttle', 'parking', 'transport', 'car', 'carrental'],
            'accessibility': ['accessible', 'mobility', 'handicap'],
            'technology': ['wifi', 'internet', 'smart', 'tech', 'digital'],
            'health': ['sanitized', 'health', 'safe', 'wellness', 'medical'],
            'event': ['meeting', 'conference', 'wedding', 'event', 'venue'],
            'pet': ['pet', 'dog', 'cat'],
            'luxury': ['luxury', 'budget', 'economy', 'basic'],
            'smoking': ['smoking', 'nonsmoking'],
            'privacy': ['private', 'shared', 'public'],
            'travelercontext': ['solo', 'family', 'business', 'group', 'context'],
            'bookingdata': ['bookingdata', 'primarycontact', 'unitoccupant'],
            'checkinprocess': ['onlinecheckin', 'skipthecounter', 'checkin'],
            'checkout': ['checkout', 'signal', 'financialadjustment', 'giftcard'],
            'payment': ['payment', 'creditcard', 'paypal', 'paymenterror'],
            'datavalues': ['true', 'false', 'yes', 'no', 'value'],
            'pricing': ['price', 'discount', 'fee', 'demandprice'],
            'timezone': ['timezone', 'utc', 'est', 'pst'],
            'cancellation': ['cancel', 'cancellation', 'penalty', 'reason'],
            'preference': ['preference', 'affinity', 'beachlover', 'adventure'],
            'review': ['review', 'verified', 'positive', 'negative', 'category', 'state'],
            'policy': ['policy', 'terms', 'cancellationpolicy', 'paymentpolicy'],
            'rating': ['rating', 'star', '5star', '3star']
        }
        for domain, keywords in domains.items():
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
        return list(self.taxonomy_parser.concepts.values())[:1]

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

    def _create_mapping_result(self, travel_concept, concept, confidence, rationale, sub_theme):
        must_not = self._find_must_not_have_concepts(travel_concept, concept)
        neg_overrides = self._find_negative_overrides(concept)
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
            'event': 'Event-Based'
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
            'Rationale': rationale,
            'Confidence (%)': confidence,
            'Must_Not_Have Concepts': "; ".join(must_not),
            'Negative_Overrides': "; ".join(neg_overrides),
            'Fallback_Logic': fallback_str
        }

def process_taxonomy(args):
    concepts_file, taxonomy_file, output_file = args
    travel_concepts = []
    if concepts_file and os.path.exists(concepts_file):
        with open(concepts_file, 'r', encoding='utf-8') as f:
            travel_concepts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        travel_concepts = ["24hourfrontdesk", "accessiblebathroom", "yoga", "àlacarterestaurant", "beachaccess"]
        print("Using default travel concepts...")

    seen = set()
    travel_concepts = [x for x in travel_concepts if not (x in seen or seen.add(x))]
    total_concepts = len(travel_concepts)
    logging.info(f"Total unique concepts to process for {taxonomy_file}: {total_concepts}")

    taxonomy_parser = TaxonomyParser()
    if not taxonomy_parser.parse_rdf(taxonomy_file):
        print(f"Skipping {taxonomy_file} due to parsing error.")
        return

    if not taxonomy_parser.concepts:
        print(f"No concepts loaded from {taxonomy_file}. Skipping.")
        return

    mapper = ConceptMapper(taxonomy_parser)
    all_mappings = []

    start_time = time.time()
    with tqdm(total=total_concepts, desc=f"Processing concepts ({os.path.basename(taxonomy_file)})", unit="concept") as pbar:
        for i, travel_concept in enumerate(travel_concepts, 1):
            mappings = mapper.map_concept(travel_concept)
            all_mappings.extend(mappings)
            pbar.update(1)

            if i % 100 == 0 or i == total_concepts:
                elapsed_time = time.time() - start_time
                percent_complete = (i / total_concepts) * 100
                concepts_remaining = total_concepts - i
                if i > 0:
                    avg_time_per_concept = elapsed_time / i
                    estimated_time_remaining = avg_time_per_concept * concepts_remaining
                else:
                    estimated_time_remaining = 0
                logging.info(
                    f"Processed {i}/{total_concepts} concepts ({percent_complete:.2f}%) for {taxonomy_file}. "
                    f"Elapsed time: {elapsed_time:.2f}s. "
                    f"Estimated time remaining: {estimated_time_remaining:.2f}s."
                )

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'Travel Concept', 'Taxonomy Definition', 'Matched Taxonomy URN', 'Matched Taxonomy UUID',
            'Concept Type', 'Matched Sub-Theme', 'Rationale', 'Confidence (%)',
            'Must_Not_Have Concepts', 'Negative_Overrides', 'Fallback_Logic'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_mappings)

    total_time = time.time() - start_time
    print(f"Completed processing {taxonomy_file}. Results written to {output_file}")
    print(f"Processed {len(travel_concepts)} unique concepts, resulting in {len(all_mappings)} mappings.")
    print(f"Total processing time for {taxonomy_file}: {total_time:.2f} seconds.")

def process_all_taxonomies(concepts_file, taxonomy_dir, output_dir):
    if not os.path.exists(taxonomy_dir):
        print(f"Error: Taxonomy directory '{taxonomy_dir}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(taxonomy_dir):
        print(f"Error: '{taxonomy_dir}' is not a directory.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    taxonomy_files = [f for f in os.listdir(taxonomy_dir) if f.endswith(('.xml', '.rdf'))]
    if not taxonomy_files:
        print(f"No .xml or .rdf files found in {taxonomy_dir}.")
        sys.exit(1)

    print(f"Found {len(taxonomy_files)} taxonomy files to process: {taxonomy_files}")

    overall_start_time = time.time()
    tasks = []
    for taxonomy_file in taxonomy_files:
        taxonomy_path = os.path.join(taxonomy_dir, taxonomy_file)
        output_file = os.path.splitext(taxonomy_file)[0] + "_mappings.csv"
        output_path = os.path.join(output_dir, output_file)
        tasks.append((concepts_file, taxonomy_path, output_path))

    with Pool(processes=5) as pool:
        pool.map(process_taxonomy, tasks)

    overall_time = time.time() - overall_start_time
    print(f"\nAll taxonomies processed. Total time: {overall_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description='Map travel concepts to taxonomy concepts across multiple taxonomies.')
    parser.add_argument('--concepts', help='Path to travel concepts file')
    parser.add_argument('--taxonomy-dir', required=True, help='Directory containing RDF/XML taxonomy files (.xml or .rdf)')
    parser.add_argument('--output-dir', required=True, help='Directory to save output CSV files')
    args = parser.parse_args()
    process_all_taxonomies(args.concepts, args.taxonomy_dir, args.output_dir)

if __name__ == '__main__':
    main()