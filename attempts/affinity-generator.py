#!/usr/bin/env python3
"""Generate affinity definitions dynamically from RDF taxonomy using traveler-prioritized themes."""

import os
import argparse
import json
from urllib.parse import quote
import multiprocessing
from rdflib import Graph, Namespace, RDF, URIRef
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

multiprocessing.set_start_method('fork', force=True)

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
AFF = Namespace("urn:expe:taxo:affinity#")
RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# Define 26 baseline themes (normalized weights sum to 1.0)
BASE_THEMES = {
    "Cleanliness & Hygiene Standards": {"type": "structural", "rule": "Must have 1", "weight": 0.085,
                                        "subScore": "CleanlinessAffinity", "hints": ["clean", "hygiene", "sanitized"]},
    "Safety & Security Measures": {"type": "structural", "rule": "Must have 1", "weight": 0.076,
                                   "subScore": "SafetyAffinity", "hints": ["safe", "secure", "locks"]},
    "Comfortable Bedding & Sleep Environment": {"type": "structural", "rule": "Must have 1", "weight": 0.068,
                                                "subScore": "BeddingAffinity", "hints": ["bed", "quiet", "sleep"]},
    "Functional, Clean Bathroom Facilities": {"type": "structural", "rule": "Must have 1", "weight": 0.068,
                                              "subScore": "BathroomAffinity", "hints": ["bathroom", "shower", "clean"]},
    "Fast, Reliable Wi-Fi and Device Charging": {"type": "technological", "rule": "Must have 1", "weight": 0.059,
                                                 "subScore": "WiFiAffinity", "hints": ["wifi", "internet", "charging"]},
    "In-Room Climate Control": {"type": "structural", "rule": "Must have 1", "weight": 0.059,
                                "subScore": "ClimateAffinity", "hints": ["heating", "cooling", "ac"]},
    "Friendly and Attentive Staff or Hosts": {"type": "service", "rule": "Optional", "weight": 0.051,
                                              "subScore": "StaffAffinity", "hints": ["friendly", "staff", "host"]},
    "Seamless Digital Experience": {"type": "technological", "rule": "Optional", "weight": 0.051,
                                    "subScore": "DigitalAffinity", "hints": ["check-in", "digital", "app"]},
    "Accurate Listings with Up-to-Date Imagery": {"type": "booking", "rule": "Must have 1", "weight": 0.051,
                                                  "subScore": "ListingAffinity",
                                                  "hints": ["imagery", "accurate", "photos"]},
    "Competitive Pricing & Perceived Value": {"type": "decision", "rule": "Optional", "weight": 0.042,
                                              "subScore": "PricingAffinity", "hints": ["price", "value", "affordable"]},
    "Location Convenience and Safety": {"type": "decision", "rule": "Must have 1", "weight": 0.042,
                                        "subScore": "LocationAffinity", "hints": ["location", "convenient", "safe"]},
    "Quiet & Peaceful Environment": {"type": "comfort", "rule": "Optional", "weight": 0.042,
                                     "subScore": "QuietAffinity", "hints": ["quiet", "peaceful", "calm"]},
    "Complimentary or Easy Meal Access": {"type": "preference", "rule": "Optional", "weight": 0.034,
                                          "subScore": "MealAffinity", "hints": ["breakfast", "food", "dining"]},
    "Flexibility in Booking and Cancellation Policies": {"type": "preference", "rule": "Optional", "weight": 0.034,
                                                         "subScore": "FlexibilityAffinity",
                                                         "hints": ["cancellation", "flexible", "policy"]},
    "In-Room Tech": {"type": "technological", "rule": "Optional", "weight": 0.034, "subScore": "TechAffinity",
                     "hints": ["tv", "smart", "lighting"]},
    "Basic Room Storage & Organization": {"type": "structural", "rule": "Optional", "weight": 0.034,
                                          "subScore": "StorageAffinity", "hints": ["storage", "closet", "space"]},
    "Pet- and Family-Friendly Accommodations": {"type": "preference", "rule": "Optional", "weight": 0.025,
                                                "subScore": "InclusiveAffinity", "hints": ["pet", "family", "kids"]},
    "Sustainability Signals or Options": {"type": "trend", "rule": "Optional", "weight": 0.025,
                                          "subScore": "SustainabilityAffinity",
                                          "hints": ["sustainable", "eco", "recycling"]},
    "Transparent Communication & Pre-Arrival Info": {"type": "booking", "rule": "Optional", "weight": 0.025,
                                                     "subScore": "CommunicationAffinity",
                                                     "hints": ["communication", "info", "pre-arrival"]},
    "Personalization or Recognition for Return Guests": {"type": "trend", "rule": "Optional", "weight": 0.017,
                                                         "subScore": "PersonalizationAffinity",
                                                         "hints": ["personalized", "loyalty", "recognition"]},
    "Sentiment": {"type": "comfort", "rule": "Optional", "weight": 0.025, "subScore": "SentimentAffinity",
                  "hints": ["luxury", "relaxing", "vibrant"]},
    "Indoor Amenities": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "IndoorAmenityAffinity",
                         "hints": ["spa", "bar", "gym"]},
    "Outdoor Amenities": {"type": "structural", "rule": "Optional", "weight": 0.025,
                          "subScore": "OutdoorAmenityAffinity", "hints": ["pool", "garden", "patio"]},
    "Activities": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "ActivityAffinity",
                   "hints": ["hiking", "nightlife", "tours"]},
    "Spaces": {"type": "structural", "rule": "Optional", "weight": 0.025, "subScore": "SpacesAffinity",
               "hints": ["suite", "balcony", "kitchen"]},
    "Events": {"type": "preference", "rule": "Optional", "weight": 0.025, "subScore": "EventAffinity",
               "hints": ["festival", "wedding", "concert"]}
}


def normalize_concept(concept: str) -> str:
    return concept.lower().replace(" ", "")


def load_configs(metadata_file: str, themes_config_file: str = "themes_config.json") -> Tuple[Dict, Dict]:
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(themes_config_file, 'r', encoding='utf-8') as f:
        themes_config = json.load(f)
        themes_config["critical_concepts"] = {normalize_concept(k): v for k, v in
                                              themes_config.get("critical_concepts", {}).items()}
    return metadata, themes_config


def parse_rdf_file(file_path: str) -> Graph:
    g = Graph()
    g.parse(file_path, format="xml")
    return g


def load_rdf_files(taxonomy_dir: str) -> Graph:
    rdf_files = [os.path.join(taxonomy_dir, f) for f in os.listdir(taxonomy_dir) if f.endswith('.rdf')]
    combined_graph = Graph()
    for f in rdf_files:
        combined_graph += parse_rdf_file(f)
    return combined_graph


class TaxonomyParser:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.concepts: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        self.label_to_concepts = defaultdict(list)
        self._parse_concepts()

    def _parse_concepts(self):
        for s, p, o in self.graph:
            uri = str(s)
            if uri not in self.concepts:
                self.concepts[uri] = {"uri": uri, "properties": {}, "lowercase": {}}
            pred_str = str(p)
            obj_str = str(o)
            if pred_str not in self.concepts[uri]["properties"]:
                self.concepts[uri]["properties"][pred_str] = []
                self.concepts[uri]["lowercase"][pred_str] = []
            self.concepts[uri]["properties"][pred_str].append(obj_str)
            self.concepts[uri]["lowercase"][pred_str].append(obj_str.lower())
            if pred_str == str(SKOS.prefLabel):
                normalized_label = normalize_concept(obj_str)
                self.label_to_concepts[normalized_label].append(self.concepts[uri])


def train_word2vec(parser: TaxonomyParser, external_corpus: str = None, critical_concepts: Dict = None) -> Word2Vec:
    all_texts = [list(set(c["properties"].get(str(SKOS.prefLabel), [""])[0].lower().split() +
                          c["lowercase"].get(str(SKOS.definition), [""])[0].split()))
                 for c in parser.concepts.values()]
    if critical_concepts:
        for keywords in critical_concepts.values():
            all_texts.extend([list(keywords)] * 5)
    if external_corpus and os.path.exists(external_corpus):
        with open(external_corpus, 'r', encoding='utf-8') as f:
            all_texts.extend([line.lower().split() for line in f if line.strip()])
    return Word2Vec(sentences=all_texts, vector_size=100, window=10, min_count=1, workers=4)


def extract_relevant_concepts(primary_concept: Dict, parser: TaxonomyParser, themes_config: Dict,
                              concept_str: str, word2vec_model: Word2Vec, critical_concepts: Dict = None) -> Dict[
    str, List[Dict]]:
    concept_normalized = normalize_concept(concept_str)
    all_concepts = parser.concepts
    is_critical = concept_normalized in critical_concepts if critical_concepts else False
    critical_keywords = set(critical_concepts.get(concept_normalized, [])) if is_critical else set()

    # Prepare embeddings
    concept_embeddings = {}
    concept_labels = {}
    for uri, concept_data in all_concepts.items():
        label = concept_data["properties"].get(str(SKOS.prefLabel), [""])[0].lower()
        words = label.split()
        if words and all(word in word2vec_model.wv for word in words):
            embedding = np.mean([word2vec_model.wv[word] for word in words], axis=0)
            concept_embeddings[uri] = embedding
            concept_labels[uri] = label

    # Normalize embeddings
    embeddings_matrix = np.array(list(concept_embeddings.values()))
    embeddings_matrix = normalize(embeddings_matrix)
    uri_list = list(concept_embeddings.keys())

    # Cluster concepts
    clusterer = DBSCAN(eps=1.5, min_samples=1, metric='euclidean')
    labels = clusterer.fit_predict(embeddings_matrix)

    # Score concepts
    concept_scores = {}
    for uri, embedding in concept_embeddings.items():
        label_words = concept_labels[uri].split()
        similarity = word2vec_model.wv.n_similarity(label_words, list(critical_keywords)) if critical_keywords and all(
            w in word2vec_model.wv for w in label_words) else 0.5
        keyword_bonus = 0.6 if any(kw in label_words for kw in critical_keywords) else 0.0
        score = (0.3 * similarity + 0.1 * keyword_bonus)
        concept_scores[uri] = score

    # Map to baseline themes first
    theme_attributes = {}
    baseline_themes = {}
    for theme_name, config in BASE_THEMES.items():
        hint_words = config["hints"] + list(critical_keywords)
        theme_centroid = np.mean([word2vec_model.wv[word] for word in hint_words if word in word2vec_model.wv], axis=0)
        baseline_themes[theme_name] = theme_centroid

    clusters = defaultdict(list)
    for uri, cluster_label in zip(uri_list, labels):
        if cluster_label != -1 and concept_scores[uri] >= 0.2:
            clusters[cluster_label].append((uri, concept_scores[uri]))

    for cluster_label, concepts in clusters.items():
        cluster_concepts = [all_concepts[uri] for uri, _ in concepts]
        cluster_labels = [concept["properties"].get(str(SKOS.prefLabel), [""])[0].lower().split() for concept in
                          cluster_concepts]
        cluster_embedding = np.mean(
            [np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
             for words in cluster_labels if words], axis=0)
        theme_similarities = {
            name: np.dot(cluster_embedding, centroid) / (np.linalg.norm(cluster_embedding) * np.linalg.norm(centroid))
            for name, centroid in baseline_themes.items()}
        best_theme = max(theme_similarities, key=theme_similarities.get) if theme_similarities else None

        if best_theme and theme_similarities[best_theme] > 0.2:  # Low threshold for baseline themes
            sorted_concepts = sorted(concepts, key=lambda x: x[1], reverse=True)[:5]
            theme_attributes[best_theme] = [{
                "skos:prefLabel": all_concepts[uri]["properties"].get(str(SKOS.prefLabel), [""])[0],
                "uri": uri,
                "concept_weight": round(score, 2)
            } for uri, score in sorted_concepts]
        elif theme_similarities.get(best_theme, 0) > 0.9:  # High threshold for dynamic themes
            top_label = all_concepts[max(concepts, key=lambda x: x[1])[0]]["properties"].get(str(SKOS.prefLabel), [""])[
                0].lower()
            theme_name = f"DynamicTheme_{cluster_label}"
            BASE_THEMES[theme_name] = {"type": "structural", "rule": "Optional", "weight": 0.01,
                                       "subScore": f"{theme_name}Affinity", "hints": []}
            sorted_concepts = sorted(concepts, key=lambda x: x[1], reverse=True)[:5]
            theme_attributes[theme_name] = [{
                "skos:prefLabel": all_concepts[uri]["properties"].get(str(SKOS.prefLabel), [""])[0],
                "uri": uri,
                "concept_weight": round(score, 2)
            } for uri, score in sorted_concepts]

    return theme_attributes


def build_themes(concept: str, matches: List[Dict], parser: TaxonomyParser, themes_config: Dict,
                 word2vec_model: Word2Vec, critical_concepts: Dict = None) -> List[Dict]:
    if not matches:
        return []  # Skip if no taxonomy match
    theme_attributes = extract_relevant_concepts(matches[0], parser, themes_config, concept, word2vec_model,
                                                 critical_concepts)
    if not theme_attributes:  # Skip if no baseline themes are matched
        return []

    themes = []
    for theme_name, rule in BASE_THEMES.items():
        attributes = theme_attributes.get(theme_name, [])
        themes.append({
            "name": theme_name,
            "type": "rule['type']",
            "rule": rule["rule"],
            "theme_weight": rule["weight"],
            "subScore": rule["subScore"],
            "attributes": attributes,
            "dynamic_rule": {
                "rule_type": rule["rule"].lower().replace(" ", "_"),
                "condition": "attribute_presence",
                "target": theme_name,
                "criteria": {"attributes": [attr["skos:prefLabel"].lower() for attr in attributes], "minimum_count": 1},
                "effect": "pass" if rule["rule"] == "Must have 1" else "boost"
            }
        })
    return themes


def process_concept(concept: str, parser: TaxonomyParser, themes_config: Dict, input_graph: Graph,
                    word2vec_model: Word2Vec, critical_concepts: Dict = None) -> Dict:
    concept_normalized = normalize_concept(concept)
    critical_keywords = set(critical_concepts.get(concept_normalized,
                                                  [])) if critical_concepts and concept_normalized in critical_concepts else set()
    matches = parser.label_to_concepts.get(concept_normalized, [])
    if not matches:
        return {}  # Skip unmapped concepts

    output_graph = Graph()
    affinity_uri = URIRef(f"urn:expe:taxo:affinity:{quote(concept_normalized, safe=':')}")
    output_graph.add((affinity_uri, RDF.type, SKOS.Concept))

    primary_match = max(matches, key=lambda m: len(m["properties"].get(str(SKOS.prefLabel), [])))
    themes = build_themes(concept, matches, parser, themes_config, word2vec_model, critical_concepts)
    if not themes:
        return {}  # Skip if no themes are populated

    for match in matches:
        output_graph.add((affinity_uri, AFF.travelCategory, URIRef(match["uri"])))

    return {
        "applicable_lodging_types": "Both",
        "concept_type": "other",
        "travel_category": {
            "name": concept.title(),
            "skos:prefLabel": primary_match["properties"].get(str(SKOS.prefLabel), [concept])[0],
            "uri": primary_match["uri"],
            "exclusionary_concepts": []
        },
        "context_settings": themes_config["context_settings"],
        "scoringFormula": "SUM(themeWeight * subScore * contextMultiplier)",
        "themes": themes,
        "must_not_have": [],
        "negative_overrides": [],
        "fallback_logic": {"condition": "At least one theme matched", "action": "Proceed with scoring",
                           "threshold": 0.0},
        "graph": output_graph
    }


def generate_affinity_definitions(concepts_file: str, taxonomy_dir: str, output_dir: str = ".",
                                  metadata_file: str = "taxonomy_metadata.json", travel_corpus: str = None):
    metadata, themes_config = load_configs(metadata_file)
    critical_concepts = themes_config.get("critical_concepts", {})
    with open(concepts_file, 'r') as f:
        concepts = [line.strip() for line in f if line.strip()]
    input_graph = load_rdf_files(taxonomy_dir)
    parser = TaxonomyParser(input_graph)
    word2vec_model = train_word2vec(parser, travel_corpus, critical_concepts)
    output_graph = Graph()
    affinity_definitions = {"travel_concepts": []}

    for concept in tqdm(concepts):
        result = process_concept(concept, parser, themes_config, input_graph, word2vec_model, critical_concepts)
        if result:
            affinity_definitions["travel_concepts"].append({k: v for k, v in result.items() if k != "graph"})
            output_graph += result["graph"]

    os.makedirs(output_dir, exist_ok=True)
    ttl_output_file = os.path.join(output_dir, "affinity_definitions.ttl")
    output_graph.serialize(destination=ttl_output_file, format="turtle")
    json_output_file = os.path.join(output_dir, "affinity_definitions.json")
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(affinity_definitions, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate affinity definitions from RDF taxonomies.")
    parser.add_argument("--concepts", required=True, help="Path to concepts file")
    parser.add_argument("--taxonomy-dir", required=True, help="Directory containing RDF taxonomies")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--metadata-file", default="taxonomy_metadata.json", help="Metadata JSON file")
    parser.add_argument("--travel-corpus", help="Path to external travel corpus file")
    args = parser.parse_args()
    generate_affinity_definitions(args.concepts, args.taxonomy_dir, args.output_dir, args.metadata_file,
                                  args.travel_corpus)


if __name__ == "__main__":
    main()