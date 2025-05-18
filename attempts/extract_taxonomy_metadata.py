#!/usr/bin/env python3
"""Extract comprehensive metadata from RDF taxonomies."""

import logging
import json
import argparse
from typing import Dict, Set, List, Optional, Tuple, DefaultDict
from rdflib import Graph, Namespace, RDF, URIRef
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
AFF = Namespace("urn:expe:taxo:affinity#")
SKOS_STR = str(SKOS)
AFF_STR = str(AFF)


def parse_rdf_file(file_path: str) -> Tuple[Optional[Graph], Optional[str]]:
    """Parse an RDF file and return the graph and filename.

    Args:
        file_path: Path to the RDF file.

    Returns:
        Tuple of (parsed graph, base filename) or (None, None) if parsing fails.
    """
    g = Graph()
    try:
        g.parse(file_path, format="xml")
        logging.info(f"Successfully parsed {file_path}")
        return g, Path(file_path).name
    except Exception as e:
        logging.error(f"Failed to parse {file_path}: {e}")
        return None, None


def process_file(file_path: str, metadata: Dict) -> Optional[Tuple[str, Dict, DefaultDict]]:
    """Process a single RDF file and extract metadata in a single pass.

    Args:
        file_path: Path to the RDF file.
        metadata: Global metadata dictionary (unused directly but passed for consistency).

    Returns:
        Tuple of (datasource_name, file_metadata, relationships) or None if parsing fails.
    """
    graph, datasource_name = parse_rdf_file(file_path)
    if not graph:
        return None

    file_metadata = {
        "namespaces": dict(graph.namespaces()),
        "concepts": [],
        "urns": set()
    }
    subjects = {}
    relationships = defaultdict(list)

    # Single pass over the graph
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)  # Convert once per triple

        # Extract URNs and initialize subjects
        if isinstance(s, URIRef):
            file_metadata["urns"].add(s_str)
            if s_str not in subjects:
                subjects[s_str] = {"uri": s_str, "properties": defaultdict(list)}

        # Extract URNs from objects and build relationships
        if isinstance(o, URIRef):
            file_metadata["urns"].add(o_str)
            relationships[s_str].append({"predicate": p_str, "object": o_str})

        # Extract properties for subjects
        if s_str in subjects and (p_str.startswith(SKOS_STR) or p_str.startswith(AFF_STR)):
            subjects[s_str]["properties"][p_str].append(o_str)

    # Finalize concepts
    for subject in subjects.values():
        if subject["properties"]:
            subject["properties"] = dict(subject["properties"])  # Convert for JSON
            file_metadata["concepts"].append(subject)

    return datasource_name, file_metadata, relationships


def extract_metadata(taxonomy_dir: str, output_file: str, max_workers: Optional[int] = None) -> None:
    """Extract metadata from all RDF files in the taxonomy directory.

    Args:
        taxonomy_dir: Directory containing RDF files.
        output_file: Path to the output JSON file.
        max_workers: Number of worker threads (defaults to CPU count if None).
    """
    rdf_files = list(Path(taxonomy_dir).glob("*.rdf"))
    if not rdf_files:
        logging.warning(f"No RDF files found in {taxonomy_dir}")
        return

    metadata = {
        "datasources": {},
        "namespaces": {},
        "urns": set(),
        "concepts": defaultdict(list),
        "relationships": defaultdict(list)
    }
    failed_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file, metadata): file for file in rdf_files}

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if not result:
                    failed_files.append(file.name)
                    continue

                datasource_name, file_metadata, relationships = result

                # Update metadata
                metadata["datasources"][datasource_name] = {"namespaces": file_metadata["namespaces"]}
                metadata["namespaces"].update(file_metadata["namespaces"])
                metadata["urns"].update(file_metadata["urns"])
                metadata["concepts"][datasource_name].extend(file_metadata["concepts"])

                for subject, rels in relationships.items():
                    metadata["relationships"][subject].extend(rels)
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
                failed_files.append(file.name)

    # Summarize failures
    if failed_files:
        logging.warning(
            f"Processed {len(rdf_files) - len(failed_files)}/{len(rdf_files)} files successfully. Failed: {failed_files}")

    # Convert sets to lists for JSON
    metadata["urns"] = list(metadata["urns"])

    # Atomic write to JSON
    output_path = Path(output_file)
    temp_file = output_path.with_suffix('.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    temp_file.replace(output_path)
    logging.info(f"Metadata written to {output_file}")


def main() -> None:
    """Main function to extract metadata."""
    parser = argparse.ArgumentParser(description="Extract metadata from RDF taxonomies.")
    parser.add_argument("--taxonomy-dir", required=True, help="Directory containing RDF taxonomies")
    parser.add_argument("--output-file", default="taxonomy_metadata.json", help="Output JSON file")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker threads (default: CPU count)")

    args = parser.parse_args()
    extract_metadata(args.taxonomy_dir, args.output_file, args.workers)


if __name__ == "__main__":
    main()