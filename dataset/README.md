# Dataset for the HERTy-Wiki Benchmarks
Our dataset is curated for the English language.
We intend to provide a language-agnostic version of the dataset, to allow for the easy population of our dataset in different languages. This will be added at a later point in time.

## Dataset Structure
### Entity Typing

The entity typing benchmark data is split into two files:

- entity_typing.parquet – The main dataset containing all text fields, metadata, and a column of image_ids (zero, one or more per datapoint). Each image_id references an entry in the image store. 
- multimodal/images.arrow – A deduplicated image store. Each row contains a unique image identified by its SHA-256 hash (image_id), along with the encoded image bytes and basic metadata (width, height, mime type).
Multiple datapoints can reference the same image without storing duplicates.

This setup keeps the dataset lightweight and modular: text and metadata can be processed independently, while images are loaded on demand using the image_id mapping.

## Data Provenance and Attribution

This dataset includes:
- Reverse-geocoded location metadata derived from OpenStreetMap (© OpenStreetMap contributors, licensed under the Open Database License, ODbL).
- Map images based on OpenStreetMap data (© OpenStreetMap contributors).
- Textual data sourced from Wikidata.
- Images collected from Wikimedia Commons (except for the aforementioned OSM-based maps).

For detailed per-image attribution, license information, and disclaimers regarding the multimodal portion of the dataset, please refer to multimodal/README.md.

## Licensing
Only our annotations, metadata, and benchmark structure are released under CC-BY-4.0.
Third-party images retain their original Creative Commons or public-domain licenses, as documented in the multimodal attribution files.