# HERTy-Wiki

HERTy-Wiki is a human-verified benchmark for evaluating hierarchical entity typing and reasoning in LLMs using Wikidata-derived entities. It includes 8,767 multiple-choice questions across five domains, plus an optional multimodal image extension. The benchmark tests whether models can select the most specific valid type with limited context—mirroring real KG-editing conditions. Baseline results show strong reliance on memorised priors, underscoring the need for reasoning-driven approaches to KGET.

## Intended Use and Licensing
HERTy-Wiki is intended solely for evaluation and analysis of hierarchical entity typing and reasoning in LLMs. It should not be used for model training or fine-tuning.

- Our contributions (including annotations, metadata, benchmark structure, code, and documentation) are released under CC-BY-4.0.
- Third-party images retain their original Creative Commons or public-domain licenses, as listed in the attribution files in dataset/multimodal/.
- Users must comply with the specific license terms attached to each image when redistributing or reusing visual content.

Textual data originates from Wikidata (CC0), geolocation information is derived from OpenStreetMap (ODbL).

Full attribution, license texts, and metadata can be found in the dataset folder.

## Structure
The full dataset, including the text-only benchmark file (entity_typing.parquet) and the multimodal image store (multimodal/images.arrow) is hosted at: Zenodo

We additionally intend to release a HuggingFace version of this dataset to allow for ease of integration in evaluation pipelines. 

Additional resources in the repository:
- Dataset documentation: dataset/README.md
- Multimodal attribution & license details: dataset/multimodal/
- Benchmark configs: configs/
- Baseline models: models/
- Evaluation pipeline: benchmark/benchmark.py and benchmark/evaluate.py

For questions, issues, or removal requests, please open an Issue on the GitHub repository.
The dataset is actively maintained by the authors

## Resource Sustainability & Maintenance Plan
We plan to continue expanding the WikiReason benchmark beyond the current entity typing task. Future releases will introduce additional reasoning tasks as well as a language-agnostic version of the dataset to support multilingual and multimodal extensions.

The main authors of the dataset will maintain and update the resource for at least three years following its publication. After this period, long-term maintenance will be overseen by the King’s College London Knowledge Graph Group, which conducts ongoing research on Wikidata and collaborates actively with the Wikimedia community. As a result, the dataset and its extensions will remain a stable, evolving resource aligned with our broader research agenda.

All updates, including corrections, attribution fixes, and additional benchmark tasks, will be released as versioned updates in the public GitHub repository and mirrored through DOI-backed archives to ensure long-term accessibility and reproducibility.

## Versioning
All releases will be versioned following semantic versioning:
- **Major (X.0.0)**: new tasks or structural changes
- **Minor (0.X.0)**: new data or metadata improvements
- **Patch (0.0.X)**: bug fixes or attribution corrections

All versions will be archived with a permanent DOI.

## Status
Please note: This resource is currently under peer review. Public release of the full dataset will occur upon acceptance