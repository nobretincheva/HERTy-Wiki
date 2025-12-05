## Preprocessing  

All images were scaled to 896px, long side. SVG images have been saved as pngs.

The below refers to preprocessing for local images.
- EXIF orientation is applied.
- GIFs are treated as still images: only the first frame is kept and saved as PNG (RGB).
- Non-GIF images are stored in their original encoding and color mode (no re-encoding or color conversion).

For hosted images:
API-ready links are provided in the main dataset. These links point to an already scaled version of the images. We do not apply additional preprocessing.

## Intended Use

This dataset has been created for evaluation purposes only. As with the main portion of our dataset, we request users do not train or fine-tune models on the provided data.


## Attribution and License information
Only our annotations and metadata are licensed under CC-BY-4.0. All third-party images retain their original licenses and are included on a “as-is” basis. This benchmark and its multimodal extension are intended solely for evaluation of LLMs and should not be used for model training or fine-tuning.

We allow the following list of licenses (please note, these codes comply with the ones used by Wikimedia Commons): 
    
"pd", "cc-zero", "cc0",
    "cc-by-1.0","cc-by-2.0","cc-by-2.5","cc-by-3.0","cc-by-4.0",
    "cc-by-sa-1.0","cc-by-sa-2.0","cc-by-sa-2.5","cc-by-sa-3.0","cc-by-sa-4.0",
    "attribution", "fal", "mit", "kogl type 1", "no restrictions",
    "licence ouverte", "isc", "bsd", "godl-india", "cc sa 1.0"


We restrict for the following categories:
restrict_categories = [ihl|nazi|communist|ai]
We additionally filtered through the images for gore/sensitive content; 

For an easy to read list of full attribution credits - refer to attribution.txt
For full metadata information including image descriptions, date of creation and date of metadata, relevant restrictions and categories - please refer to attribution csv. 

Full license texts for third-party images are in the `licenses/` directory.
Each image’s specific license is listed in `attribution.csv` and `attribution.txt`,
along with author and source URL.

## Disclaimers
### Trademarks and Protected Symbols
Some images in this dataset may depict trademarks or other protected symbols.  
Such trademarks are included for informational and research purposes only.  
Their presence does not imply any affiliation with or endorsement by the trademark owner.  
Copyright for these images is governed by their respective Creative Commons licenses,  
as detailed in `attribution.csv`.

This benchmark includes images depicting national flags, emblems, and other official insignia.
These are included solely for informational and research purposes.
Their inclusion does not imply any affiliation with, or endorsement by, the respective governments or organisations.

### Personality and Privacy Rights

Some images in this dataset depict identifiable individuals.
While these images are freely licensed or in the public domain under Wikimedia Commons terms,
the individuals shown may have rights that legally restrict certain reuses,
particularly for commercial or promotional purposes.

This dataset is provided for informational and research use only.
No endorsement or affiliation by any individual depicted is implied.
Users are responsible for ensuring compliance with applicable privacy and personality rights laws
in their jurisdictions.

### Excluded Media

Certain entities in this benchmark correspond to historically significant
or politically sensitive organizations and symbols.

To ensure compliance with international content laws,
**images depicting restricted insignia or symbols** (such as Nazi or Communist emblems)
have been **excluded** from the distributed dataset.

The entities themselves remain included to preserve coverage and factual completeness.
Affected items are annotated as `restricted` or `no_image` in the dataset metadata.

No political stance, endorsement, or opinion is implied by the inclusion of any entity.

## Feedback and Removal Requests

We have made every effort to ensure that all images in this benchmark 
are appropriately licensed, ethically sourced, and used in accordance with
Wikimedia Commons’ reuse guidelines and applicable law.

However, if you believe that any image or metadata entry in this dataset 
infringes rights, includes sensitive content, or should otherwise be removed, 
please contact us by raising an issue.

We will review all requests promptly and, where appropriate, remove or 
revise the affected items in future releases.