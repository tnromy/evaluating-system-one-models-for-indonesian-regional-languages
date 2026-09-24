"""Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2023, Table 3.

Copied from references/pdf2md/md/j08_nusax_winata_eacl2023.md (journal folder). Columns in ISO 639-3.
"""
LANGS = ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]

_ROWS = {
    "nb":           [72.5, 72.6, 73.0, 71.9, 73.7, 76.5, 73.1, 69.4, 66.8, 73.2, 68.8, 71.9],
    "svm":          [75.7, 75.3, 76.7, 74.8, 77.2, 75.0, 78.7, 71.3, 73.8, 76.7, 75.1, 74.3],
    "lr":           [77.4, 76.3, 76.3, 75.0, 77.2, 75.9, 74.7, 73.7, 74.7, 74.8, 73.4, 75.8],
    "indobert_base": [75.4, 74.8, 70.0, 83.1, 73.9, 79.5, 90.0, 81.7, 77.8, 82.5, 75.8, 77.5],
    "xlmr_base":    [73.9, 72.8, 62.3, 76.6, 66.6, 90.8, 88.4, 78.9, 69.7, 79.1, 75.0, 80.1],
}

TABLE3 = {model: dict(zip(LANGS, row)) for model, row in _ROWS.items()}
