import nbformat

# Load your notebook (as v4)
nb = nbformat.read("sentence_transformer.ipynb", as_version=4)

# Remove any metadata.widgets entirely
for cell in nb.cells:
    if "widgets" in cell.get("metadata", {}):
        cell["metadata"].pop("widgets")

# Write out a clean copy
nbformat.write(nb, "sentence_transformer_cleaned.ipynb")