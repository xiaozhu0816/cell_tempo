#!/usr/bin/env python3
"""
Fill the Cell Reports Key Resources Table (KRT) for the cell_tempo paper.
Reads the empty template, populates Table 0 with our paper's entries,
saves as a new docx (KRT_cell_tempo_filled.docx).
"""

from docx import Document
from copy import deepcopy
from pathlib import Path

SRC = Path(__file__).parent / "Table_Template_Cell_Reports.docx"
DST = Path(__file__).parent / "KRT_cell_tempo_filled.docx"

# ── KRT entries grouped by subheading ─────────────────────────────────
# Each entry is (REAGENT or RESOURCE, SOURCE, IDENTIFIER)
KRT = {
    "Bacterial and virus strains": [
        ("Venezuelan equine encephalitis virus (VEEV), vaccine strain TC-83",
         "P1 laboratory stock generated from an infectious cDNA clone provided by I. Frolov",
         "Kinney et al., J. Virol. 1993"),
    ],
    "Chemicals, peptides, and recombinant proteins": [
        ("Dulbecco's phosphate-buffered saline (DPBS)",
         "Corning",
         "Cat#21-030-CV"),
        ("EGM-2 Endothelial Cell Growth Medium-2 BulletKit",
         "Lonza",
         "Cat#CC-3162"),
        ("Fetal bovine serum (FBS)",
         "Gibco",
         "Cat#A5256801"),
        ("Dulbecco's Modified Eagle Medium (DMEM)",
         "Quality Biological",
         "Cat#112-014-101CS"),
        ("Trypsin (0.25%)-EDTA (0.02%)",
         "Quality Biological",
         "Cat#118-093-721"),
        ("Minimum Essential Medium (EMEM) (2X) without Phenol Red and L-Glutamine",
         "Quality Biological",
         "Cat#115-073-101"),
        ("Penicillin/Streptomycin solution (100X)",
         "Corning",
         "Cat#30-002-CI"),
    ],
    "Experimental models: Cell lines": [
        ("Human: Primary brain microvascular endothelial cells (HBMVEC), male pediatric donor, passages 5-8",
         "Cell Systems (an AnaBios company)",
         "Cat#ACBRI 376; Lot#376.07.05.01.2F"),
        ("African green monkey kidney epithelial cells (Vero)",
         "ATCC",
         "Cat#CCL-81"),
    ],
    "Software and algorithms": [
        ("CELLCYTE Studio",
         "CYTENA",
         "https://www.cytena.com/products/cellcyte-x/"),
        ("GraphPad Prism v11.0.0",
         "GraphPad Software",
         "https://www.graphpad.com"),
        ("Python (v3.10)",
         "Python Software Foundation",
         "https://www.python.org"),
        ("PyTorch (v2.0)",
         "PyTorch",
         "https://pytorch.org"),
        ("torchvision (ResNet50 with ImageNet pretrained weights)",
         "PyTorch",
         "https://pytorch.org/vision"),
        ("ResNet50 architecture",
         "He et al.19",
         "https://pytorch.org/vision/stable/models/resnet.html"),
        ("scikit-learn",
         "Pedregosa et al.",
         "https://scikit-learn.org"),
        ("t-SNE (visualization)",
         "van der Maaten and Hinton20",
         "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"),
        ("cell_tempo analysis pipeline",
         "This paper",
         "https://github.com/CAIR-LAB-WFUSM/cell_tempo"),
    ],
    "Other": [
        ("CELLCYTE X live-cell imaging system (10x objective, Enhanced Contour mode)",
         "CYTENA GmbH, Freiburg, Germany",
         "CELLCYTE X Quick Start Guide; https://www.cytena.com/products/cellcyte-x/"),
        ("12-well tissue-culture plates (CELLSTAR)",
         "Greiner Bio-One",
         "Cat#665180"),
    ],
}

# Subheadings present in template (in order); ones not in KRT will be removed.
SUBHEADINGS_IN_TEMPLATE = [
    "Antibodies",
    "Bacterial and virus strains",
    "Biological samples",
    "Chemicals, peptides, and recombinant proteins",
    "Critical commercial assays",
    "Deposited data",
    "Experimental models: Cell lines",
    "Experimental models: Organisms/strains",
    "Oligonucleotides",
    "Recombinant DNA",
    "Software and algorithms",
    "Other",
]


def first_cell_text(row):
    return row.cells[0].text.strip()


def is_subheading_row(row):
    """Heading rows have all 3 cells == the same heading text."""
    txts = [c.text.strip() for c in row.cells]
    return txts[0] == txts[1] == txts[2] != "" and txts[0] != "REAGENT or RESOURCE"


def remove_row(table, row):
    """Remove a row from a python-docx table."""
    row._element.getparent().remove(row._element)


def insert_row_after(table, ref_row):
    """Insert a new empty row immediately after ref_row, return the new row."""
    new_tr = deepcopy(ref_row._element)
    # Clear cell text in the copied row
    for tc in new_tr.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc'):
        for p in tc.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            for r in list(p):
                tag = r.tag.split('}')[-1]
                if tag == 'r':
                    p.remove(r)
    ref_row._element.addnext(new_tr)
    # Return the wrapper Row object
    for r in table.rows:
        if r._element is new_tr:
            return r
    return None


def set_row_text(row, values):
    for cell, val in zip(row.cells, values):
        # Clear existing paragraph runs but keep at least one paragraph
        p = cell.paragraphs[0]
        for run in list(p.runs):
            run.text = ""
        if p.runs:
            p.runs[0].text = val
        else:
            p.add_run(val)


def main():
    doc = Document(str(SRC))
    table = doc.tables[0]  # Table 0 = the empty template

    # Walk through rows; identify each subheading and the empty rows beneath it,
    # up to the next subheading. Replace those empty rows with our entries.
    rows_snapshot = list(table.rows)

    # First pass: build a map subheading -> list of empty content rows directly under it
    section_map = {}
    heading_rows = {}
    current_heading = None
    for row in rows_snapshot:
        txts = [c.text.strip() for c in row.cells]
        if txts[0] == "REAGENT or RESOURCE":
            continue
        if is_subheading_row(row):
            current_heading = txts[0]
            heading_rows[current_heading] = row
            section_map[current_heading] = []
        else:
            if current_heading is not None:
                section_map[current_heading].append(row)

    # Second pass: fill / add / delete rows under each subheading
    for heading in SUBHEADINGS_IN_TEMPLATE:
        empty_rows = section_map.get(heading, [])
        entries = KRT.get(heading, [])

        if not entries:
            # Section unused: leave one empty row (for user to delete in Word) or remove all extras
            for er in empty_rows:
                remove_row(table, er)
            if heading in heading_rows:
                remove_row(table, heading_rows[heading])
            continue

        # Fill in the first len(entries) rows; add more if needed; delete leftovers
        for i, entry in enumerate(entries):
            if i < len(empty_rows):
                set_row_text(empty_rows[i], entry)
            else:
                # Need a new row — copy the last empty row's structure
                ref = empty_rows[-1] if empty_rows else None
                if ref is None:
                    # Shouldn't happen — every heading has empty rows in template
                    print(f"WARN: no template rows under {heading}")
                    continue
                new_row = insert_row_after(table, empty_rows[-1] if i > 0 else empty_rows[0])
                set_row_text(new_row, entry)
                empty_rows.append(new_row)

        # Delete leftover empty rows beyond what we filled
        for j in range(len(entries), len(empty_rows)):
            remove_row(table, empty_rows[j])

    # Final cleanup: remove any unused heading rows or fully empty rows that
    # survived template manipulation.
    for row in list(table.rows):
        txts = [c.text.strip() for c in row.cells]
        if txts[0] == "REAGENT or RESOURCE":
            continue
        if is_subheading_row(row) and txts[0] not in KRT:
            remove_row(table, row)
            continue
        if all(t == "" for t in txts):
            remove_row(table, row)

    doc.save(str(DST))
    print(f"Saved: {DST}")


if __name__ == "__main__":
    main()
