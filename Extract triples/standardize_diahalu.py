import ast
import json
import re
from pathlib import Path

import pandas as pd


def extract_line_number(type_str):
    """Extract line number from type string like 'Line 1 - Extrinsic - Entity' or '[[1,"Extrinsic"]]'"""
    if not isinstance(type_str, str):
        return None, type_str

    # Handle format like '[[7,"Incoherence"]]'
    try:
        parsed = ast.literal_eval(type_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list) and len(parsed[0]) >= 2:
                line_num = str(parsed[0][0])
                type_name = parsed[0][1]
                return line_num, type_name
    except:
        pass

    # Handle format like 'Line 1 - Extrinsic - Entity'
    match = re.match(r"Line (\d+)\s*-\s*(.+)", type_str)
    if match:
        line_num = match.group(1)
        type_without_line = match.group(2).strip()
        return line_num, type_without_line
    return None, type_str


def standardize_diahalu(input_xlsx: str, output_json: str):
    """
    Convert DiaHalu_V2.xlsx to JSON format similar to PHD_benchmark.

    Fields mapping:
    - Column B: AI (the text)
    - Column C: label (0 = "factual", 1 = "non-factual")
    - Column D: type (contains line number and type)
    - Column E: domain
    - Column F: explanation
    - Column G: source
    - Column H: LLM
    """
    # Read the Excel file
    df = pd.read_excel(input_xlsx)

    # Get column names (assuming they are A, B, C, D, E, F, G or 0-indexed)
    # Excel columns: A=0, B=1, C=2, D=3, E=4, F=5, G=6
    print(f"Columns in Excel file: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")

    # Assuming the columns are in order or we need to map by index
    # Let's check the first few rows to understand the structure
    print("\nFirst few rows:")
    print(df.head())

    result = []

    for idx, row in df.iterrows():
        # Get values by column index (0-based)
        # Column B = index 1, C = 2, D = 3, E = 4, F = 5, G = 6, H = 7
        ai_text = row.iloc[1] if len(row) > 1 else None  # Column B
        label_val = row.iloc[2] if len(row) > 2 else None  # Column C
        type_str = row.iloc[3] if len(row) > 3 else None  # Column D
        domain = row.iloc[4] if len(row) > 4 else None  # Column E
        explanation = row.iloc[5] if len(row) > 5 else None  # Column F
        source = row.iloc[6] if len(row) > 6 else None  # Column G
        llm = row.iloc[7] if len(row) > 7 else None  # Column H

        # Skip rows with missing essential data
        if pd.isna(ai_text) or ai_text == "":
            continue

        # Convert label: 0 = "factual", 1 = "non-factual"
        if pd.isna(label_val):
            label = "factual"  # default
        else:
            label = "factual" if int(label_val) == 0 else "non-factual"

        # Extract line number and clean type
        line_num, clean_type = extract_line_number(type_str) if not pd.isna(type_str) else (None, None)

        # Parse explanation if it's in array format like '["text"]'
        explanation_text = None
        if not pd.isna(explanation):
            explanation_str = str(explanation)
            try:
                parsed_exp = ast.literal_eval(explanation_str)
                if isinstance(parsed_exp, list) and len(parsed_exp) > 0:
                    explanation_text = parsed_exp[0]
                else:
                    explanation_text = explanation_str
            except:
                explanation_text = explanation_str

        # Build wrong_part field
        wrong_part = None
        if label == "non-factual":
            if line_num and explanation_text:
                wrong_part = f"{line_num}: {explanation_text}"
            elif explanation_text:
                wrong_part = explanation_text
            elif line_num:
                wrong_part = line_num

        # Create entry
        entry = {
            "AI": str(ai_text).strip(),
            "label": label,
            "wrong_part": wrong_part,
            "type": clean_type if clean_type and not pd.isna(clean_type) else None,
            "domain": str(domain).strip() if not pd.isna(domain) else None,
            "source": str(source).strip() if not pd.isna(source) else None,
            "LLM": str(llm).strip() if not pd.isna(llm) else None,
        }

        result.append(entry)

    # Save to JSON
    output_path = Path(output_json)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Converted {len(result)} entries")
    print(f"✓ Saved to {output_json}")

    # Print some statistics
    factual_count = sum(1 for e in result if e["label"] == "factual")
    non_factual_count = len(result) - factual_count
    print("\nStatistics:")
    print(f"  Factual: {factual_count}")
    print(f"  Non-factual: {non_factual_count}")

    # Print domains distribution
    domains = {}
    for entry in result:
        domain = entry.get("domain", "Unknown")
        domains[domain] = domains.get(domain, 0) + 1
    print("\nDomains:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    import sys

    # Default paths
    input_file = "Extract triples/dataset/DiaHalu_V2.xlsx"
    output_file = "Extract triples/dataset/DiaHalu_V2.json"

    # Allow command-line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print(f"Converting {input_file} to {output_file}...")
    standardize_diahalu(input_file, output_file)
