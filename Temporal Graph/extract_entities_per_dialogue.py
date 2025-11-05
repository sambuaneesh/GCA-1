import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

MODEL = "gemini-2.5-flash-lite"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
client = genai.GenerativeModel(MODEL)

GPT_DIALOG_ENTITY_LIST_PROMPT = """
You will be given a multi-turn dialogue formatted as lines that begin with a speaker tag
(e.g., A1:, B1:, A2:, B2:, ...). For EACH line in the dialogue, extract a list of entities/
key concepts that explicitly appear in that line.

Return ONLY a Python-style list of lists, where:
- There is exactly one inner list per dialogue line, in the original order.
- Each inner list contains unique entities/concepts from that line only (no cross-line memory).
- Preserve the original casing found in the line; remove surrounding punctuation.
- Include proper nouns (e.g., Neptune, Kuiper Belt), domain terms (e.g., dwarf planet,
  elliptical orbits), specific numeric quantities with units (e.g., 5.9 billion kilometers),
  and well-defined ordinal/role phrases (e.g., eighth planet, ninth planet).
- Exclude pronouns (he, she, it, they, this, that, etc.), filler words, generic verbs,
  and vague terms without domain meaning (e.g., tell, about, is, consider).
- If a line has no extractable entities/concepts, return an empty list [] for that line.
- Do NOT merge, summarize, or infer beyond what the line states.
- Do NOT include speaker tags (A1:, B1:, etc.) in the content of the lists.

In-context example:

<input>
A1: Tell me about the solar system.
B1: The solar system consists of the sun, planets, moons, asteroids, and comets orbiting around the sun.
A2: Tell me about Pluto.
B2: Pluto is a dwarf planet in the Kuiper Belt, and it used to be considered the ninth planet in our solar system before being reclassified in 2006.
A3: Which is the eighth planet?
B3: Neptune.
A4: What is the distance between Neptune and Pluto?
B4: The distance between Neptune and Pluto can vary as both planets have elliptical orbits, but on average, it's about 4.5 billion kilometers (2.8 billion miles).
A5: What is the distance between Sun and Pluto?
B5: The distance between the Sun and Pluto also varies due to their elliptical orbits. On average, it's about 5.9 billion kilometers (3.67 billion miles).
A6: Which is closer from Neptune? Sun or Pluto?
B6: The Sun is closer to Neptune than Pluto.
A7: Whether Pluto is in the solar system?
B7: Yes, Pluto is in the solar system.
</input>

<output>
[
  ["solar system"],
  ["solar system", "sun", "planets", "moons", "asteroids", "comets"],
  ["Pluto"],
  ["Pluto", "dwarf planet", "Kuiper Belt", "ninth planet", "solar system", "2006"],
  ["eighth planet"],
  ["Neptune"],
  ["Neptune", "Pluto"],
  ["Neptune", "Pluto", "elliptical orbits", "4.5 billion kilometers", "2.8 billion miles"],
  ["Sun", "Pluto"],
  ["Sun", "Pluto", "elliptical orbits", "5.9 billion kilometers", "3.67 billion miles"],
  ["Neptune", "Sun", "Pluto"],
  ["Sun", "Neptune", "Pluto"],
  ["Pluto", "solar system"],
  ["Pluto", "solar system"]
]
</output>

Now extract entities for the following dialogue:

<input>
{dialogue_text}
</input>
"""


def parse_dialogue_text(ai_text):
    """
    Parse the AI dialogue text into individual turns.
    Format: A1: question \nB1: answer \nA2: question \nB2: answer ...
    Returns a tuple: (list of dialogue strings, original dialogue text with tags)
    """
    dialogues = []

    # Split by newline and process each line
    lines = ai_text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match pattern like "A1: " or "B1: " at the start
        match = re.match(r"^[AB]\d+:\s*(.+)$", line)
        if match:
            dialogue_content = match.group(1).strip()
            dialogues.append(dialogue_content)

    # Return both parsed dialogues and original text
    return dialogues, ai_text


def request_api(prompt, temperature=0.0):
    """Make API request with retry logic."""
    flag = True
    while flag:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2000,
            )
            response = client.generate_content(prompt, generation_config=generation_config)
            text_response = response.text.strip()
            flag = False
            return text_response
        except Exception as e:
            print("Exception:", e)
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                print("Rate limit exceeded, waiting...")
                time.sleep(1)
            else:
                print(f"Error: {e}")
                time.sleep(0.5)


def extract_entities_for_dialogue(dialogue_text):
    """
    Extract entities for all dialogue turns at once using the API.
    Returns a list of lists, one per dialogue line.
    """
    prompt = GPT_DIALOG_ENTITY_LIST_PROMPT.format(dialogue_text=dialogue_text)
    response = request_api(prompt, temperature=0.0)

    # Try to parse the response as a Python list of lists
    try:
        # Clean up the response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```python"):
            response = response[len("```python") :].strip()
        if response.startswith("```"):
            response = response[3:].strip()
        if response.endswith("```"):
            response = response[:-3].strip()

        # Parse as Python literal
        import ast

        entities_list = ast.literal_eval(response)

        # Validate it's a list of lists
        if not isinstance(entities_list, list):
            print(f"Warning: Response is not a list: {type(entities_list)}")
            return []

        # Ensure each element is a list
        for i, item in enumerate(entities_list):
            if not isinstance(item, list):
                print(f"Warning: Item {i} is not a list: {type(item)}")
                entities_list[i] = []

        return entities_list

    except Exception as e:
        print(f"Error parsing entity list: {e}")
        print(f"Response was: {response[:200]}...")
        return []


def process_single_sample(entry):
    """Process a single sample to extract dialogues and entities."""
    ai_text = entry.get("AI", "")

    # Parse dialogues
    dialogues, original_text = parse_dialogue_text(ai_text)

    if not dialogues:
        print("Warning: No dialogues parsed from AI text")
        return None

    # Extract entities for all dialogue turns at once
    print(f"  Extracting entities for {len(dialogues)} dialogue turns...")
    entities_list = extract_entities_for_dialogue(original_text)

    # Validate that we got the right number of entity lists
    if len(entities_list) != len(dialogues):
        print(f"  Warning: Got {len(entities_list)} entity lists but expected {len(dialogues)}")
        # Pad with empty lists or truncate as needed
        if len(entities_list) < len(dialogues):
            entities_list.extend([[] for _ in range(len(dialogues) - len(entities_list))])
        else:
            entities_list = entities_list[: len(dialogues)]

    # Print summary
    total_entities = sum(len(ents) for ents in entities_list)
    print(f"  ✓ Extracted {total_entities} total entities across {len(dialogues)} turns")

    # Build output sample
    output_sample = {
        "dialogues": dialogues,
        "entities": entities_list,
        "label": entry.get("label"),
        "wrong_part": entry.get("wrong_part"),
        "type": entry.get("type"),
        "domain": entry.get("domain"),
        "source": entry.get("source"),
        "LLM": entry.get("LLM"),
    }

    return output_sample


def load_progress(save_path):
    """Load previously processed data if it exists."""
    try:
        with open(save_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data, save_path):
    """Save data to JSON file."""
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def process_dataset(input_path, output_path, max_workers=5):
    """
    Process entire dataset: parse dialogues and extract entities per turn.

    Args:
        input_path: Path to input JSON file with DiaHalu format
        output_path: Path to save output JSON file
        max_workers: Number of parallel workers for processing samples
    """
    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples from {input_path}")

    # Load any existing progress
    processed_data = load_progress(output_path)
    current_length = len(processed_data)

    print(f"Resuming from sample {current_length}")
    print(f"Processing with {max_workers} parallel workers")

    # Process remaining samples in parallel
    remaining_samples = dataset[current_length:]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all samples for processing
        future_to_idx = {
            executor.submit(process_single_sample, entry): (idx + current_length, entry)
            for idx, entry in enumerate(remaining_samples)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, entry = future_to_idx[future]
            print(f"\nProcessing sample {idx + 1}/{len(dataset)}...")

            try:
                output_sample = future.result()

                if output_sample:
                    processed_data.append(output_sample)

                    # Save progress after each sample
                    save_data(processed_data, output_path)

                    print(f"  ✓ Saved sample {idx + 1}: {len(output_sample['dialogues'])} turns")
                else:
                    print(f"  ✗ Failed to process sample {idx + 1}")

            except Exception as e:
                print(f"  ✗ Error processing sample {idx + 1}: {e}")
                continue

    print(f"\n✓ Processing complete! Saved {len(processed_data)} samples to {output_path}")
    return processed_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract entities per dialogue turn from DiaHalu dataset")
    parser.add_argument(
        "--input", type=str, default="Extract triples/dataset/DiaHalu_V2.json", help="Input JSON file path"
    )
    parser.add_argument(
        "--output", type=str, default="Temporal Graph/processed/diahalu_temporal.json", help="Output JSON file path"
    )
    parser.add_argument("--max-workers", type=int, default=5, help="Number of parallel workers for processing samples")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process the dataset
    process_dataset(args.input, args.output, max_workers=args.max_workers)
