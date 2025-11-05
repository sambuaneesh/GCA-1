from concurrent.futures import ThreadPoolExecutor, as_completed

from extract_triples import *

# data_path = "/home/stealthspectre/iiith/GCA/Extract triples/dataset/WikiBio_dataset/wikibio.json"
# data_path = "/home/stealthspectre/iiith/GCA/Extract triples/dataset/PHD_benchmark.json"
data_path = "/home/stealthspectre/iiith/GCA/Extract triples/dataset/DiaHalu_V2.json"

suffix = "diahalu"
save_path = f"/home/stealthspectre/iiith/GCA/Extract triples/processed/out_triplets_{suffix}.json"

max_workers = 100

with open(data_path, "r", encoding="utf-8") as f:
    content = f.read()
dataset = json.loads(content)


def storage_triples_diahalu(dataset):
    process_data = load_progress(save_path)
    current_data_length = len(process_data)
    new_dataset = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        entry_futures = {
            executor.submit(init_triples_response, entry.get("gpt3_text") or entry.get("AI")): entry
            for entry in dataset[current_data_length:]
        }
        for future in as_completed(entry_futures):
            response = future.result()
            entry = entry_futures[future]
            revise_response = update_triples_response(entry.get("gpt3_text") or entry.get("AI"), response)
            print(revise_response)
            triples = process_triples_response(revise_response)
            new_entry = {
                "gpt3_text": entry.get("gpt3_text") or entry.get("AI"),
                "label": entry["label"],
                "wrong_part": entry["wrong_part"],
                "type": entry["type"],
                "domain": entry["domain"],
                "source": entry["source"],
                "LLM": entry["LLM"],
                "triples": triples,
            }
            print(len(new_entry["triples"]))
            print(new_entry)
            new_dataset.append(new_entry)
            save_data(new_dataset, save_path)
    return new_dataset


def storage_triples_phd(dataset):
    process_data = load_progress(save_path)
    current_data_length = len(process_data)
    new_dataset = process_data.copy()

    all_entries = []
    keys_to_search = ["PHD-Medium", "PHD-Low", "PHD-High"]
    for key in keys_to_search:
        for entry in dataset.get(key, []):
            entry_with_category = entry.copy()
            entry_with_category["category"] = key
            all_entries.append(entry_with_category)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        entry_futures = {
            executor.submit(init_triples_response, entry.get("gpt3_text") or entry.get("AI")): entry
            for entry in all_entries[current_data_length:]
        }
        for future in as_completed(entry_futures):
            response = future.result()
            entry = entry_futures[future]
            revise_response = update_triples_response(entry.get("gpt3_text") or entry.get("AI"), response)
            print(revise_response)
            triples = process_triples_response(revise_response)
            new_entry = {
                "entity": entry["entity"],
                "gpt3_text": entry.get("gpt3_text") or entry.get("AI"),
                "label": entry["label"],
                "category": entry["category"],
                "triples": triples,
            }
            print(f"Processed: {entry['entity']} ({entry['category']}) - {len(new_entry['triples'])} triples")
            print(new_entry)
            new_dataset.append(new_entry)
            save_data(new_dataset, save_path)
    return new_dataset


def storage_triples_wikibio(dataset):
    process_data = load_progress(save_path)
    current_data_length = len(process_data)
    new_dataset = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        entry_futures = {
            executor.submit(init_triples_response, entry.get("gpt3_text") or entry.get("AI")): entry
            for entry in dataset[current_data_length:]
        }
        for future in as_completed(entry_futures):
            response = future.result()
            entry = entry_futures[future]
            revise_response = update_triples_response(entry.get("gpt3_text") or entry.get("AI"), response)
            print(revise_response)
            triples = process_triples_response(revise_response)
            new_entry = {
                "entity": entry["entity"],
                "gpt3_text": entry.get("gpt3_text") or entry.get("AI"),
                "label": entry["label"],
                "triples": triples,
            }
            print(len(new_entry["triples"]))
            print(new_entry)
            new_dataset.append(new_entry)
            save_data(new_dataset, save_path)
    return new_dataset


if __name__ == "__main__":
    init_dataset = dataset
    if suffix == "diahalu":
        new_dataset = storage_triples_diahalu(init_dataset)
    elif suffix == "phd":
        new_dataset = storage_triples_phd(init_dataset)
    elif suffix == "wiki":
        new_dataset = storage_triples_wikibio(init_dataset)
    else:
        raise ValueError(f"Unknown suffix: {suffix}")
