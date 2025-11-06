import json
from collections import Counter

with open("Temporal Graph/processed/diahalu_temporal.json", "r", encoding="utf-8") as f:
    data = json.load(f)

types = set()
type_counts = Counter()

for entry in data:
    type_value = entry.get("type")
    if type_value is not None:
        types.add(type_value)
        type_counts[type_value] += 1

print(f"Total samples: {len(data)}")
print(f"\nUnique 'type' values found: {len(types)}")
print("\nAll unique types:")
print("-" * 50)

for type_val in sorted(types):
    print(f"  - {type_val}")

print("\n" + "=" * 50)
print("Type distribution (with counts):")
print("=" * 50)

for type_val, count in type_counts.most_common():
    print(f"  {type_val}: {count}")

print("\n" + "=" * 50)
print(f"Samples with null/missing type: {sum(1 for e in data if e.get('type') is None)}")
