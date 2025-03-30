import os
from collections import Counter


def log_tile_summary(hr_dir, lr_dir, iteration=1):
    def count_tiles(directory):
        category_counter = Counter()
        city_counter = Counter()
        total = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".tif"):
                    total += 1
                    category = os.path.basename(root)
                    category_counter[category] += 1

                    if "_delft.tif" in file:
                        city_counter["delft"] += 1
                    elif "_rotterdam.tif" in file:
                        city_counter["rotterdam"] += 1
                    else:
                        city_counter["unknown"] += 1

        return total, category_counter, city_counter

    print(f"\nSummary of categorized tiles (Iteration {iteration}):")

    total_hr, hr_by_cat, hr_by_city = count_tiles(hr_dir)
    total_lr, lr_by_cat, lr_by_city = count_tiles(lr_dir)

    print(f"\nTotal HR tiles: {total_hr}")
    for cat, count in sorted(hr_by_cat.items()):
        print(f"  {cat}: {count}")
    for city, count in hr_by_city.items():
        print(f"  From {city}: {count}")

    print(f"\nTotal LR tiles: {total_lr}")
    for cat, count in sorted(lr_by_cat.items()):
        print(f"  {cat}: {count}")
    for city, count in lr_by_city.items():
        print(f"  From {city}: {count}")
