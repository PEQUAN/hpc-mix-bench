import json
import csv
import os
import re
import sys

CATEGORY_DISPLAY_NAMES = {
    'double': 'FP64',
    'float': 'FP32',
    'flx::floatx<5, 10>': 'FP16',
    'flx::floatx<8, 7>': 'BF16',
    'flx::floatx<4, 3>': 'E4M3',
    'flx::floatx<5, 2>': 'E5M2'
}

ROOT_DIR = '.'
JSON_PATTERN = re.compile(r"prec_setting_([0-9]+)\.json$")


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_stats.py folder1 folder2 ... folderk")
        sys.exit(1)

    target_folders = sys.argv[1:]
    summary_data = []

    # 用于统计“每行占比”的平均值
    ratio_sums = {display_name: 0.0 for display_name in CATEGORY_DISPLAY_NAMES.values()}
    valid_ratio_rows = 0

    for folder in target_folders:
        folder_path = os.path.join(ROOT_DIR, folder)

        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid folder, skipping.")
            continue

        json_files = []
        for file in os.listdir(folder_path):
            match = JSON_PATTERN.match(file)
            if match:
                index = int(match.group(1))
                json_files.append((index, file))

        if not json_files:
            print(f"No prec_setting_<i>.json files found in {folder_path}, skipping.")
            continue

        json_files.sort(key=lambda x: x[0])

        for precision_setting, json_file in json_files:
            file_path = os.path.join(folder_path, json_file)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Failed to read {file_path}, skipping.")
                continue

            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list, skipping.")
                continue

            for significant_digits, entry in enumerate(data, start=1):
                if not isinstance(entry, dict):
                    print(f"Warning: Non-dict entry in {file_path}, skipping.")
                    continue

                counts = []
                row = [folder, precision_setting, significant_digits]

                for key in CATEGORY_DISPLAY_NAMES:
                    value = entry.get(key, [])
                    count = len(value) if isinstance(value, list) else 0
                    counts.append(count)
                    row.append(count)

                summary_data.append(row)

                # 计算这一行各精度占比
                total_count = sum(counts)
                if total_count > 0:
                    valid_ratio_rows += 1
                    display_names = list(CATEGORY_DISPLAY_NAMES.values())
                    for i, display_name in enumerate(display_names):
                        ratio_sums[display_name] += counts[i] / total_count

    if not summary_data:
        print("No valid data found across the specified folders.")
        return

    # 写原始汇总 CSV
    headers = [
        'Folder',
        'Precision Setting',
        'Significant Digits',
        'FP64',
        'FP32',
        'FP16',
        'BF16',
        'E4M3',
        'E5M2'
    ]

    output_csv = os.path.join(ROOT_DIR, 'fp_counts_summary.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(summary_data)

    print(f"Generated Summary CSV: {output_csv}")

    # 计算平均占比
    if valid_ratio_rows == 0:
        print("No valid rows with non-zero counts, cannot compute average ratios.")
        return

    avg_ratios = {
        name: ratio_sums[name] / valid_ratio_rows
        for name in ratio_sums
    }

    # 打印结果
    print("\nAverage ratio of each precision across all valid rows:")
    for name, avg_ratio in avg_ratios.items():
        print(f"{name}: {avg_ratio:.6f} ({avg_ratio * 100:.2f}%)")

    # 写平均占比 CSV
    ratio_csv = os.path.join(ROOT_DIR, 'fp_ratio_averages.csv')
    with open(ratio_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Precision', 'Average Ratio', 'Average Percentage'])
        for name, avg_ratio in avg_ratios.items():
            writer.writerow([name, avg_ratio, avg_ratio * 100])

    print(f"Generated Average Ratio CSV: {ratio_csv}")


if __name__ == "__main__":
    main()