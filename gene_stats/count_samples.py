import argparse
import csv
import json
import os


def main():
    # 获取脚本所在目录的上级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    default_json = os.path.join(parent_dir, 'hest_human_visium_subtypes.json')
    default_output = os.path.join(script_dir, 'dataset_sample_counts.csv')

    parser = argparse.ArgumentParser(description="Count samples per dataset group from a subtype JSON.")
    parser.add_argument(
        '--json',
        dest='json_path',
        default=default_json,
        help=f"Path to subtype JSON (default: {default_json})"
    )
    parser.add_argument(
        '--output',
        dest='output_path',
        default=default_output,
        help=f"Where to write the CSV (default: {default_output})"
    )
    args = parser.parse_args()

    # 读取JSON文件
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # 统计每个数据集的样本数量
    results = []
    for dataset_name, subtypes in data.items():
        total_samples = 0
        # 遍历所有子类型，累加样本数量
        for subtype, samples in subtypes.items():
            total_samples += len(samples)

        results.append({
            'dataset': dataset_name,
            'sample_count': total_samples
        })

    # 保存为CSV（保存在当前脚本所在目录）
    with open(args.output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'sample_count'])
        writer.writeheader()
        writer.writerows(results)

    # 打印结果
    print(f"Dataset Sample Counts from {args.json_path}:")
    print("-" * 60)
    for result in results:
        print(f"{result['dataset']:<50} {result['sample_count']:>5}")
    print("-" * 60)
    print(f"Total datasets: {len(results)}")
    print(f"Total samples: {sum(r['sample_count'] for r in results)}")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == '__main__':
    main()
