import json


def filter_treatment_data(input_file, output_file):
    # 读取原始JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {input_file}")
        return

    filtered_data = []

    # 遍历数据进行筛选
    for item in data:
        instruction = item.get('instruction', '')

        # 筛选逻辑：
        # 1. 保留包含“如何”和“治疗”的条目 (对应“如何用XX治疗XX？”)
        # 2. 排除包含“部门”或“气味”的条目 (双重确认，虽然上面的条件通常已经排除了这些)
        if "如何" in instruction and "治疗" in instruction:
            if "部门" not in instruction and "气味" not in instruction:
                filtered_data.append(item)

    # 将筛选后的数据写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成。")
    print(f"原始数据条数: {len(data)}")
    print(f"提取数据条数: {len(filtered_data)}")
    print(f"新文件已保存为: {output_file}")


# 执行函数
input_filename = 'herb_instruction_dataset_cleaned_filtered.json'
output_filename = 'herb_treatment_subset.json'

if __name__ == '__main__':
    filter_treatment_data(input_filename, output_filename)