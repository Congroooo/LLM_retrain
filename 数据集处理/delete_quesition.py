import json

# 读取JSON文件
with open('herb_instruction_dataset_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义要删除的问题关键词模式
patterns_to_remove = [
    "可以治疗哪些症状？",
    "可以治疗哪些症状"
]

# 筛选数据，保留不包含这些模式的问题
filtered_data = [
    item for item in data
    if not any(pattern in item.get('instruction', '') for pattern in patterns_to_remove)
]

print(f"原始数据量: {len(data)}")
print(f"过滤后数据量: {len(filtered_data)}")
print(f"删除了 {len(data) - len(filtered_data)} 条记录")

# 保存过滤后的数据
with open('herb_instruction_dataset_cleaned_filtered.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print("过滤后的数据已保存到 'herb_structured_filtered.json'")