import json
import random

# 读取数据集
with open('D:/python_project/LLM_retrain/final_version/data/herb_instruction_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 提取所有问题
all_questions = [item["instruction"] for item in dataset]

# 随机抽取100个问题（如果不够就全部抽取）
num_to_extract = min(200, len(all_questions))
random_questions = random.sample(all_questions, num_to_extract)

# 保存到txt文件（带序号，无空行）
with open('random_questions.txt', 'w', encoding='utf-8') as f:
    for i, question in enumerate(random_questions, 1):
        f.write(f"{i}. {question}\n")

print(f"已提取 {num_to_extract} 个随机问题到 random_questions.txt")