import json
import random


def extract_qa_pairs(dataset_path, questions_path, answers_path):
    """
    根据问题文件提取对应的答案并保存

    Args:
        dataset_path: 数据集JSON文件路径
        questions_path: 问题txt文件路径
        answers_path: 答案txt文件路径
    """

    # 1. 读取数据集并建立instruction到output的映射
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 创建问题到答案的字典映射
    question_to_answer = {item["instruction"]: item["output"] for item in dataset}

    # 2. 读取问题文件
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_content = f.readlines()

    # 解析问题（移除序号前缀）
    questions = []
    for line in questions_content:
        line = line.strip()
        if line and '. ' in line:
            # 移除序号和点号，例如 "1. " -> ""
            question = line.split('. ', 1)[1] if '. ' in line else line
            questions.append(question)

    # 3. 提取对应答案并保存
    answers_found = 0
    with open(answers_path, 'w', encoding='utf-8') as f:
        for i, question in enumerate(questions, 1):
            if question in question_to_answer:
                answer = question_to_answer[question]
                f.write(f"{i}. {answer}\n")
                answers_found += 1
            else:
                # 如果问题在数据集中找不到，记录一条提示
                f.write(f"{i}. [未在数据集中找到对应答案]\n")
                print(f"警告：问题 '{question}' 在数据集中未找到")

    print(f"\n成功提取 {answers_found}/{len(questions)} 个问题的答案")
    print(f"已保存到: {answers_path}")



# 使用示例
if __name__ == "__main__":
    # 文件路径
    dataset_file = "D:/python_project/LLM_retrain/final_version/data/herb_instruction_dataset.json"
    questions_file = "random_questions.txt"
    answers_file = "corresponding_answers.txt"

    # 提取答案
    extract_qa_pairs(dataset_file, questions_file, answers_file)