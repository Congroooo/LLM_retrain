import json

# 读取 JSON 文件
with open('herb_data.json', 'r', encoding='utf-8') as f:
    herb_data = json.load(f)

# 准备存储所有转换后的数据
dataset = []
# 遍历每一种草药
for herb_name, herb_info in herb_data.items():

    # 1. 生成"属于哪个部门"的问题（仅当部门非空）
    department = herb_info.get('部门', '')
    if department:
        instruction = f"{herb_name}属于哪个部门？"
        input_text = ""
        output_text = f"{herb_name}属于{department}。"
        dataset.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    # 2. 生成"有哪些释名"的问题（仅当释名列表非空）
    alternative_names = herb_info.get('释名', [])
    if alternative_names:
        # 过滤掉空字符串的释名
        valid_names = [name for name in alternative_names if name.strip()]
        if valid_names:
            instruction = f"{herb_name}有哪些释名？"
            input_text = ""
            output_text = f"{herb_name}的释名包括：{'、'.join(valid_names)}。"
            dataset.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })

    # 3. 生成"气味是什么"的问题（仅当气味非空）
    smell = herb_info.get('气味', '')
    if smell and smell.strip():
        instruction = f"{herb_name}的气味是什么？"
        input_text = ""
        output_text = f"{herb_name}的气味是：{smell}。"
        dataset.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    # 4. 生成"可以治疗哪些症状"的问题（仅当有主治条目且症状非空）
    all_symptoms = []
    treatments = herb_info.get('主治', [])

    if treatments:
        for treatment in treatments:
            symptoms = treatment.get('症状', [])
            if symptoms:
                all_symptoms.extend(symptoms)

        # 去重并过滤空症状
        all_symptoms = list(dict.fromkeys([s for s in all_symptoms if s.strip()]))

        if all_symptoms:
            instruction = f"{herb_name}可以治疗哪些症状？"
            input_text = ""
            # 构建带序号的分行输出
            symptom_lines = []
            for i, symptom in enumerate(all_symptoms, 1):
                symptom_lines.append(f"{i}. {symptom}")
            output_text = f"{herb_name}可以治疗以下症状：\n" + "\n".join(symptom_lines)
            dataset.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })

    # 5. 针对每一个主治条目，生成"治疗XX症状的方法是什么"（仅当症状和方法都非空）
    for treatment in treatments:
        symptoms = treatment.get('症状', [])
        method = treatment.get('方法', '')

        # 过滤掉空症状
        valid_symptoms = [s for s in symptoms if s.strip()]

        if valid_symptoms and method and method.strip():
            instruction = f"如何用{herb_name}治疗{'、'.join(valid_symptoms)}？"
            input_text = ""
            output_text = f"治疗{'、'.join(valid_symptoms)}的方法是：{method}。"
            dataset.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })

# 保存为新的 JSON 文件
with open('herb_instruction_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"共生成 {len(dataset)} 条指令数据，已保存到 herb_instruction_dataset.json")
print(f"处理的草药种类：{len(herb_data)}")

# 显示一些统计信息
print("\n数据统计：")
print(f"总问答对数量：{len(dataset)}")
print(f"平均每种草药生成 {len(dataset) / len(herb_data):.1f} 个问答对")

