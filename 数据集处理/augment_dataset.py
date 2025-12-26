import json
import time
import os
from openai import OpenAI, APIError, RateLimitError
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 在这里填入你的 DeepSeek API Key
DEEPSEEK_API_KEY = "sk-6e786118f0e94495a839dbdaa00a8e7e"

# 2. 输入和输出文件名
INPUT_FILE = "herb_treatment_subset.json"
OUTPUT_FILE = "herb_instruction_dataset_augmented_deepseek.json"

# 3. 模型选择
# 推荐使用 "deepseek-chat" (即 V3)，它支持 JSON Mode 且速度快
# "deepseek-reasoner" (R1) 目前对 JSON Mode 支持不如 V3 稳定，且适合复杂推理
MODEL_NAME = "deepseek-chat"
# ===========================================

# 初始化 OpenAI 客户端 (DeepSeek 兼容)
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"  # DeepSeek 的官方接口地址
)


def build_system_prompt():
    """
    系统提示词：设定角色并强制 JSON 输出
    DeepSeek 的 JSON Mode 要求 Prompt 中必须包含 'json' 字样
    """
    return """
    你是一位精通中医的数据增强专家。你的任务是将给定的中医问答对改写成 3 种不同的风格。
    请务必输出合法的 JSON 格式。
    """


def build_user_prompt(instruction, output):
    """
    构建用户提示词
    """
    return f"""
    【原始数据】
    指令: "{instruction}"
    回答: "{output}"

    【任务要求】
    请基于上述数据，生成 3 个新的问答对，并以 JSON 列表 (Array) 格式返回。

    1. **意思不变**：核心药理、方剂、剂量和主治绝对不能篡改。
    2. **语气多样化**：
       - 第1条：模拟【患者】（口语化，描述症状）。
       - 第2条：模拟【中医学生/医生】（专业，探讨药理，学术化）。
       - 第3条：模拟【搜索引擎/简略】（关键词查询，直接给出结论）。
    3. **输出格式**：
       请直接返回一个 JSON 列表，列表包含 3 个对象，每个对象有 "instruction" 和 "output" 两个字段。

    【JSON 示例】
    [
        {{"instruction": "...", "output": "..."}},
        {{"instruction": "...", "output": "..."}},
        {{"instruction": "...", "output": "..."}}
    ]
    """


def process_dataset():
    # 1. 读取原始数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            original_data = json.load(f)
        except json.JSONDecodeError:
            print("错误：无法解析输入的 JSON 文件。")
            return

    print(f"成功读取 {len(original_data)} 条原始数据。正在使用 DeepSeek ({MODEL_NAME})...")

    augmented_results = []

    # 断点续传逻辑
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list) and len(existing_data) > 0:
                    print(f"发现已有输出文件，包含 {len(existing_data)} 条数据。将追加写入。")
                    # 这里简单处理：如果你想完全去重，需要更复杂的逻辑
                    # 现在的逻辑是：追加模式，最后你可以手动去重
        except:
            pass

    SAVE_EVERY = 10

    # 遍历数据
    for i, entry in tqdm(enumerate(original_data), total=len(original_data)):
        original_instruction = entry.get("instruction", "")
        original_output = entry.get("output", "")

        if not original_instruction or not original_output:
            continue

        # 1. 保留原始数据
        augmented_results.append(entry)

        # 2. 调用 API 生成增强数据
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": build_system_prompt()},
                        {"role": "user", "content": build_user_prompt(original_instruction, original_output)}
                    ],
                    response_format={"type": "json_object"},  # 关键：强制 JSON 模式
                    temperature=1.1,  # DeepSeek 建议 V3 可以适当提高温度以增加多样性
                    max_tokens=2048
                )

                content = response.choices[0].message.content

                # 解析返回的 JSON
                try:
                    generated_entries = json.loads(content)

                    # 兼容处理：DeepSeek 有时会把列表包在一个 key 里（如 {"data": [...]}）
                    # 也有可能直接返回列表 [...]
                    if isinstance(generated_entries, dict):
                        # 尝试寻找列表值的 key
                        for key, value in generated_entries.items():
                            if isinstance(value, list):
                                generated_entries = value
                                break

                    if isinstance(generated_entries, list):
                        for new_entry in generated_entries:
                            if "instruction" in new_entry and "output" in new_entry:
                                merged_entry = entry.copy()
                                merged_entry["instruction"] = new_entry["instruction"]
                                merged_entry["output"] = new_entry["output"]
                                augmented_results.append(merged_entry)
                    else:
                        # 格式不对，跳过
                        pass

                except json.JSONDecodeError:
                    print(f"JSON 解析失败: {content[:50]}...")
                    continue

                break  # 成功则跳出重试

            except RateLimitError:
                print(f"\n[Warning] DeepSeek 限流，等待 10 秒...")
                time.sleep(10)
            except APIError as e:
                print(f"\n[Error] DeepSeek API 错误: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"\n[Error] 未知错误: {e}")
                time.sleep(1)

        # 3. 定期保存
        if (i + 1) % SAVE_EVERY == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
                json.dump(augmented_results, out_f, ensure_ascii=False, indent=2)

    # 最后保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        json.dump(augmented_results, out_f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！")
    print(f"原始数据量: {len(original_data)}")
    print(f"增强后数据量: {len(augmented_results)}")
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dataset()