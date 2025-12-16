import json
import re
from collections import defaultdict

# ========== 配置区域 ==========
INPUT_FILE = "herb_instruction_dataset.json"   # 你的原始文件
OUTPUT_FILE = "herb_instruction_dataset_cleaned.jsonl"

MIN_OUTPUT_LEN = 5        # 输出太短视为噪声
ENABLE_MERGE_DUP = True   # 若有重复 instruction，是否合并多个输出
MERGE_WITH_SEPARATOR = "\n\n【另一方案】\n"   # 多个方案之间的拼接格式

# 常见异常模式（可继续扩展）
BAD_PATTERNS = [
    r"洗去酸汁.*切细",      # 来自你数据的错误 instruction
    r"，。",
    r"。。",
    r"，，",
    r"^\s*$"
]

def looks_bad_text(text: str):
    """判断一段文本是否存在噪声或明显错误"""
    if len(text.strip()) < MIN_OUTPUT_LEN:
        return True
    for pat in BAD_PATTERNS:
        if re.search(pat, text):
            return True
    return False

def normalize_text(t: str):
    """简单文本规整：去空行、多空格、修正标点"""
    t = t.replace("。，", "。").replace(",,", "，")
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

# ========== 主流程 ==========

def load_dataset(path):
    """支持 JSON 或 JSONL 两种格式"""
    try:
        # 尝试读取 JSON list
        data = json.load(open(path, "r", encoding="utf8"))
        if isinstance(data, dict):
            raise ValueError
        return data
    except:
        # 尝试读取 JSONL
        data = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

def main():
    data = load_dataset(INPUT_FILE)
    print(f"Loaded {len(data)} records.")

    inst_dict = defaultdict(list)
    cleaned = []

    # 1) 预清洗与收集
    for d in data:
        inst = d.get("instruction", "").strip()
        out = d.get("output", "").strip()

        # 跳过 instruction 或 output 明显异常的样本
        if looks_bad_text(inst) or looks_bad_text(out):
            continue

        inst = normalize_text(inst)
        out = normalize_text(out)

        inst_dict[inst].append(out)

    # 2) 去重 / 合并重复 instruction
    for inst, outs in inst_dict.items():
        if ENABLE_MERGE_DUP:
            # 多输出时合并
            final_out = MERGE_WITH_SEPARATOR.join(sorted(set(outs)))
        else:
            # 只保留第一条
            final_out = outs[0]

        cleaned.append({
            "instruction": inst,
            "input": "",
            "output": final_out
        })

    # 3) 输出为 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf8") as fw:
        for d in cleaned:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Cleaned dataset size: {len(cleaned)}")
    print(f"Saved cleaned file to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
