import csv
import json
import os
import time
import random
from tqdm import tqdm
from google import genai
from matplotlib import rcParams

# 中文字体支持
rcParams['font.sans-serif'] = ['PingFang', 'STHeiti', 'Arial']
rcParams['axes.unicode_minus'] = False

# 配置项
DISAMBIGUATION_FIELDS = ['歧义句消岐1', '歧义句消岐2', '歧义句消岐3', '歧义句消岐4']
AI_FIELD = 'ai理解'

# Gemini API Key（可从环境变量读取）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 调用 Gemini API
def call_gemini_api(prompt, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
    )
    return response.text

# 带重试机制的API调用
def process_prompt_with_retry(prompt, api_key, retries=3, delay=5):
    time.sleep(random.uniform(1, 3))
    for i in range(retries):
        try:
            return call_gemini_api(prompt, api_key)
        except Exception as e:
            print(f"[Retry {i+1}/{retries}] Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"[Error] Failed after multiple retries: {prompt}")
    return None

# 读取CSV并收集数据
def read_csv(input_csv):
    pairs = []
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dis_list = []
            for field in DISAMBIGUATION_FIELDS:
                value = (row.get(field) or '').strip()
                if value and value != '无':
                    dis_list.append(value)
            ai_value = (row.get(AI_FIELD) or '').strip()
            if dis_list and ai_value:
                dis_value = '; '.join(dis_list)
                pairs.append((dis_value, ai_value))
    return pairs


# 构造 prompt 并让 Gemini 判断答对几个
def evaluate_match(dis_values, ai_value, api_key):
    prompt = f"""请你判断下列AI的理解与原理解的匹配程度。

原理解选项:
{dis_values}

AI的理解:
{ai_value}

请回答匹配的个数，不需要解释原因。"""
    response = process_prompt_with_retry(prompt, api_key)
    try:
        matched = int(''.join(c for c in response if c.isdigit()))
        return matched
    except:
        print(f"无法解析响应: {response}")
        return 0

# 计算precision和recall，并写入结果文件
def calculate_precision_recall(pairs, api_key, output_file):
    total_correct = 0
    total_ai = 0
    total_gold = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (dis, ai) in enumerate(tqdm(pairs), 1):
            dis_list = [x.strip() for x in dis.split(';') if x.strip()]
            gold_count = len(dis_list)
            ai_count = 1

            matched = evaluate_match(dis, ai, api_key)

            total_correct += matched
            total_gold += gold_count
            total_ai += ai_count

            f.write(f"样本 {idx}:\n")
            f.write(f"原理解: {dis}\n")
            f.write(f"AI理解: {ai}\n")
            f.write(f"匹配个数: {matched} / {gold_count}\n")
            f.write("-" * 40 + "\n")

        precision = total_correct / total_ai if total_ai else 0
        recall = total_correct / total_gold if total_gold else 0

        f.write("\n======= 汇总结果 =======\n")
        f.write(f"✅ Precision: {precision:.3f}\n")
        f.write(f"✅ Recall: {recall:.3f}\n")

    print(f"✅ Precision: {precision:.3f}")
    print(f"✅ Recall: {recall:.3f}")
    print(f"📄 已写入结果到 {output_file}")

    # ✅ 写入 JSON 文件
    metrics_data = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "total_samples": len(pairs),
        "total_correct": total_correct
    }

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as json_file:
        json.dump(metrics_data, json_file, ensure_ascii=False, indent=2)

# 主运行逻辑
if __name__ == "__main__":
    CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")
    OUTPUT_FILE = "results/match_result.txt"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if CSV_FILE_PATH is None:
        raise EnvironmentError("请设置环境变量 CSV_FILE_PATH")
    if GEMINI_API_KEY is None:
        raise EnvironmentError("请设置环境变量 GEMINI_API_KEY")

    pairs = read_csv(CSV_FILE_PATH)
    calculate_precision_recall(pairs, GEMINI_API_KEY, OUTPUT_FILE)
