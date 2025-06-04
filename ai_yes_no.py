import os
import time
import random
import csv
import re
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_env_config():
    load_dotenv()
    prompt_file = os.getenv("PROMPT1_FILE_PATH")
    csv_file = os.getenv("CSV_FILE_PATH")
    api_key = os.getenv("GEMINI_API_KEY")
    output_csv = "results/ai_judgment.csv"

    if not api_key:
        print("❌ GEMINI_API_KEY not found. Please check your .env file.")
        exit(1)

    return prompt_file, csv_file, api_key, output_csv

def load_prompt_template(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.readline().strip()

# ==== 读取句子 ====
def load_sentences_from_csv(csv_path):
    sentences = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for idx, row in enumerate(reader, start=1):
            if row and row[0].strip():
                sentence = row[0].strip()
                if not sentence.endswith(('。', '.', '！', '!', '？', '?')):
                    sentence += "。"
                sentences.append((idx, sentence))
    return sentences

def call_gemini_api(prompt, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
    )
    return response.text

def process_prompt_with_retry(prompt, api_key, retries=3, delay=5):
    time.sleep(random.uniform(1, 2))
    for i in range(retries):
        try:
            return call_gemini_api(prompt, api_key)
        except ServerError as e:
            print(f"[Retry {i+1}/{retries}] Server error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"[Error] Failed after multiple retries: {prompt}")
    return None

def process_all_prompts(prompts, api_key, max_workers=2):
    results = [None] * len(prompts)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_prompt_with_retry, prompt, api_key): i
            for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc="Processing"):
            i = future_to_index[future]
            try:
                result = future.result()
                results[i] = result.strip() if result else None
            except Exception as e:
                print(f"[Error] Prompt failed: {prompts[i]} -> {e}")
                results[i] = None
    return results

def build_prompts(template, indexed_sentences):
    return [template.format(sentence) for _, sentence in indexed_sentences]

def write_results_to_csv(csv_path, indexed_results):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        rows = []

    if not rows:
        rows = [["原句", "AI理解"]]

    if len(rows[0]) < 2:
        rows[0].append("AI理解")

    for idx, sentence, result in indexed_results:
        while len(rows) <= idx:
            rows.append([""] * len(rows[0]))
        while len(rows[idx]) < 2:
            rows[idx].append("")
        rows[idx][0] = sentence
        rows[idx][1] = result

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def write_results_to_txt(indexed_sentences, results, failed_path="results/failed.txt"):
    os.makedirs(os.path.dirname(failed_path), exist_ok=True)
    with open(failed_path, 'a', encoding='utf-8') as f:
        for (idx, sentence), result in zip(indexed_sentences, results):
            if not result:
                f.write(f"[{idx}] {sentence}\n")

def load_failed_sentences(failed_path="results/failed.txt"):
    failed = []
    try:
        with open(failed_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    m = re.match(r'\[(\d+)\]\s+(.*)', line)
                    if m:
                        idx = int(m.group(1))
                        sentence = m.group(2)
                        failed.append((idx, sentence))
    except FileNotFoundError:
        pass
    return failed

def update_failed_txt(failed_path, failed_sentences, succeeded_indices):
    new_failed = [fs for fs in failed_sentences if fs[0] not in succeeded_indices]
    with open(failed_path, 'w', encoding='utf-8') as f:
        for idx, sentence in new_failed:
            f.write(f"[{idx}] {sentence}\n")

def main():
    prompt_path, csv_path, api_key, output_csv= load_env_config()
    template = load_prompt_template(prompt_path)
    sentences = load_sentences_from_csv(csv_path)

    batch_size = 10
    total = len(sentences)
    current_index = 0

    while current_index < total:
        batch = sentences[current_index:current_index + batch_size]
        prompts = build_prompts(template, batch)

        print(f"\n📦 Processing batch {current_index + 1} to {current_index + len(batch)} of {total}...")

        results = process_all_prompts(prompts, api_key)
        indexed_results = [(idx, sentence, result) for (idx, sentence), result in zip(batch, results)]

        # 写入成功的
        write_results_to_csv(output_csv, [r for r in indexed_results if r[2]])

        # 处理失败的
        failed = [(idx, sentence) for (idx, sentence), result in zip(batch, results) if not result]
        if failed:
            print("\n❌ 以下句子处理失败：")
            for idx, sentence in failed:
                print(f"[{idx}] {sentence}")
            retry_choice = input("\n🔁 是否尝试重新处理这些失败的句子？输入 y 继续，其他键跳过：")
            if retry_choice.lower() == 'y':
                retry_prompts = build_prompts(template, failed)
                retry_results = process_all_prompts(retry_prompts, api_key)

                indexed_retry_results = [
                    (idx, sentence, result) for (idx, sentence), result in zip(failed, retry_results) if result
                ]
                succeeded_indices = [idx for idx, _, _ in indexed_retry_results]

                write_results_to_csv(output_csv, indexed_retry_results)
                update_failed_txt("results/failed.txt", failed, succeeded_indices)

                # 打印 retry 成果
                for idx, _, result in indexed_retry_results:
                    preview = result[:100] + ('...' if len(result) > 100 else '')
                    print(f"[Retry Result {idx}] ✅ {preview}")

        else:
            print("✅ 本批次全部成功。")

        # 是否继续
        if current_index + batch_size < total:
            cont = input("\n👉 是否继续处理下一批？输入 y 继续，其他键退出：")
            if cont.lower() != 'y':
                print("✅ 处理已手动终止。")
                break

        current_index += batch_size


if __name__ == "__main__":
    main()
