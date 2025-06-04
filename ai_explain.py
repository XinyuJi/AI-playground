import os
import time
import random
import csv
import re
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_env_config():
    load_dotenv()
    return (os.getenv("PROMPT_FILE_PATH"), os.getenv("CSV_FILE_PATH"), os.getenv("GEMINI_API_KEY"))

def load_prompt_template(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.readline().strip()

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

def build_prompts(template, indexed_sentences):
    return [template.format(sentence) for _, sentence in indexed_sentences]

def call_gemini_api(prompt, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
    return response.text

def process_prompt_with_retry(prompt, api_key, retries=3, delay=5):
    time.sleep(random.uniform(1, 2))
    for i in range(retries):
        try:
            return call_gemini_api(prompt, api_key)
        except (ServerError) as e:
            print(f"[Retry {i+1}/{retries}] API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return None

def clean_formatting(text):
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text

def split_result_lines(text):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return None, None
    judgment = lines[0]
    explanation = '\n'.join(lines[1:]) if len(lines) > 1 else ''
    return judgment, explanation

def write_results_to_csv(csv_path, results, offset=0):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    header, data_rows = reader[0], reader[1:]
    if "ai判断" not in header:
        header.append("ai判断")
    if "ai理解" not in header:
        header.append("ai理解")
    for i, result in enumerate(results):
        if result:
            row_index = offset + i
            judgment, explanation = split_result_lines(result)
            while len(data_rows[row_index]) < len(header):
                data_rows[row_index].append("")
            data_rows[row_index][header.index("ai判断")] = judgment or ""
            data_rows[row_index][header.index("ai理解")] = explanation or ""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([header] + data_rows)

def write_success_results_to_txt(indexed_sentences, results, path="results/success_results.txt"):
    with open(path, 'a', encoding='utf-8') as f:
        for (idx, sentence), result in zip(indexed_sentences, results):
            if result is not None:
                judgment, explanation = split_result_lines(result)
                f.write(f"[行号 {idx}] {sentence}\n")
                f.write(f"AI判断: {judgment or ''}\n")
                f.write(f"AI理解: {explanation or ''}\n")
                f.write("-" * 40 + "\n")

def write_failed_to_txt(indexed_sentences, results, path="results/failed.txt"):
    with open(path, 'w', encoding='utf-8') as f:
        for (idx, sentence), result in zip(indexed_sentences, results):
            if result is None:
                f.write(f"[{idx}] {sentence}\n")

def load_failed_sentences(path="results/failed.txt"):
    failed = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r'\[(\d+)\]\s+(.*)', line.strip())
                if m:
                    failed.append((int(m.group(1)), m.group(2)))
    except FileNotFoundError:
        pass
    return failed

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
                result = clean_formatting(future.result())
                results[i] = result
            except Exception as e:
                print(f"[Error] Prompt failed: {prompts[i]} -> {e}")
    return results

def main():
    prompt_path, csv_path, api_key = load_env_config()
    template = load_prompt_template(prompt_path)
    indexed_sentences = load_sentences_from_csv(csv_path)

    batch_size = 10
    total = len(indexed_sentences)
    current_index = 0

    while current_index < total:
        batch = indexed_sentences[current_index:current_index + batch_size]
        prompts = build_prompts(template, batch)

        print(f"\n📦 Processing batch {current_index + 1} to {current_index + len(batch)} of {total}...")

        results = process_all_prompts(prompts, api_key)
        #write_results_to_txt(batch, results)
        write_results_to_csv(csv_path, results, offset=current_index)

        # 找出失败项
        failed_indices = [i for i, r in enumerate(results) if r is None]

        if failed_indices:
            print("\n❌ 以下句子处理失败：")
            for i in failed_indices:
                idx, sentence = batch[i]
                print(f"  [行号 {idx}] {sentence}")

            retry_choice = input("\n🔁 是否重试失败的句子？输入 y 继续，其他键跳过：")
            if retry_choice.lower() == 'y':
                # 只重试失败的句子
                failed_batch = [batch[i] for i in failed_indices]
                failed_prompts = build_prompts(template, failed_batch)
                retry_results = process_all_prompts(failed_prompts, api_key)
                #write_results_to_txt(failed_batch, retry_results)
                # 更新 CSV
                for i, result in enumerate(retry_results):
                    if result is not None:
                        idx_in_csv = failed_batch[i][0] - 1
                        judgment, explanation = split_result_lines(result)
                        # 读写CSV更新该行
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            reader = list(csv.reader(f))
                        header = reader[0]
                        data_rows = reader[1:]
                        while len(data_rows[idx_in_csv]) < len(header):
                            data_rows[idx_in_csv].append("")
                        if "ai判断" not in header:
                            header.append("ai判断")
                        if "ai理解" not in header:
                            header.append("ai理解")
                        data_rows[idx_in_csv][header.index("ai判断")] = judgment or ""
                        data_rows[idx_in_csv][header.index("ai理解")] = explanation or ""
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(header)
                            writer.writerows(data_rows)
                print("✅ 重试完成，已更新CSV。")
            else:
                print("⚠️ 跳过重试。")
        else:
            print("✅ 本批句子全部处理成功。")

        # 是否继续下一批
        if current_index + batch_size < total:
            cont = input("\n👉 是否继续处理下一批？输入 y 继续，其他键退出：")
            if cont.lower() != 'y':
                print("✅ 处理已手动终止。")
                break

        current_index += batch_size

    print("🎉 全部完成或中途终止。")


if __name__ == "__main__":
    main()
