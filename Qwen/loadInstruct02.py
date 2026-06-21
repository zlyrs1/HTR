import os
import re
import time
import json
import ollama

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

JSON_FILE = 'val_unseen.json'
MODEL_NAME = "qwen02:latest"
THINK_TAG = "</think>"
SAVE_INTERVAL = 5

# 定义特殊标记
NULL_MARKER = "[NULL]"
INVALID_INPUT_MARKER = "[INVALID_INPUT]"


def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {filepath} 不是有效的JSON格式")
        return None

def save_data(filepath, data):
    # ... (与第一个脚本完全相同) ...
    temp_filepath = filepath + '.tmp'
    with open(temp_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temp_filepath, filepath)
    print(f"\n[进度已保存] 数据已成功保存到 {filepath}")

def nA_extraction(component):
    pattern_nA = r"n\.A\s*\((.*?)\)"
    nA = re.findall(pattern_nA, component, flags=re.MULTILINE)

    Ae = []
    for action_group in nA:
        sub_actions = [part.strip() for part in re.split(r'\s+and\s+', action_group)]
        Ae.append(sub_actions)

    return nA, Ae

def nL_extraction(component):
    pattern_nL = r"^\s*n\.A\s*\((?:[^()]*|\((?:[^()]*|\((?:[^()]*)*\))*\))*\)\s*,?\s*"
    processed_text = re.sub(pattern_nL, '', component, flags=re.MULTILINE)
    lines = processed_text.strip().splitlines()

    pattern_extract = r"[a-zA-Z]\.L\s*\(([^)]*)\)"
    nL = []
    for line in lines:
        extractions = re.findall(pattern_extract, line)
        joined_string = ', '.join(extractions)
        nL.append(joined_string)
    nL_str = '\n'.join(nL)

    return nL, nL_str


# ------------------------------------   02 提取导航元素  ------------------------------------
def main():
    data = load_data(JSON_FILE)
    if data is None or 'episodes' not in data:
        print("JSON数据不合法或缺少 'episodes'，程序退出！！！")
        return

    episodes = data['episodes']
    total_items = len(episodes)
    processed_count = 0
    skipped_count = 0

    print("--- STAGE 2: Elements Extraction ---")
    print(f"总共找到 {total_items} 条指令需要检查。")
    print(f"将每处理 {SAVE_INTERVAL} 条指令保存一次进度。")

    script_start_time = time.time()

    for i, item in enumerate(episodes):
        print(f"\n--- 正在处理第 {i + 1}/{total_items} 条 ---")
        instruction = item.get("instruction", {})
        preprocessed_text = instruction.get("preprocessed_text")

        # 检查前置条件：preprocessed_text是否有效
        if preprocessed_text in [None, NULL_MARKER, INVALID_INPUT_MARKER]:
            print(f"输入文本无效 ({preprocessed_text}) 或不存在，无法进行进行导航元素提取。跳过。")
            skipped_count += 1
            continue

        # 检查前置条件：Component是否有效
        Component = instruction.get("Component")
        if Component in [None, NULL_MARKER]:
            print(f"输入文本无效 ({Component}) 或不存在，无法进行导航元素提取。跳过。")
            skipped_count += 1
            continue

        # 跳过已处理部分
        Le = instruction.get("Le")
        if Le is not None:
            print("元素已成功提取，跳过。")
            skipped_count += 1
            continue

        nA, Ae = nA_extraction(Component)
        item["instruction"]["nA"] = nA
        item["instruction"]["Ae"] = Ae

        nL, nL_str = nL_extraction(Component)
        item["instruction"]["nL"] = nL

        print(f"输入的nL:\n {nL_str}")
        Input02 = [{"role": "user", "content": nL_str + '/no_think'}]
        item_start_time = time.time()
        response_data = ollama.chat(model=MODEL_NAME, stream=False, messages=Input02)
        response_content = response_data['message']['content']
        item_end_time = time.time()
        print(f"LLM调用耗时: {item_end_time - item_start_time:.4f} 秒")

        # --- 解析并存储结果 ---
        if THINK_TAG in response_content:
            Le_str = response_content.split(THINK_TAG, 1)[-1].strip()
            Le = [[item.strip() for item in line.split(',')] for line in Le_str.splitlines()]
            item["instruction"]["Le"] = Le
            print("提取Le结果:")
            for sublist in Le:
                print(sublist)
            processed_count += 1
        else:
            print(f"警告: 响应中未找到 '{THINK_TAG}' 标签,标记为 {NULL_MARKER} 以便下次重试。")
            print(f"错误输出为 '{response_content}' ")
            item["instruction"]["Le"] = NULL_MARKER

        if processed_count > 0 and processed_count % SAVE_INTERVAL == 0:
            save_data(JSON_FILE, data)

    script_end_time = time.time()
    # --- 结果汇总与最终保存 ---
    print("\n--- 成分解析完成 ---")
    print(f"总耗时: {script_end_time - script_start_time:.4f} 秒")
    print(f"本次新增处理: {processed_count} 条")
    print(f"跳过已处理: {skipped_count} 条")
    print(f"当前总进度: {processed_count + skipped_count}/{total_items}")

    if processed_count > 0:
        save_data(JSON_FILE, data)
    else:
        print("\n本次运行没有新的指令被处理，无需保存文件。")

    save_data(JSON_FILE, data)


if __name__ == "__main__":
    main()
