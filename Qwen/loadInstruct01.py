import os
import re
import time
import json
import ollama

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

JSON_FILE = 'val_unseen.json'
MODEL_NAME = "qwen01:latest"
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

def generate_subtasks(raw_input):
    # --- 步骤 0: 成分提取 ---
    pattern = r"(n\.A|b\.L|o\.L)\((.*?)\)"
    raw_components = re.findall(pattern, raw_input)
    components = [f"{tag}({arg})" for tag, arg in raw_components]

    # --- 以and合并连续的A.e构造n.A ---
    consolidated_components = []
    i = 0
    while i < len(components):
        current_comp = components[i]
        if current_comp.startswith('n.A'):
            na_group_args = []
            na_group_args.append(re.search(r'n\.A\((.*)\)', current_comp).group(1))
            j = i + 1
            while j < len(components) and components[j].startswith('n.A'):
                na_group_args.append(re.search(r'n\.A\((.*)\)', components[j]).group(1))
                j += 1

            merged_arg = " and ".join(na_group_args)
            consolidated_components.append(f"n.A({merged_arg})")
            i = j
        else:
            consolidated_components.append(current_comp)
            i += 1

    # --- 分割成子任务行 (Segmentation) ---
    subtask_lines_of_components = []
    if consolidated_components:
        current_line_components = []
        for comp in consolidated_components:
            if comp.startswith('n.A'):
                if current_line_components:
                    subtask_lines_of_components.append(current_line_components)
                current_line_components = [comp]
            else:
                current_line_components.append(comp)
        if current_line_components:
            subtask_lines_of_components.append(current_line_components)

    # 输出导航子任务
    Subtask = []
    arg_pattern = re.compile(r'\((.*)\)')
    for line_comps in subtask_lines_of_components:
        args = []
        for comp in line_comps:
            match = arg_pattern.search(comp)
            if match:
                args.append(match.group(1))
        Subtask.append(", ".join(args))

    # 输出导航成分
    subtask_str_lines = []
    for line_comps in subtask_lines_of_components:
        subtask_str_lines.append(", ".join(line_comps) + ",")
    Component = "\n".join(subtask_str_lines)

    return Subtask, Component


# ------------------------------------   01 成分解析   ------------------------------------
def main():
    data = load_data(JSON_FILE)
    if data is None or 'episodes' not in data:
        print("JSON数据不合法或缺少 'episodes'，程序退出！！！")
        return

    episodes = data['episodes']
    total_items = len(episodes)
    processed_count = 0
    skipped_count = 0

    print("--- STAGE 1: Component Parsing ---")
    print(f"总共找到 {total_items} 条指令需要检查。")
    print(f"将每处理 {SAVE_INTERVAL} 条指令保存一次进度。")

    script_start_time = time.time()

    for i, item in enumerate(episodes):
        print(f"\n--- 正在处理第 {i + 1}/{total_items} 条 ---")
        instruction = item.get("instruction", {})
        preprocessed_text = instruction.get("preprocessed_text")

        # 检查前置条件：preprocessed_text 是否有效
        if preprocessed_text in [None, NULL_MARKER, INVALID_INPUT_MARKER]:
            print(f"输入文本无效 ({preprocessed_text}) 或不存在，无法进行成分解析。跳过。")
            skipped_count += 1
            continue

        # 跳过已处理部分
        Subtask_status = instruction.get("Subtask")
        # if Subtask_status is not None and Subtask_status != NULL_MARKER:
        if Subtask_status is not None:
            print("该成分已解析，跳过。")
            skipped_count += 1
            continue

        # if Subtask_status == NULL_MARKER:
        #     print(f"检测到上次处理失败 ({NULL_MARKER})，正在重试...")

        print(f"预处理指令: {preprocessed_text}")
        Input01 = [{"role": "user", "content": preprocessed_text}]
        item_start_time = time.time()
        response_data = ollama.chat(model=MODEL_NAME, stream=False, messages=Input01)
        response_content = response_data['message']['content']
        item_end_time = time.time()
        print(f"LLM调用耗时: {item_end_time - item_start_time:.4f} 秒")

        # --- 解析并存储结果 ---
        if THINK_TAG in response_content:
            # 分割并获取</think>之后的内容
            component_raw = response_content.split(THINK_TAG, 1)[-1].strip()
            Subtask, Component = generate_subtasks(component_raw)
            item["instruction"]["Subtask"] = Subtask
            item["instruction"]["Component"] = Component
            print(f"提取Subtask结果: {Subtask}")
            print(f"提取Component结果: {Component}")
            processed_count += 1
        else:
            print(f"警告: 响应中未找到 '{THINK_TAG}' 标签,标记为 {NULL_MARKER} 以便下次重试。")
            print(f"错误输出为 '{response_content}' ")
            item["instruction"]["Subtask"] = NULL_MARKER

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


if __name__ == "__main__":
    main()
