import os
import time
import json
import ollama

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

JSON_FILE = 'test.json'
MODEL_NAME = "qwen00:latest"
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
    # 使用临时文件和重命名来确保原子性写入，防止保存过程中断导致文件损坏
    temp_filepath = filepath + '.tmp'
    with open(temp_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temp_filepath, filepath)
    print(f"\n[进度已保存] 数据已成功保存到 {filepath}")


# ---------------------------   主程序   ---------------------------
def main():
    data = load_data(JSON_FILE)
    if data is None or 'episodes' not in data:
        print("JSON数据不合法或缺少 'episodes'，程序退出！！！")
        return

    episodes = data['episodes']
    total_items = len(episodes)
    processed_count = 0
    skipped_count = 0

    print("--- 开始预处理导航指令 ---")
    print(f"总共找到 {total_items} 条指令需要检查。")
    print(f"将每处理 {SAVE_INTERVAL} 条指令保存一次进度。")

    script_start_time = time.time()

    for i, item in enumerate(episodes):
        instruction = item.get("instruction", {})
        original_text = instruction.get("instruction_text") + '/no_think'
        print(f"\n--- 正在处理第 {i + 1}/{total_items} 条 ---")

        current_preprocessed_text = instruction.get("preprocessed_text")
        if current_preprocessed_text is not None and current_preprocessed_text != NULL_MARKER:
            if current_preprocessed_text == INVALID_INPUT_MARKER:
                print(f"指令已标记为 {INVALID_INPUT_MARKER}，跳过。")
            else:
                print(f"指令已成功预处理，跳过。")
            skipped_count += 1
            continue

        if current_preprocessed_text == NULL_MARKER:
            # print(f"检测到上次处理失败 ({NULL_MARKER})，正在重试...")
            print(f"检测到上次处理失败 ({NULL_MARKER})，暂时跳过...")
            skipped_count += 1
            continue

        print(f"原始导航指令: {original_text}")
        messages = [{"role": "user", "content": original_text}]

        item_start_time = time.time()
        response_data = ollama.chat(model=MODEL_NAME, stream=False, messages=messages)
        response_content = response_data['message']['content']
        if THINK_TAG in response_content:
            # 分割并获取</think>之后的内容
            preprocessed_text = response_content.split(THINK_TAG, 1)[-1].strip()
            item["instruction"]["preprocessed_text"] = preprocessed_text
            print(f"提取结果: {preprocessed_text}")
            processed_count += 1
        else:
            print(f"警告: 响应中未找到 '{THINK_TAG}' 标签。将标记为 {NULL_MARKER} 以便下次重试。")
            print(f"错误输出为 '{response_content}' ")
            item["instruction"]["preprocessed_text"] = NULL_MARKER

        item_end_time = time.time()
        print(f"本条耗时: {item_end_time - item_start_time:.4f} 秒")

        if processed_count > 0 and processed_count % SAVE_INTERVAL == 0:
            save_data(JSON_FILE, data)


    script_end_time = time.time()
    # --- 结果汇总与保存 ---
    print("\n--- 预处理完成 ---")
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
