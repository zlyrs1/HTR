import os
import time
import json
import ollama

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

JSON_FILE = 'val_seen.json'
MODEL_NAME = "qwenS"
THINK_TAG = "</think>"
SAVE_INTERVAL = 5

# 定义特殊标记
NULL_MARKER = "[NULL]"
INVALID_INPUT_MARKER = "[INVALID_INPUT]"
REQUIRED_KEYS = ["subtask", "nA", "Ae", "nL", "Le"]


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
    temp_filepath = filepath + '.tmp'
    with open(temp_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temp_filepath, filepath)
    print(f"\n[进度已保存] 数据已成功保存到 {filepath}")


def extract_json_text(response_content):
    if THINK_TAG in response_content:
        candidate = response_content.split(THINK_TAG, 1)[-1].strip()
    else:
        candidate = response_content.strip()

    if candidate.startswith('{') and candidate.endswith('}'):
        return candidate

    start = candidate.find('{')
    end = candidate.rfind('}')
    if start != -1 and end != -1 and start < end:
        return candidate[start:end + 1]

    return candidate


def validate_result(parsed_json):
    if not isinstance(parsed_json, dict):
        return False

    for key in REQUIRED_KEYS:
        if key not in parsed_json:
            return False

    if not all(isinstance(parsed_json[key], list) for key in REQUIRED_KEYS):
        return False

    return True


# ------------------------------------   Simple Parsing   ------------------------------------
def main():
    data = load_data(JSON_FILE)
    if data is None or 'episodes' not in data:
        print("JSON数据不合法或缺少 'episodes'，程序退出！！！")
        return

    episodes = data['episodes']
    total_items = len(episodes)
    processed_count = 0
    skipped_count = 0

    print("--- SIMPLE STAGE: Direct Parsing ---")
    print(f"总共找到 {total_items} 条指令需要检查。")
    print(f"将每处理 {SAVE_INTERVAL} 条指令保存一次进度。")

    script_start_time = time.time()

    for i, item in enumerate(episodes):
        print(f"\n--- 正在处理第 {i + 1}/{total_items} 条 ---")
        instruction = item.get("instruction", {})
        original_text = instruction.get("instruction_text")

        if not original_text:
            print("原始指令不存在，跳过。")
            skipped_count += 1
            continue

        if instruction.get("subtask") is not None:
            print("该指令已完成简单解析，跳过。")
            skipped_count += 1
            continue

        print(f"原始导航指令: {original_text}")
        messages = [{"role": "user", "content": original_text}]

        item_start_time = time.time()
        response_data = ollama.chat(model=MODEL_NAME, stream=False, messages=messages)
        response_content = response_data['message']['content']
        item_end_time = time.time()
        print(f"LLM调用耗时: {item_end_time - item_start_time:.4f} 秒")

        if INVALID_INPUT_MARKER in response_content:
            print(f"模型返回 {INVALID_INPUT_MARKER}，跳过。")
            instruction["subtask"] = INVALID_INPUT_MARKER
            instruction["nA"] = INVALID_INPUT_MARKER
            instruction["Ae"] = INVALID_INPUT_MARKER
            instruction["nL"] = INVALID_INPUT_MARKER
            instruction["Le"] = INVALID_INPUT_MARKER
            processed_count += 1
        else:
            json_text = extract_json_text(response_content)
            try:
                parsed_json = json.loads(json_text)
                if not validate_result(parsed_json):
                    raise ValueError("输出JSON缺少必要字段或字段类型错误")

                instruction["subtask"] = parsed_json["subtask"]
                instruction["nA"] = parsed_json["nA"]
                instruction["Ae"] = parsed_json["Ae"]
                instruction["nL"] = parsed_json["nL"]
                instruction["Le"] = parsed_json["Le"]

                print(f"提取subtask结果: {instruction['subtask']}")
                print(f"提取nA结果: {instruction['nA']}")
                print(f"提取Ae结果: {instruction['Ae']}")
                print(f"提取nL结果: {instruction['nL']}")
                print(f"提取Le结果: {instruction['Le']}")
                processed_count += 1
            except (json.JSONDecodeError, ValueError) as e:
                print(f"警告: 响应解析失败，标记为 {NULL_MARKER} 以便下次重试。")
                print(f"解析错误: {e}")
                print(f"错误输出为 '{response_content}' ")
                instruction["subtask"] = NULL_MARKER

        if processed_count > 0 and processed_count % SAVE_INTERVAL == 0:
            save_data(JSON_FILE, data)

    script_end_time = time.time()

    print("\n--- 简单解析完成 ---")
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
