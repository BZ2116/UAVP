"""
共享的 VLM 推理与 JSON 解析工具
"""
import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VLMAnalyzer:
    """Qwen2.5-VL 推理封装"""

    def __init__(self, model_path: str, torch_dtype="auto"):
        print(f"正在加载模型: {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def analyze(self, image_path: str, prompt: str, max_new_tokens: int = 1024,
                do_sample: bool = True, temperature: float = 0.2, top_p: float = 0.9):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p if do_sample else None,
        )
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text


def extract_json_after_assistant_codeblock(text: str):
    """
    从 VLM 输出中提取 JSON。
    查找 assistant 标记后的 ```json 代码块。
    """
    assistant_idx = text.find("assistant")
    if assistant_idx == -1:
        return None

    search_from = text[assistant_idx:]
    codeblock_start = search_from.find("```json")
    if codeblock_start == -1:
        return None

    json_text = search_from[codeblock_start + len("```json"):].strip()
    brace_start = json_text.find("{")
    if brace_start == -1:
        return None

    stack = 0
    json_str = ""
    for char in json_text[brace_start:]:
        json_str += char
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
            if stack == 0:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_from_text(text: str):
    """备用 JSON 提取：直接找第一个和最后一个花括号"""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return None


def clean_and_save_json(raw_text: str, save_path: str) -> bool:
    """解析 VLM 输出并保存为 JSON 文件，失败时保留原始输出"""
    data = extract_json_after_assistant_codeblock(raw_text)
    if data is not None:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    else:
        log_path = save_path + ".raw.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        print(f"⚠️ JSON 提取失败，已保存原始输出: {log_path}")
        return False
