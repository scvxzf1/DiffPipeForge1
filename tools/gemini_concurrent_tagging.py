import os
import base64
import requests
import glob
import time
import json
import re
import threading
import queue
import timeit
import argparse

# 默认配置
DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# 默认提示词
DEFAULT_PROMPT = """
请作为专业动漫图像打标器，基于输入的动漫风格图像，生成简明扼要的中文自然语言打标文件，需覆盖动漫风格类型、角色特征、场景元素、艺术细节（画风如手绘 / 像素 / 3D 渲染、色彩基调如明亮 / 暗黑 / 柔和、笔触风格、画质分辨率）、额外关键信息（是否为原创角色、是否有特效元素如魔法 / 光影 / 烟雾），所有打标文件用中文逗号分隔，确保打标文件精准贴合图像内容，不遗漏核心特征，用一段自然语言来描述，而非标签，不要分段。
"""

def encode_image(image_path):
    """读取并对图片进行 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def worker(api_key, image_queue, worker_id, model_name, prompt_text, sleep_duration, api_type, base_url):
    print(f"[线程 {worker_id}] 启动，使用 {api_type} 模式")
    
    headers = {"Content-Type": "application/json"}
    if api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
        if base_url:
            if "/chat/completions" not in base_url:
                api_url = base_url.rstrip("/") + "/chat/completions"
            else:
                api_url = base_url
        else:
            api_url = "https://api.openai.com/v1/chat/completions"
    else:
        api_url = GEMINI_URL_TEMPLATE.format(model=model_name, key=api_key)

    while not image_queue.empty():
        try:
            image_path = image_queue.get_nowait()
        except queue.Empty:
            break

        base_name = os.path.splitext(image_path)[0]
        txt_path = f"{base_name}.txt"

        if os.path.exists(txt_path):
            print(f"[线程 {worker_id}] 跳过 {image_path} (标签文件已存在)")
            image_queue.task_done()
            continue

        print(f"[线程 {worker_id}] 开始处理 {image_path}...")
        start_time = timeit.default_timer()
        
        should_mark_done = True
        should_retry = False

        try:
            base64_image = encode_image(image_path)
            mime_type = "image/png"
            ext = image_path.lower()
            if ext.endswith(".jpg") or ext.endswith(".jpeg"):
                mime_type = "image/jpeg"
            elif ext.endswith(".webp"):
                mime_type = "image/webp"

            if api_type == "openai":
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2048
                }
            else:
                # Gemini format
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": prompt_text},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "topK": 32,
                        "topP": 1,
                        "maxOutputTokens": 2048,
                    }
                }

            max_retries = 3
            response = None
            for retry in range(max_retries):
                try:
                    response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 3
                        print(f"[线程 {worker_id}] 连接错误或超时，{wait_time}秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            if response and response.status_code == 200:
                result = response.json()
                
                if api_type == "openai":
                    if "choices" in result and len(result["choices"]) > 0:
                        tags = result["choices"][0]["message"]["content"].strip()
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(tags)
                        elapsed = timeit.default_timer() - start_time
                        print(f"[线程 {worker_id}] 成功处理 {image_path}，耗时: {elapsed:.2f} 秒")
                    else:
                        print(f"[线程 {worker_id}] OpenAI 响应中无内容: {result}")
                else:
                    # Gemini parsing
                    if "promptFeedback" in result and "blockReason" in result["promptFeedback"]:
                        print(f"[线程 {worker_id}] 内容被阻止 {image_path}: {result['promptFeedback']['blockReason']}")
                        continue
                    
                    if "candidates" not in result or len(result["candidates"]) == 0:
                        print(f"[线程 {worker_id}] 无候选响应 {image_path}")
                        continue
                    
                    candidate = result["candidates"][0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    
                    if parts:
                        tags = parts[0]["text"].strip()
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(tags)
                        elapsed = timeit.default_timer() - start_time
                        print(f"[线程 {worker_id}] 成功处理 {image_path}，耗时: {elapsed:.2f} 秒")
                    else:
                        print(f"[线程 {worker_id}] Gemini 响应格式异常 {image_path}: {result}")

            elif response and response.status_code == 429:
                print(f"[线程 {worker_id}] 触发速率限制 (429) {image_path}。")
                should_retry = True
            else:
                s_code = response.status_code if response else "No Response"
                print(f"[线程 {worker_id}] API 错误 {image_path}: {s_code}")
                if response:
                    print(response.text[:200])

        except Exception as e:
            print(f"[线程 {worker_id}] 处理异常 {image_path}: {e}")
        
        finally:
            if should_mark_done:
                image_queue.task_done()
            
            if should_retry:
                image_queue.put(image_path)
                time.sleep(sleep_duration * 2)
            else:
                time.sleep(sleep_duration)

    print(f"[线程 {worker_id}] 任务结束。")

def main():
    parser = argparse.ArgumentParser(description="Gemini/OpenAI 并发图片打标")
    parser.add_argument("--dir", type=str, default=".", help="图片文件夹路径")
    parser.add_argument("--api_keys", type=str, required=True, help="API Keys，用逗号分隔")
    parser.add_argument("--api_type", type=str, default="gemini", choices=["gemini", "openai"], help="接口类型")
    parser.add_argument("--base_url", type=str, default="", help="OpenAI 兼容接口的 Base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="模型名称")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="提示词")
    parser.add_argument("--threads", type=int, default=4, help="每个 Key 的线程数")
    parser.add_argument("--sleep", type=float, default=1.0, help="每次请求后的休眠时间(秒)")
    
    args = parser.parse_args()
    
    # 解析 API Keys
    api_keys = [k.strip() for k in args.api_keys.split(',') if k.strip()]
    if not api_keys:
        print("错误: 未提供有效的 API Key")
        return

    directory = os.path.abspath(args.dir)
    print(f"扫描目录: {directory}")
    print(f"API 类型: {args.api_type}")
    print(f"进程模型: {args.model}")
    
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    image_files.sort(key=lambda x: natural_keys(os.path.basename(x)))
    
    print(f"总共发现 {len(image_files)} 张图片。")

    img_queue = queue.Queue()
    count_skipped = 0
    
    for img_path in image_files:
        base_name = os.path.splitext(img_path)[0]
        txt_path = f"{base_name}.txt"
        if not os.path.exists(txt_path):
            img_queue.put(img_path)
        else:
            count_skipped += 1
            
    print(f"跳过 {count_skipped} 张已处理图片，剩余 {img_queue.qsize()} 张待处理。")

    if img_queue.empty():
        print("所有图片都已处理完毕！")
        return

    threads = []
    thread_id = 1
    
    total_threads = len(api_keys) * args.threads
    print(f"\n准备启动 {total_threads} 个线程 ({len(api_keys)} keys * {args.threads} threads)...")
    
    for _, api_key in enumerate(api_keys):
        for _ in range(args.threads):
            t = threading.Thread(target=worker, args=(api_key, img_queue, thread_id, args.model, args.prompt, args.sleep, args.api_type, args.base_url))
            t.start()
            threads.append(t)
            thread_id += 1
            if thread_id % 5 == 0:
                time.sleep(0.1)
    
    for t in threads:
        t.join()

    print("所有任务完成。")

if __name__ == "__main__":
    main()
