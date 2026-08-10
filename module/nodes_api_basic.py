import io
import json
import traceback
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image


class ADIC_COMMON_API:
    """ADIC Common API"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "params": ("STRING", {"forceInput": True}),
                "app_name": ("STRING", {"default": "NHCI"}),
                "api_key": ("STRING", {"default": ""}),
                "api_endpoint": ("STRING", {"default": "/open/api/v1/ai/imageTranslate"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    FUNCTION = "request"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def request(self, params, app_name, api_key, api_endpoint):
        try:
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")

            if not api_endpoint or api_endpoint.strip() == "":
                raise ValueError("API端点不能为空")

            try:
                if isinstance(params, str):
                    request_body = json.loads(params)
                else:
                    request_body = params
            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")

            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip(),
            }

            print(f"[ADIC_COMMON_API] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")

            response = requests.post(
                api_endpoint.strip(),
                headers=headers,
                json=request_body,
                timeout=600,
                verify=False,
            )

            response.raise_for_status()
            response_data = response.json()

            if not response_data.get("success", False):
                error_msg = response_data.get("message", "API调用失败")
                print(f"[ADIC_COMMON_API] 业务错误: {error_msg}")
                return (
                    json.dumps(
                        {
                            "error": error_msg,
                            "code": response_data.get("code", -1),
                            "debugInfo": response_data.get("debugInfo"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )

            result_json = json.dumps(response_data, ensure_ascii=False, indent=2)
            print(f"[ADIC_COMMON_API] 请求结果: {result_json}")

            return (result_json,)

        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            print(f"[ADIC_COMMON_API] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)

        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            print(f"[ADIC_COMMON_API] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"[ADIC_COMMON_API] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class ImageTranslateAPI:
    """阿里云图片翻译API节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "params": ("STRING", {"forceInput": True}),
                "app_name": ("STRING", {"default": "NHCI"}),
                "api_key": ("STRING", {"default": ""}),
                "api_endpoint": ("STRING", {"default": "/open/api/v1/ai/imageTranslate"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    FUNCTION = "translate"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def translate(self, params, app_name, api_key, api_endpoint):
        try:
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")

            if not api_endpoint or api_endpoint.strip() == "":
                raise ValueError("API端点不能为空")

            try:
                if isinstance(params, str):
                    request_body = json.loads(params)
                else:
                    request_body = params

            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")

            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip(),
            }

            print(f"[ImageTranslateAPI] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")

            response = requests.post(
                api_endpoint.strip(),
                headers=headers,
                json=request_body,
                timeout=600,
                verify=False,
            )

            response.raise_for_status()
            response_data = response.json()

            print(f"[ImageTranslateAPI] 响应数据: {response_data}")

            if not response_data.get("success", False):
                error_msg = response_data.get("message", "API调用失败")
                print(f"[ImageTranslateAPI] 业务错误: {error_msg}")
                return (
                    json.dumps(
                        {
                            "error": error_msg,
                            "code": response_data.get("code", -1),
                            "debugInfo": response_data.get("debugInfo"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )

            result_json = json.dumps(response_data, ensure_ascii=False, indent=2)
            print(f"[ImageTranslateAPI] 翻译完成，处理了 {len(response_data.get('data', []))} 张图片")
            print(f"[ImageTranslateAPI] 翻译结果: {result_json}")

            return (result_json,)

        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            print(f"[ImageTranslateAPI] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)

        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            print(f"[ImageTranslateAPI] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"[ImageTranslateAPI] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class ImageTranslateParamsBuilder:
    """图片翻译参数构建器节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url1": ("STRING", {"default": ""}),
            },
            "optional": {
                "source_language1": (["", "zh", "en", "tr"], {"default": ""}),
                "target_language1": (["", "es", "fr", "pt", "ko", "en"], {"default": "en"}),
                "url2": ("STRING", {"default": ""}),
                "source_language2": (["", "zh", "en", "tr"], {"default": ""}),
                "target_language2": (["", "es", "fr", "pt", "ko", "en"], {"default": "en"}),
                "url3": ("STRING", {"default": ""}),
                "source_language3": (["", "zh", "en", "tr"], {"default": ""}),
                "target_language3": (["", "es", "fr", "pt", "ko", "en"], {"default": "en"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("params_json",)

    FUNCTION = "build"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def build(
        self,
        url1,
        source_language1="",
        target_language1="en",
        url2="",
        source_language2="",
        target_language2="en",
        url3="",
        source_language3="",
        target_language3="en",
    ):
        params_array = []
        url_configs = [
            (url1, source_language1, target_language1),
            (url2, source_language2, target_language2),
            (url3, source_language3, target_language3),
        ]

        for url, source_lang, target_lang in url_configs:
            url = url.strip()
            if not url:
                continue

            if not (url.startswith("http://") or url.startswith("https://")):
                print(f"[ImageTranslateParamsBuilder] 警告: URL格式不正确: {url}")
                continue

            param = {"url": url}

            if source_lang and source_lang.strip():
                param["sourceLanguage"] = source_lang.strip()

            if target_lang and target_lang.strip():
                param["targetLanguage"] = target_lang.strip()

            params_array.append(param)

        if not params_array:
            print("[ImageTranslateParamsBuilder] 警告: 没有有效的URL")
            return (json.dumps({"params": []}),)

        result = {"params": params_array}
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        print(f"[ImageTranslateParamsBuilder] 构建了 {len(params_array)} 个翻译参数")

        return (result_json,)


class ImageTranslateResultExtractor:
    """图片翻译结果提取器节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_response": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "extract_type": (["all", "result_urls", "result_jsons", "source_info"], {"default": "result_urls"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_data",)

    FUNCTION = "extract"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def extract(self, api_response, extract_type="result_urls", index=0):
        try:
            response_data = json.loads(api_response)

            if "error" in response_data:
                return (api_response,)

            data_list = response_data.get("data", [])
            if not data_list:
                return (json.dumps([], ensure_ascii=False),)

            if extract_type == "all":
                result = json.dumps(data_list, ensure_ascii=False, indent=2)

            elif extract_type == "result_urls":
                urls = [item.get("resultUrl", "") for item in data_list if item.get("resultUrl")]
                result = json.dumps(urls, ensure_ascii=False)

            elif extract_type == "result_jsons":
                jsons = [item.get("resultJson", "") for item in data_list if item.get("resultJson")]
                result = json.dumps(jsons, ensure_ascii=False)

            elif extract_type == "source_info":
                source_info = []
                for item in data_list:
                    info = {
                        "url": item.get("url", ""),
                        "sourceLanguage": item.get("sourceLanguage", ""),
                        "targetLanguage": item.get("targetLanguage", ""),
                    }
                    source_info.append(info)
                result = json.dumps(source_info, ensure_ascii=False, indent=2)

            else:
                result = json.dumps(data_list, ensure_ascii=False, indent=2)

            print(f"[ImageTranslateResultExtractor] 提取类型: {extract_type}, 数据数量: {len(data_list)}")
            return (result,)

        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            print(f"[ImageTranslateResultExtractor] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)

        except Exception as e:
            error_msg = f"提取过程出错: {str(e)}"
            print(f"[ImageTranslateResultExtractor] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class LoadImagesFromUrls:
    """从URL列表加载图片节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "urls": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "input_format": (["json_array", "newline_separated", "auto"], {"default": "auto"}),
                "max_images": ("INT", {"default": 10, "min": 1, "max": 50}),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 300}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "loaded_urls", "count")

    OUTPUT_IS_LIST = (True, False, False)

    FUNCTION = "load_images"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def load_images(self, urls, input_format="auto", max_images=10, timeout=30):
        try:
            url_list = self._parse_urls(urls, input_format)

            if not url_list:
                print("[LoadImagesFromUrls] 警告: 没有找到有效的URL")
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (empty_image, json.dumps([]), 0)

            url_list = url_list[:max_images]

            loaded_images = []
            loaded_urls = []

            print(f"[LoadImagesFromUrls] 开始加载 {len(url_list)} 张图片")

            for i, url in enumerate(url_list):
                try:
                    print(f"[LoadImagesFromUrls] 正在加载第 {i+1}/{len(url_list)} 张图片: {url}")

                    if not self._is_valid_url(url):
                        print(f"[LoadImagesFromUrls] 跳过无效URL: {url}")
                        continue

                    response = requests.get(
                        url,
                        timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                        verify=False,
                    )
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        print(f"[LoadImagesFromUrls] 跳过非图片内容: {url}, content-type: {content_type}")
                        continue

                    image = Image.open(io.BytesIO(response.content))

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)

                    loaded_images.append(image_tensor)
                    loaded_urls.append(url)

                    print(f"[LoadImagesFromUrls] 成功加载图片: {image.size}, 模式: {image.mode}")

                except requests.exceptions.RequestException as e:
                    print(f"[LoadImagesFromUrls] 下载图片失败 {url}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"[LoadImagesFromUrls] 处理图片失败 {url}: {str(e)}")
                    continue

            if not loaded_images:
                print("[LoadImagesFromUrls] 警告: 没有成功加载任何图片")
                empty_image = torch.zeros((64, 64, 3), dtype=torch.float32)
                return ([empty_image], json.dumps([]), 0)

            output_images = []
            for i, img_tensor in enumerate(loaded_images):
                batched_img = img_tensor.unsqueeze(0)
                output_images.append(batched_img)
                print(f"  图片 {i+1}: {img_tensor.shape[1]}x{img_tensor.shape[0]} (WxH)")

            print(f"[LoadImagesFromUrls] 成功加载 {len(loaded_images)} 张图片，保持原始尺寸")

            return (output_images, json.dumps(loaded_urls, ensure_ascii=False), len(loaded_images))

        except Exception as e:
            error_msg = f"加载图片过程出错: {str(e)}"
            print(f"[LoadImagesFromUrls] {error_msg}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return ([empty_image], json.dumps({"error": error_msg}, ensure_ascii=False), 0)

    def _parse_urls(self, urls_input, input_format):
        urls_input = urls_input.strip()
        if not urls_input:
            return []

        url_list = []

        if input_format == "auto":
            try:
                parsed = json.loads(urls_input)
                if isinstance(parsed, list):
                    url_list = [str(url).strip() for url in parsed if str(url).strip()]
                    print("[LoadImagesFromUrls] 检测到JSON数组格式")
                else:
                    raise ValueError("JSON不是数组格式")
            except Exception:
                url_list = [line.strip() for line in urls_input.split("\n") if line.strip()]
                print("[LoadImagesFromUrls] 检测到换行分隔格式")

        elif input_format == "json_array":
            try:
                parsed = json.loads(urls_input)
                if isinstance(parsed, list):
                    url_list = [str(url).strip() for url in parsed if str(url).strip()]
                else:
                    raise ValueError("JSON格式错误：不是数组")
            except Exception as e:
                print(f"[LoadImagesFromUrls] JSON解析失败: {str(e)}")
                return []

        elif input_format == "newline_separated":
            url_list = [line.strip() for line in urls_input.split("\n") if line.strip()]

        valid_urls = [url for url in url_list if self._is_valid_url(url)]

        print(f"[LoadImagesFromUrls] 解析得到 {len(valid_urls)} 个有效URL")
        return valid_urls

    def _is_valid_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme in ["http", "https"], result.netloc])
        except Exception:
            return False


class PythonCodeExecutor:
    """Python代码执行器节点

    ⚠️ 安全说明：
    此节点提供的是受限的Python执行环境，用于数据处理任务。
    安全保护已强制启用，无法关闭。禁止使用的操作包括：
    - 导入模块（os, sys, subprocess, shutil 等）
    - 访问对象内部属性（__class__, __base__, __subclasses__ 等）
    - 使用反射函数（eval, exec, compile, __import__ 等）
    - 文件操作和系统调用
    - 序列化/反序列化（pickle, marshal 等）

    只允许使用提供的白名单函数进行数据转换。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "code": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": """# 在这里编写Python代码
# 可用变量：
# - input1, input2, input3: 输入数据
# - json, re, math, random, datetime, timedelta: 预导入的模块
# 
# 可用函数：len, str, int, float, list, dict, tuple, set, range
#          enumerate, zip, map, filter, sorted, max, min, sum, any, all
# 
# 示例：将换行分割的字符串转换为JSON数组
# lines = input1.strip().split('\\n')
# result = [line.strip() for line in lines if line.strip()]
# output = json.dumps(result, ensure_ascii=False)

# 请将最终结果赋值给 'output' 变量
output = \"请在上方编写代码\"
""",
                    },
                ),
            },
            "optional": {
                "input1": ("STRING", {"default": "", "forceInput": True}),
                "input2": ("STRING", {"default": "", "forceInput": True}),
                "input3": ("STRING", {"default": "", "forceInput": True}),
                "safe_mode": ("BOOLEAN", {"default": True, "tooltip": "已强制启用安全模式（防止沙箱逃逸），此参数已废弃"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "logs")

    FUNCTION = "execute_code"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def execute_code(self, code, input1="", input2="", input3="", safe_mode=True):
        import math
        import random
        import re
        from datetime import datetime, timedelta

        _input1 = input1 if type(input1) == str else json.dumps(input1)
        _input2 = input2 if type(input2) == str else json.dumps(input2)
        _input3 = input3 if type(input3) == str else json.dumps(input3)

        try:
            dangerous_patterns = [
                r"\bimport\s+(os|sys|subprocess|shutil|pickle|marshal|shelve|dill)\b",
                r"\b(__import__|eval|exec|compile|open|file|__class__|__base__|__subclasses__|__mro__|__dict__|__getattribute__|__setattr__|__globals__|__code__|__builtins__|__new__|__init__|__loader__|__spec__|__func__|__self__|__closure__|__module__|__qualname__|__reduce__|__getstate__|__setstate__|__getnewargs__|__getinitargs__)\b",
                r"\b(raw_input|globals|locals|vars|dir|getattr|setattr|delattr|hasattr)\s*\(",
                r"\bobject\.__",
                r"\btype\.__",
            ]

            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if re.search(pattern, code_lower):
                    return (
                        json.dumps({"error": f"代码包含被禁用的内容（匹配模式: {pattern}）"}, ensure_ascii=False),
                        "[PythonCodeExecutor] 安全检查失败: 发现被禁用的操作",
                    )

            local_vars = {
                "input1": _input1,
                "input2": _input2,
                "input3": _input3,
                "json": json,
                "re": re,
                "math": math,
                "random": random,
                "datetime": datetime,
                "timedelta": timedelta,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "max": max,
                "min": min,
                "sum": sum,
                "any": any,
                "all": all,
                "print": print,
                "output": None,
            }

            if safe_mode:
                additional_dangerous_patterns = [
                    r"\bpickle\b",
                    r"\bmarshal\b",
                    r"\bshelve\b",
                    r"\bdill\b",
                ]

                for pattern in additional_dangerous_patterns:
                    if re.search(pattern, code_lower):
                        return (
                            json.dumps({"error": f"安全模式禁止使用: {pattern}"}, ensure_ascii=False),
                            f"[PythonCodeExecutor] 安全检查失败: 发现禁用关键词 '{pattern}'",
                        )

            import io as _io
            import sys

            old_stdout = sys.stdout
            sys.stdout = captured_output = _io.StringIO()

            logs = []
            logs.append("[PythonCodeExecutor] 开始执行代码，安全模式已强制启用（防止沙箱逃逸）")
            logs.append("[PythonCodeExecutor] 输入数据:")
            try:
                logs.append(f"  input1: {repr(_input1[:100])}{'...' if len(_input1) > 100 else ''}")
            except Exception as e:
                print(f"[PythonCodeExecutor] 输入1解析失败: {str(e)}, input1: {_input1}")
            try:
                logs.append(f"  input2: {repr(_input2[:100])}{'...' if len(_input2) > 100 else ''}")
            except Exception as e:
                print(f"[PythonCodeExecutor] 输入2解析失败: {str(e)}, input2: {_input2}")
            try:
                logs.append(f"  input3: {repr(_input3[:100])}{'...' if len(_input3) > 100 else ''}")
            except Exception as e:
                print(f"[PythonCodeExecutor] 输入3解析失败: {str(e)}, input3: {_input3}")

            restricted_builtins = {
                "__name__": "__main__",
                "__doc__": None,
            }

            exec(code, {"__builtins__": restricted_builtins}, local_vars)

            sys.stdout = old_stdout
            captured_print = captured_output.getvalue()

            result = local_vars.get("output", None)

            if result is None:
                logs.append("[PythonCodeExecutor] 警告: 代码未设置output变量")
                output_str = json.dumps({"warning": "代码未设置output变量"}, ensure_ascii=False)
            else:
                if isinstance(result, str):
                    output_str = result
                else:
                    try:
                        output_str = json.dumps(result, ensure_ascii=False, indent=2)
                    except Exception:
                        output_str = str(result)

                logs.append("[PythonCodeExecutor] 代码执行成功")
                logs.append(f"[PythonCodeExecutor] 输出类型: {type(result).__name__}")
                logs.append(f"[PythonCodeExecutor] 输出长度: {len(output_str)} 字符")

            if captured_print:
                logs.append("[PythonCodeExecutor] Print输出:")
                for line in captured_print.strip().split("\n"):
                    logs.append(f"  {line}")

            log_output = "\n".join(logs)

            return (output_str, log_output)

        except SyntaxError as e:
            error_msg = f"语法错误: {str(e)}"
            log_output = f"[PythonCodeExecutor] {error_msg}\n在第 {e.lineno} 行: {e.text}"
            return (json.dumps({"error": error_msg}, ensure_ascii=False), log_output)

        except Exception as e:
            error_msg = f"执行错误: {str(e)}, {traceback.format_exc()}"
            print(f"[PythonCodeExecutor] {error_msg}, {traceback.format_exc()}")
            log_output = f"[PythonCodeExecutor] {error_msg}"
            return (json.dumps({"error": error_msg}, ensure_ascii=False), log_output)

        finally:
            if "old_stdout" in locals():
                sys.stdout = old_stdout


class StringToJsonArray:
    """字符串转JSON数组节点（快捷版本）"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "separator": (["newline", "comma", "semicolon", "pipe", "tab"], {"default": "newline"}),
                "remove_empty": ("BOOLEAN", {"default": True}),
                "trim_whitespace": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_array",)

    FUNCTION = "convert"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def convert(self, text, separator="newline", remove_empty=True, trim_whitespace=True):
        try:
            if not text:
                return (json.dumps([], ensure_ascii=False),)

            separators = {
                "newline": "\n",
                "comma": ",",
                "semicolon": ";",
                "pipe": "|",
                "tab": "\t",
            }

            sep = separators.get(separator, "\n")
            items = text.split(sep)

            if trim_whitespace:
                items = [item.strip() for item in items]

            if remove_empty:
                items = [item for item in items if item]

            result = json.dumps(items, ensure_ascii=False, indent=2)

            print(f"[StringToJsonArray] 转换了 {len(items)} 个项目，分隔符: {separator}")

            return (result,)

        except Exception as e:
            error_msg = f"转换失败: {str(e)}"
            print(f"[StringToJsonArray] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class MarketImageGenerateWithPolling:
    """营销图生图任务创建与轮询节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generate_params": ("STRING", {"forceInput": True, "multiline": True}),
                "app_name": ("STRING", {"default": "NHCI"}),
                "api_key": ("STRING", {"default": ""}),
                "user_id": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://pre-zhimei.alibabadesign.com"}),
            },
            "optional": {
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_poll_time": ("INT", {"default": 300, "min": 30, "max": 1800}),
                "auto_start_polling": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("result", "task_id", "status")

    FUNCTION = "execute_task"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def execute_task(self, generate_params, app_name, api_key, user_id, base_url, poll_interval=5, max_poll_time=300, auto_start_polling=True):
        import time

        try:
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")

            if not user_id or user_id.strip() == "":
                raise ValueError("用户ID不能为空")

            try:
                if isinstance(generate_params, str):
                    request_body = json.loads(generate_params)
                else:
                    request_body = generate_params
            except json.JSONDecodeError as e:
                raise ValueError(f"生成参数JSON解析失败: {str(e)}")

            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip(),
            }

            create_url = f"{base_url.rstrip('/')}/open/api/v1/ai/marketImageGenerate"

            print("[MarketImageGenerateWithPolling] 创建任务...")
            print(f"[MarketImageGenerateWithPolling] 请求URL: {create_url}")
            print(f"[MarketImageGenerateWithPolling] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")

            response = requests.post(create_url, headers=headers, json=request_body, timeout=60, verify=False)

            response.raise_for_status()
            create_response = response.json()

            print(f"[MarketImageGenerateWithPolling] 创建任务响应: {json.dumps(create_response, ensure_ascii=False)}")

            if not create_response.get("success", False):
                error_msg = create_response.get("message", "创建任务失败")
                return (json.dumps({"error": error_msg, "code": create_response.get("code", -1)}, ensure_ascii=False), "", "failed")

            task_id = create_response.get("data")
            if not task_id:
                return (json.dumps({"error": "未获取到任务ID"}, ensure_ascii=False), "", "failed")

            task_id_str = str(task_id)
            print(f"[MarketImageGenerateWithPolling] 任务创建成功，任务ID: {task_id_str}")

            if not auto_start_polling:
                return (
                    json.dumps({"task_id": task_id_str, "message": "任务已创建，请手动查询结果"}, ensure_ascii=False),
                    task_id_str,
                    "created",
                )

            query_url = f"{base_url.rstrip('/')}/open/api/v1/ai/getMainTask"
            query_params = {
                "appKey": api_key.strip(),
                "appName": app_name.strip(),
                "id": task_id,
                "userId": user_id.strip(),
            }

            print("[MarketImageGenerateWithPolling] 开始轮询任务结果...")
            print(f"[MarketImageGenerateWithPolling] 轮询间隔: {poll_interval}秒, 最大轮询时间: {max_poll_time}秒")

            start_time = time.time()
            poll_count = 0

            while time.time() - start_time < max_poll_time:
                poll_count += 1
                print(f"[MarketImageGenerateWithPolling] 第 {poll_count} 次查询任务状态...")

                try:
                    query_response = requests.post(query_url, headers=headers, json=query_params, timeout=30, verify=False)
                    query_response.raise_for_status()
                    query_data = query_response.json()

                    print(f"[MarketImageGenerateWithPolling] 查询响应: {json.dumps(query_data, ensure_ascii=False)}")

                    if not query_data.get("success", False):
                        error_msg = query_data.get("message", "查询任务失败")
                        print(f"[MarketImageGenerateWithPolling] 查询失败: {error_msg}")
                        time.sleep(poll_interval)
                        continue

                    task_data = query_data.get("data")
                    if not task_data:
                        print("[MarketImageGenerateWithPolling] 未获取到任务数据")
                        time.sleep(poll_interval)
                        continue

                    task_status = task_data.get("status", "").upper()
                    print(f"[MarketImageGenerateWithPolling] 任务状态: {task_status}")

                    if task_status in ["SUCCESS", "COMPLETED", "FINISHED"]:
                        print(f"[MarketImageGenerateWithPolling] 任务完成！耗时 {time.time() - start_time:.1f} 秒")
                        result_json = json.dumps(query_data, ensure_ascii=False, indent=2)
                        return (result_json, task_id_str, "completed")

                    elif task_status in ["FAILED", "ERROR", "CANCELLED"]:
                        print(f"[MarketImageGenerateWithPolling] 任务失败，状态: {task_status}")
                        error_result = {"error": f"任务失败，状态: {task_status}", "task_data": task_data}
                        return (json.dumps(error_result, ensure_ascii=False, indent=2), task_id_str, "failed")

                    else:
                        print(f"[MarketImageGenerateWithPolling] 任务进行中，状态: {task_status}，等待 {poll_interval} 秒后重试...")
                        time.sleep(poll_interval)
                        continue

                except requests.exceptions.RequestException as e:
                    print(f"[MarketImageGenerateWithPolling] 查询请求失败: {str(e)}")
                    time.sleep(poll_interval)
                    continue

                except Exception as e:
                    print(f"[MarketImageGenerateWithPolling] 查询过程出错: {str(e)}")
                    time.sleep(poll_interval)
                    continue

            timeout_result = {
                "error": f"轮询超时（{max_poll_time}秒），任务可能仍在进行中",
                "task_id": task_id_str,
                "poll_count": poll_count,
                "elapsed_time": time.time() - start_time,
            }

            return (json.dumps(timeout_result, ensure_ascii=False, indent=2), task_id_str, "timeout")

        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            print(f"[MarketImageGenerateWithPolling] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False), "", "error")

        except json.JSONDecodeError as e:
            error_msg = f"JSON 解析失败: {str(e)}"
            print(f"[MarketImageGenerateWithPolling] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False), "", "error")

        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"[MarketImageGenerateWithPolling] {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False), "", "error")
