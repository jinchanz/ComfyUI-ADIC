import json
import requests
from typing import List, Dict, Any
import torch
import numpy as np
from PIL import Image
import io
from urllib.parse import urlparse



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

    FUNCTION = "translate"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def translate(self, params, app_name, api_key, api_endpoint):
        try:
            # 验证必填参数
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")
            
            if not api_endpoint or api_endpoint.strip() == "":
                raise ValueError("API端点不能为空")
            
            # 解析params参数
            try:
                if isinstance(params, str):
                    request_body = json.loads(params)
                else:
                    request_body = params
                
                
            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip()
            }
            
            print(f"[ADIC_COMMON_API] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")
            
            # 发送POST请求
            response = requests.post(
                api_endpoint.strip(),
                headers=headers,
                json=request_body,
                timeout=600,
                verify=False  # 忽略SSL证书验证
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()
            
            # 检查业务状态
            if not response_data.get("success", False):
                error_msg = response_data.get("message", "API调用失败")
                print(f"[ADIC_COMMON_API] 业务错误: {error_msg}")
                return (json.dumps({
                    "error": error_msg,
                    "code": response_data.get("code", -1),
                    "debugInfo": response_data.get("debugInfo")
                }, ensure_ascii=False, indent=2),)
            
            # 返回成功结果
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
            # 验证必填参数
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")
            
            if not api_endpoint or api_endpoint.strip() == "":
                raise ValueError("API端点不能为空")
            
            # 解析params参数
            try:
                if isinstance(params, str):
                    request_body = json.loads(params)
                else:
                    request_body = params
                
            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip()
            }
            
            print(f"[ImageTranslateAPI] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")
            
            # 发送POST请求
            response = requests.post(
                api_endpoint.strip(),
                headers=headers,
                json=request_body,
                timeout=600,
                verify=False  # 忽略SSL证书验证
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()

            print(f"[ImageTranslateAPI] 响应数据: {response_data}")
            
            # 检查业务状态
            if not response_data.get("success", False):
                error_msg = response_data.get("message", "API调用失败")
                print(f"[ImageTranslateAPI] 业务错误: {error_msg}")
                return (json.dumps({
                    "error": error_msg,
                    "code": response_data.get("code", -1),
                    "debugInfo": response_data.get("debugInfo")
                }, ensure_ascii=False, indent=2),)
            
            # 返回成功结果
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("params_json",)

    FUNCTION = "build"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def build(self, url1, source_language1="", target_language1="en", 
              url2="", source_language2="", target_language2="en",
              url3="", source_language3="", target_language3="en"):
        """构建符合API要求的params JSON字符串"""
        params_array = []
        
        # 处理所有URL和对应的语言设置
        url_configs = [
            (url1, source_language1, target_language1),
            (url2, source_language2, target_language2),
            (url3, source_language3, target_language3),
        ]
        
        for url, source_lang, target_lang in url_configs:
            url = url.strip()
            if not url:
                continue
                
            # 验证URL格式
            if not (url.startswith('http://') or url.startswith('https://')):
                print(f"[ImageTranslateParamsBuilder] 警告: URL格式不正确: {url}")
                continue
            
            param = {"url": url}
            
            # 添加源语言（如果指定了）
            if source_lang and source_lang.strip():
                param["sourceLanguage"] = source_lang.strip()
            
            # 添加目标语言（如果指定了）
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_data",)

    FUNCTION = "extract"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def extract(self, api_response, extract_type="result_urls", index=0):
        """从API响应中提取特定数据"""
        try:
            # 解析API响应
            response_data = json.loads(api_response)
            
            # 检查是否有错误
            if "error" in response_data:
                return (api_response,)
            
            # 获取数据列表
            data_list = response_data.get("data", [])
            if not data_list:
                return (json.dumps([], ensure_ascii=False),)
            
            if extract_type == "all":
                # 返回完整的数据列表
                result = json.dumps(data_list, ensure_ascii=False, indent=2)
            
            elif extract_type == "result_urls":
                # 提取所有结果图片URL
                urls = [item.get("resultUrl", "") for item in data_list if item.get("resultUrl")]
                result = json.dumps(urls, ensure_ascii=False)
            
            elif extract_type == "result_jsons":
                # 提取所有结果JSON
                jsons = [item.get("resultJson", "") for item in data_list if item.get("resultJson")]
                result = json.dumps(jsons, ensure_ascii=False)
            
            elif extract_type == "source_info":
                # 提取源信息（URL和语言）
                source_info = []
                for item in data_list:
                    info = {
                        "url": item.get("url", ""),
                        "sourceLanguage": item.get("sourceLanguage", ""),
                        "targetLanguage": item.get("targetLanguage", "")
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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "loaded_urls", "count")
    
    OUTPUT_IS_LIST = (True, False, False)

    FUNCTION = "load_images"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def load_images(self, urls, input_format="auto", max_images=10, timeout=30):
        """从URL列表加载图片"""
        try:
            # 解析URL列表
            url_list = self._parse_urls(urls, input_format)
            
            if not url_list:
                print("[LoadImagesFromUrls] 警告: 没有找到有效的URL")
                # 返回空图片数据
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (empty_image, json.dumps([]), 0)
            
            # 限制图片数量
            url_list = url_list[:max_images]
            
            loaded_images = []
            loaded_urls = []
            
            print(f"[LoadImagesFromUrls] 开始加载 {len(url_list)} 张图片")
            
            for i, url in enumerate(url_list):
                try:
                    print(f"[LoadImagesFromUrls] 正在加载第 {i+1}/{len(url_list)} 张图片: {url}")
                    
                    # 验证URL格式
                    if not self._is_valid_url(url):
                        print(f"[LoadImagesFromUrls] 跳过无效URL: {url}")
                        continue
                    
                    # 下载图片
                    response = requests.get(
                        url, 
                        timeout=timeout,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        },
                        verify=False
                    )
                    response.raise_for_status()
                    
                    # 检查内容类型
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        print(f"[LoadImagesFromUrls] 跳过非图片内容: {url}, content-type: {content_type}")
                        continue
                    
                    # 将响应内容转换为PIL图片
                    image = Image.open(io.BytesIO(response.content))
                    
                    # 确保图片是RGB模式
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # 转换为numpy数组
                    image_np = np.array(image).astype(np.float32) / 255.0
                    
                    # 转换为torch张量 (H, W, C)
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
                # 返回空图片列表
                empty_image = torch.zeros((64, 64, 3), dtype=torch.float32)
                return ([empty_image], json.dumps([]), 0)
            
            # 保持原始尺寸，将每张图片作为独立的张量返回
            # 为每张图片添加batch维度，但保持原始尺寸
            output_images = []
            for i, img_tensor in enumerate(loaded_images):
                # 图片格式为 (H, W, C)，添加batch维度变为 (1, H, W, C)
                batched_img = img_tensor.unsqueeze(0)
                output_images.append(batched_img)
                print(f"  图片 {i+1}: {img_tensor.shape[1]}x{img_tensor.shape[0]} (WxH)")
            
            print(f"[LoadImagesFromUrls] 成功加载 {len(loaded_images)} 张图片，保持原始尺寸")
            
            return (output_images, json.dumps(loaded_urls, ensure_ascii=False), len(loaded_images))
            
        except Exception as e:
            error_msg = f"加载图片过程出错: {str(e)}"
            print(f"[LoadImagesFromUrls] {error_msg}")
            # 返回错误时的默认数据
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return ([empty_image], json.dumps({"error": error_msg}, ensure_ascii=False), 0)
    
    def _parse_urls(self, urls_input, input_format):
        """解析URL输入"""
        urls_input = urls_input.strip()
        if not urls_input:
            return []
        
        url_list = []
        
        if input_format == "auto":
            # 自动检测格式
            try:
                # 尝试解析为JSON
                parsed = json.loads(urls_input)
                if isinstance(parsed, list):
                    url_list = [str(url).strip() for url in parsed if str(url).strip()]
                    print("[LoadImagesFromUrls] 检测到JSON数组格式")
                else:
                    raise ValueError("JSON不是数组格式")
            except:
                # 解析为换行分隔
                url_list = [line.strip() for line in urls_input.split('\n') if line.strip()]
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
            url_list = [line.strip() for line in urls_input.split('\n') if line.strip()]
        
        # 过滤有效的URL
        valid_urls = [url for url in url_list if self._is_valid_url(url)]
        
        print(f"[LoadImagesFromUrls] 解析得到 {len(valid_urls)} 个有效URL")
        return valid_urls
    
    def _is_valid_url(self, url):
        """验证URL格式"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False


class PythonCodeExecutor:
    """Python代码执行器节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "code": ("STRING", {"multiline": True, "default": """# 在这里编写Python代码
# 可用变量：
# - input1, input2, input3: 输入数据
# - json, re: 预导入的模块
# 
# 示例：将换行分割的字符串转换为JSON数组
# lines = input1.strip().split('\\n')
# result = [line.strip() for line in lines if line.strip()]
# output = json.dumps(result, ensure_ascii=False)

# 请将最终结果赋值给 'output' 变量
output = "请在上方编写代码"
"""}),
            },
            "optional": {
                "input1": ("STRING", {"default": "", "forceInput": True}),
                "input2": ("STRING", {"default": "", "forceInput": True}),
                "input3": ("STRING", {"default": "", "forceInput": True}),
                "safe_mode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "logs")

    FUNCTION = "execute_code"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def execute_code(self, code, input1="", input2="", input3="", safe_mode=True):
        """执行Python代码"""
        import re
        import math
        import random
        from datetime import datetime, timedelta
        
        try:
            # 准备执行环境
            local_vars = {
                'input1': input1,
                'input2': input2, 
                'input3': input3,
                'json': json,
                're': re,
                'math': math,
                'random': random,
                'datetime': datetime,
                'timedelta': timedelta,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'max': max,
                'min': min,
                'sum': sum,
                'any': any,
                'all': all,
                'print': print,
                'output': None
            }
            
            # 安全模式检查
            if safe_mode:
                dangerous_keywords = [
                    'import os', 'import sys', 'import subprocess', 'import shutil',
                    '__import__', 'eval', 'exec', 'compile', 'open', 'file',
                    'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                    'getattr', 'setattr', 'delattr', 'hasattr'
                ]
                
                code_lower = code.lower()
                for keyword in dangerous_keywords:
                    if keyword in code_lower:
                        return (json.dumps({"error": f"安全模式禁止使用: {keyword}"}, ensure_ascii=False), 
                               f"[PythonCodeExecutor] 安全检查失败: 发现禁用关键词 '{keyword}'")
            
            # 捕获print输出
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            logs = []
            logs.append(f"[PythonCodeExecutor] 开始执行代码，安全模式: {'开启' if safe_mode else '关闭'}")
            logs.append(f"[PythonCodeExecutor] 输入数据:")
            logs.append(f"  input1: {repr(input1[:100])}{'...' if len(input1) > 100 else ''}")
            logs.append(f"  input2: {repr(input2[:100])}{'...' if len(input2) > 100 else ''}")
            logs.append(f"  input3: {repr(input3[:100])}{'...' if len(input3) > 100 else ''}")
            
            # 执行用户代码
            exec(code, {"__builtins__": {}}, local_vars)
            
            # 恢复stdout
            sys.stdout = old_stdout
            captured_print = captured_output.getvalue()
            
            # 获取输出结果
            result = local_vars.get('output', None)
            
            if result is None:
                logs.append("[PythonCodeExecutor] 警告: 代码未设置output变量")
                output_str = json.dumps({"warning": "代码未设置output变量"}, ensure_ascii=False)
            else:
                # 将结果转换为字符串
                if isinstance(result, str):
                    output_str = result
                else:
                    try:
                        output_str = json.dumps(result, ensure_ascii=False, indent=2)
                    except:
                        output_str = str(result)
                
                logs.append(f"[PythonCodeExecutor] 代码执行成功")
                logs.append(f"[PythonCodeExecutor] 输出类型: {type(result).__name__}")
                logs.append(f"[PythonCodeExecutor] 输出长度: {len(output_str)} 字符")
            
            # 添加print输出到日志
            if captured_print:
                logs.append("[PythonCodeExecutor] Print输出:")
                for line in captured_print.strip().split('\n'):
                    logs.append(f"  {line}")
            
            log_output = '\n'.join(logs)
            
            return (output_str, log_output)
            
        except SyntaxError as e:
            error_msg = f"语法错误: {str(e)}"
            log_output = f"[PythonCodeExecutor] {error_msg}\n在第 {e.lineno} 行: {e.text}"
            return (json.dumps({"error": error_msg}, ensure_ascii=False), log_output)
            
        except Exception as e:
            error_msg = f"执行错误: {str(e)}"
            log_output = f"[PythonCodeExecutor] {error_msg}"
            return (json.dumps({"error": error_msg}, ensure_ascii=False), log_output)
        
        finally:
            # 确保恢复stdout
            if 'old_stdout' in locals():
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_array",)

    FUNCTION = "convert"

    OUTPUT_NODE = False

    CATEGORY = "Malette"

    def convert(self, text, separator="newline", remove_empty=True, trim_whitespace=True):
        """将字符串转换为JSON数组"""
        try:
            if not text:
                return (json.dumps([], ensure_ascii=False),)
            
            # 选择分隔符
            separators = {
                "newline": "\n",
                "comma": ",",
                "semicolon": ";",
                "pipe": "|",
                "tab": "\t"
            }
            
            sep = separators.get(separator, "\n")
            
            # 分割字符串
            items = text.split(sep)
            
            # 处理选项
            if trim_whitespace:
                items = [item.strip() for item in items]
            
            if remove_empty:
                items = [item for item in items if item]
            
            # 转换为JSON
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("result", "task_id", "status")

    FUNCTION = "execute_task"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def execute_task(self, generate_params, app_name, api_key, user_id, base_url, 
                     poll_interval=5, max_poll_time=300, auto_start_polling=True):
        """执行营销图生图任务并轮询结果"""
        import time
        
        try:
            # 验证必填参数
            if not api_key or api_key.strip() == "":
                raise ValueError("API密钥不能为空")
            
            if not user_id or user_id.strip() == "":
                raise ValueError("用户ID不能为空")
            
            # 解析生成参数
            try:
                if isinstance(generate_params, str):
                    request_body = json.loads(generate_params)
                else:
                    request_body = generate_params
            except json.JSONDecodeError as e:
                raise ValueError(f"生成参数JSON解析失败: {str(e)}")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip()
            }
            
            # 步骤1: 创建营销图生图任务
            create_url = f"{base_url.rstrip('/')}/open/api/v1/ai/marketImageGenerate"
            
            print(f"[MarketImageGenerateWithPolling] 创建任务...")
            print(f"[MarketImageGenerateWithPolling] 请求URL: {create_url}")
            print(f"[MarketImageGenerateWithPolling] 请求参数: {json.dumps(request_body, ensure_ascii=False)}")
            
            # 发送创建任务请求
            response = requests.post(
                create_url,
                headers=headers,
                json=request_body,
                timeout=60,
                verify=False
            )
            
            response.raise_for_status()
            create_response = response.json()
            
            print(f"[MarketImageGenerateWithPolling] 创建任务响应: {json.dumps(create_response, ensure_ascii=False)}")
            
            # 检查创建任务是否成功
            if not create_response.get("success", False):
                error_msg = create_response.get("message", "创建任务失败")
                return (
                    json.dumps({"error": error_msg, "code": create_response.get("code", -1)}, ensure_ascii=False),
                    "",
                    "failed"
                )
            
            # 获取任务ID
            task_id = create_response.get("data")
            if not task_id:
                return (
                    json.dumps({"error": "未获取到任务ID"}, ensure_ascii=False),
                    "",
                    "failed"
                )
            
            task_id_str = str(task_id)
            print(f"[MarketImageGenerateWithPolling] 任务创建成功，任务ID: {task_id_str}")
            
            # 如果不自动轮询，直接返回任务ID
            if not auto_start_polling:
                return (
                    json.dumps({"task_id": task_id_str, "message": "任务已创建，请手动查询结果"}, ensure_ascii=False),
                    task_id_str,
                    "created"
                )
            
            # 步骤2: 轮询查询任务结果
            query_url = f"{base_url.rstrip('/')}/open/api/v1/ai/getMainTask"
            
            # 构建查询参数
            query_params = {
                "appKey": api_key.strip(),
                "appName": app_name.strip(),
                "id": task_id,
                "userId": user_id.strip()
            }
            
            print(f"[MarketImageGenerateWithPolling] 开始轮询任务结果...")
            print(f"[MarketImageGenerateWithPolling] 轮询间隔: {poll_interval}秒, 最大轮询时间: {max_poll_time}秒")
            
            start_time = time.time()
            poll_count = 0
            
            while time.time() - start_time < max_poll_time:
                poll_count += 1
                print(f"[MarketImageGenerateWithPolling] 第 {poll_count} 次查询任务状态...")
                
                try:
                    # 查询任务状态
                    query_response = requests.post(
                        query_url,
                        headers=headers,
                        json=query_params,
                        timeout=30,
                        verify=False
                    )
                    
                    query_response.raise_for_status()
                    query_data = query_response.json()
                    
                    print(f"[MarketImageGenerateWithPolling] 查询响应: {json.dumps(query_data, ensure_ascii=False)}")
                    
                    # 检查查询是否成功
                    if not query_data.get("success", False):
                        error_msg = query_data.get("message", "查询任务失败")
                        print(f"[MarketImageGenerateWithPolling] 查询失败: {error_msg}")
                        time.sleep(poll_interval)
                        continue
                    
                    # 获取任务数据
                    task_data = query_data.get("data")
                    if not task_data:
                        print("[MarketImageGenerateWithPolling] 未获取到任务数据")
                        time.sleep(poll_interval)
                        continue
                    
                    # 检查任务状态
                    task_status = task_data.get("status", "").upper()
                    print(f"[MarketImageGenerateWithPolling] 任务状态: {task_status}")
                    
                    # 任务完成状态判断
                    if task_status in ["SUCCESS", "COMPLETED", "FINISHED"]:
                        print(f"[MarketImageGenerateWithPolling] 任务完成！耗时 {time.time() - start_time:.1f} 秒")
                        result_json = json.dumps(query_data, ensure_ascii=False, indent=2)
                        return (result_json, task_id_str, "completed")
                    
                    elif task_status in ["FAILED", "ERROR", "CANCELLED"]:
                        print(f"[MarketImageGenerateWithPolling] 任务失败，状态: {task_status}")
                        error_result = {
                            "error": f"任务失败，状态: {task_status}",
                            "task_data": task_data
                        }
                        return (json.dumps(error_result, ensure_ascii=False, indent=2), task_id_str, "failed")
                    
                    else:
                        # 任务仍在进行中
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
            
            # 轮询超时
            timeout_result = {
                "error": f"轮询超时（{max_poll_time}秒），任务可能仍在进行中",
                "task_id": task_id_str,
                "poll_count": poll_count,
                "elapsed_time": time.time() - start_time
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
        
class ImageStitch:
    """Upstreamed from https://github.com/kijai/ComfyUI-KJNodes"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "step": 2},
                ),
                "spacing_color": (
                    ["white", "black", "red", "green", "blue"],
                    {"default": "white"},
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "image/transform"
    DESCRIPTION = """
Stitches image2 to image1 in the specified direction.
If image2 is not provided, returns image1 unchanged.
Optional spacing can be added between images.
"""

    def stitch(
        self,
        image1,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        image2=None,
    ):
        if image2 is None:
            return (image1,)

        # Handle batch size differences
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < max_batch:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
                )

        # Match image sizes if requested
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2

            if direction in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:  # up, down
                target_w, target_h = w1, int(w1 / aspect_ratio)

            image2 = comfy.utils.common_upscale(
                image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
            ).movedim(1, -1)

        color_map = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        color_val = color_map[spacing_color]

        # When not matching sizes, pad to align non-concat dimensions
        if not match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            pad_value = 0.0
            if not isinstance(color_val, tuple):
                pad_value = color_val

            if direction in ["left", "right"]:
                # For horizontal concat, pad heights to match
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        pad_h = target_h - h1
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
            else:  # up, down
                # For vertical concat, pad widths to match
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)

        # Ensure same number of channels
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1],
                            max_channels - image1.shape[-1],
                            device=image1.device,
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < max_channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1],
                            max_channels - image2.shape[-1],
                            device=image2.device,
                        ),
                    ],
                    dim=-1,
                )

        # Add spacing if specified
        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)  # Ensure even

            if direction in ["left", "right"]:
                spacing_shape = (
                    image1.shape[0],
                    max(image1.shape[1], image2.shape[1]),
                    spacing_width,
                    image1.shape[-1],
                )
            else:
                spacing_shape = (
                    image1.shape[0],
                    spacing_width,
                    max(image1.shape[2], image2.shape[2]),
                    image1.shape[-1],
                )

            spacing = torch.full(spacing_shape, 0.0, device=image1.device)
            if isinstance(color_val, tuple):
                for i, c in enumerate(color_val):
                    if i < spacing.shape[-1]:
                        spacing[..., i] = c
                if spacing.shape[-1] == 4:  # Add alpha
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color_val
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0

        # Concatenate images
        images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        return (torch.cat(images, dim=concat_dim),)
    
def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c


class ReferenceLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                            },
                "optional": {"latent": ("LATENT", ),}
               }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/edit_models"
    DESCRIPTION = "This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images."

    def append(self, conditioning, latent=None):
        if latent is not None:
            conditioning = conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        return (conditioning, )


NODE_CLASS_MAPPINGS = {
    "ReferenceLatent": ReferenceLatent,
}



# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageTranslateAPI": ImageTranslateAPI,
    "ImageTranslateParamsBuilder": ImageTranslateParamsBuilder,
    "ImageTranslateResultExtractor": ImageTranslateResultExtractor,
    "ADIC_COMMON_API": ADIC_COMMON_API,
    "LoadImagesFromUrls": LoadImagesFromUrls,
    "PythonCodeExecutor": PythonCodeExecutor,
    "StringToJsonArray": StringToJsonArray,
    "MarketImageGenerateWithPolling": MarketImageGenerateWithPolling,
    "ImageStitch": ImageStitch,
    "ReferenceLatent": ReferenceLatent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTranslateAPI": "图片翻译 API",
    "ImageTranslateParamsBuilder": "图片翻译参数构建器",
    "ImageTranslateResultExtractor": "图片翻译结果提取器",
    "ADIC_COMMON_API": "ADIC Common API",
    "LoadImagesFromUrls": "从URL列表加载图片",
    "PythonCodeExecutor": "Python代码执行器",
    "StringToJsonArray": "字符串转JSON数组",
    "MarketImageGenerateWithPolling": "营销图生图任务（带轮询）",
    "ImageStitch": "图片拼接",
    "ReferenceLatent": "参考潜变量"
} 