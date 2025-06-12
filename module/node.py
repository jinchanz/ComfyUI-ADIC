import json
import requests
from typing import List, Dict, Any



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
                
                # 验证params格式
                if not isinstance(request_body, dict) or "params" not in request_body:
                    raise ValueError("params参数格式错误，应该包含params数组")
                
                params_array = request_body["params"]
                if not isinstance(params_array, list):
                    raise ValueError("params.params 应该是一个数组")
                
                if len(params_array) == 0:
                    raise ValueError("params数组不能为空")
                
                # 验证每个参数项
                for i, param in enumerate(params_array):
                    if not isinstance(param, dict):
                        raise ValueError(f"params[{i}] 应该是一个对象")
                    
                    if "url" not in param or not param["url"]:
                        raise ValueError(f"params[{i}] 缺少必需的url字段")
                    
                    url = param["url"].strip()
                    if not (url.startswith('http://') or url.startswith('https://')):
                        raise ValueError(f"params[{i}] url格式不正确: {url}")
                
            except json.JSONDecodeError as e:
                raise ValueError(f"params参数JSON解析失败: {str(e)}")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Authorization": f"Bearer {api_key.strip()}",
                "X-App-Name": app_name.strip()
            }
            
            print(f"[ImageTranslateAPI] 发送翻译请求，图片数量: {len(params_array)}")
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


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageTranslateAPI": ImageTranslateAPI,
    "ImageTranslateParamsBuilder": ImageTranslateParamsBuilder,
    "ImageTranslateResultExtractor": ImageTranslateResultExtractor,
    "ADIC_COMMON_API": ADIC_COMMON_API
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTranslateAPI": "图片翻译 API",
    "ImageTranslateParamsBuilder": "图片翻译参数构建器",
    "ImageTranslateResultExtractor": "图片翻译结果提取器",
    "ADIC_COMMON_API": "ADIC Common API"
} 