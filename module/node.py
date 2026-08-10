from .nodes_ai_generation import ADICOpenAIGPTImage1, IdeaLabImageGenerate
from .nodes_api_basic import (
    ADIC_COMMON_API,
    ImageTranslateAPI,
    ImageTranslateParamsBuilder,
    ImageTranslateResultExtractor,
    LoadImagesFromUrls,
    MarketImageGenerateWithPolling,
    PythonCodeExecutor,
    StringToJsonArray,
)
from .nodes_image_utils import (
    FluxKontextImageScale,
    ImageConcatFromBatch,
    ImageStitch,
    ReferenceLatent,
)
from .nodes_storage import AliCloudOSSUpload
from .nodes_template_compose import RemoteTemplateBatchCompose


NODE_CLASS_MAPPINGS = {
    "ImageTranslateAPI": ImageTranslateAPI,
    "ImageTranslateParamsBuilder": ImageTranslateParamsBuilder,
    "ImageTranslateResultExtractor": ImageTranslateResultExtractor,
    "ADIC_COMMON_API": ADIC_COMMON_API,
    "LoadImagesFromUrls": LoadImagesFromUrls,
    "PythonCodeExecutor": PythonCodeExecutor,
    "StringToJsonArray": StringToJsonArray,
    "MarketImageGenerateWithPolling": MarketImageGenerateWithPolling,
    "MaletteImageStitch": ImageStitch,
    "MaletteReferenceLatent": ReferenceLatent,
    "MaletteFluxKontextImageScale": FluxKontextImageScale,
    "MaletteImageConcatFromBatch": ImageConcatFromBatch,
    "AliCloudOSSUpload": AliCloudOSSUpload,
    "ADICOpenAIGPTImage1": ADICOpenAIGPTImage1,
    "IdeaLabImageGenerate": IdeaLabImageGenerate,
    "RemoteTemplateBatchCompose": RemoteTemplateBatchCompose,
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
    "MaletteImageStitch": "图片拼接",
    "MaletteReferenceLatent": "参考潜变量",
    "MaletteFluxKontextImageScale": "Flux Kontext 图片缩放",
    "MaletteImageConcatFromBatch": "图片拼接",
    "AliCloudOSSUpload": "阿里云OSS文件上传",
    "OpenAIGPTImage1": "ADIC OpenAIGPTImage1",
    "ADICOpenAIGPTImage1": "ADIC OpenAIGPTImage1",
    "IdeaLabImageGenerate": "IdeaLab Image Generate",
    "RemoteTemplateBatchCompose": "批量套版（远端合成）",
}
