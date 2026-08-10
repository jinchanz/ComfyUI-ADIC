import io
import json
import os
import time
import traceback

import numpy as np
from PIL import Image

try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False
    print("[AliCloudOSSUpload] 警告: oss2 库未安装，请运行 'pip install oss2' 安装")


class AliCloudOSSUpload:
    """阿里云OSS文件上传节点（支持批量上传）"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {"default": "https://oss-cn-hangzhou.aliyuncs.com"}),
                "region": ("STRING", {"default": "cn-hangzhou"}),
                "domain": ("STRING", {"default": "oss-cn-hangzhou.aliyuncs.com"}),
                "bucket_name": ("STRING", {"default": ""}),
                "object_key": ("STRING", {"default": "uploads/image_{timestamp}_{index}"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "file_paths": ("STRING", {"default": "", "multiline": True}),
                "access_key_id": ("STRING", {"default": ""}),
                "access_key_secret": ("STRING", {"default": ""}),
                "content_type": ("STRING", {"default": "image/png"}),
                "use_timestamp": ("BOOLEAN", {"default": True}),
                "make_public": ("BOOLEAN", {"default": False}),
                "batch_upload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("oss_urls", "public_urls", "upload_info")

    FUNCTION = "upload_to_oss"

    OUTPUT_NODE = True

    CATEGORY = "Malette"

    def upload_to_oss(
        self,
        endpoint,
        region,
        domain,
        bucket_name,
        object_key,
        image=None,
        file_paths="",
        access_key_id="",
        access_key_secret="",
        content_type="image/png",
        use_timestamp=True,
        make_public=False,
        batch_upload=True,
    ):
        try:
            if not OSS_AVAILABLE:
                error_msg = "oss2 库未安装，请运行 'pip install oss2' 安装"
                print(f"[AliCloudOSSUpload] {error_msg}")
                return (
                    json.dumps({"error": error_msg}, ensure_ascii=False),
                    "",
                    json.dumps({"error": error_msg}, ensure_ascii=False),
                )

            domain = domain.strip() if domain else os.getenv("OSS_DOMAIN", domain)
            bucket_name = bucket_name.strip() if bucket_name else os.getenv("OSS_BUCKET", "")
            ak_id = access_key_id.strip() if access_key_id else os.getenv("OSS_ACCESS_KEY", "")
            ak_secret = access_key_secret.strip() if access_key_secret else os.getenv("OSS_ACCESS_SECRET", "")

            if not ak_id:
                raise ValueError("ACCESS_KEY_ID不能为空，请在参数中设置或设置环境变量 OSS_ACCESS_KEY")

            if not ak_secret:
                raise ValueError("ACCESS_KEY_SECRET不能为空，请在参数中设置或设置环境变量 OSS_ACCESS_SECRET")

            if not bucket_name:
                raise ValueError("存储桶名称不能为空，请在参数中设置或设置环境变量 OSS_BUCKET")

            if not object_key.strip():
                raise ValueError("对象键不能为空")

            upload_tasks = []

            if image is not None:
                print(f"[AliCloudOSSUpload] 处理图片上传，图片形状: {image.shape}")

                if len(image.shape) == 4 and batch_upload:
                    for i in range(image.shape[0]):
                        image_np = image[i].cpu().numpy()
                        upload_tasks.append({"type": "image", "data": image_np, "index": i, "name": f"image_{i}"})
                else:
                    if len(image.shape) == 4:
                        image_np = image[0].cpu().numpy()
                    else:
                        image_np = image.cpu().numpy()

                    upload_tasks.append({"type": "image", "data": image_np, "index": 0, "name": "image"})

            if file_paths and file_paths.strip():
                paths = [path.strip() for path in file_paths.strip().split("\n") if path.strip()]

                for i, file_path in enumerate(paths):
                    if not os.path.exists(file_path):
                        print(f"[AliCloudOSSUpload] 警告: 文件不存在，跳过: {file_path}")
                        continue

                    upload_tasks.append({"type": "file", "data": file_path, "index": i, "name": os.path.basename(file_path)})

                    if not batch_upload and i == 0:
                        break

            if not upload_tasks:
                raise ValueError("没有找到有效的上传文件")

            print(f"[AliCloudOSSUpload] 创建OSS连接: {domain}")
            auth = oss2.Auth(ak_id, ak_secret)
            bucket = oss2.Bucket(auth, domain, bucket_name)

            oss_urls = []
            public_urls = []
            upload_results = []

            print(f"[AliCloudOSSUpload] 开始批量上传，共 {len(upload_tasks)} 个文件")

            for task in upload_tasks:
                try:
                    current_object_key = object_key

                    if use_timestamp:
                        timestamp = str(int(time.time()))
                        current_object_key = current_object_key.replace("{timestamp}", timestamp)

                    current_object_key = current_object_key.replace("{index}", str(task["index"]))

                    if "{name}" in current_object_key:
                        current_object_key = current_object_key.replace("{name}", task["name"])

                    upload_data = None
                    actual_content_type = content_type

                    if task["type"] == "image":
                        current_object_key = current_object_key + ".png"
                        image_np = task["data"]

                        if image_np.max() <= 1.0:
                            image_np = (image_np * 255).astype(np.uint8)
                        else:
                            image_np = image_np.astype(np.uint8)

                        pil_image = Image.fromarray(image_np)
                        img_buffer = io.BytesIO()

                        if current_object_key.lower().endswith((".jpg", ".jpeg")):
                            pil_image.save(img_buffer, format="JPEG", quality=95)
                            actual_content_type = "image/jpeg"
                        elif current_object_key.lower().endswith(".webp"):
                            pil_image.save(img_buffer, format="WEBP", quality=95)
                            actual_content_type = "image/webp"
                        else:
                            pil_image.save(img_buffer, format="PNG")
                            actual_content_type = "image/png"

                        upload_data = img_buffer.getvalue()

                    elif task["type"] == "file":
                        ext = os.path.splitext(task["data"])[1]
                        current_object_key = current_object_key + ext
                        print(f"[AliCloudOSSUpload] current_object_key: {current_object_key}")
                        file_path = task["data"]

                        with open(file_path, "rb") as f:
                            upload_data = f.read()

                        if content_type == "image/png":
                            _, ext = os.path.splitext(file_path)
                            content_type_map = {
                                ".png": "image/png",
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".gif": "image/gif",
                                ".webp": "image/webp",
                                ".pdf": "application/pdf",
                                ".txt": "text/plain",
                                ".json": "application/json",
                            }
                            actual_content_type = content_type_map.get(ext.lower(), "application/octet-stream")

                    headers = {
                        "Content-Type": actual_content_type,
                    }

                    if make_public:
                        headers["x-oss-object-acl"] = "public-read"

                    print(f"[AliCloudOSSUpload] 上传第 {task['index']+1} 个文件: {current_object_key}")

                    result = bucket.put_object(current_object_key, upload_data, headers=headers)

                    oss_url = current_object_key
                    public_url = self.get_download_url(ak_id, ak_secret, domain, bucket_name, current_object_key)

                    oss_urls.append(oss_url)
                    public_urls.append(public_url)

                    upload_result = {
                        "success": True,
                        "index": task["index"],
                        "object_key": current_object_key,
                        "etag": result.etag,
                        "request_id": result.request_id,
                        "content_type": actual_content_type,
                        "size": len(upload_data),
                        "oss_url": oss_url,
                        "public_url": public_url,
                        "type": task["type"],
                    }
                    upload_results.append(upload_result)

                    print(f"[AliCloudOSSUpload] 文件 {task['index']+1} 上传成功: {oss_url}")

                except Exception as e:
                    error_msg = f"文件 {task['index']+1} 上传失败: {str(e)}"
                    print(f"[AliCloudOSSUpload] {error_msg}")

                    upload_result = {
                        "success": False,
                        "index": task["index"],
                        "error": error_msg,
                        "type": task["type"],
                    }
                    upload_results.append(upload_result)

                    oss_urls.append("")
                    public_urls.append("")

            success_count = sum(1 for result in upload_results if result.get("success", False))
            total_count = len(upload_results)

            summary_info = {
                "success": success_count > 0,
                "total_files": total_count,
                "success_count": success_count,
                "failed_count": total_count - success_count,
                "bucket": bucket_name,
                "endpoint": endpoint,
                "domain": domain,
                "make_public": make_public,
                "results": upload_results,
            }

            print(f"[AliCloudOSSUpload] 批量上传完成！成功: {success_count}/{total_count}")

            return (
                json.dumps(oss_urls, ensure_ascii=False),
                json.dumps(public_urls, ensure_ascii=False),
                json.dumps(summary_info, ensure_ascii=False, indent=2),
            )

        except oss2.exceptions.OssError as e:
            error_msg = f"OSS错误: {e.code} - {e.message}"
            print(f"[AliCloudOSSUpload] {e} {error_msg}, traceback: {traceback.format_exc()}")
            error_info = {
                "error": error_msg,
                "code": e.code,
                "message": e.message,
                "request_id": getattr(e, "request_id", ""),
            }
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                "",
                json.dumps(error_info, ensure_ascii=False, indent=2),
            )

        except Exception as e:
            error_msg = f"上传失败: {str(e)}"
            print(f"[AliCloudOSSUpload] {error_msg}, traceback: {traceback.format_exc()}")
            error_info = {"error": error_msg}
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                "",
                json.dumps(error_info, ensure_ascii=False, indent=2),
            )

    def get_download_url(self, ak_id, ak_secret, domain, bucket_name, file_id, expires=3600):
        auth = oss2.Auth(ak_id, ak_secret)
        print("[AliCloudOSSUpload] file_id: %s , expires: %s" % (file_id, expires))
        bucket = oss2.Bucket(auth, domain, bucket_name)
        url = bucket.sign_url("GET", file_id, expires, params={})
        print("[AliCloudOSSUpload] oss file url is %s" % url)
        return url
