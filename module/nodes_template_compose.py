import copy
import json
import traceback
import uuid

import requests


class RemoteTemplateBatchCompose:
    """批量套版节点

    接收一份模板 JSON 和多条表格数据，为每条数据生成预览图 URL 和
    可重新打开编辑的画布 JSON（最外层 type 为 page）。

    模板替换在本地完成；预览图通过远端套版服务
    `POST {base_url}/open/api/agent/v1/template/replace` 合成。
    `base_url` 留空时回退到 Mock 固定测试图，方便本地调试。
    """

    REPLACE_API_PATH = "/open/api/agent/v1/template/replace"

    # Mock 阶段返回的固定预览图信息
    MOCK_PREVIEW_URL = "https://placehold.co/1200x1200.png"
    MOCK_PREVIEW_WIDTH = 1200
    MOCK_PREVIEW_HEIGHT = 1200

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template_json": ("STRING", {"forceInput": True, "tooltip": "完整模板 JSON 字符串，结构遵循拓版协议"}),
                "items_json": ("STRING", {"forceInput": True, "tooltip": "待处理数据数组的 JSON 字符串"}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "", "tooltip": "远端套版服务基础地址，例如 https://abcd.example；留空时使用 Mock 预览图"}),
                "auth_token": ("STRING", {"default": "", "tooltip": "Bearer AK，调用远端服务时写入 Authorization 头"}),
                "app_name": ("STRING", {"default": "local", "tooltip": "远端服务 appName"}),
                "biz_code": ("STRING", {"default": "", "tooltip": "远端服务 bizCode"}),
                "file_type": (["png", "jpg", "webp"], {"default": "png", "tooltip": "预览图文件格式"}),
                "store_type": ("STRING", {"default": "oss", "tooltip": "预览图存储类型"}),
                "timeout": ("INT", {"default": 300, "min": 5, "max": 1800, "tooltip": "单条远端调用超时（秒）"}),
                "extra_headers": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "自定义请求头，支持 JSON 对象或每行一条 'Key: Value'；"
                                   "与 Bearer Token 合并，auth_token 非空时 Authorization 以 auth_token 为准",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "compose"
    CATEGORY = "Malette"

    def compose(
        self,
        template_json,
        items_json,
        base_url="",
        auth_token="",
        app_name="local",
        biz_code="",
        file_type="png",
        store_type="oss",
        timeout=300,
        extra_headers="",
    ):
        config = {
            "base_url": (base_url or "").strip(),
            "auth_token": (auth_token or "").strip(),
            "app_name": (app_name or "local").strip(),
            "biz_code": (biz_code or "").strip(),
            "file_type": (file_type or "png").strip(),
            "store_type": (store_type or "oss").strip(),
            "timeout": timeout or 300,
            # 节点级配置错误直接抛错，使本次工作流失败
            "extra_headers": self._parse_extra_headers(extra_headers),
        }

        # templateJson / itemsJson 本身无法解析时直接报错，使本次工作流失败
        try:
            template_data = json.loads(template_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"templateJson 解析失败: {str(e)}")

        try:
            items = json.loads(items_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"itemsJson 解析失败: {str(e)}")

        if not isinstance(template_data, dict):
            raise ValueError("templateJson 必须是 JSON 对象")
        if not isinstance(items, list):
            raise ValueError("itemsJson 必须是 JSON 数组")

        # 模板校验只做一次；模板不合法时所有条目返回同样的失败结果，
        # 保证结果数量与输入数量一致
        template_error = None
        try:
            self._validate_template(template_data)
        except ValueError as e:
            template_error = str(e)
            print(f"[RemoteTemplateBatchCompose] 模板校验失败: {template_error}")

        results = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                results.append(self._failed_result({}, "INVALID_ITEM", f"第 {index + 1} 条数据不是 JSON 对象"))
                continue

            if template_error is not None:
                results.append(self._failed_result(item, "TEMPLATE_INVALID", template_error))
                continue

            # 单条失败只返回该条失败，不影响其他数据
            try:
                results.append(self._compose_one(template_data, item, config))
            except Exception as e:
                print(f"[RemoteTemplateBatchCompose] 第 {index + 1} 条套版失败: {str(e)}")
                traceback.print_exc()
                results.append(self._failed_result(item, "COMPOSE_FAILED", str(e)))

        print(f"[RemoteTemplateBatchCompose] 处理完成: 共 {len(results)} 条，"
              f"成功 {sum(1 for r in results if r.get('status') == 'SUCCESS')} 条")
        return (json.dumps(results, ensure_ascii=False),)

    def _validate_template(self, template_data):
        """校验模板结构与 templateRules 的一致性"""
        layer = template_data.get("layer")
        if not isinstance(layer, dict):
            raise ValueError("模板缺少 layer 对象")
        if layer.get("type") not in ("template", "frame"):
            raise ValueError(f"layer.type 必须为 template 或 frame，当前为: {layer.get('type')}")

        rules = template_data.get("templateRules")
        if not isinstance(rules, dict):
            raise ValueError("模板缺少 templateRules 对象")

        for field_name, rule in rules.items():
            if not isinstance(rule, dict):
                raise ValueError(f"规则 '{field_name}' 必须是 JSON 对象")
            rule_id = rule.get("id")
            rule_type = rule.get("type")
            if not rule_id:
                raise ValueError(f"规则 '{field_name}' 缺少 id")

            nodes = self._find_nodes_by_id(layer, rule_id)
            if len(nodes) == 0:
                raise ValueError(f"规则 '{field_name}' 引用的节点 '{rule_id}' 在模板中不存在")
            if len(nodes) > 1:
                raise ValueError(f"规则 '{field_name}' 引用的节点 '{rule_id}' 在模板中不唯一")

            node = nodes[0]
            if node.get("type") != rule_type:
                raise ValueError(
                    f"规则 '{field_name}' 类型不一致: 规则为 '{rule_type}'，节点为 '{node.get('type')}'"
                )
            # 规则键与节点 columnName 的一致性不在本地校验，由远端接口负责

    def _compose_one(self, template_data, item, config):
        """处理单条数据，返回成功结果"""
        work_id = item.get("workId")
        input_values = item.get("input") or {}
        if not isinstance(input_values, dict):
            raise ValueError("input 必须是 JSON 对象")

        # 深拷贝模板，不修改原始模板
        layer = copy.deepcopy(template_data["layer"])
        rules = copy.deepcopy(template_data["templateRules"])

        # 为模板根节点及所有子节点生成全局唯一的新 ID
        id_mapping = self._regenerate_ids(layer)

        # 只更新规则中引用的节点 ID，规则其余字段（含 comfyConfig）完整保留。
        # 值替换由远端按 materialJson 完成，节点内不做本地替换：
        # 既避免对不认识的节点类型写错字段，也不会因类型不在白名单而报错。
        for rule in rules.values():
            rule["id"] = id_mapping[rule["id"]]

        # 本地拼装的画布仅作兜底（Mock 或远端未返回 templateJson 时使用），
        # 其中的节点值仍是模板默认值；远端返回结果时以远端为准。
        # colorPalette / colorPalettes 是可选字段，原样透传，缺失时不补默认值
        canvas = {
            key: copy.deepcopy(value)
            for key, value in template_data.items()
            if key not in ("layer", "templateRules")
        }
        canvas["type"] = "page"
        canvas["layers"] = [layer]
        canvas["templateRules"] = {layer.get("name", ""): rules}

        palette_key, palette = self._get_palette(template_data)
        preview = self._compose_remote(layer, rules, input_values, palette_key, palette, item, config)

        result = {
            "workId": work_id,
            "rowNo": item.get("rowNo"),
            "attempt": item.get("attempt"),
            "status": "SUCCESS",
            "url": preview["url"],
            "filename": preview["filename"],
            "width": preview["width"],
            "height": preview["height"],
            # 远端返回的结果 page JSON 优先，Mock/未返回时用本地拼装的画布
            "value": preview.get("canvas") or canvas,
        }
        return result

    def _compose_remote(self, layer, rules, input_values, palette_key, color_palette, item, config):
        """调用远端套版服务完成替换与合图。

        请求 `POST {base_url}/open/api/agent/v1/template/replace`：
        - templateJson: 模板（字符串化 JSON）
        - rulesJson: 以节点 ID 为键的规则映射，附带 frameId（字符串化 JSON）
        - materialJson: 字段名 -> 表格值（字符串化 JSON）
        - colorPaletteJson: 模板的配方数组（可选，模板未带配方时不传该字段）

        响应 `data.url` 为预览图，`data.templateJson` 为替换完成的 frame 协议
        （或整页 page JSON），`data.colorPaletteJson` 为原样带回的配方数组（可选）。
        base_url 为空时返回 Mock 固定测试图，方便本地调试。
        """
        work_id = item.get("workId") or f"work_{uuid.uuid4().hex[:8]}"
        file_type = config.get("file_type") or "png"
        fallback_filename = f"{work_id}.{file_type}"

        base_url = config.get("base_url") or ""
        if not base_url:
            return {
                "url": self.MOCK_PREVIEW_URL,
                "filename": fallback_filename,
                "width": self.MOCK_PREVIEW_WIDTH,
                "height": self.MOCK_PREVIEW_HEIGHT,
            }

        # rulesJson 以节点 ID 为键，保留规则原有字段（含 comfyConfig），补充 frameId
        frame_id = layer.get("id")
        rules_payload = {}
        for rule in rules.values():
            rule_entry = dict(rule)
            rule_entry["frameId"] = frame_id
            rules_payload[rule["id"]] = rule_entry

        request_body = {
            "appName": config.get("app_name") or "local",
            "bizCode": config.get("biz_code") or "",
            "fileType": file_type,
            "materialJson": json.dumps(input_values, ensure_ascii=False),
            "needPreview": True,
            "rulesJson": json.dumps(rules_payload, ensure_ascii=False),
            "storeType": config.get("store_type") or "oss",
            "templateJson": json.dumps(layer, ensure_ascii=False),
        }
        # colorPaletteJson 非必填，模板未带配方时不传
        if color_palette is not None:
            request_body["colorPaletteJson"] = json.dumps(color_palette, ensure_ascii=False)

        # 合并顺序：默认头 -> 自定义头 -> Bearer Token；
        # auth_token 非空时 Authorization 以 auth_token 为准
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        headers.update(config.get("extra_headers") or {})
        auth_token = config.get("auth_token") or ""
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        url = base_url.rstrip("/") + self.REPLACE_API_PATH
        print(f"[RemoteTemplateBatchCompose] 调用套版服务: {url} (workId: {work_id})")

        response = requests.post(url, headers=headers, json=request_body, timeout=config.get("timeout") or 300)
        if response.status_code >= 400:
            raise RuntimeError(
                f"套版服务请求失败: {response.status_code} {response.reason}, "
                f"响应: {self._snippet(response.text)}"
            )

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"套版服务返回非 JSON 内容: {self._snippet(response.text)}")

        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"套版服务返回失败: {data.get('message') or self._snippet(response.text)}")

        payload = data.get("data") if isinstance(data, dict) else None
        preview = self._extract_preview_info(payload if payload is not None else data)
        if not preview.get("url"):
            raise RuntimeError(f"套版服务未返回预览图 URL: {self._snippet(response.text)}")

        # data.templateJson 是替换完成的结果（frame 协议或整页 page）；
        # data.colorPaletteJson 非必填，未返回时沿用模板自带的配方
        canvas = self._build_result_canvas(
            self._parse_json_field(payload, "templateJson"),
            self._parse_optional_palette(payload),
            layer,
            rules,
            palette_key,
            color_palette,
        )

        # 尺寸缺失时回退到模板根节点尺寸
        width = preview.get("width") or self._as_int(layer.get("width")) or self.MOCK_PREVIEW_WIDTH
        height = preview.get("height") or self._as_int(layer.get("height")) or self.MOCK_PREVIEW_HEIGHT
        return {
            "url": preview["url"],
            "filename": preview.get("filename") or fallback_filename,
            "width": width,
            "height": height,
            "canvas": canvas,
        }

    @staticmethod
    def _get_palette(template_data):
        """读取模板的配方字段，兼容 colorPalette / colorPalettes 两种命名。

        返回 (使用的键名, 配方值)；两者都没有时返回 (默认键名, None)。
        """
        for key in ("colorPalette", "colorPalettes"):
            if key in template_data:
                return key, template_data[key]
        return "colorPalette", None

    @classmethod
    def _parse_optional_palette(cls, payload):
        """解析响应中的配方字段（非必填），兼容单复数命名；解析不了就当作未返回"""
        if not isinstance(payload, dict):
            return None
        for key in ("colorPaletteJson", "colorPalettesJson", "colorPalette", "colorPalettes"):
            if key not in payload:
                continue
            try:
                parsed = cls._parse_json_field(payload, key)
            except RuntimeError as e:
                print(f"[RemoteTemplateBatchCompose] 忽略无法解析的 {key}: {str(e)}")
                return None
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_json_field(payload, key):
        """解析响应中的字符串化 JSON 字段，缺失或为空时返回 None"""
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"套版服务返回的 {key} 解析失败: {str(e)}")
        raise RuntimeError(f"套版服务返回的 {key} 类型不支持: {type(value).__name__}")

    @staticmethod
    def _build_result_canvas(result_root, result_palette, layer, rules, palette_key, local_palette):
        """把远端返回的结果组装成可编辑画布（最外层 type 为 page）。

        远端 templateJson 既可能是替换后的 frame 协议，也可能已是整页 page；
        配方优先用远端带回的，其次回退模板自带的，都没有时不写该字段
        （colorPalette / colorPalettes 均为可选，沿用入参使用的命名）。
        """
        if not isinstance(result_root, dict):
            # 远端未返回结果，交由调用方使用本地兜底画布
            return None

        palette = result_palette if result_palette is not None else local_palette

        if result_root.get("type") == "page":
            canvas = result_root
            canvas.setdefault("templateRules", {layer.get("name", ""): rules})
        else:
            # frame / template 根：包装成 page，规则沿用替换后的节点 ID
            canvas = {
                "type": "page",
                "layers": [result_root],
                "templateRules": {result_root.get("name") or layer.get("name", ""): rules},
            }

        if palette is not None:
            canvas[palette_key] = palette
        return canvas

    def _extract_preview_info(self, payload):
        """从响应中提取预览图信息，兼容字符串 URL 与常见字段命名"""
        if isinstance(payload, str):
            return {"url": payload}
        if not isinstance(payload, dict):
            return {}

        url = None
        for key in ("url", "previewUrl", "preview_url", "imageUrl", "image_url", "fileUrl", "file_url", "previewImage"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                url = value
                break

        filename = None
        for key in ("filename", "fileName", "name"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                filename = value
                break

        return {
            "url": url,
            "filename": filename,
            "width": self._as_int(payload.get("width")),
            "height": self._as_int(payload.get("height")),
        }

    @staticmethod
    def _parse_extra_headers(extra_headers):
        """解析自定义请求头，支持 JSON 对象或每行一条 'Key: Value'"""
        text = (extra_headers or "").strip()
        if not text:
            return {}

        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"extra_headers JSON 解析失败: {str(e)}")
            if not isinstance(parsed, dict):
                raise ValueError("extra_headers JSON 必须是对象")
            headers = {}
            for key, value in parsed.items():
                if not str(key).strip():
                    raise ValueError("extra_headers 中存在空的 header 名")
                headers[str(key).strip()] = str(value)
            return headers

        headers = {}
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            key, sep, value = line.partition(":")
            if not sep or not key.strip():
                raise ValueError(f"extra_headers 第 {line_no} 行格式不正确，应为 'Key: Value': {line}")
            headers[key.strip()] = value.strip()
        return headers

    @staticmethod
    def _as_int(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(round(value))
        return None

    @staticmethod
    def _snippet(text, limit=500):
        text = text or ""
        return text[:limit] + ("...（已截断）" if len(text) > limit else "")

    def _regenerate_ids(self, root):
        """为节点树生成全局唯一的新 ID，返回 旧ID->新ID 映射"""
        id_mapping = {}

        def walk(node):
            old_id = node.get("id")
            if old_id is not None:
                new_id = f"{node.get('type', 'layer')}_{uuid.uuid4().hex}"
                node["id"] = new_id
                id_mapping[old_id] = new_id
            children = node.get("layers")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        walk(child)

        walk(root)
        return id_mapping

    def _find_nodes_by_id(self, root, node_id):
        """在模板子树中查找指定 ID 的全部节点"""
        found = []

        def walk(node):
            if node.get("id") == node_id:
                found.append(node)
            children = node.get("layers")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        walk(child)

        walk(root)
        return found

    def _failed_result(self, item, error_code, error_message):
        return {
            "workId": item.get("workId"),
            "rowNo": item.get("rowNo"),
            "attempt": item.get("attempt"),
            "status": "FAILED",
            "errorCode": error_code,
            "errorMessage": error_message,
        }
