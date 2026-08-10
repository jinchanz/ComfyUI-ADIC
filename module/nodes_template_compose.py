import copy
import json
import traceback
import uuid


class RemoteTemplateBatchCompose:
    """批量套版节点

    接收一份模板 JSON 和多条表格数据，为每条数据生成预览图 URL 和
    可重新打开编辑的画布 JSON（最外层 type 为 page）。

    模板替换与合图以后会调用远端 HTTP 服务，当前在 `_compose_remote`
    中 Mock 实现；远端服务上线后只替换该方法内部实现，输入输出格式不变。
    """

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
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "compose"
    CATEGORY = "Malette"

    def compose(self, template_json, items_json):
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
                results.append(self._compose_one(template_data, item))
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
        if layer.get("type") != "template":
            raise ValueError(f"layer.type 必须为 template，当前为: {layer.get('type')}")

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
            if node.get("columnName") != field_name:
                raise ValueError(
                    f"规则键 '{field_name}' 与节点 columnName '{node.get('columnName')}' 不一致"
                )

    def _compose_one(self, template_data, item):
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

        # 根据规则替换数据并更新规则中引用的 ID；
        # 规则完整保留（包括 comfyConfig 等字段），只更新其中的节点 ID。
        # 存在 comfyConfig 时当前不做 AI 调用，同样把当前表格值当作 Mock 结果写入。
        for field_name, rule in rules.items():
            new_id = id_mapping[rule["id"]]
            rule["id"] = new_id

            if field_name not in input_values:
                continue
            value = input_values[field_name]
            node = self._find_nodes_by_id(layer, new_id)[0]
            if rule.get("type") == "text":
                node["content"] = value
            elif rule.get("type") == "image":
                node["src"] = value
            else:
                raise ValueError(f"规则 '{field_name}' 类型 '{rule.get('type')}' 不支持替换")

        # 生成可编辑画布 JSON：最外层 type 为 page，模板放入 layers；
        # 规则按 templateRules[模板名称][字段名] 输出；
        # colorPalette 与模板中不认识的顶层字段原样保留
        canvas = {
            key: copy.deepcopy(value)
            for key, value in template_data.items()
            if key not in ("layer", "templateRules")
        }
        canvas["type"] = "page"
        canvas["layers"] = [layer]
        canvas["templateRules"] = {layer.get("name", ""): rules}
        canvas.setdefault("colorPalette", [])

        preview = self._compose_remote(canvas, item)

        result = {
            "workId": work_id,
            "rowNo": item.get("rowNo"),
            "attempt": item.get("attempt"),
            "status": "SUCCESS",
            "url": preview["url"],
            "filename": preview["filename"],
            "width": preview["width"],
            "height": preview["height"],
            "value": canvas,
        }
        return result

    def _compose_remote(self, canvas, item):
        """调用远端服务合成预览图。

        远端 HTTP 服务暂未完成，当前返回固定测试图；
        服务上线后只替换此方法内部实现，返回格式保持不变。
        """
        work_id = item.get("workId") or f"work_{uuid.uuid4().hex[:8]}"
        return {
            "url": self.MOCK_PREVIEW_URL,
            "filename": f"{work_id}.png",
            "width": self.MOCK_PREVIEW_WIDTH,
            "height": self.MOCK_PREVIEW_HEIGHT,
        }

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
