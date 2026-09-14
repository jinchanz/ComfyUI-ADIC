"""画布模板工作流协议（templateProtocolVersion = 2）的解析逻辑。

只处理协议语义，不含任何 HTTP 调用，供批量套版 AI hook 与单元格生成共用同一套解析器。
"""

import copy
import hashlib
import json

# 批量路由与执行上下文的保留 key，不能作为 templateRules key 或 binding inputKey；
# 存量数据可能把它们泄漏在 input 中，解析时必须忽略
RESERVED_INPUT_KEYS = frozenset({"workId", "rowNo", "sourceKey", "attempt", "templateKey", "templateId"})

PROTOCOL_VERSION = 2
SUPPORTED_WORKFLOW_TYPE = "COMFY_APP"

# 引用图层时属性与图层类型的对应关系
LAYER_PROPERTY_TYPES = {"src": "image", "content": "text"}
# AI 结果图层类型与取值属性的对应关系
RESULT_TYPE_PROPERTIES = {"image": "src", "text": "content"}

_MISSING = object()


class WorkflowProtocolError(Exception):
    """带协议错误码与定位信息的校验/解析失败"""

    def __init__(self, code, message, **location):
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = {key: value for key, value in location.items() if value is not None}


def strip_reserved_keys(input_values):
    """剔除行数据中的保留 key，得到只含模板字段的输入"""
    return {key: value for key, value in input_values.items() if key not in RESERVED_INPUT_KEYS}


def index_layers(root):
    """按 ID 索引图层树，返回 (id -> 图层, 重复出现的 ID 集合)"""
    index = {}
    duplicated = set()

    def walk(node):
        node_id = node.get("id")
        if node_id is not None:
            if node_id in index:
                duplicated.add(node_id)
            else:
                index[node_id] = node
        children = node.get("layers")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child)

    walk(root)
    return index, duplicated


def has_ai_rules(template_rules):
    return bool(ai_result_keys(template_rules))


def ai_result_keys(template_rules):
    """按模板声明顺序返回所有 AI 结果列的 rule key（workflows 非空）"""
    keys = []
    for key, rule in template_rules.items():
        if isinstance(rule, dict) and rule.get("workflows"):
            keys.append(key)
    return keys


def get_workflow(template_rules, result_rule_key):
    """取出第 2 版协议要求的唯一 WorkflowConfig"""
    rule = template_rules.get(result_rule_key)
    if not isinstance(rule, dict):
        raise WorkflowProtocolError("TEMPLATE_INVALID", f"规则 '{result_rule_key}' 不存在", resultRuleKey=result_rule_key)

    workflows = rule.get("workflows")
    if not isinstance(workflows, list) or len(workflows) != 1:
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 workflows 必须恰好包含一项，当前为 "
            f"{len(workflows) if isinstance(workflows, list) else type(workflows).__name__}",
            resultRuleKey=result_rule_key,
        )

    workflow = workflows[0]
    if not isinstance(workflow, dict):
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID", f"规则 '{result_rule_key}' 的 workflows[0] 必须是 JSON 对象", resultRuleKey=result_rule_key
        )
    return workflow


def parse_pointer(pointer, result_rule_key=None):
    """解析 binding 的 JSON Pointer，返回 token 列表"""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"binding 路径必须是以 / 开头的 JSON Pointer，当前为: {pointer!r}",
            resultRuleKey=result_rule_key,
            parameterPath=pointer if isinstance(pointer, str) else None,
        )
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def pointer_get(root, pointer, default=_MISSING, result_rule_key=None):
    """按 JSON Pointer 读取值，路径不存在时返回 default"""
    current = root
    for token in parse_pointer(pointer, result_rule_key):
        if isinstance(current, dict):
            if token not in current:
                return default
            current = current[token]
        elif isinstance(current, list):
            index = _as_list_index(token)
            if index is None or index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def pointer_set(root, pointer, value, result_rule_key=None):
    """按 JSON Pointer 写入值；父级为 dict 且缺失时自动补空对象"""
    tokens = parse_pointer(pointer, result_rule_key)
    current = root

    for token in tokens[:-1]:
        if isinstance(current, dict):
            child = current.get(token, _MISSING)
            if child is _MISSING or not isinstance(child, (dict, list)):
                current[token] = {}
            current = current[token]
        elif isinstance(current, list):
            index = _as_list_index(token)
            if index is None or index >= len(current):
                raise WorkflowProtocolError(
                    "PARAMETER_INVALID",
                    f"binding 路径 '{pointer}' 的数组下标 '{token}' 越界",
                    resultRuleKey=result_rule_key,
                    parameterPath=pointer,
                )
            current = current[index]
        else:
            raise WorkflowProtocolError(
                "PARAMETER_INVALID",
                f"binding 路径 '{pointer}' 的父级不是对象或数组",
                resultRuleKey=result_rule_key,
                parameterPath=pointer,
            )

    last = tokens[-1]
    if isinstance(current, dict):
        current[last] = value
        return

    if isinstance(current, list):
        index = _as_list_index(last)
        if index is None or index >= len(current):
            raise WorkflowProtocolError(
                "PARAMETER_INVALID",
                f"binding 路径 '{pointer}' 的数组下标 '{last}' 越界",
                resultRuleKey=result_rule_key,
                parameterPath=pointer,
            )
        current[index] = value
        return

    raise WorkflowProtocolError(
        "PARAMETER_INVALID",
        f"binding 路径 '{pointer}' 的父级不是对象或数组",
        resultRuleKey=result_rule_key,
        parameterPath=pointer,
    )


def _as_list_index(token):
    try:
        index = int(token)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def validate_workflows(template_data, layer_root):
    """校验模板中的 workflows 配置，不合法时抛带协议错误码的异常"""
    template_rules = template_data.get("templateRules") or {}
    result_keys = ai_result_keys(template_rules)
    if not result_keys:
        return

    version = template_data.get("templateProtocolVersion")
    if version != PROTOCOL_VERSION:
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"存在非空 workflows 时 templateProtocolVersion 必须为 {PROTOCOL_VERSION}，当前为: {version!r}",
        )

    layer_index, duplicated_ids = index_layers(layer_root)
    result_key_set = set(result_keys)

    for result_rule_key in result_keys:
        workflow = get_workflow(template_rules, result_rule_key)
        _validate_workflow(
            workflow, result_rule_key, template_rules, result_key_set, layer_index, duplicated_ids
        )

    # 依赖成环在这里暴露，避免执行时才失败
    build_execution_order(template_rules)


def _validate_workflow(workflow, result_rule_key, template_rules, result_key_set, layer_index, duplicated_ids):
    rule_type = template_rules[result_rule_key].get("type")
    if rule_type not in RESULT_TYPE_PROPERTIES:
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"AI 结果规则 '{result_rule_key}' 的 type 必须是 image 或 text，当前为: {rule_type!r}",
            resultRuleKey=result_rule_key,
        )

    workflow_key = workflow.get("workflowKey")
    if not isinstance(workflow_key, str) or not workflow_key.strip():
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID", f"规则 '{result_rule_key}' 的 workflowKey 不能为空", resultRuleKey=result_rule_key
        )

    workflow_type = workflow.get("type")
    if workflow_type != SUPPORTED_WORKFLOW_TYPE:
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 workflow type 暂不支持: {workflow_type!r}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    tool_name = workflow.get("toolName")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 toolName 不能为空",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    if not isinstance(workflow.get("params"), dict):
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 params 必须是 JSON 对象",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    output = workflow.get("output")
    if not isinstance(output, dict) or not output.get("key"):
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 缺少有效的 output.key",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )
    item_index = output.get("itemIndex", 0)
    if not isinstance(item_index, int) or isinstance(item_index, bool) or item_index < 0:
        raise WorkflowProtocolError(
            "OUTPUT_INVALID",
            f"规则 '{result_rule_key}' 的 output.itemIndex 必须是非负整数，当前为: {item_index!r}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    bindings = workflow.get("bindings")
    if not isinstance(bindings, dict):
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 bindings 必须是 JSON 对象",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    parsed_paths = []
    for pointer, binding in bindings.items():
        tokens = parse_pointer(pointer, result_rule_key)
        for existing_pointer, existing_tokens in parsed_paths:
            if _is_prefix(tokens, existing_tokens) or _is_prefix(existing_tokens, tokens):
                raise WorkflowProtocolError(
                    "TEMPLATE_INVALID",
                    f"规则 '{result_rule_key}' 的 binding 路径重复或父子重叠: '{existing_pointer}' 与 '{pointer}'",
                    resultRuleKey=result_rule_key,
                    workflowKey=workflow_key,
                    parameterPath=pointer,
                )
        parsed_paths.append((pointer, tokens))

        if not isinstance(binding, dict):
            raise WorkflowProtocolError(
                "TEMPLATE_INVALID",
                f"规则 '{result_rule_key}' 的 binding '{pointer}' 必须是 JSON 对象",
                resultRuleKey=result_rule_key,
                workflowKey=workflow_key,
                parameterPath=pointer,
            )

        source = binding.get("source")
        if source == "constant":
            continue

        if source == "row":
            _validate_row_binding(binding, pointer, result_rule_key, workflow_key, template_rules, result_key_set)
            continue

        if source == "layer":
            _validate_layer_binding(
                binding, pointer, result_rule_key, workflow_key, layer_index, duplicated_ids
            )
            continue

        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 binding '{pointer}' source 不合法: {source!r}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            parameterPath=pointer,
        )


def _validate_row_binding(binding, pointer, result_rule_key, workflow_key, template_rules, result_key_set):
    input_key = binding.get("inputKey")
    if not isinstance(input_key, str) or not input_key:
        raise WorkflowProtocolError(
            "TEMPLATE_INVALID",
            f"规则 '{result_rule_key}' 的 binding '{pointer}' 缺少 inputKey",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            parameterPath=pointer,
        )
    if input_key in RESERVED_INPUT_KEYS:
        raise WorkflowProtocolError(
            "INPUT_KEY_CONFLICT",
            f"inputKey '{input_key}' 与保留 key 冲突",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            inputKey=input_key,
            parameterPath=pointer,
        )
    if input_key in result_key_set:
        raise WorkflowProtocolError(
            "INPUT_KEY_CONFLICT",
            f"inputKey '{input_key}' 是 AI 结果列，AI 结果之间的依赖必须使用 source=layer",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            inputKey=input_key,
            parameterPath=pointer,
        )
    # inputKey 允许等于普通模板字段：该列同时用于画布替换和 AI 参数
    rule = template_rules.get(input_key)
    if isinstance(rule, dict) and rule.get("workflows"):
        raise WorkflowProtocolError(
            "INPUT_KEY_CONFLICT",
            f"inputKey '{input_key}' 绑定的规则含 workflows",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            inputKey=input_key,
            parameterPath=pointer,
        )


def _validate_layer_binding(binding, pointer, result_rule_key, workflow_key, layer_index, duplicated_ids):
    layer_id = binding.get("layerId")
    property_name = binding.get("property")

    if property_name not in LAYER_PROPERTY_TYPES:
        raise WorkflowProtocolError(
            "LAYER_REFERENCE_INVALID",
            f"规则 '{result_rule_key}' 的 binding '{pointer}' property 必须是 src 或 content，当前为: {property_name!r}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            layerId=layer_id,
            parameterPath=pointer,
        )

    node = layer_index.get(layer_id)
    if node is None:
        raise WorkflowProtocolError(
            "LAYER_REFERENCE_INVALID",
            f"规则 '{result_rule_key}' 引用的图层 '{layer_id}' 不存在",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            layerId=layer_id,
            parameterPath=pointer,
        )
    if layer_id in duplicated_ids:
        raise WorkflowProtocolError(
            "LAYER_REFERENCE_INVALID",
            f"规则 '{result_rule_key}' 引用的图层 '{layer_id}' 在模板中不唯一",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            layerId=layer_id,
            parameterPath=pointer,
        )

    expected_type = LAYER_PROPERTY_TYPES[property_name]
    if node.get("type") != expected_type:
        raise WorkflowProtocolError(
            "LAYER_REFERENCE_INVALID",
            f"规则 '{result_rule_key}' 的 binding '{pointer}' 要求 {expected_type} 图层，"
            f"图层 '{layer_id}' 类型为 {node.get('type')!r}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
            layerId=layer_id,
            parameterPath=pointer,
        )


def _is_prefix(tokens, other):
    return len(tokens) <= len(other) and other[: len(tokens)] == tokens


def build_execution_order(template_rules):
    """按 source=layer 依赖对 AI 规则拓扑排序，成环时抛 DEPENDENCY_CYCLE"""
    result_keys = ai_result_keys(template_rules)
    owner_by_layer = {}
    for key in result_keys:
        layer_id = template_rules[key].get("id")
        if layer_id is not None:
            owner_by_layer[layer_id] = key

    dependencies = {}
    for key in result_keys:
        workflow = get_workflow(template_rules, key)
        upstream_keys = []
        for pointer, binding in (workflow.get("bindings") or {}).items():
            if not isinstance(binding, dict) or binding.get("source") != "layer":
                continue
            upstream = owner_by_layer.get(binding.get("layerId"))
            if upstream is None:
                continue
            if upstream == key:
                raise WorkflowProtocolError(
                    "DEPENDENCY_CYCLE",
                    f"规则 '{key}' 的 binding '{pointer}' 引用了自身图层",
                    resultRuleKey=key,
                    layerId=binding.get("layerId"),
                    parameterPath=pointer,
                )
            upstream_keys.append(upstream)
        dependencies[key] = upstream_keys

    order = []
    state = {}

    def visit(key, path):
        status = state.get(key)
        if status == "done":
            return
        if status == "visiting":
            cycle = " -> ".join(path + [key])
            raise WorkflowProtocolError("DEPENDENCY_CYCLE", f"AI 图层依赖成环: {cycle}", resultRuleKey=key)
        state[key] = "visiting"
        for upstream in dependencies[key]:
            visit(upstream, path + [key])
        state[key] = "done"
        order.append(key)

    for key in result_keys:
        visit(key, [])
    return order


def resolve_workflow_params(template_rules, layer_index, input_values, result_rule_key, upstream_results):
    """按 bindings 解析出本行的工作流参数，返回深拷贝后的 params"""
    workflow = get_workflow(template_rules, result_rule_key)
    params = copy.deepcopy(workflow.get("params") or {})

    for pointer, binding in (workflow.get("bindings") or {}).items():
        source = binding.get("source")
        if source == "constant":
            # 固定值即 params 中已保存的原生值，保持不动
            continue

        if source == "row":
            input_key = binding.get("inputKey")
            if input_key not in input_values:
                # key 缺失才回退已保存值；null / 空串 / 0 / false / 空数组都是有效值
                continue
            value = input_values[input_key]
        elif source == "layer":
            value = _resolve_layer_value(
                binding, pointer, result_rule_key, template_rules, layer_index, input_values, upstream_results
            )
        else:
            raise WorkflowProtocolError(
                "TEMPLATE_INVALID",
                f"规则 '{result_rule_key}' 的 binding '{pointer}' source 不合法: {source!r}",
                resultRuleKey=result_rule_key,
                parameterPath=pointer,
            )

        pointer_set(params, pointer, value, result_rule_key)

    return params


def _resolve_layer_value(
    binding, pointer, result_rule_key, template_rules, layer_index, input_values, upstream_results
):
    layer_id = binding.get("layerId")
    property_name = binding.get("property")
    node = layer_index.get(layer_id)
    if node is None or property_name not in LAYER_PROPERTY_TYPES:
        raise WorkflowProtocolError(
            "LAYER_REFERENCE_INVALID",
            f"规则 '{result_rule_key}' 的 binding '{pointer}' 引用图层无效: {layer_id!r}.{property_name!r}",
            resultRuleKey=result_rule_key,
            layerId=layer_id,
            parameterPath=pointer,
        )
    if node.get("type") != LAYER_PROPERTY_TYPES[property_name]:
        raise WorkflowProtocolError(
            "LAYER_REFERENCE_INVALID",
            f"规则 '{result_rule_key}' 的 binding '{pointer}' 引用图层类型不匹配: {node.get('type')!r}",
            resultRuleKey=result_rule_key,
            layerId=layer_id,
            parameterPath=pointer,
        )

    owner_key = None
    for rule_key, rule in template_rules.items():
        if isinstance(rule, dict) and rule.get("id") == layer_id:
            owner_key = rule_key
            break

    if owner_key is None:
        # 未绑定任何规则的图层，直接用模板默认值
        return node.get(property_name)

    if template_rules[owner_key].get("workflows"):
        if owner_key not in upstream_results:
            raise WorkflowProtocolError(
                "WORKFLOW_EXECUTION_FAILED",
                f"规则 '{result_rule_key}' 依赖的上游 AI 结果 '{owner_key}' 尚未生成",
                resultRuleKey=result_rule_key,
                layerId=layer_id,
                parameterPath=pointer,
            )
        return upstream_results[owner_key]

    # 普通素材图层先应用本行值，缺失时回退图层自身值
    if owner_key in input_values:
        return input_values[owner_key]
    return node.get(property_name)


def select_workflow_output(result_data, output, result_type, result_rule_key=None, workflow_key=None):
    """按 output.key + itemIndex 从工具结果中选出最终值"""
    output_key = output.get("key")
    item_index = output.get("itemIndex") or 0

    items = _locate_output_items(result_data, output_key)
    if items is None:
        raise WorkflowProtocolError(
            "OUTPUT_INVALID",
            f"工具结果中找不到输出 '{output_key}'",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    if item_index >= len(items):
        raise WorkflowProtocolError(
            "OUTPUT_INVALID",
            f"输出 '{output_key}' 只有 {len(items)} 项，取不到第 {item_index} 项",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    value = _extract_scalar(items[item_index], result_type)
    if value is None:
        raise WorkflowProtocolError(
            "OUTPUT_INVALID",
            f"输出 '{output_key}' 第 {item_index} 项不是可用的 {result_type} 结果: "
            f"{_snippet(items[item_index])}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    if result_type == "image" and not value.startswith(("http://", "https://")):
        raise WorkflowProtocolError(
            "OUTPUT_INVALID",
            f"输出 '{output_key}' 的图片结果不是可访问的 URL: {_snippet(value)}",
            resultRuleKey=result_rule_key,
            workflowKey=workflow_key,
        )

    return value


# 结果结构暂未固定，按常见容器逐层查找 output.key
_OUTPUT_CONTAINER_KEYS = ("outputs", "outputNodes", "output", "result", "results", "data")
_ITEM_LIST_KEYS = ("images", "texts", "items", "values", "list", "data", "results")
_IMAGE_VALUE_KEYS = ("url", "imageUrl", "image_url", "fileUrl", "file_url", "ossUrl", "src", "value")
_TEXT_VALUE_KEYS = ("text", "content", "value", "string")


def _locate_output_items(result_data, output_key):
    if not isinstance(result_data, dict) or not output_key:
        return None

    containers = [result_data]
    for key in _OUTPUT_CONTAINER_KEYS:
        nested = result_data.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
            for inner_key in _OUTPUT_CONTAINER_KEYS:
                inner = nested.get(inner_key)
                if isinstance(inner, dict):
                    containers.append(inner)

    for container in containers:
        if output_key not in container:
            continue
        return _as_item_list(container[output_key])
    return None


def _as_item_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in _ITEM_LIST_KEYS:
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [value]
    return None


def _extract_scalar(item, result_type):
    if isinstance(item, str):
        return item or None
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return str(item) if result_type == "text" else None
    if not isinstance(item, dict):
        return None

    keys = _IMAGE_VALUE_KEYS if result_type == "image" else _TEXT_VALUE_KEYS
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        if result_type == "text" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def resolve_compose_input(input_values, generated_patch):
    """把 AI 结果合并进行数据，得到交给旧替换/合图链路的 resolvedInput"""
    resolved = dict(input_values)
    resolved.update(generated_patch)
    return resolved


def effective_input_hash(workflow, params):
    """幂等身份的输入部分：工具、版本、解析后参数和输出选择共同决定"""
    payload = {
        "toolName": workflow.get("toolName"),
        "workflowCode": workflow.get("workflowCode"),
        "workflowVersion": workflow.get("workflowVersion"),
        "params": params,
        "output": workflow.get("output"),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remap_rule_layer_ids(template_rules, id_mapping):
    """重新分配图层 ID 后，同步更新 rule.id 与 binding.layerId"""
    for rule in template_rules.values():
        if not isinstance(rule, dict):
            continue
        if rule.get("id") in id_mapping:
            rule["id"] = id_mapping[rule["id"]]
        for workflow in rule.get("workflows") or []:
            if not isinstance(workflow, dict):
                continue
            for binding in (workflow.get("bindings") or {}).values():
                if isinstance(binding, dict) and binding.get("layerId") in id_mapping:
                    binding["layerId"] = id_mapping[binding["layerId"]]


def _snippet(value, limit=200):
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit] + ("...（已截断）" if len(text) > limit else "")
