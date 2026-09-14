"""Agent 工具任务调用适配器。

提交 `POST {base_url}/open/api/agent/v1/tool-tasks/submit` 得到 taskId，
再轮询 `GET {base_url}/open/api/agent/v1/tool-tasks/{taskId}` 直到终态，
返回 StandaloneToolCallResult。提交即返回终态时不再轮询。
"""

import copy
import json
import time

import requests


# 状态字段的取值暂未固定，按语义归类；未命中的状态按“仍在执行”处理
TERMINAL_SUCCESS_STATUSES = frozenset({"SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE", "FINISHED", "DONE", "OK"})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"FAILED", "FAILURE", "ERROR", "CANCELLED", "CANCELED", "TIMEOUT", "TIMED_OUT", "REJECTED", "EXPIRED"}
)

_STAGE_DONE = "DONE"
_STAGE_FAILED = "FAILED"
_STAGE_PENDING = "PENDING"

# 服务端业务错误码 -> 套版工作流协议错误码，值为 (协议码, 是否可重试)
_BUSINESS_ERROR_CODES = {
    "UNAUTHORIZED": ("WORKFLOW_FORBIDDEN", False),
    "AUTH_FAILED": ("WORKFLOW_FORBIDDEN", False),
    "SHARE_INVALID": ("WORKFLOW_FORBIDDEN", False),
    "TOOL_NOT_BOUND": ("WORKFLOW_UNAVAILABLE", False),
    "TOOL_NOT_FOUND": ("WORKFLOW_UNAVAILABLE", False),
    "TASK_NOT_FOUND": ("WORKFLOW_UNAVAILABLE", False),
    "INVALID_ARGUMENTS": ("PARAMETER_INVALID", False),
    "INVALID_PARAM": ("PARAMETER_INVALID", False),
    # 每用户每分钟有调用上限，批量场景下属于可恢复错误
    "RATE_LIMITED": ("WORKFLOW_EXECUTION_FAILED", True),
    "POINT_INSUFFICIENT": ("WORKFLOW_EXECUTION_FAILED", False),
}


def _map_business_error(code, retryable_hint=False):
    mapped = _BUSINESS_ERROR_CODES.get((code or "").strip().upper())
    if mapped is None:
        return "WORKFLOW_EXECUTION_FAILED", bool(retryable_hint)
    return mapped[0], mapped[1] or bool(retryable_hint)


class ToolTaskError(Exception):
    """工具任务调用失败，code 使用套版工作流协议的错误码"""

    def __init__(self, code, message, retryable=False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ToolTaskClient:
    SUBMIT_PATH = "/open/api/agent/v1/tool-tasks/submit"
    QUERY_PATH = "/open/api/agent/v1/tool-tasks/{task_id}"

    def __init__(
        self,
        base_url,
        headers=None,
        context=None,
        poll_interval=5,
        max_poll_time=600,
        timeout=60,
        log_prefix="[ToolTaskClient]",
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.headers = dict(headers or {})
        # 提交体中除 requestId / toolCall 之外的上下文由调用方整块透传
        self.context = copy.deepcopy(context or {})
        self.poll_interval = max(1, int(poll_interval or 5))
        self.max_poll_time = max(1, int(max_poll_time or 600))
        self.timeout = max(5, int(timeout or 60))
        self.log_prefix = log_prefix

    def call_tool(self, tool_name, arguments, request_id):
        """提交并等待工具任务完成，返回 StandaloneToolCallResult"""
        body = copy.deepcopy(self.context)
        body["requestId"] = request_id
        body["toolCall"] = {"name": tool_name, "arguments": arguments, "id": request_id}

        print(f"{self.log_prefix} 提交工具任务: {tool_name} (requestId: {request_id})")
        result = self._request("POST", self.SUBMIT_PATH, json_body=body)

        stage = self._stage(result)
        if stage == _STAGE_DONE:
            return result
        if stage == _STAGE_FAILED:
            raise self._result_error(result)

        task_id = result.get("taskId")
        if not task_id:
            raise ToolTaskError("WORKFLOW_EXECUTION_FAILED", f"提交工具任务未返回 taskId: {_snippet(result)}")
        return self._poll(task_id)

    def _poll(self, task_id):
        path = self.QUERY_PATH.format(task_id=task_id)
        deadline = time.time() + self.max_poll_time
        attempt = 0

        while True:
            attempt += 1
            try:
                result = self._request("GET", path)
            except ToolTaskError as error:
                # 查询接口的瞬时故障不算任务失败，超时前继续重试
                if not error.retryable or time.time() >= deadline:
                    raise
                print(f"{self.log_prefix} 第 {attempt} 次查询失败，稍后重试: {error.message}")
            else:
                stage = self._stage(result)
                if stage == _STAGE_DONE:
                    print(f"{self.log_prefix} 任务完成: {task_id} (轮询 {attempt} 次)")
                    return result
                if stage == _STAGE_FAILED:
                    raise self._result_error(result)
                print(f"{self.log_prefix} 任务进行中: {task_id} status={result.get('status')!r}")

            if time.time() >= deadline:
                raise ToolTaskError(
                    "WORKFLOW_EXECUTION_FAILED",
                    f"工具任务轮询超时（{self.max_poll_time} 秒未到终态）: {task_id}",
                )
            time.sleep(self.poll_interval)

    def _request(self, method, path, json_body=None):
        if not self.base_url:
            raise ToolTaskError("WORKFLOW_UNAVAILABLE", "未配置工具任务服务地址")

        url = self.base_url + path
        # 请求上下文随日志与错误信息一起给出，便于对照排查（headers 含 token，不打印）
        where = f"{method} {url}"
        if json_body is not None:
            print(f"{self.log_prefix} 请求 {where} body={_snippet(json_body, 1000)}")
        else:
            print(f"{self.log_prefix} 请求 {where}")

        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
            else:
                response = requests.post(url, headers=self.headers, json=json_body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ToolTaskError("WORKFLOW_EXECUTION_FAILED", f"工具任务请求失败: {where} {exc}", retryable=True)

        print(f"{self.log_prefix} 响应 {where} -> {response.status_code} {_snippet(response.text, 1000)}")

        if response.status_code >= 400:
            raise ToolTaskError(
                self._http_error_code(response.status_code),
                f"工具任务请求失败: {where} -> {response.status_code} {response.reason}, "
                f"响应: {_snippet(response.text)}",
                retryable=response.status_code >= 500,
            )

        try:
            envelope = response.json()
        except ValueError:
            raise ToolTaskError(
                "WORKFLOW_EXECUTION_FAILED",
                f"工具任务返回非 JSON 内容: {where} -> {_snippet(response.text)}",
                retryable=True,
            )

        if not isinstance(envelope, dict):
            raise ToolTaskError(
                "WORKFLOW_EXECUTION_FAILED", f"工具任务返回结构不支持: {where} -> {_snippet(envelope)}"
            )

        if envelope.get("success") is False:
            code, retryable = _map_business_error(envelope.get("code"))
            raise ToolTaskError(
                code,
                f"工具任务返回失败: {envelope.get('message') or envelope.get('code') or '未提供 message'}"
                f"（{where}, code: {envelope.get('code')}"
                + (f", 提交体: {_snippet(json_body, 400)}" if json_body is not None else "")
                + f", 响应: {_snippet(envelope, 400)}）",
                retryable=retryable,
            )

        result = envelope.get("data")
        if not isinstance(result, dict):
            raise ToolTaskError(
                "WORKFLOW_EXECUTION_FAILED", f"工具任务未返回 data: {where} -> {_snippet(envelope)}"
            )
        return result

    @staticmethod
    def _http_error_code(status_code):
        if status_code in (401, 403):
            return "WORKFLOW_FORBIDDEN"
        if status_code == 404:
            return "WORKFLOW_UNAVAILABLE"
        return "WORKFLOW_EXECUTION_FAILED"

    @staticmethod
    def _stage(result):
        status = (result.get("status") or "").strip().upper()
        if status in TERMINAL_SUCCESS_STATUSES:
            return _STAGE_DONE
        if status in TERMINAL_FAILURE_STATUSES:
            return _STAGE_FAILED
        if status:
            return _STAGE_PENDING

        # 没有 status 时按 success 与结果体判断，兼容同步返回
        if result.get("success") is False or result.get("error"):
            return _STAGE_FAILED
        if result.get("success") is True and result.get("data"):
            return _STAGE_DONE
        return _STAGE_PENDING

    @staticmethod
    def _result_error(result):
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        message = error.get("message") or f"工具任务执行失败: status={result.get('status')!r}"
        code, retryable = _map_business_error(error.get("code"), bool(error.get("retryable")))
        return ToolTaskError(
            code,
            f"{message}（taskId: {result.get('taskId')}, code: {error.get('code')}, "
            f"结果: {_snippet(result, 400)}）",
            retryable=retryable,
        )


def _snippet(value, limit=500):
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit] + ("...（已截断）" if len(text) > limit else "")
