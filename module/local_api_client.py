from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type
from urllib.parse import urljoin

import asyncio
import time
import requests
from pydantic import BaseModel

# 与旧版 comfy_api_nodes client 保持一致的默认值
DEFAULT_TIMEOUT = 7200
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"


@dataclass
class ApiEndpoint:
    path: str
    method: HttpMethod
    request_model: Optional[Type[BaseModel]] = None
    response_model: Optional[Type[BaseModel]] = None


class SynchronousOperation:
    def __init__(
        self,
        api_base: str,
        endpoint: ApiEndpoint,
        request: Optional[BaseModel] = None,
        files: Optional[Any] = None,
        content_type: str = "application/json",
        auth_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.api_base = (api_base or "").rstrip("/")
        self.endpoint = endpoint
        self.request = request
        self.files = files
        self.content_type = content_type or "application/json"
        self.auth_kwargs = auth_kwargs or {}

    def _build_url(self) -> str:
        if not self.api_base:
            raise ValueError("api_base 不能为空")
        path = self.endpoint.path if self.endpoint.path.startswith("/") else f"/{self.endpoint.path}"
        return urljoin(f"{self.api_base}/", path.lstrip("/"))

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}

        auth_token = (
            self.auth_kwargs.get("auth_token")
            or self.auth_kwargs.get("api_key")
            or self.auth_kwargs.get("comfy_api_key")
        )
        if isinstance(auth_token, str) and auth_token.strip():
            headers["Authorization"] = f"Bearer {auth_token.strip()}"

        if self.files:
            return headers

        headers["Content-Type"] = self.content_type
        return headers

    def _request_payload(self) -> Dict[str, Any]:
        if self.request is None:
            return {}
        if hasattr(self.request, "model_dump"):
            return self.request.model_dump(exclude_none=True)
        return dict(self.request)

    def _execute_sync(self):
        url = self._build_url()
        method = self.endpoint.method
        headers = self._build_headers()
        payload = self._request_payload()

        timeout = self.auth_kwargs.get("timeout", DEFAULT_TIMEOUT)
        verify_ssl = self.auth_kwargs.get("verify", True)

        response = self._send_with_retries(method, url, headers, payload, timeout, verify_ssl)

        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text}

        response_model = self.endpoint.response_model
        if response_model:
            if hasattr(response_model, "model_validate"):
                return response_model.model_validate(data)
            return response_model(**data)

        return data

    def _send_once(self, method, url, headers, payload, timeout, verify_ssl):
        if method == HttpMethod.GET:
            return requests.get(url, headers=headers, params=payload, timeout=timeout, verify=verify_ssl)

        if method == HttpMethod.POST:
            if self.files:
                # 多段上传前需要把文件流指针复位，否则重试时会发出空文件
                self._rewind_files()
                return requests.post(
                    url,
                    headers=headers,
                    data=payload,
                    files=self.files,
                    timeout=timeout,
                    verify=verify_ssl,
                )
            return requests.post(url, headers=headers, json=payload, timeout=timeout, verify=verify_ssl)

        raise ValueError(f"不支持的HTTP方法: {method}")

    def _rewind_files(self) -> None:
        file_iter = self.files if isinstance(self.files, list) else self.files.items()
        for _, file_value in file_iter:
            stream = file_value[1] if isinstance(file_value, (tuple, list)) else file_value
            if hasattr(stream, "seek"):
                stream.seek(0)

    def _send_with_retries(self, method, url, headers, payload, timeout, verify_ssl):
        delay = DEFAULT_RETRY_DELAY
        last_error: Optional[Exception] = None

        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                response = self._send_once(method, url, headers, payload, timeout, verify_ssl)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    return response

                error = RuntimeError(
                    f"API 请求失败: {response.status_code} {response.reason} "
                    f"{method.value if hasattr(method, 'value') else method} {url}\n"
                    f"响应内容: {self._response_snippet(response)}"
                )
                # 4xx 属于确定性失败，重试没有意义
                if response.status_code < 500:
                    raise error
                last_error = error

            if attempt < DEFAULT_MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= DEFAULT_RETRY_BACKOFF

        raise last_error if last_error else RuntimeError(f"API 请求失败: {url}")

    @staticmethod
    def _response_snippet(response, limit: int = 2000) -> str:
        try:
            text = response.text or ""
        except Exception:
            return "<无法读取响应内容>"
        return text[:limit] + ("...（已截断）" if len(text) > limit else "")

    async def execute(self):
        return await asyncio.to_thread(self._execute_sync)
