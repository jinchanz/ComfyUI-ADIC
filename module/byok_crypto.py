"""Agent AIGC BYOK ``encryptedAk`` 解密工具。

与 Java ``AigcEncryptedAkCodec`` 保持兼容：

    Base64(12 字节 IV + AES-256-GCM 密文 + 16 字节 auth tag)

共享 32 字节 AES 密钥只从环境变量 ``AIGC_BYOK_ENCRYPTION_KEY`` 读取，
不接受工作流入参，避免密钥被持久化进 workflow JSON。

依赖: pip install cryptography
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Optional

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except ImportError:  # 缺少依赖时插件仍可加载，仅 BYOK 解密不可用
    InvalidTag = Exception  # type: ignore[assignment,misc]
    AESGCM = None  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False
    print("[BYOK] 警告: cryptography 库未安装，encryptedAk 解密不可用，请运行 'pip install cryptography' 安装")


IV_LENGTH = 12
TAG_LENGTH = 16
KEY_ENV_NAME = "AIGC_BYOK_ENCRYPTION_KEY"

# 合法密文载荷至少为 IV + tag + 1 字节明文
MIN_PAYLOAD_LENGTH = IV_LENGTH + TAG_LENGTH + 1

# ak 解析模式：自动判定 / 强制按明文 / 强制按密文
AK_MODES = ["auto", "plaintext", "encrypted"]


class EncryptedAkError(ValueError):
    """encryptedAk 无法被安全解密时抛出。"""


def _parse_key(key_hex: str) -> bytes:
    value = (key_hex or "").strip()
    if len(value) != 64:
        raise EncryptedAkError("AIGC BYOK encryption key must be 64 hex characters")
    try:
        key = bytes.fromhex(value)
    except ValueError as exc:
        raise EncryptedAkError("AIGC BYOK encryption key is not valid hexadecimal") from exc
    if len(key) != 32:
        raise EncryptedAkError("AIGC BYOK encryption key must decode to 32 bytes")
    return key


def is_key_configured() -> bool:
    """共享密钥是否已配置（不暴露密钥内容）。"""
    return bool((os.getenv(KEY_ENV_NAME) or "").strip())


def looks_like_encrypted_ak(value: str) -> bool:
    """粗筛：形态上是否可能是 encryptedAk 载荷。

    明文 ak（如百炼 ``sk-xxx``）含 Base64 字母表之外的字符，会在此直接被判为明文；
    只有严格 Base64 且解码长度足够的值才会进入真正的解密尝试。
    """
    text = (value or "").strip()
    if len(text) < 4 or len(text) % 4 != 0:
        return False
    try:
        payload = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(payload) >= MIN_PAYLOAD_LENGTH


def decrypt_encrypted_ak(encrypted_ak: str, key_hex: Optional[str] = None) -> str:
    """返回 ``encryptedAk`` 携带的明文 API Key。

    Args:
        encrypted_ak: Java ``AigcEncryptedAkCodec`` 产出的 Base64 载荷。
        key_hex: 可选的 64 位十六进制密钥。生产调用应省略该参数，
            由 ``AIGC_BYOK_ENCRYPTION_KEY`` 提供。

    Raises:
        EncryptedAkError: 配置、Base64、认证或 UTF-8 任一环节不合法。
            错误信息不包含密钥与明文。
    """
    value = (encrypted_ak or "").strip()
    if not value:
        raise EncryptedAkError("encryptedAk is empty")

    if not CRYPTO_AVAILABLE:
        raise EncryptedAkError("cryptography 库未安装，无法解密 encryptedAk")

    configured_key = key_hex if key_hex is not None else os.getenv(KEY_ENV_NAME, "")
    key = _parse_key(configured_key)

    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EncryptedAkError("encryptedAk is not valid Base64") from exc

    if len(payload) < MIN_PAYLOAD_LENGTH:
        raise EncryptedAkError("encryptedAk payload is too short")

    iv = payload[:IV_LENGTH]
    ciphertext_and_tag = payload[IV_LENGTH:]
    try:
        plaintext = AESGCM(key).decrypt(iv, ciphertext_and_tag, None)
    except InvalidTag as exc:
        raise EncryptedAkError(
            "encryptedAk authentication failed; check the shared key and payload"
        ) from exc
    except ValueError as exc:
        raise EncryptedAkError("encryptedAk payload is invalid") from exc

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EncryptedAkError("encryptedAk plaintext is not valid UTF-8") from exc


def resolve_ak(value: str, mode: str = "auto", log_prefix: str = "BYOK") -> str:
    """把节点入参里的 ak 归一成明文，供 Authorization: Bearer 使用。

    同一个入参既可以是用户明文 ak，也可以是 BYOK 密文：

    - ``auto``（默认）：形态像密文且能通过 GCM 认证时解密，否则原样当明文使用；
    - ``plaintext``：完全不解密，原样使用；
    - ``encrypted``：强制解密，失败即抛错（便于暴露密钥配置问题）。

    日志只输出判定结果，不输出 ak、密钥或明文本身。
    """
    text = (value or "").strip()
    if not text:
        return text

    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode == "plaintext":
        return text

    if normalized_mode == "encrypted":
        plaintext = decrypt_encrypted_ak(text)
        print(f"[{log_prefix}] encryptedAk 已解密（强制密文模式）")
        return plaintext

    # auto：先做形态粗筛，避免对明文 ak 做无意义的解密尝试
    if not looks_like_encrypted_ak(text):
        return text

    if not is_key_configured():
        print(f"[{log_prefix}] 入参形似 encryptedAk，但未配置 {KEY_ENV_NAME}，按明文使用")
        return text

    try:
        plaintext = decrypt_encrypted_ak(text)
    except EncryptedAkError as exc:
        # 认证失败无法区分“密钥不对”与“恰好像 Base64 的明文 ak”，
        # 保守按明文继续，并给出可排查的提示（不含敏感内容）
        print(f"[{log_prefix}] encryptedAk 解密未通过，按明文使用: {exc}")
        return text

    print(f"[{log_prefix}] encryptedAk 已解密")
    return plaintext
