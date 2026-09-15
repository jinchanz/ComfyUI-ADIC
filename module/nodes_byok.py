"""AIGC BYOK 密钥解密节点。

把工作流里的 ``encryptedAk`` 解成明文 API Key 输出，供任意需要 ak 的节点连线使用，
这样新增节点不必各自处理明文/密文兼容。

解密密钥可通过节点参数动态配置；未提供或留空时依次回退到
``AIGC_BYOK_ENCRYPTION_KEY`` 环境变量和内置默认值。
"""

import os

from .byok_crypto import AK_MODES, KEY_ENV_NAME, resolve_ak


DEFAULT_ENCRYPTION_KEY = "3c54c0245384676b9e2cb7c2565d82d974914cbb8a70a57199b3795ce7cdd6f5"


class AigcByokDecryptNode:
    """输出解密后的 API Key（STRING），可直接连到其他节点的 auth_token / api_key 输入。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encryptedAk": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "BYOK 密文，或用户明文 ak（auto 模式下原样透传）",
                    },
                )
            },
            "optional": {
                # 保持 ak_mode 在前，避免旧 workflow 的 widget_values 错位成 encryption_key
                "ak_mode": (
                    AK_MODES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: 自动判断是否需要解密，明文原样透传；"
                            "plaintext: 完全不解密；"
                            "encrypted: 强制解密，密钥或载荷有问题时直接报错"
                        ),
                    },
                ),
                "encryption_key": (
                    "STRING",
                    {
                        "default": DEFAULT_ENCRYPTION_KEY,
                        "multiline": False,
                        "tooltip": (
                            "AES-256-GCM 解密密钥（64 位十六进制）。"
                            f"留空时依次读取环境变量 {KEY_ENV_NAME} 和内置默认值；"
                            "该值会保存在工作流 JSON 中"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "decrypt"
    CATEGORY = "Agent/BYOK"
    OUTPUT_NODE = False

    def decrypt(
        self,
        encryptedAk: str,
        ak_mode: str = "auto",
        encryption_key: str = DEFAULT_ENCRYPTION_KEY,
    ):
        # 兼容曾短暂发布过的错误 widget 顺序：当旧 workflow 把密钥映射到
        # ak_mode、把模式映射到 encryption_key 时，自动交换回来。
        mode_value = (ak_mode or "auto").strip()
        key_value = (encryption_key or "").strip()
        if mode_value not in AK_MODES and key_value in AK_MODES:
            mode_value, key_value = key_value, mode_value

        # 不要打印 encryptedAk、共享密钥或解密后的明文。
        # 回退顺序：节点参数 -> 环境变量 -> 内置默认值。
        key_hex = (
            key_value
            or (os.getenv(KEY_ENV_NAME) or "").strip()
            or DEFAULT_ENCRYPTION_KEY
        )
        return (
            resolve_ak(
                encryptedAk,
                mode_value,
                log_prefix="AigcByokDecrypt",
                key_hex=key_hex,
            ),
        )
