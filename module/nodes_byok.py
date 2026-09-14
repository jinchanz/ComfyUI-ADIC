"""AIGC BYOK 密钥解密节点。

把工作流里的 ``encryptedAk`` 解成明文 API Key 输出，供任意需要 ak 的节点连线使用，
这样新增节点不必各自处理明文/密文兼容。

共享密钥只从 ``AIGC_BYOK_ENCRYPTION_KEY`` 环境变量读取，不作为工作流入参。
"""

from .byok_crypto import AK_MODES, KEY_ENV_NAME, resolve_ak


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
                "ak_mode": (
                    AK_MODES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: 自动判断是否需要解密，明文原样透传；"
                            "plaintext: 完全不解密；"
                            f"encrypted: 强制解密，密钥({KEY_ENV_NAME})或载荷有问题时直接报错"
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

    def decrypt(self, encryptedAk: str, ak_mode: str = "auto"):
        # 不要打印 encryptedAk、共享密钥或解密后的明文
        return (resolve_ak(encryptedAk, ak_mode, log_prefix="AigcByokDecrypt"),)
