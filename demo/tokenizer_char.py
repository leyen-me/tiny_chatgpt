# =============================
# 字符级 Tokenizer（最简单版）
# =============================

class CharTokenizer:
    def __init__(self, text):
        """
        text: 一整段字符串，用来统计有哪些字符
        """
        # 去重，排序
        chars = sorted(list(set(text)))

        # 建立两个“字典”
        self.char_to_id = {}
        self.id_to_char = {}

        for i, ch in enumerate(chars):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch

        self.vocab_size = len(chars)

    def encode(self, s):
        """
        把字符串 → 数字列表
        """
        return [self.char_to_id[ch] for ch in s]

    def decode(self, ids):
        """
        把数字列表 → 字符串
        """
        return "".join(self.id_to_char[i] for i in ids)


# =============================
# 测试
# =============================
if __name__ == "__main__":
    text = "你好你好世界"

    tokenizer = CharTokenizer(text)

    print("词表大小:", tokenizer.vocab_size)
    print("字符 → 数字:", tokenizer.char_to_id)

    encoded = tokenizer.encode("你好世界")
    print("编码结果:", encoded)

    decoded = tokenizer.decode(encoded)
    print("解码结果:", decoded)
