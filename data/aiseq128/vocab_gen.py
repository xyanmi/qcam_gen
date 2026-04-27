import json
import string # 导入 string 模块以方便地获取 a-z

# 1. 定义你的字符集
# 'endl' 是一个特殊标记，代表句子结束或换行
special_tokens = {' ': 0, ',': 27, '.': 28, '?': 29, '!': 30, 'endl': 31}
chars = string.ascii_lowercase # 'a' 到 'z'

# 2. 创建字符到整数的映射 (char -> token_id)
char_to_token = {}
char_to_token.update(special_tokens)

# a-z 对应 1-26
for i, char in enumerate(chars):
    char_to_token[char] = i + 1

# 3. 创建整数到字符的映射 (token_id -> char)
token_to_char = {token_id: char for char, token_id in char_to_token.items()}


# 4. 将两个映射一起保存到一个 JSON 文件中
vocab = {
    'char_to_token': char_to_token,
    'token_to_char': token_to_char
}

# 5. 写入文件
filepath = 'vocab.json'
with open(filepath, 'w', encoding='utf-8') as f:
    # indent=2 让 JSON 文件格式更美观，易于阅读
    json.dump(vocab, f, indent=2)

print(f"词汇表已成功生成并保存到: {filepath}")
print("char_to_token 内容示例:", list(char_to_token.items())[:5]) # 打印前5个看看
print("token_to_char 内容示例:", list(token_to_char.items())[:5])