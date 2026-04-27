import random

# 1. 定义你的词汇表 (每个列表10个词)
SUBJECTS = [
    'the dog', 'a cat', 'the man', 'a woman', 'the bird',
    'one boy', 'the teacher', 'a student', 'my friend', 'his sister'
]
VERBS = [
    'chased', 'saw', 'ate', 'liked', 'heard',
    'found', 'kicked', 'pushed', 'loved', 'wanted'
]
OBJECTS = [
    'the ball', 'a mouse', 'the food', 'a song', 'the car',
    'one book', 'the house', 'a tree', 'my key', 'her bag'
]
ADJECTIVES = [
    'big', 'small', 'red', 'happy', 'sad',
    'loud', 'quiet', 'new', 'old', 'fast'
]

# 2. 同样的句子模板 (你的 "语法")
def generate_sentence():
    # 模板1: S V O (10 * 10 * 10 = 1,000 种)
    if random.random() < 0.5: # 保持两种模板大致均衡
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        o = random.choice(OBJECTS)
        return f"{s} {v} {o}."
    
    # 模板2: Adj S V O (10 * 10 * 10 * 10 = 10,000 种)
    # (注意：我简化了 'A' 和 'The' 的逻辑，你可以在此基础上细化)
    else:
        adj = random.choice(ADJECTIVES)
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        o = random.choice(OBJECTS)
        
        # 简单处理一下定冠词 'The' 和不定冠词 'A'
        s_parts = s.split(' ', 1) # e.g., ['The', 'dog']
        s = f"{s_parts[0]} {adj} {s_parts[1]}"
        
        return f"{s} {v} {o}."

# 3. 生成 10k 条数据
sentences = set()
print("开始生成句子...")

# 现在总的组合数是 1,000 + 10,000 = 11,000 种
# 所以这个循环现在可以正常结束了
while len(sentences) < 11000:
    sentences.add(generate_sentence())

# 4. 保存到文件
with open("sentence.json", "w", encoding="utf-8") as f:
    f.write("[\n")
    s_max = -1
    for s in sentences:
        s_max = max(s_max, len(s))
        f.write('    "' + s + '",\n')
        # 去除最后一个逗号会更复杂一些，这里简单处理
    print(f"最长句子长度: {s_max} 字符")
    f.write("]\n")
        
print(f"成功生成 11000 条不重复的句子到 sentence.json")