import random

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


def generate_sentence():
    if random.random() < 0.5: 
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        o = random.choice(OBJECTS)
        return f"{s} {v} {o}."
    else:
        adj = random.choice(ADJECTIVES)
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        o = random.choice(OBJECTS)

        s_parts = s.split(' ', 1) 
        s = f"{s_parts[0]} {adj} {s_parts[1]}"
        
        return f"{s} {v} {o}."

sentences = set()
print("开始生成句子...")

while len(sentences) < 11000:
    sentences.add(generate_sentence())

with open("sentence.json", "w", encoding="utf-8") as f:
    f.write("[\n")
    s_max = -1
    for s in sentences:
        s_max = max(s_max, len(s))
        f.write('    "' + s + '",\n')
    print(f"最长句子长度: {s_max} 字符")
    f.write("]\n")
        
print(f"成功生成 11000 条不重复的句子到 sentence.json")