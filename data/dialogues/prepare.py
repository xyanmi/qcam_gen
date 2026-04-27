import csv
import re

def process_dailydialog_strict(input_txt, output_txt):
# 严格的 32 字符白名单
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz \n.,?'")
    processed_dialogues = []

    with open(input_txt, 'r', encoding='utf-8') as f:
        # 逐行读取，每一行就是一个独立对话
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 使用 __eou__ 切分出每个人说的话
            utterances = line.split('__eou__')
            
            clean_utterances = []
            for u in utterances:
                u = u.lower()
                
                # 1. 基础字符替换
                u = u.replace('’', "'")  # 统一单引号格式
                u = u.replace('。', '.')  # 修复混入的中文句号
                u = u.replace('-', '')   # 删掉连字符 (push-ups -> pushups)
                
                # 2. 缝合被打散的缩写 (e.g., "let ' s" -> "let's")
                u = re.sub(r"\b([a-z]+)\s+'\s+([smtd]|re|ll|ve)\b", r"\1'\2", u)
                
                # 3. 剥离无用的引用单引号
                u = re.sub(r"(?<![a-z])'|'(?![a-z])", "", u)
                
                # 4. 清理数字残留的序数词后缀 (15th -> th -> 清除)
                u = re.sub(r'\b(th|st|nd|rd)\b', '', u)
                
                # 5. 核心过滤：只保留 32 个白名单字符
                u = "".join([c for c in u if c in allowed_chars])
                
                # 6. 标点排版修复
                u = re.sub(r'([.,?])([a-z])', r'\1 \2', u)  # 标点后加空格
                u = re.sub(r'\s+([.,?])', r'\1', u)        # 标点前去空格
                u = re.sub(r' +', ' ', u).strip()          # 合并多余空格
                
                if u:
                    clean_utterances.append(u)
            
            # 组装单个对话（内部单回车）
            if clean_utterances:
                processed_dialogues.append("\n".join(clean_utterances))

    # 组装所有对话（之间双回车）
    final_output = "\n\n".join(processed_dialogues)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(final_output)
    
    print("Finish!")

# 假设你的原始文件名为 raw_dialogues.txt

# Usage:
process_dailydialog_strict('dialogues_test.txt', 'training_test.txt')