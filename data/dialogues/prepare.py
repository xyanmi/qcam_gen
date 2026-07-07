import csv
import re

def process_dailydialog_strict(input_txt, output_txt):
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz \n.,?'")
    processed_dialogues = []

    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterances = line.split('__eou__')
            
            clean_utterances = []
            for u in utterances:
                u = u.lower()
            
                u = u.replace('’', "'")  
                u = u.replace('。', '.')  
                u = u.replace('-', '')  
                
                u = re.sub(r"\b([a-z]+)\s+'\s+([smtd]|re|ll|ve)\b", r"\1'\2", u)
                u = re.sub(r"(?<![a-z])'|'(?![a-z])", "", u)
                u = re.sub(r'\b(th|st|nd|rd)\b', '', u)
                
                u = "".join([c for c in u if c in allowed_chars])
                
                u = re.sub(r'([.,?])([a-z])', r'\1 \2', u) 
                u = re.sub(r'\s+([.,?])', r'\1', u)        
                u = re.sub(r' +', ' ', u).strip()       
                
                if u:
                    clean_utterances.append(u)
            
            if clean_utterances:
                processed_dialogues.append("\n".join(clean_utterances))
    final_output = "\n\n".join(processed_dialogues)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(final_output)
    
    print("Finish!")

process_dailydialog_strict('dialogues_test.txt', 'training_test.txt')