import os
import json
import nltk
import pymorphy2


db_fileName = "./data_cl.json"

def add_data(data):
    import pathlib
    path = pathlib.Path(db_fileName)
    content = []
    if path.exists():
        with open(db_fileName, "r", encoding="UTF8") as file:
            jsoncontent = file.read()
        content = json.loads(jsoncontent)
        content.append(data)
        jsonstring = json.dumps(content, ensure_ascii=False)
        with open(db_fileName, "w", encoding="UTF8") as file:
            file.write(jsonstring)
    else:
        content.append(data)
        jsonstring = json.dumps(content, ensure_ascii=False)
        with open(db_fileName, "w", encoding="UTF8") as file:
            file.write(jsonstring)
    return content


def load_db():
    import pathlib
    path = pathlib.Path(db_fileName)
    if path.exists():
        with open(db_fileName, "r", encoding="UTF8") as file:
            jsoncontent = file.read()
        content = json.loads(jsoncontent)
        return content
    else:
        return [{}]
    

def clear_db():
    import pathlib
    path = pathlib.Path(db_fileName)
    if path.exists():
        os.remove(db_fileName)


def data_proc(filename):
    with open("./uploads/"+filename+".json", "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    text = ""
    count_messages = len(messages)
    print(count_messages)
    texts = []
    for m in messages:
        text = m['text']
        # texts.append(remove_all_mas(text))
    num = 0
    # ltexts = d2lemmatize(texts)
    proc_messages = []
    for m in messages:
        line = {}
        line['date'] = m['date']
        text = m['text']
        line['text'] = text
        # line['remove_all'] = remove_all(text)
        # print(str(texts[num]))
        # print(text)
        str1 = ""
        # for item in ltexts[num]:
        #     if len(str(item)) > 3:
        #         str1 += item
        line['normal_form'] = str(str1).strip()
        # line['get_normal_form'] = get_normal_form(remove_all(data))
        # line['Rake_Summarizer'] = Rake_Summarizer(data)
        # line['YakeSummarizer'] = YakeSummarizer(data)
        line['message_id'] = m['message_id']
        line['user_id'] = m['user_id']
        line['reply_message_id'] = m['reply_message_id']
        proc_messages.append(line)
        print(f"{num} / {count_messages}")
        num += 1

    jsonstring = json.dumps(proc_messages, ensure_ascii=False)
    # print(jsonstring)
    name = filename.split(".")[0]
    with open(f"./uploads/{name}_proc.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)
    return proc_messages


def get_pattern(text):
    line = {}
    # line['text'] = text.strip()
    # line['remove_all'] = remove_all(text).strip()
    # line['normal_form'] = get_normal_form(remove_all(text)).strip()
    # line['Rake_Summarizer'] = Rake_Summarizer(text).strip()
    # line['YakeSummarizer'] = YakeSummarizer(text).strip()
    # line['BERT_Summarizer'] = BERT_Summarizer(text).strip()
    return line


def get_normal_form_mas(words):
    morph = pymorphy2.MorphAnalyzer()
    result = []
    for word in words.split():
        p = morph.parse(word)[0]
        result.append(p.normal_form)
    return result


def get_normal_form(words):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(words)[0]
    return p.normal_form


def load_data(filename='data.txt'):
    with open(filename, "r", encoding='utf-8') as file:
        data = file.read()
    return data

def remove_from_patterns(text, pattern):
    str2 = ''
    for c in text:
        if c not in pattern:
            str2 = str2 + c
    return str2

def display(text):
    print(text) 
    print("--------------------------------")

def remove_paragraf_and_toLower(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = ' '.join([k for k in text.split(" ") if k])
    return text


def get_RAKE(text):
    from rake_nltk import Metric, Rake
    r = Rake(language="russian")
    r.extract_keywords_from_text(text)
    numOfKeywords = 20
    keywords = r.get_ranked_phrases()[:numOfKeywords]
    return keywords


def get_YAKE(text):
    import yake
    language = "ru"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords


def get_KeyBERT(text):
    from keybert import KeyBERT
    kw_model = KeyBERT()
    # keywords = kw_model.extract_keywords(doc)
    numOfKeywords = 20
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english',
                            use_maxsum=True, nr_candidates=20, top_n=numOfKeywords)
    return keywords



def nltk_download():
    nltk.download('stopwords')
    nltk.download('punkt')
    
    

def find_cl(filename):
    
    proc_messages = data_proc(filename)
    data_cl = load_db()
    ae_messages = []

    def calc_intersection_one(text1, text2):
        count = 0
        for item1 in text1.split():
            for item2 in text2.split():
                if item1 == item2:
                    count += 1
        return count

    def calc_intersection_all(text1, l2):
        max_counts = 0
        for item in l2:
            current_counts = calc_intersection_one(text1, item['normal_form'])
            if current_counts > max_counts:
                max_counts = current_counts
        return max_counts
    counts = []
    for m in proc_messages:
        intersect = calc_intersection_all(m['normal_form'], data_cl)
        counts.append(intersect)
    max_counts=max(counts)
    indices = [i for i, x in enumerate(counts) if x == max_counts]
    print(max(counts))
    print(indices)
    print(len(indices))
    for ind in indices:
        m = proc_messages[ind]
        if len(m['text'])>30:
            line = {}
            line['text'] = m['text']
            line['date'] = m['date']
            line['remove_all'] =  m['remove_all']
            line['normal_form'] =  m['normal_form']
            line['message_id'] = m['message_id']
            line['user_id'] = m['user_id']
            line['reply_message_id'] = m['reply_message_id']
        ae_messages.append(line)   
    jsonstring = json.dumps(ae_messages, ensure_ascii=False)
    name = filename.split(".")[0]
    with open(f"./uploads/{name}_ae.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)
    return jsonstring   





if __name__ == '__main__':
    nltk_download()
    data = "«Два самых важных дня в твоей жизни: день, когда ты появился на свет, и день, когда ты понял зачем!». — Марк Твен"
    # # t = get_normal_form(remove_all(data))
    # t = get_pattern(data)
    # print(t)

    # t = remove_all(data)
    # print("remove_all")
    # print(t)
    # t = get_normal_form(remove_all(data))
    # print("norm")
    # print(t)
    # t = Rake_Summarizer(data)
    # print("Rake_Summarizer")
    # print(t)
    # # t = BERT_Summarizer(data)
    # # print("BERT_Summarizer")
    # # print(t)
    # t = YakeSummarizer(data)
    # print("YakeSummarizer")
    # print(t)
    # data_proc("d:/ml/chat/andromedica.json")
    filename="d:/ml/chat/andromedica.json"
    # ae = find_ae(filename)
    # print(ae)