from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
import nltk
from nltk import sent_tokenize
nltk.download('punkt')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from setfit import SetFitModel, SetFitTrainer
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
import datasets
from datasets import Dataset, DatasetDict
from sklearn.metrics.pairwise import cosine_similarity
import trafilatura
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer)
from transformers.pipelines import AggregationStrategy
import numpy as np
import pickle
import re
import boto3
import json

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model_ner = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
model_sim = SentenceTransformer('bert-base-nli-mean-tokens')
nlp = pipeline('ner', model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple")
model = SetFitModel.from_pretrained("kowshik/upsc-classification-model-v1")
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
map_ = {'agriculture': 0,'culture': 1,'defence': 2,'economy': 3,'environment': 4,'geography': 5,'governance': 6,
'health': 7,'history': 8,'international relations': 9,'polity': 10,'science&technology': 11,'society': 12,'sports': 13}
inv_map = {v: k for k, v in map_.items()}

__TableName__ = 'prod1_app_data'
client  = boto3.client('dynamodb',region_name = 'ap-south-1')
DB  = boto3.resource('dynamodb',region_name = 'ap-south-1')
table = DB.Table(__TableName__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def get_summary(url,pid,text,summarizer,th=120):
    if len(text) > 1200:
        summary = summarizer(text, max_length= th, min_length=120, do_sample=False)[0]['summary_text']
        flag = 3 + pid*10
        data = summary
        item_summary = create_item(url,flag, data)
        response = table.put_item(Item  = item_summary)
        return(summary)
    else:
        flag = 3 + pid*10
        data = text
        item_summary = create_item(url,flag, data)
        response = table.put_item(Item  = item_summary)
        return(text)

f = open('rm_model.pkl', 'rb')
clf = pickle.load(f)
f.close()

def create_item(url,flag, data):
    '''  
    flag = 0 > text
    flag = 1 > sentence
    flag = 2 > key Phrase 
    flag = 3 > summary 
    '''
    item = {
        'url': url,
        'flag':flag,
        'data': data,
    }
    return(item)


def get_data_url(url):
    downloaded = trafilatura.fetch_url(url)
    text_original = trafilatura.extract(downloaded)
    text_extracted = text_original.replace('\n',' ')
    flag = 0
    data = text_original
    item_complete = create_item(url,flag, data)
    response = table.put_item(Item  = item_complete)
    return(text_extracted, text_original)

def get_label(word,model_sim):
    labels = ['Environment','Geography','International Relations',
    'Polity','Governance','Health','Society','Economy','Science&Technology','Agriculture','sports']
    labels = [i.lower() for i in labels]
    embeddings_tags = model_sim.encode(labels)
    embeddings_key = model_sim.encode(word)
    probs = cosine_similarity([embeddings_key],embeddings_tags)
    label_index = np.argmax(probs)
    return(labels[label_index])

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])
extractor = KeyphraseExtractionPipeline(model=model_name)

def get_keywords_text(url,pid, sentences, extractor):
    keywords_ = []
    for te in sentences:
        keywords_ = keywords_+ list(extractor(te))
    keywords_unq = np.unique(keywords_)
    flag = 2 +pid*10
    data = json.dumps(list(keywords_unq))
    item_key = create_item(url,  flag, data,)
    response = table.put_item(Item  = item_key)
    return(keywords_unq)

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    sentences_final = []
    for sent in sentences:
        if len(sent)>= 120:
            if ',' in sent:
                pos_comma = np.array([i for i in range(len(sent)) if sent.startswith(',', i)])
                to_split = np.argmin(np.abs(pos_comma - (len(sent) - pos_comma)))
                if (pos_comma[to_split])<= 50 or (len(sent)-pos_comma[to_split] <= 50):
                    sentences_final.append(sent)
                else:
                    sentences_final.append(sent[:pos_comma[to_split]])
                    sentences_final.append(sent[pos_comma[to_split] +1:])
            else:
                sentences_final.append(sent)
        else:
            sentences_final.append(sent)

    return sentences_final


def get_sentence_labels(url, pid, sentences, clf, model, sent_no=4):
    prediction_probas = clf.predict_proba(model.predict_proba(sentences))
    df = pd.DataFrame()
    df['sentences'] = sentences
    df['labels_1'] = np.argmax(prediction_probas,axis=1)
    df['prob_1'] = np.max(prediction_probas,axis=1)
    df['label_text_1'] = df['labels_1'].replace(inv_map)
    df['labels_2'] = [[list(p).index(i) for i in sorted(p, reverse=True)][1]  for p in prediction_probas]
    df['prob_2'] = [p[[list(p).index(i) for i in sorted(p, reverse=True)][1]]  for p in prediction_probas]
    df['label_text_2'] = df['labels_2'].replace(inv_map)
    df = df.sort_values('prob_1',ascending=False)
    labels = df[['sentences','label_text_1','label_text_2']][:sent_no]
    flag = 1  + pid*10
    data = json.dumps(labels.set_index('sentences').to_dict('index'))
    item_sentence = create_item(url, flag, data)
    response = table.put_item(Item  = item_sentence)

    return(labels)



def get_cuts(text, sentences_all):
    if len(text) > 3000:
        cumsum_ = np.cumsum([len(i) for i in sentences_all])
        chunks = np.round(len(text)/2500)
        cutoff_ = int(len(text)/chunks)
        cuts = [0]
        for i in np.arange(1,chunks):
            cutoff = cutoff_*i
            cut = np.argmin(np.abs(cumsum_ - cutoff))
            cuts.append(cut)
        cuts.append(len(sentences_all))

        sentences_chunks = []
        for c in range(0,len(cuts)-1):
            sentences_chunks.append(sentences_all[cuts[c]:cuts[c+1]])
        return(sentences_chunks)
    else:
        return([sentences_all])
