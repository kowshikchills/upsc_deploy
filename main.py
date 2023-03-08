from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import boto3
from boto3.dynamodb.conditions import Key
from app_utils import *

app = FastAPI()
class UpscDeploy(BaseModel):
    url: str 

@app.post('/url')
async def update_upsc_db(upsc: UpscDeploy):
    data = upsc.dict()
    url = data['url']
    try: 
        text, text_act = get_data_url(url)
        if len(text) > 120:
            sentences_all = split_into_sentences(text)
            sentences_chunks = get_cuts(text, sentences_all)
            pid = 0
            for payload in sentences_chunks:
                sentence_labels = get_sentence_labels(url,pid,payload,clf,model,sent_no=4)
                sentence_keywords = list(set(payload) -  set(sentence_labels.sentences.values))
                keyphrases = get_keywords_text(url,pid,sentence_keywords,extractor)
                text = ' '.join(payload)
                summary = get_summary(url,pid,text, summarizer, th= min(int(len(text)/10),240))
                pid = pid+1
        response_flag = 'success'
    except:
        response_flag = 'Failed'

    return {
        'flag': response_flag
    }

@app.post('/fetch')
async def fetch_data_db(upsc: UpscDeploy):
    data = upsc.dict()
    url = data['url']
    __TableName__ = 'prod1_app_data'
    client  = boto3.client('dynamodb',region_name = 'ap-south-1')
    DB  = boto3.resource('dynamodb',region_name = 'ap-south-1')
    table = DB.Table(__TableName__)
    response = table.query(
    KeyConditionExpression = Key('url').eq(url))

    return {
        'response': response
    }