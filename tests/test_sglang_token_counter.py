from shopping_grpo.environment.context import SGLangChatTokenCounter, SGLangTextTokenCounter

def test_sglang_chat_tokenizer_contract():
    calls=[]
    def transport(url,payload,headers,timeout):
        calls.append((url,payload)); return {'count':123,'tokens':[1,2]}
    counter=SGLangChatTokenCounter('shopping-agent','http://localhost:8000/v1','EMPTY',transport=transport)
    messages=[{'role':'user','content':'buy a cup'}]; tools=[{'type':'function','function':{'name':'search'}}]
    assert counter(messages,tools)==123
    assert calls[0][0]=='http://localhost:8000/v1/tokenize'
    assert calls[0][1]['messages']==messages and calls[0][1]['tools']==tools

def test_sglang_text_tokenizer_contract():
    def transport(url,payload,headers,timeout):
        assert url.endswith('/v1/tokenize')
        assert payload['prompt']=='hello' and payload['add_special_tokens'] is False
        return {'count':1}
    assert SGLangTextTokenCounter('agent','http://localhost:8000','EMPTY',transport=transport)('hello')==1
