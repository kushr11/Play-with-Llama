import requests
import json
url_generate = "http://10.21.4.21:11434/api/generate"
def get_response(url, data):
    response = requests.post(url, json=data)
    response_dict = json.loads(response.text)
    response_content = response_dict["response"]
    return response_content

data = {   "model": "llama3.2",  
         "messages": [     { "role": "user", "content": "why is the sky blue?" }   ] }


res = get_response(url_generate,data)
print(res)