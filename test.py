from torch.cuda.amp import autocast
import json
from transformers import pipeline

pipe = pipeline("question-answering", model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# import ipdb
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"

#### load model
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
# model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

# tokenizer = AutoTokenizer.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
# model = AutoModelForCausalLM.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")



file_path = '/home/lyli/ssd/AI4security/LLM4mmc/table_info.json'
with open(file_path, 'r', encoding='utf-8') as file:
    rules = file.read()
log_path="/home/lyli/ssd/AI4security/camflow 2.log"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

model.generation_config.pad_token_id = tokenizer.pad_token_id
model=model.to(device)
output_f_name='output2'
with open(log_path, 'r', encoding='utf-8') as file:
    # 遍历文件中的每一行
    check=0
    for line in file:
        # 去除行尾的换行符（如果有的话）
        log_line = line.strip()
        
        # 打印当前行或者进行其他处理
        # check+=1
        # if check<2:
        #     continue
        print(log_line)
        # logfile = '''{"type": "Entity", "id": "EAAAAAAAABQFFQAAAAAAAAsAAABu50AbAQAAAAAAAAA=", "annotations": {"object_id": "5381", "object_type": "machine", "boot_id": 11, "cf:machine_id": "cf:457238382", "version": 1, "cf:date": "2024:09:11T03:29:35", "cf:taint": "0", "cf:jiffies": "0", "cf:epoch": 0, "u_sysname": "Linux", "u_nodename": "fedora", "u_release": "6.0.5-200.camflow.fc36.x86_64", "u_version": "#1 SMP PREEMPT_DYNAMIC Mon Oct 31 15:11:11 UTC 2022", "u_machine": "x86_64", "u_domainname": "(none)", "k_version": "0.9.0", "l_version": "v0.5.5", "l_commit": "46255580589d1d5c751cebe960daedc4c5724b27"}}'''



        # question=f"Read the table and understand the rules for each id, which consist of description and subfeatures. \
        # 	Next, read the camflow log databelow and determine which id it mostlikely belongs to according to the rules in the table. That means, only ONE ID(format like T1234) you should answer.\
        # 	{logfile}"

        question=f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\
        Question: {rules}\
        Read the table of rules above and understand the rules for each id, which consist of description and subfeatures. \
        Next, read the camflow log data below and determine which id it mostlikely belongs to according to the rules in the table. That means, only ONE ID(format like T1234) you should answer.\
        {log_line}\
        You MUST decide to return ONE id with the highest probability, NO other text SHOULD be included.<|eot_id|>"


        inputs = tokenizer(question, return_tensors="pt").to(device)
        prompt_len=inputs.input_ids.shape[1]
        # with autocast(device):
        try:
            generate_ids = model.generate(inputs.input_ids,attention_mask=inputs.attention_mask, max_new_tokens=50)
        except Exception as e:
            print("ERROR ",e)
            answer="NULL"
            data = {
                'log': log_line,
                'label': answer
                }

                # print(answer)
            
            output_file_path = f'{output_f_name}.json'
            with open(output_file_path, 'a+', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
                file.write('\n')
            continue
        answer = tokenizer.batch_decode(generate_ids[:,prompt_len+1:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # ipdb.set_trace()
        data = {
        'log': log_line,
        'label': answer
        }
        print(answer)
        
        
        output_file_path = f'{output_f_name}.json'
        with open(output_file_path, 'a+', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.write('\n')
            


    # answer_start_index = outputs.start_logits.argmax()
    # answer_end_index = outputs.end_logits.argmax()

    # predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    # print("------",tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))
