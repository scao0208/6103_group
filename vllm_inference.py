from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
import re
import string
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument("--model_path", type=str, default="./model/Qwen1.5-4B")
parser.add_argument("--adapter_path", type=str, default="./4b_lora_ckpt_1epoch")
parser.add_argument("--data_path", type=str, default="./test_data/")
parser.add_argument("--data_name", type=str, default="mit-movie")
parser.add_argument("--output_path", type=str, default="./output/results")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path=args.model_path
adapter_path=args.adapter_path
data_path=args.data_path
data_name=args.data_name
output_path=args.output_path


role_mapping = {"human":"user", "gpt":"assistant"}

def convert_data(example: list, role_mapping = role_mapping) -> list:
    res = []
    for text in example["conversations"]:
        converted_data = {}
        converted_data['role'] = role_mapping[text['from']]
        converted_data['content'] = text['value']
        res.append(converted_data)
    return res

def format_input(tokenizer, chat):
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False)
    input_chat = formatted_chat.split("assistant")
    input_chat[-1] = "\n"
    input_chat = "assistant".join(input_chat)
    return input_chat

def get_response(responses):
    responses = [r.split('assistant')[-1].strip().rstrip('<|im_end|>') for r in responses]
    return responses

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parsers(text):
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([normalize_answer(element) for element in item])
            else:
                item = normalize_answer(item)
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []

class NEREvaluator:
    def evaluate(self, preds: list, golds: list):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for pred, gold in zip(preds, golds):
            gold_tuples = parsers(gold)
            pred_tuples = parsers(pred)
            for t in pred_tuples:
                if t in gold_tuples:
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
        }


def inference():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # lora_path = "/kaggle/input/qwen1-5-4b-base-ckpt/4b_lora_ckpt_1epoch"

    llm = LLM(
        model="Qwen/Qwen1.5-4B-Chat", 
        #dtype="half",
        dtype="auto",
        enforce_eager=True,
        gpu_memory_utilization=0.99,
        swap_space=8, # The size (GiB) of CPU memory per GPU to use as swap space.
        enable_lora=True,
        #tensor_parallel_size=2,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        max_tokens=64,
        use_beam_search=True,
        best_of=3, #beam width
        temperature=0.2,
        repetition_penalty=1.5,
        top_k=50,
        top_p=0.8,
        stop_token_ids=[151645],
        seed = 42,
    )

    with open(os.path.join(data_path, f"{data_name}.json"), "r") as f:
        examples = json.load(f)
    print(tokenizer.apply_chat_template(convert_data(examples[0]), tokenize=False))

    converted_examples =  [convert_data(chat) for chat in examples]
    golds = [example[-1]['content'] for example in converted_examples]
    prompts = [format_input(tokenizer, example) for example in converted_examples]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("ner_adapter", 1, adapter_path)
    )

    responses = get_response([output.outputs[0].text for output in responses])


    with open(os.path.join(output_path, f"{data_name}_responses.json"), "w") as json_file:
        json.dump(responses, json_file, ensure_ascii=False)

    with open(f"./output/{data_name}_outputs_raw.json", "w") as json_file:
        json.dump(outputs, json_file, ensure_ascii=False)

    evaluator = NEREvaluator()
    eval_result = evaluator.evaluate(responses, golds)
    print(f"DATA NAME: {data_name}")
    print(f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}')

    eval_result['data_name'] = data_name
    eval_result['model_name'] = model_path.split("/")[-1]
    with open(os.path.join(output_path, f"{data_name}_eval_results.json", "w+")) as json_file:
        json.dump(eval_result, json_file, ensure_ascii=False, indent=4)

inference()