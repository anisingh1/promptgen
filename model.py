import os
from openai import OpenAI
import timeout_decorator
from tqdm import tqdm
from utils import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
if torch.cuda.is_available():
    from vllm import LLM, SamplingParams
import traceback
import math
import ray
import json
import transformers


models = [
    #"WizardLM/WizardCoder-Python-13B-V1.0"
    #"WizardLM/WizardCoder-15B-V1.0",
    #"bigcode/starcoder",
    #"Salesforce/codegen25-7b-mono",
    #"codellama/CodeLlama-13b-Instruct-hf",
    #"meta-llama/Llama-2-13b-hf",
    #"meta-llama/Llama-2-13b-chat-hf",
    #"bigcode/starcoder2-15b",
    #"mistralai/Mixtral-8x7B-Instruct-v0.1",
    #"Salesforce/xgen-7b-8k-inst",
    #"google/gemma-7b-it",
    #"meta-llama/Meta-Llama-3-8B-Instruct",
    #"meta-llama/Meta-Llama-3-70B-Instruct",
    #"microsoft/Phi-3-mini-128k-instruct",
    "CohereForAI/c4ai-command-r-plus"
]

class RemoteModel(object):
    def __init__(self, model) -> None:
        if "gpt" in model:
            if "OPENAI_KEY" not in os.environ:
                print("Cannot find OPENAI_KEY, please set this variable in your shell.")
                exit()
            self.apikey = os.environ["OPENAI_KEY"]
            self.client = OpenAI(api_key = self.apikey)
        self.temperature = 0.7
        self.model = model

    def upload_file(self, filename):
        response = self.client.files.create(
            file=open(filename, "rb"),
            purpose="batch"
        )
        print(response)
        return response

    def delete_file(self, file_id):
        response = self.client.files.delete(file_id)
        print(response)
        return response

    def download_file(self, batch_id):
        response = self.client.batches.retrieve(batch_id)
        if response.status != "completed":
            raise ValueError("The batch is not completed and its status is {}.".format(response.status))
        output_file = response.output_file_id
        content = self.client.files.content(output_file)
        contents = content.content.splitlines()
        outputs = []
        for line in contents:
            outputs.append(json.loads(line))
        return outputs

    def collect_errors(self, batch_id):
        response = self.client.batches.retrieve(batch_id)
        if response.status != "completed":
            raise ValueError("The batch is not completed and its status is {}.".format(response.status))
        error_file = response.error_file_id
        if error_file == None:
            return []
        content = self.client.files.content(error_file)
        contents = content.content.splitlines()
        outputs = []
        for line in contents:
            outputs.append(json.loads(line))
        return outputs
        
    def batch_run(self, file_id):
        response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(response)
        return response

    def batch_status(self, batch_id):
        response = self.client.batches.retrieve(batch_id)
        print(response)
        return response

    def cancel_batch(self, batch_id):
        response = self.client.batches.cancel(batch_id)
        print(response)
        return response

    def delete_batch(self, batch_id):
        response = self.client.batches.retrieve(batch_id)
        if response.input_file_id != None:
            r = self.client.files.delete(response.input_file_id )
            print(r)
        if response.error_file_id != None:
            r = self.client.files.delete(response.error_file_id)
            print(r)
        if response.output_file_id != None:
            r = self.client.files.delete(response.output_file_id)
            print(r)


    def build_messages(self, prompt):
        messages = [
            {"role": "system", "content": "You are an excellent code programmer."},
            {"role": "user", "content": prompt}
        ]

        return messages

    @timeout_decorator.timeout(60)
    def generate(self, prompt, n = 20, temperature = 0.7):
        if not isinstance(prompt, list):
            messages = self.build_messages(prompt)
        else:
            messages = prompt

        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = temperature,
            n = n,
        )
        results = []
        for c in response.choices:
            results.append(c.message.content)
        return results
    

    def run(self, prompt, times = 0, n = 20, temperature = 0.7):
        if times > 4:
            results = self.generate(prompt)
            return results
        else:
            try:
                results = self.generate(prompt, n = n, temperature = temperature)
                return results
            except Exception as e:
                logger.error(str(e))
                self.run(prompt, times = times + 1)




class LocalModel(object):
    def __init__(self, model, device_num, llama = False, tensor_parallel_size = 4, temperature=0.7, n = 200) -> None:
        self.name = model
        self.sampling_params = SamplingParams(temperature=temperature, n = n, max_tokens = 512)
        #self.sampling_params = SamplingParams(temperature=0, n = 1, max_tokens = 512)
        self.model = LLM(model, trust_remote_code = True, tensor_parallel_size= tensor_parallel_size)
        self.tokenizer = self.model.get_tokenizer()

    def get_prompt_length(self, prompt):
        return len(self.tokenizer(prompt).input_ids)


    def format_chats(self, dialogs, pad = False):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "SYS\n", "\n<</SYS>>\n\n"


        prompt_tokens = []


        for dialog in dialogs:
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                    [msg["role"] == "assistant" for msg in dialog[1::2]]
                ), (
                    "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )

            dialog_tokens = sum(
                [
                    [
                        [self.tokenizer.bos_token_id]
                        + self.tokenizer(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                        ).input_ids
                        + [self.tokenizer.eos_token_id]
                    ]
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += [
                [self.tokenizer.bos_token_id]
                + self.tokenizer(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
                ).input_ids
            ]
            prompt_tokens.append(dialog_tokens)

        return prompt_tokens
        
    def infer_one(self, prompt, n = 10):
        try:
            if n <= 10:
                outputs = self.model.generate(prompt, self.sampling_params)
                predictions = []
                for o in outputs[0].outputs:
                    predictions.append(o.text)
            else:
                iters = math.ceil(n / 10)
                predictions = []
                for i in range(0, iters):
                    outputs = self.model.generate(prompt, self.sampling_params)
                    for o in outputs[0].outputs:
                        predictions.append(o.text)
            return predictions
        except Exception as e:
            logger.error("Error occurred with reason {} for prompt:\n{}".format(str(e), prompt))
            return str(e)


    def infer_one_chat(self, dialog, n = 10):
        prompt_tokens = self.format_chats(dialog)[0]
        try:
            if n <= 10:
                outputs = self.model.generate(prompt_token_ids = prompt_tokens, sampling_params=self.sampling_params)
                predictions = []
                for o in outputs[0].outputs:
                    predictions.append(o.text)
            else:
                iters = math.ceil(n/10)
                predictions = []
                for i in range(0, iters):
                    outputs = self.model.generate(prompt, self.sampling_params)
                    for o in outputs[0].outputs:
                        predictions.append(o.text)
            return predictions
        except Exception as e:
            logger.error("Error occurred with reason {} for prompt:\n{}".format(str(e), dialog[0]["content"]))
            return str(e)




    def infer_many(self, prompts, temperature = 0.7, n = 200):
        self.sampling_params = SamplingParams(temperature=temperature, n = n, max_tokens = 512)
        predictions = {}
        try:
            outputs = self.model.generate(prompts, self.sampling_params)
            for index, out in enumerate(outputs):
                preds = []
                for o in out.outputs:
                    preds.append(o.text)
                predictions[prompts[index]] = [preds, True]
            return predictions
        except Exception as e:
            logger.error("Error occurred with reason {} when processing multiple prompts".format(str(e)))
            print(traceback.print_exc())
            return str(e)

    def infer_many_chats(self, dialogs, n = 10):
        prompt_tokens = self.format_chats(dialogs, pad = True)
        predictions = {}
        try:
            for index, prompt_token in enumerate(prompt_tokens):
                if n <= 10:
                    outputs = self.model.generate(prompt_token_ids = prompt_token, sampling_params=self.sampling_params)
                    preds = []
                    for o in outputs[0].outputs:
                        preds.append(o.text)
                    predictions[dialogs[index][0]["content"]] = [preds, True]
                else:
                    iters = math.ceil(n/10)
                    preds = []
                    for i in range(0, iters):
                        outputs = self.model.generate(prompt_token_ids = prompt_token, sampling_params=self.sampling_params)
                        for o in outputs[0].outputs:
                            preds.append(o.text)
                    predictions[dialogs[index][0]["content"]] = [preds, True]
            return predictions
        except Exception as e:
            logger.error("Error occurred with reason {} when processing multiple prompts".format(str(e)))
            print(traceback.print_exc())
            return str(e)
