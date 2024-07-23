from model import RemoteModel, LocalModel
from dataset import Dataset
from utils import logger
from prompt import Prompt
import os
import json
import traceback
import pynvml
import time
from tqdm import tqdm
import sys



def infer_openai_model(model, dataset, output_path):
    model = RemoteModel(model)
    dataset = Dataset(dataset, data_path = os.path.join("test_datasets", dataset.replace("/", "_")), full = False)

    res_dir_path = os.path.join(output_path, dataset.name.replace("/", "_"))
    res_path = os.path.join(res_dir_path, model.model + ".json")

    if os.path.exists(res_path):
        results = json.load(open(res_path, "r"))
    else:
        results = {}

    finish = False
    index = 0

    while(not finish):
        index += 1
        instance, finish = dataset.next()
        prompt = dataset.get_prompt(instance)
        if prompt in results and results[prompt][-1] == True and isinstance(results[prompt][0], list):
            continue
        try:
            res = model.run(prompt)
            results[prompt] = [res, True]
        except Exception as e:
            logger.error("Dataset: {}\nModel: {}\nPrompt:\n{}\n".format(dataset.name, model.model, prompt) + str(e))
            results[prompt] = [str(e), False]
        print("\r{}/{}         ".format(index, dataset.length()), end="", flush=True)

    

    if not os.path.exists(res_dir_path):
        os.mkdir(res_dir_path)
    
    with open(res_path, "w", encoding="utf-8") as of:
        of.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))


def infer_openai_model_for_all(model, output_path):
    code_datasets = [
        #"openai_humaneval",
        "codeparrot/apps",
        #"mbpp",
        #"deepmind/code_contests",
        #"BAAI/TACO",
        #"NTU-NLP-sg/xCodeEval"
    ]

    for d in code_datasets:
        infer_openai_model(model, d, output_path)


def infer_openai_model_prompts(model, mode, rd, output_path, inputs_file, sample_num, 
                               history_file = None, reviewer_history_file = None, 
                               leader_history_file = None, checked_inputs_file = None, 
                               demo_num = 3, similar_example_file = None, 
                               batch = True, batch_split = True):
    
    """
    Generate the batch input file for inference.

    Args:
        model (str): The path where the Hugging Face model is stored.
        mode (str): The mode in which the model should operate, typically set to "base".
        rd (int): Round of PerfCodeGen.
        output_path (str): The file path where the generated results will be saved.

        sample_num (int): The number of samples to generate. Set to 20 in the paper.

        batch (bool, optional): Whether to process the inputs in batches. Defaults to True. 
                                Note that when using non-public models like OpenAI API, this should be left to True
                                if you want to use the batch inference capability. 
                                Setting to False will begin the inference process immediately, but without batching it takes 
                                much longer to complete the inference process.

        batch_split (bool, optional):   Whether to split batches during processing. Defaults to True.
                                        Set to False for open models (hosted locally). Set to True for OpenAI API-like models
                                        A batch can't contain more than 10k lines in the batch inference scheme from OpenAI.

    """

    outputs = {}
    prompt_generator = Prompt(chat = True)
    prompt_generator.load_templates()

    if "gpt" not in model:
        model = model.replace("/", "_")

    model = RemoteModel(model)

    no_system = False
    if "Mixtral" in model.model:
        no_system = True

    
    inputs = json.load(open(inputs_file, "r"))
    correctness_categories = [
        #"correctness_testcase_feedback",
        "correctness_reflection_and_feedback"
    ]
    time_categories = [
        #"time_simple_instruction",
        #"time_in_context_learning",
        #"time_pre_defined_strategy",
        #"time_chain_of_thought",
        #"time_time_complexity_reduction",
        #"time_simple_execution_feedback",
        "time_execution_feedback_with_testcase",
        #"time_multiple_agents_with_reviewer",
        #"time_multiple_agents_with_team"
    ]
    if "gpt" in model.model:
        output_dir = os.path.join(output_path, "gpt-4" if "gpt-4" in model.model else "gpt-3.5-turbo", mode)
    else:
        output_dir = os.path.join(output_path, model.model, mode)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kwargs = {}

    hash_file = os.path.join(output_path, "hash_map.json")
    if not os.path.exists(hash_file):
        hash_map = {}
        for dataset in inputs:
            for prompt in inputs[dataset]:
                cur_hash = str(hash(prompt))
                hash_map[cur_hash] = prompt
                hash_map[prompt] = cur_hash
        with open(hash_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(hash_map, sort_keys=True, indent=4, separators=(',', ': ')))
    else:
        hash_map = json.load(open(hash_file, "r"))

    #Base Prompt
    if mode == "base":
        if not batch:
            for dataset in inputs:
                for prompt in inputs[dataset]:
                    if "base" in inputs[dataset][prompt] and isinstance(inputs[dataset][prompt]["base"], list):
                        continue
                    try:
                        inputs[dataset][prompt]["base"] = [model.run(prompt, n = sample_num)]
                    except Exception as e:
                        logger.error("Dataset: {}\nModel: {}\nPrompt:\n{}\n".format(dataset.name, model.model, prompt) + str(e))
                        inputs[dataset][prompt]["base"] = [None]*sample_num
            outputs_file = os.path.join(output_dir, f"rd{rd}.json")
            with open(outputs_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(inputs, sort_keys=True, indent=4, separators=(',', ': ')))
        else:
            real_inputs = {}
            data = []
            for dataset in inputs:
                real_inputs[dataset] = {}
                for prompt in inputs[dataset]:
                    real_inputs[dataset][prompt] = {}
                    data.append({
                        "custom_id": f"{dataset}@base@{hash_map[prompt]}@0",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model.model,
                            "messages": [
                                {"role": "system", "content": "You are an excellent code programmer."},
                                {"role": "user", "content": prompt}
                            ] if not no_system else [{"role": "user", "content": prompt.replace("\nPython Solution:\n```python", "\nGive your solution as follows. Wrap it with ```python```. Do not add __name__ check.")}],
                            "n": 20,
                            "temperature": 0.7
                        }
                    })
                    real_inputs[dataset][prompt]["base"] = [["<PENDING>"] * sample_num]
            if batch_split:
                datalines = []
                for d in data:
                    datalines.append(json.dumps(d) + "\n")
                num = len(datalines) // 10000
                if num * 10000 < len(datalines):
                    num += 1
                for i in range(0, num):
                    data_file = os.path.join(output_dir, f"rd{rd}_batch_inputs_{i}.jsonl")
                    with open(data_file, "w", encoding = "utf-8") as f:
                        if i < num - 1:
                            f.writelines(datalines[(len(datalines) // num) * i:(len(datalines) // num) * (i+1)])
                        else:
                            f.writelines(datalines[(len(datalines) // num) * i:])
            else:
                data_file = os.path.join(output_dir, f"rd{rd}_batch_inputs.json")
                with open(data_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            outputs_file = os.path.join(output_dir, f"rd{rd}.json")
            with open(outputs_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(real_inputs, sort_keys=True, indent=4, separators=(',', ': ')))

    #Correctness Prompt
    if mode == "correctness":
        kwargs = {}
        if checked_inputs_file == None:
            raise ValueError("Missing checked inputs file!")
        checked_inputs = json.load(open(checked_inputs_file, "r"))
        if rd > 0 and history_file == None:
            raise ValueError("Missing history file!")
        if rd > 0:
            history = json.load(open(history_file, "r"))
        else:
            history = {}
            for dataset in inputs:
                history[dataset] = {}
                for prompt in inputs[dataset]:
                    for name in correctness_categories:
                        if name not in history[dataset]:
                            history[dataset][name] = {}
                        history[dataset][name][prompt] = []
                        for out in inputs[dataset][prompt]["base"][0]:
                            history[dataset][name][prompt].append([
                                {"role": "system", "content": "You are an excellent code programmer."},
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": out}
                            ] if not no_system else [{"role": "user", "content": prompt.replace("\nPython Solution:\n```python", "\nGive your solution as follows. Wrap it with ```python```. Do not add __name__ check.")}, {"role": "assistant", "content": out}])
        for dataset in inputs:
            kwargs[dataset] = {}
            for prompt in inputs[dataset]:
                if prompt not in checked_inputs[dataset]:
                    continue
                for name in correctness_categories:
                    if name not in kwargs[dataset]:
                        kwargs[dataset][name] = {}
                    if name not in inputs[dataset][prompt]:
                        inputs[dataset][prompt][name] = []
                    if name == "correctness_testcase_feedback":
                        if "testcases" not in kwargs[dataset][name]:
                            kwargs[dataset][name]["testcases"] = {}
                        if rd == 0:
                            data = checked_inputs[dataset][prompt]["base"][0]
                        else:
                            data = checked_inputs[dataset][prompt][name][rd-1]
                        kwargs[dataset][name]["testcases"][prompt] = []
                        for status in data:
                            if isinstance(status, str) or status["status"] in ["PASSED", "QUERY_ERROR"]:
                                kwargs[dataset][name]["testcases"][prompt].append(None)
                            elif status["status"] in ["TIMEOUT", "PARSE_ERROR"]:
                                kwargs[dataset][name]["testcases"][prompt].append(status["status"])
                            else:
                                kwargs[dataset][name]["testcases"][prompt].append(status["failed_testcase"])
                    elif name == "correctness_reflection_and_feedback":
                        if "testcases" not in kwargs[dataset][name]:
                            kwargs[dataset][name]["testcases"] = {}
                        if rd == 0:
                            data = checked_inputs[dataset][prompt]["base"][0]
                        else:
                            data = checked_inputs[dataset][prompt][name][rd-1]
                        if rd%2 == 0:
                            kwargs[dataset][name]["testcases"][prompt] = []
                            for status in data:
                                if isinstance(status, str) or status["status"] in ["PASSED", "QUERY_ERROR"]:
                                    kwargs[dataset][name]["testcases"][prompt].append(None)
                                elif status["status"] in ["TIMEOUT", "PARSE_ERROR"]:
                                    kwargs[dataset][name]["testcases"][prompt].append(status["status"])
                                else:
                                    kwargs[dataset][name]["testcases"][prompt].append(status["failed_testcase"])
                        else:
                            if "indicators" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["indicators"] = {}
                            kwargs[dataset][name]["indicators"][prompt] = []
                            for status in data:
                                if (isinstance(status, str) and status.startswith("<") and status.endswith(">")) or (not isinstance(status, str) and status["status"] in ["PASSED", "QUERY_ERROR"]):
                                    kwargs[dataset][name]["indicators"][prompt].append(False)
                                else:
                                    kwargs[dataset][name]["indicators"][prompt].append(True)

        if not batch:
            for dataset in kwargs:
                for name in kwargs[dataset]:
                    if not prompt_generator.round_exist(rd, name):
                        continue
                    prompts = prompt_generator.get_chat_prompts(name, rd, history[dataset][name], **kwargs[dataset][name])
                    index = 0
                    for prompt in prompts:
                        print("Handling {}/{} prompts for {} in dataset {}".format(index, len(list(prompts.keys())), name, dataset), end = "\r", file = sys.stderr)
                        index += 1
                        outputs = []
                        for i, message in enumerate(prompts[prompt]):
                            if message != None:
                                history[dataset][name][prompt][i] = message
                                try:
                                    outputs.append(model.run(message, n = 1, temperature = 1)[0])
                                except Exception as e:
                                    logger.error("Dataset: {}\nModel: {}\nPrompt:\n{}\n".format(dataset.name, model.model, message) + str(e))
                                    outputs.append(None)
                            else:
                                outputs.append("<END>")
                        inputs[dataset][prompt][name].append(outputs)
        else:
            data = []
            for dataset in kwargs:
                for name in kwargs[dataset]:
                    if not prompt_generator.round_exist(rd, name):
                        continue
                    prompts = prompt_generator.get_chat_prompts(name, rd, history[dataset][name], **kwargs[dataset][name])
                    for prompt in prompts:
                        outputs = []
                        for i, message in enumerate(prompts[prompt]):
                            if message != None:
                                history[dataset][name][prompt][i] = message
                                data.append({
                                    "custom_id": f"{dataset}@{name}@{hash_map[prompt]}@{len(inputs[dataset][prompt][name])}@{i}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": model.model,
                                        "messages": message,
                                        "n": 1,
                                        "temperature": 0
                                    }
                                })
                                outputs.append("<PENDING>")
                            else:
                                outputs.append("<END>")
                        inputs[dataset][prompt][name].append(outputs)
            
            if batch_split:
                datalines = []
                for d in data:
                    datalines.append(json.dumps(d) + "\n")
                num = len(datalines) // 10000
                if num * 10000 < len(datalines):
                    num += 1
                for i in range(0, num):
                    data_file = os.path.join(output_dir, f"rd{rd}_batch_inputs_{i}.jsonl")
                    with open(data_file, "w", encoding = "utf-8") as f:
                        f.writelines(datalines[(len(datalines) // num) * i:(len(datalines) // num) * (i+1)])
            else:
                data_file = os.path.join(output_dir, f"rd{rd}_batch_inputs.json")
                with open(data_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            


        outputs_file = os.path.join(output_dir, f"rd{rd}.json")
        with open(outputs_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(inputs, sort_keys=True, indent=4, separators=(',', ': ')))
        
        history_file = os.path.join(output_dir, f"rd{rd}_HISTORY.json")
        with open(history_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(history, sort_keys=True, indent=4, separators=(',', ': ')))
        
        
    #Time Prompt
    if mode == "time":
        kwargs = {}
        if checked_inputs_file == None:
            raise ValueError("Missing checked inputs file!")
        checked_inputs = json.load(open(checked_inputs_file, "r"))
        if history_file == None:
            raise ValueError("Missing history file!")
        history = json.load(open(history_file, "r"))
        if reviewer_history_file:
            reviewer_history = json.load(open(reviewer_history_file, "r"))
        else:
            reviewer_history = {}
        if leader_history_file:
            leader_history = json.load(open(leader_history_file, "r"))
        else:
            leader_history = {}
        for dataset in inputs:
            kwargs[dataset] = {}
            for prompt in inputs[dataset]:
                if prompt not in checked_inputs[dataset]:
                    continue
                for name in time_categories:
                    if not prompt_generator.round_exist(rd, name):
                        continue
                    if "correct_solutions" not in checked_inputs[dataset][prompt]:
                        continue
                    if name not in kwargs[dataset]:
                        kwargs[dataset][name] = {}
                    if name not in inputs[dataset][prompt]:
                        inputs[dataset][prompt][name] = []
                    if "correct_solutions" not in inputs[dataset][prompt]:
                        inputs[dataset][prompt]["correct_solutions"] = checked_inputs[dataset][prompt]["correct_solutions"]
                    if rd == 0:
                        data = checked_inputs[dataset][prompt]["correct_solutions"]
                    else:
                        data = checked_inputs[dataset][prompt][name][-1]
                    if "indicators" not in kwargs[dataset][name]:
                        kwargs[dataset][name]["indicators"] = {}
                    kwargs[dataset][name]["indicators"][prompt] = [] 
                    if name in ["time_simple_execution_feedback", "time_execution_feedback_with_testcase"]:
                        for status in data:
                            if (isinstance(status, str) and status.startswith("<") and status.endswith(">")):
                                kwargs[dataset][name]["indicators"][prompt].append(False)
                            elif rd > 0 and isinstance(status, dict) and "status" in status and status["status"] != "TIME_MEASURED":
                                kwargs[dataset][name]["indicators"][prompt].append(False)
                            else:
                                kwargs[dataset][name]["indicators"][prompt].append(True)
                    else:
                        for status in data:
                            if (isinstance(status, str) and status.startswith("<") and status.endswith(">")):
                                kwargs[dataset][name]["indicators"][prompt].append(False)
                            elif rd > 0 and isinstance(status, dict) and "status" in status and status["status"] not in ["TIME_MEASURED", "PASSED"]:
                                kwargs[dataset][name]["indicators"][prompt].append(False)
                            else:
                                kwargs[dataset][name]["indicators"][prompt].append(True)
                    if name == "time_in_context_learning":
                        if "demos" in kwargs[dataset][name]:
                            continue
                        prompt_generator.load_examples(file_path = similar_example_file)
                        fixed_programs = []
                        for example in prompt_generator.examples:
                            fixed_programs.append([example["ori_program"], example["opt_program"]])
                        kwargs[dataset][name]["demos"] = fixed_programs
                        kwargs[dataset][name]["demo_num"] = demo_num
                    elif name == "time_simple_execution_feedback":
                        if rd == 1:
                            correct_solutions = checked_inputs[dataset][prompt]["correct_solutions"]
                            if "times" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["times"] = {}
                            kwargs[dataset][name]["times"][prompt] = []
                            for i, status in enumerate(correct_solutions):
                                if isinstance(status, str) or isinstance(data[i], str):
                                    kwargs[dataset][name]["times"][prompt].append(None)
                                elif isinstance(status, dict) and "status" in status and status["status"] != "TIME_MEASURED":
                                    kwargs[dataset][name]["times"][prompt].append(None)
                                    print("Cannot get run-time information for correct solutions, skipping.")
                                elif isinstance(data[i], dict) and "status" in data[i] and data[i]["status"] != "TIME_MEASURED":
                                    kwargs[dataset][name]["times"][prompt].append(None)
                                else:
                                    if status["exec_time"] == None or data[i]["exec_time"] == None:
                                        print(dataset, name, prompt)
                                    kwargs[dataset][name]["times"][prompt].append([status["exec_time"], data[i]["exec_time"]])
                    elif name == "time_execution_feedback_with_testcase":
                        if rd == 1:
                            if "testcases" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["testcases"] = {}
                            kwargs[dataset][name]["testcases"][prompt] = []
                            for status in data:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["testcases"][prompt].append(None)
                                elif isinstance(status, dict) and "status" in status and status["status"] != "TIME_MEASURED":
                                    kwargs[dataset][name]["testcases"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["testcases"][prompt].append(status["large_testcase"])
                    elif name == "time_multiple_agents_with_reviewer":
                        if rd == 0:
                            pass
                        elif rd == 1:
                            if dataset not in reviewer_history:
                                reviewer_history[dataset] = {}
                            if name not in reviewer_history[dataset]:
                                reviewer_history[dataset][name] = {}
                            if prompt not in reviewer_history[dataset][name]:
                                if no_system:
                                    reviewer_history[dataset][name][prompt] = [[]] * sample_num
                                else:
                                    reviewer_history[dataset][name][prompt] = [[{"role": "system", "content": "You are a code reviewer and your job is to check the quality of code written by others."}]] * sample_num
                            if "programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["programs"] = {}
                            if "opt_programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["opt_programs"] = {}
                            if "reviewer_history" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["reviewer_history"] = {}
                            kwargs[dataset][name]["programs"][prompt] = []
                            kwargs[dataset][name]["opt_programs"][prompt] = []
                            kwargs[dataset][name]["reviewer_history"][prompt] = reviewer_history[dataset][name][prompt]
                            for status in checked_inputs[dataset][prompt]["correct_solutions"]:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["programs"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["programs"][prompt].append(status["passed_solution"])
                            for status in data:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["opt_programs"][prompt].append(None)
                                elif isinstance(status, dict) and "status" in status and status["status"] not in ["TIME_MEASURED", "PASSED"]:
                                    kwargs[dataset][name]["opt_programs"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["opt_programs"][prompt].append(status["passed_solution"])
                        elif rd == 2:
                            if "decisions" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["decisions"] = {}
                            if "comments" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["comments"] = {}
                            kwargs[dataset][name]["decisions"][prompt] = []
                            kwargs[dataset][name]["comments"][prompt] = []
                            for status in data:
                                if status.startswith("<") and status.endswith(">"):
                                    kwargs[dataset][name]["decisions"][prompt].append(None)
                                    kwargs[dataset][name]["comments"][prompt].append(None)
                                elif isinstance(status, dict) and "status" in status and status["status"] != "TIME_MEASURED":
                                    kwargs[dataset][name]["decisions"][prompt].append(None)
                                    kwargs[dataset][name]["comments"][prompt].append(None)
                                else:
                                    if status.startswith("[Agree]"):
                                        kwargs[dataset][name]["comments"][prompt].append(status.split("[Agree]")[-1])
                                        kwargs[dataset][name]["decisions"][prompt].append("successful")
                                    else:
                                        kwargs[dataset][name]["comments"][prompt].append(status.split("[Disagree]")[-1])
                                        kwargs[dataset][name]["decisions"][prompt].append("unsuccessful")
                    elif name == "time_multiple_agents_with_team":
                        if rd == 0:
                            if dataset not in leader_history:
                                leader_history[dataset] = {}
                            if name not in leader_history[dataset]:
                                leader_history[dataset][name] = {}
                            if prompt not in leader_history[dataset][name]:
                                if no_system:
                                    leader_history[dataset][name][prompt] = [[]]*sample_num
                                else:
                                    leader_history[dataset][name][prompt] = [[{"role": "system", "content": "You are a coding team leader and your job is to give plans for code optimization."}]]*sample_num
                            if "problems" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["problems"] = {}
                            if "programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["programs"] = {}
                            if "leader_history" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["leader_history"] = {}
                            kwargs[dataset][name]["leader_history"][prompt] = leader_history[dataset][name][prompt]
                            kwargs[dataset][name]["problems"][prompt] = prompt.replace("\nDo not give explanations, only give the Python code.\nPython Solution:\n```python", "").replace("You are an expert Python programmer, and here is your task:", "")
                            kwargs[dataset][name]["programs"][prompt] = []
                            for status in data:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["programs"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["programs"][prompt].append(status["passed_solution"])
                        elif rd == 1:
                            if "plans" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["plans"] = {}
                            if "programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["programs"] = {}
                            kwargs[dataset][name]["plans"][prompt] = []
                            kwargs[dataset][name]["programs"][prompt] = []
                            for status in checked_inputs[dataset][prompt]["correct_solutions"]:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["programs"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["programs"][prompt].append(status["passed_solution"])
                            for status in data:
                                if isinstance(status, str) and status.startswith("<") and status.endswith(">"):
                                    kwargs[dataset][name]["plans"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["plans"][prompt].append(status)
                        elif rd == 2:
                            if dataset not in reviewer_history:
                                reviewer_history[dataset] = {}
                            if name not in reviewer_history[dataset]:
                                reviewer_history[dataset][name] = {}
                            if prompt not in reviewer_history[dataset][name]:
                                if no_system:
                                    reviewer_history[dataset][name][prompt] = [[]]* sample_num
                                else:
                                    reviewer_history[dataset][name][prompt] = [[{"role": "system", "content": "You are a code reviewer and your job is to check the quality of code written by others."}]]* sample_num
                            if "programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["programs"] = {}
                            if "opt_programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["opt_programs"] = {}
                            if "reviewer_history" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["reviewer_history"] = {}
                            if "plans" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["plans"] = {}
                            kwargs[dataset][name]["reviewer_history"][prompt] = reviewer_history[dataset][name][prompt]
                            kwargs[dataset][name]["programs"][prompt] = []
                            kwargs[dataset][name]["opt_programs"][prompt] = []
                            kwargs[dataset][name]["plans"][prompt] = []
                            for status in checked_inputs[dataset][prompt][name][0]:
                                if isinstance(status, str) and status.startswith("<") and status.endswith(">"):
                                    kwargs[dataset][name]["plans"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["plans"][prompt].append(status)
                            for status in checked_inputs[dataset][prompt]["correct_solutions"]:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["programs"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["programs"][prompt].append(status["passed_solution"])
                            for status in data:
                                if isinstance(status, str):
                                    kwargs[dataset][name]["opt_programs"][prompt].append(None)
                                elif isinstance(status, dict) and "status" in status and status["status"] not in ["TIME_MEASURED", "PASSED"]:
                                    kwargs[dataset][name]["opt_programs"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["opt_programs"][prompt].append(status["passed_solution"])
                        elif rd == 3:
                            if "decisions" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["decisions"] = {}
                            if "comments" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["comments"] = {}
                            if "opt_programs" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["opt_programs"] = {}
                            if "leader_history" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["leader_history"] = {}
                            kwargs[dataset][name]["leader_history"][prompt] = leader_history[dataset][name][prompt]
                            kwargs[dataset][name]["decisions"][prompt] = []
                            kwargs[dataset][name]["comments"][prompt] = []
                            kwargs[dataset][name]["opt_programs"][prompt] = []
                            for i, status in enumerate(data):
                                if isinstance(status, str) and status.startswith("<") and status.endswith(">"):
                                    kwargs[dataset][name]["decisions"][prompt].append(None)
                                    kwargs[dataset][name]["comments"][prompt].append(None)
                                    kwargs[dataset][name]["opt_programs"][prompt].append(None)
                                elif isinstance(status, dict) and "status" in status and status["status"] not in ["TIME_MEASURED", "PASSED"]:
                                    kwargs[dataset][name]["decisions"][prompt].append(None)
                                    kwargs[dataset][name]["comments"][prompt].append(None)
                                    kwargs[dataset][name]["opt_programs"][prompt].append(None)
                                else:
                                    if status.startswith("[Agree]"):
                                        kwargs[dataset][name]["comments"][prompt].append(status.split("[Agree]")[-1])
                                        kwargs[dataset][name]["decisions"][prompt].append("successful")
                                    else:
                                        kwargs[dataset][name]["comments"][prompt].append(status.split("[Disagree]")[-1])
                                        kwargs[dataset][name]["decisions"][prompt].append("unsuccessful")
                                    kwargs[dataset][name]["opt_programs"][prompt].append(checked_inputs[dataset][prompt][name][rd-2][i]["passed_solution"])
                        elif rd == 4:
                            if "plans" not in kwargs[dataset][name]:
                                kwargs[dataset][name]["plans"] = {}
                            kwargs[dataset][name]["plans"][prompt] = []
                            for status in data:
                                if isinstance(status, str) and status.startswith("<") and status.endswith(">"):
                                    kwargs[dataset][name]["plans"][prompt].append(None)
                                else:
                                    kwargs[dataset][name]["plans"][prompt].append(status)
        if not batch:
            for dataset in kwargs:
                for name in kwargs[dataset]:
                    if not prompt_generator.round_exist(rd, name):
                        continue
                    prompts = prompt_generator.get_chat_prompts(name, rd, history[dataset][name], **kwargs[dataset][name])
                    index = 0
                    for prompt in prompts:
                        print("Handling {}/{} prompts for {} in dataset {}".format(index, len(list(prompts.keys())), name, dataset), end = "\r", file = sys.stderr)
                        index += 1
                        outputs = []
                        for i, message in enumerate(prompts[prompt]):
                            if message != None:
                                if "leader_history" in kwargs[dataset][name]:
                                    leader_history[dataset][name][prompt][i] = message
                                elif "reviewer_history" in kwargs[dataset][name]:
                                    reviewer_history[dataset][name][prompt][i] = message
                                else:
                                    history[dataset][name][prompt][i] = message
                                try:
                                    outputs.append(model.run(message, n = 1, temperature = 1)[0])
                                except Exception as e:
                                    logger.error("Dataset: {}\nModel: {}\nPrompt:\n{}\n".format(dataset.name, model.model, message) + str(e))
                                    outputs.append(None)
                            else:
                                outputs.append("<END>")
                        inputs[dataset][prompt][name].append(outputs)
        else:
            data = []
            test = {}
            for dataset in kwargs:
                test[dataset] = {}
                for name in kwargs[dataset]:
                    test[dataset][name] = {}
                    if not prompt_generator.round_exist(rd, name):
                        continue
                    prompts = prompt_generator.get_chat_prompts(name, rd, history[dataset][name], **kwargs[dataset][name])
                    for prompt in prompts:
                        outputs = []
                        for i, message in enumerate(prompts[prompt]):
                            if message != None:
                                if len(test[dataset][name]) == 0:
                                    test[dataset][name][prompt] = message
                                if "leader_history" in kwargs[dataset][name]:
                                    leader_history[dataset][name][prompt][i] = message
                                elif "reviewer_history" in kwargs[dataset][name]:
                                    reviewer_history[dataset][name][prompt][i] = message
                                else:
                                    history[dataset][name][prompt][i] = message
                                data.append({
                                    "custom_id": f"{dataset}@{name}@{hash_map[prompt]}@{len(inputs[dataset][prompt][name])}@{i}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": model.model,
                                        "messages": message,
                                        "n": 1,
                                        "temperature": 0
                                    }
                                })
                                outputs.append("<PENDING>")
                            else:
                                outputs.append("<END>")
                        inputs[dataset][prompt][name].append(outputs)
            if batch_split:
                datalines = []
                for d in data:
                    datalines.append(json.dumps(d) + "\n")
                num = len(datalines) // 10000
                if num * 10000 < len(datalines):
                    num += 1
                for i in range(0, num):
                    data_file = os.path.join(output_dir, f"rd{rd}_batch_inputs_{i}.jsonl")
                    with open(data_file, "w", encoding = "utf-8") as f:
                        if i < num - 1:
                            f.writelines(datalines[(len(datalines) // num) * i:(len(datalines) // num) * (i+1)])
                        else:
                            f.writelines(datalines[(len(datalines) // num) * i:])
            else:
                data_file = os.path.join(output_dir, f"rd{rd}_batch_inputs.json")
                with open(data_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))


        for name in correctness_categories + ["base"]:
            for dataset in inputs:
                for prompt in inputs[dataset]:
                    if name in inputs[dataset][prompt]:
                        del inputs[dataset][prompt][name]
        

        with open("test_file.json", "w", encoding = "utf-8") as f:
            f.write(json.dumps(test, sort_keys=True, indent=4, separators=(',', ': ')))


        outputs_file = os.path.join(output_dir, f"rd{rd}.json")
        with open(outputs_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(inputs, sort_keys=True, indent=4, separators=(',', ': ')))
        
        history_file = os.path.join(output_dir, f"rd{rd}_HISTORY.json")
        with open(history_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(history, sort_keys=True, indent=4, separators=(',', ': ')))
        
        reviewer_history_file = os.path.join(output_dir, f"rd{rd}_HISTORY_REVIEWER.json")
        if len(reviewer_history) > 0:
            with open(reviewer_history_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(reviewer_history, sort_keys=True, indent=4, separators=(',', ': ')))
        
        leader_history_file = os.path.join(output_dir, f"rd{rd}_HISTORY_LEADER.json")
        if len(leader_history) > 0:
            with open(leader_history_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(leader_history, sort_keys=True, indent=4, separators=(',', ': ')))




def wait_available_gpu(num=4):
    pynvml.nvmlInit()
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()
    available_gpus = []
    while True:
        available_gpus = []
        for i in range(gpuDeviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            pidAllInfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if len(pidAllInfo) == 0:
                available_gpus.append(i)
        if len(available_gpus) >= num:
            print("Found available GPUs: {}!".format(available_gpus))
            break
        else:
            print("Not enough GPUs...")
            time.sleep(600)
    string = ""
    for i in available_gpus:
        string += ","
        string += str(i)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = string[1:]
    os.system("export CUDA_VISIBLE_DEVICES={}".format(string[1:]))


def infer_local_model(model, dataset, output_path, device_num, prefix = None, suffix = None, chat = False):
    model = LocalModel(model, device_num, tensor_parallel_size = 8 if "Mixtral" in model else 4)
    if dataset != "mbpp":
        dataset = Dataset(dataset, data_path = os.path.join("test_datasets/", dataset.replace("/", "_")), full = False)
    else:
        dataset = Dataset(dataset, data_path = os.path.join("test_datasets/", dataset.replace("/", "_")), testfile_path = "datasets/MbppPlus.jsonl", full = False)

    results = {}

    finish = False
    index = 0

    while(not finish):
        index += 1
        instance, finish = dataset.next()
        if not chat:
            prompt = dataset.get_prompt(instance)
            if prefix != None or suffix != None:
                prompt = (prefix if prefix != None else "") + prompt + (suffix if suffix != None else "")
        else:
            dialog = dataset.get_chat(instance)
        try:
            if not chat:
                res = model.infer_one(prompt)
                if isinstance(res, list):
                    results[prompt] = [res, True]
                else:
                    results[prompt] = [res, False]
                print(prompt)
                print(res)
                exit()
            else:
                res = model.infer_one_chat([dialog])
                if isinstance(res, list):
                    results[dialog[0]["content"]] = [res, True]
                else:
                    results[dialog[0]["content"]] = [res, False]
                print(dialog)
                print(res)
                exit()

        except Exception as e:
            logger.error("Dataset: {}\nModel: {}\nPrompt:\n{}\n".format(dataset.name, model.name, prompt if not chat else dialog[0]["content"]) + str(e))
            results[prompt] = [str(e), False]
        print("\r{}/{}".format(index, dataset.length()), end="", flush=True)

    res_dir_path = os.path.join(output_path, dataset.name.replace("/", "_"))
    res_path = os.path.join(res_dir_path, model.name.replace("/", "_") + ".json")

    if not os.path.exists(res_dir_path):
        os.mkdir(res_dir_path)
    
    with open(res_path, "w", encoding="utf-8") as of:
        of.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))


def batch_infer_local_model(model, dataset, output_path, device_num, n = 200, temperature = 0.7, dataset_repo = "test_dataset", prefix = None, suffix = None, chat = False, context_length = None):
    if isinstance(model, str):
        model = LocalModel(model, device_num)
    
    if dataset != "mbpp":
        dataset = Dataset(dataset, data_path = os.path.join(dataset_repo, dataset.replace("/", "_")), full = False)
    else:
        dataset = Dataset(dataset, data_path = os.path.join(dataset_repo, dataset.replace("/", "_")), testfile_path = "datasets/MbppPlus.jsonl", full = False)

    if not chat:
        prompts, overlong_prompts = dataset.get_all_prompts(model = model, context_length = context_length)
        if prefix != None or suffix != None:
            new_prompts = []
            for prompt in prompts:
                new_prompts.append((prefix if prefix != None else "") + prompt + (suffix if suffix != None else ""))
            prompts = new_prompts
        print("{} overlong prompts found.".format(len(overlong_prompts)))
    else:
        dialogs = dataset.get_all_chats()
    try:
        if not chat:
            results = model.infer_many(prompts, n = n, temperature = temperature)
            for p in overlong_prompts:
                results[p] = ["Overlong input prompts.", False]
        else:
            results = model.infer_many_chats(dialogs)
    except Exception as e:
        logger.error("Error occurred with reason: {}. INFO Dataset: {}\nModel: {}\n".format(str(e), dataset.name, model.name))
        traceback.print_exc()

    res_dir_path = os.path.join(output_path, dataset.name.replace("/", "_"))
    res_path = os.path.join(res_dir_path, model.name.replace("/", "_") + ".json")

    if not os.path.exists(res_dir_path):
        os.mkdir(res_dir_path)
    
    with open(res_path, "w", encoding="utf-8") as of:
        of.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))


def batch_infer_local_chat_model(model, mode, rd, input_file, output_path):
    if isinstance(model, str):
        model = LocalModel(model, "0", tensor_parallel_size = 8 if "70B" in model or "Mixtral" in model or "command" in model else 4)

    inputs = json.load(open(input_file, "r"))

    prompts = []
    overlong_prompts = []
    prompt_map = {}

    print("Generating prompts...")

    context_length = 8192 if "llama" in model.name else 128000

    for d in inputs:
        prompt = model.tokenizer.apply_chat_template(
            d["body"]["messages"],
            tokenize = False,
            add_generation_prompt=True
        )
        if model.get_prompt_length(prompt) < context_length and prompt not in prompt_map:
            prompts.append(prompt)
        elif prompt not in prompt_map:
            overlong_prompts.append(prompt)
        if prompt not in prompt_map:
            prompt_map[prompt] = []
        prompt_map[prompt].append(d["custom_id"])

    print("{} overlong prompts".format(len(overlong_prompts)))

    num = 0
    for prompt in prompt_map:
        num += len(prompt_map[prompt])

    if num != len(inputs):
        raise ValueError("The instances in the prompt map does not match the input size!")

    if len(prompt_map) != len(prompts) + len(overlong_prompts):
        raise ValueError("The number of prompt in the prompt map does not match the input prompts.")

        
    try:
        results = model.infer_many(prompts, n = inputs[0]["body"]["n"], temperature = inputs[0]["body"]["temperature"])
    except Exception as e:
        logger.error("Error occurred with reason: {}. INFO Dataset: {}\nModel: {}\n".format(str(e), dataset.name, model.name))
        traceback.print_exc()

    data = {}

    for prompt in results:
        for i in prompt_map[prompt]:
            if i.count("@") == 4:
                data[i] = {
                    "content": results[prompt][0][0],
                    "role": "assistant"
                }
            else:
                data[i] = {
                    "content": results[prompt][0],
                    "role": "assistant"
                }

    for prompt in overlong_prompts:
        for i in prompt_map[prompt]:
            if i.count("@") == 4:
                data[i] = None
            else:
                data[i] = None

    if len(data) != len(inputs):
        raise ValueError("The output data size does not match the input size.")


    res_dir_path = os.path.join(output_path, model.name.replace("/", "_"), mode)
    res_path = os.path.join(res_dir_path, f"rd{rd}_batch_outputs.json")

    if not os.path.exists(res_dir_path):
        os.makedirs(res_dir_path)
    
    with open(res_path, "w", encoding="utf-8") as of:
        of.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
    
def batch_infer_local_prompt_model(model, mode, rd, input_file, output_path):
    if isinstance(model, str):
        model = LocalModel(model, "0", tensor_parallel_size = 8 if "70B" in model or "Mixtral" in model else 4)

    inputs = json.load(open(input_file, "r"))

    prompts = []
    prompt_map = {}

    print("Generating prompts...")

    overlong_prompts = []

    context_length = 2048 if "codegen25" in model.name else 8192

    for d in inputs:
        prompt = d["body"]["prompt"]
        if model.get_prompt_length(prompt) < context_length and prompt not in prompt_map:
            prompts.append(prompt)
        elif prompt not in prompt_map:
            overlong_prompts.append(prompt)
        if prompt not in prompt_map:
            prompt_map[prompt] = []
        else:
            prompt_map[prompt].append(d["custom_id"])
    
    print("{} overlong prompts found.".format(len(overlong_prompts)))

    try:
        results = model.infer_many(prompts, n = inputs[0]["body"]["n"], temperature = inputs[0]["body"]["temperature"])
    except Exception as e:
        logger.error("Error occurred with reason: {}. INFO Dataset: {}\nModel: {}\n".format(str(e), dataset.name, model.name))
        traceback.print_exc()

    data = {}

    for prompt in results:
        for i in prompt_map[prompt]:
            if i.count("@") == 4:
                data[i] = {
                    "content": results[prompt][0][0],
                    "role": "assistant"
                }
            else:
                data[i] = {
                    "content": results[prompt][0],
                    "role": "assistant"
                }
    for prompt in overlong_prompts:
        for i in prompt_map[prompt]:
            if i.count("@") == 4:
                data[i] = None
            else:
                data[i] = None


        

    res_dir_path = os.path.join(output_path, model.name.replace("/", "_"), mode)
    res_path = os.path.join(res_dir_path, f"rd{rd}_batch_outputs.json")

    if not os.path.exists(res_dir_path):
        os.makedirs(res_dir_path)
    
    with open(res_path, "w", encoding="utf-8") as of:
        of.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))


def infer_local_model_for_all(model, output_path, device_num):
    code_datasets = [
        "openai_humaneval",
        "codeparrot/apps",
        "mbpp",
        "deepmind/code_contests",
        "BAAI/TACO",
        "NTU-NLP-sg/xCodeEval"
    ]

    prefix = "<s>[INST] " if "Mixtral" in model else None
    suffix = " [/INST]" if "Mixtral" in model else None
    model = LocalModel(model, device_num, tensor_parallel_size = 8 if "Mixtral" in model else 4)


    for d in code_datasets:
        batch_infer_local_model(model, d, output_path, device_num, prefix = prefix, suffix = suffix, chat = True if "chat" in model.name else False)


def infer_all_local_models_for_all(output_path, device_num, n = 200, temperature = 0.7, dataset_repo = "test_datasets"):
    code_datasets = [
        "openai_humaneval",
        "mbpp",
        "codeparrot/apps",
        #"deepmind/code_contests",
        #"BAAI/TACO",
        #"NTU-NLP-sg/xCodeEval"
    ]
    models = [
        #"/export/code-pretrain/yun/transformers/models--WizardLM--WizardCoder-15B-V1.0/snapshots/9c177589dec389eac2c8de51cbc371d45e47984e",
        #"/export/code-pretrain/yun/transformers/models--WizardLM--WizardCoder-Python-13B-V1.0/snapshots/5ac6748b1f5a4c282107ddc7d3b69fdc4a686d75",
        #"/export/code-pretrain/yun/transformers/models--codellama--CodeLlama-13b-Instruct-hf/snapshots/e9066d1322d2aba257d935c3e30e1ca483b84d1f",
        #"/export/code-pretrain/yun/transformers/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55",
        #"/export/code-pretrain/yun/transformers/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496",
        #"/export/code-pretrain/yun/transformers/models--bigcode--starcoder/snapshots/b1af7f63dfbe5f2989b33399f1b99b58ff80a7d4",
        #"/export/code-pretrain/yun/transformers/models--bigcode--starcoder2-15b/snapshots/e18a21d65bd99f46b2d0f3b5913ccd386744c0b5",
        #"/export/code-pretrain/yun/transformers/models--Salesforce--xgen-7b-8k-inst/snapshots/943f44c31ffc2667253efca08a0cae7963333ce5",
        #"/export/code-pretrain/yun/transformers/models--Salesforce--codegen25-7b-mono/snapshots/94ea6ac9084111cc36856624c11fc6f51a37e7de",
        #"/export/code-pretrain/yun/transformers/models--google--gemma-7b-it/snapshots/b078e50c64458cf42df13eb97e368c00ce3a7b00",
        #"/export/code-pretrain/yun/transformers/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83",
        "/export/code-pretrain/yun/transformers/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
    ]

    for m in models:
        model = LocalModel(m, device_num, tensor_parallel_size = 8 if "Mixtral" in m else 4, n = n, temperature = temperature)
        for d in code_datasets:
            batch_infer_local_model(model, d, output_path, device_num, n = n, temperature = temperature, dataset_repo = dataset_repo, chat = True if "chat" in model.name else False, context_length = 2048 if "codegen25" in model.name else 8192)

def infer_all_local_models_for_all_prompts(output_path, rd, inputs_file = None, sample_num = 20):
    models = [
        "/export/code-pretrain/yun/transformers/models--WizardLM--WizardCoder-15B-V1.0/snapshots/9c177589dec389eac2c8de51cbc371d45e47984e",
        #"/export/code-pretrain/yun/transformers/models--WizardLM--WizardCoder-Python-13B-V1.0/snapshots/5ac6748b1f5a4c282107ddc7d3b69fdc4a686d75",
        #"/export/code-pretrain/yun/transformers/models--codellama--CodeLlama-13b-Instruct-hf/snapshots/e9066d1322d2aba257d935c3e30e1ca483b84d1f",
        #"/export/code-pretrain/yun/transformers/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55",
        #"/export/code-pretrain/yun/transformers/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496",
        #"/export/code-pretrain/yun/transformers/models--bigcode--starcoder/snapshots/b1af7f63dfbe5f2989b33399f1b99b58ff80a7d4",
        #"/export/code-pretrain/yun/transformers/models--bigcode--starcoder2-15b/snapshots/e18a21d65bd99f46b2d0f3b5913ccd386744c0b5",
        #"/export/code-pretrain/yun/transformers/models--Salesforce--xgen-7b-8k-inst/snapshots/943f44c31ffc2667253efca08a0cae7963333ce5",
        #"/export/code-pretrain/yun/transformers/models--Salesforce--codegen25-7b-mono/snapshots/94ea6ac9084111cc36856624c11fc6f51a37e7de",
        #"/export/code-pretrain/yun/transformers/models--google--gemma-7b-it/snapshots/b078e50c64458cf42df13eb97e368c00ce3a7b00",
        #"/export/code-pretrain/yun/transformers/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83"
    ]

    for m in models:
        if inputs_file == None:
            inputs_file = os.path.join(output_path, f"rd{rd-1}", m.replace("/", "_") + "_next_inputs.json")
        model = LocalModel(m, "0", tensor_parallel_size = 8 if "Mixtral" in m else 4, n = sample_num)
        #model = None
        batch_infer_local_model_prompts(model, rd, output_path, inputs_file, sample_num, context_length = 2048 if "codegen25" in model.name else 8192)

def test():
    data = json.load(open("prompt_chat_results/gpt-3.5-turbo/time/rd0_batch_outputs.json", "r"))
    for i in range(0, 5):
        print(f"in {i}")
        input_file = f"prompt_chat_results/gpt-3.5-turbo/time/rd0_batch_inputs_{i}.jsonl"
        with open(input_file, "r") as f:
            for line in f.readlines():
                input_data = json.loads(line)
                print(input_data["custom_id"])
                if "time_simple_execution_feedback" in input_data["custom_id"] or "time_execution_feedback_with_testcase" in input_data["custom_id"]:
                    print("!!!!")


def upload_file(model, prefix, num):
    model = RemoteModel(model)
    ids = []
    for i in range(0, num):
        response = model.upload_file(f"{prefix}{i}.jsonl")
        ids.append(response.id)
    print(ids)

def delete_file(model, ids):
    model = RemoteModel(model)
    for i in ids:
        model.delete_file(i)

def download_file(model, ids):
    model = RemoteModel(model)
    contents = []
    for i in ids:
        contents += model.download_file(i)
    data = {}
    for d in contents:
        data[d["custom_id"]] = d["response"]["body"]["choices"][0]["message"]
    
    errors = []
    for i in ids:
        errors += model.collect_errors(i)
    
    for e in errors:
        data[e["custom_id"]] = None

    with open("prompt_chat_results/gpt-4/time/rd1_batch_outputs.json", "w", encoding = "utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))


def batch_run(model, ids):
    model = RemoteModel(model)
    batch_ids = []
    for i in ids:
        response = model.batch_run(i)
        batch_ids.append(response.id)
    print(batch_ids)

def batch_status(model, ids):
    model = RemoteModel(model)
    for i in ids:
        model.batch_status(i)

def cancel_batch(model, ids):
    model = RemoteModel(model)
    for i in ids:
        response = model.batch_status(i)
        if response.status not in ["failed", "cancelling"]:
            model.cancel_batch(i)

def delete_batch(model, ids):
    model = RemoteModel(model)
    for i in ids:
        response = model.delete_batch(i)






if __name__ == "__main__":
    model = "gpt-4"
    #model = "/export/code-pretrain/yun/transformers/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298"
    #model = "/export/code-pretrain/yun/transformers/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/e8cf5276ae3e97cfde8a058e64a636f2cde47820"
    #model = "/export/code-pretrain/yun/transformers/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83"
    #model = "/export/code-pretrain/yun/transformers/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f"
    #model = "/export/code-pretrain/yun/transformers/models--WizardLM--WizardCoder-Python-13B-V1.0/snapshots/5ac6748b1f5a4c282107ddc7d3b69fdc4a686d75"
    #model = "/export/code-pretrain/yun/transformers/models--WizardLM--WizardCoder-15B-V1.0/snapshots/9c177589dec389eac2c8de51cbc371d45e47984e"
    #model = "/export/code-pretrain/yun/transformers/models--Salesforce--codegen25-7b-mono/snapshots/94ea6ac9084111cc36856624c11fc6f51a37e7de"
    #model = "/export/code-pretrain/yun/transformers/models--CohereForAI--c4ai-command-r-plus/snapshots/ba7f1d954c9d1609013677d87e4142ab95c34e62"
    #wait_available_gpu(num=4)
    #infer_openai_model("gpt-4", "openai_humaneval", "./results_temp0")
    #infer_openai_model_for_all("gpt-4", "./results")
    #infer_local_model("./transformers/models--WizardLM--WizardCoder-15B-V1.0/snapshots/9c177589dec389eac2c8de51cbc371d45e47984e", "codeparrot/apps", "./results", "0", chat = False)
    #batch_infer_local_model("./transformers/models--WizardLM--WizardCoder-15B-V1.0/snapshots/9c177589dec389eac2c8de51cbc371d45e47984e", "openai_humaneval", "./results", "0")
    #batch_infer_local_chat_model(model, "time", 1, "prompt_chat_results/_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62/time/rd1_batch_inputs.json", "prompt_chat_results")
    #batch_infer_local_prompt_model(model, "correctness", 0, "prompt_model_results/_export_code-pretrain_yun_transformers_models--WizardLM--WizardCoder-Python-13B-V1.0_snapshots_5ac6748b1f5a4c282107ddc7d3b69fdc4a686d75/correctness/rd0_batch_inputs.json", "prompt_model_results")
    #infer_local_model_for_all("./transformers/models--WizardLM--WizardCoder-15B-V1.0/snapshots/9c177589dec389eac2c8de51cbc371d45e47984e", "./results", "0")
    #infer_all_local_models_for_all("./results", "0", n = 20, temperature = 0.7, dataset_repo = "test_datasets")
    #infer_all_local_models_for_all_prompts("prompt_results2", 2)#, "results_sample1000/best_inputs.json")
    infer_openai_model_prompts(model, "time", 1, "prompt_chat_results", f"prompt_chat_results/gpt-4/time/rd0.json", 20, batch_split = True, checked_inputs_file = "prompt_chat_results/gpt-4/time/rd0_CHECKED.json", history_file = "prompt_chat_results/gpt-4/time/rd1_HISTORY.json")
    #infer_openai_model_prompts(model, "time", 4, "prompt_chat_results", f"prompt_chat_results/gpt-3.5-turbo/time/rd3.json", 20, batch_split = True, checked_inputs_file = f"prompt_chat_results/gpt-3.5-turbo/time/rd3_CHECKED.json", history_file = "prompt_chat_results/gpt-3.5-turbo/time/rd4_HISTORY.json", leader_history_file = "prompt_chat_results/gpt-3.5-turbo/time/rd4_HISTORY_LEADER.json", reviewer_history_file = "prompt_chat_results/gpt-3.5-turbo/time/rd4_HISTORY_REVIEWER.json")
    #upload_file(model, "prompt_chat_results/gpt-4/time/rd1_batch_inputs_", 10)
    #delete_file(model, ['file-wH0IcgnhE6oLJO9bCVHLdWl4', 'file-c4HKmGZccF5XBlqooeCbM9Bt', 'file-5louWObZlF9p8z2DuuJthd9U', 'file-L1nM1ae0akq4u6BWzl8Ody0y', 'file-rfrZpdRB8SbOBygGqKMnsKqL', 'file-ChsV2ILc9ydpV8zi3L90dstK', 'file-FAPGPuHrHLFSRldnEFcJ6QMy', 'file-w48yGmWUbMmi8h4Wz3VpP40L', 'file-MN0ijRSPmfYMWEiznMNP6hoo', 'file-pq9q46WjSfrxOlRl3gVuM0Of'])
    #batch_run(model,['file-xNOdVXvwvdM3GNcbZe9bitjd'])
    #batch_status(model,['batch_WOhKEZnW9VnMIkQDf3SyLnvC', 'batch_0FUTxfvuZ3weL94Fo9TQ8v7d', 'batch_pDuDmLnGKa6VI1QSLujZ5dLE', 'batch_bCpR9gYjabkyT6v66t8zHMG1', 'batch_W1Wofb802JywtmfuLT1ibXDA', 'batch_50QIw8xzJtot1gRJjORWZ2kY'])
    #batch_status(model,  ['batch_rcMhrnxmRj0LWJvoDegdOBvM', 'batch_iePweAExDrOEMnWvLiini5GJ', 'batch_lzmIlcLtWoC2O9EL2KLXTm79', 'batch_LPujDfW2qLAQm8NBV5wX4iKq'])
    #cancel_batch(model,['batch_6emAoOwjgLrtRsypIMHFVDqC', 'batch_UMxigylOeWczDEJXON7I5cXY', 'batch_2hsROerAAt0i9EKGnWSysEfl', 'batch_RZ4ocvGbQazAVTOmg4TBYRrT', 'batch_X5nferOsNspSAxfsZfAdtafc', 'batch_9IBf77qysLsuRcIuva0jR7EI', 'batch_DTsePOZPoJfFwq2ilXtkVlFD', 'batch_cOHCwMCUGGWvOZRLwm6jzerP'])
    #download_file(model, ['batch_rcMhrnxmRj0LWJvoDegdOBvM', 'batch_WOhKEZnW9VnMIkQDf3SyLnvC', 'batch_0FUTxfvuZ3weL94Fo9TQ8v7d', 'batch_pDuDmLnGKa6VI1QSLujZ5dLE', 'batch_bCpR9gYjabkyT6v66t8zHMG1', 'batch_W1Wofb802JywtmfuLT1ibXDA', 'batch_50QIw8xzJtot1gRJjORWZ2kY', 'batch_LPujDfW2qLAQm8NBV5wX4iKq'])
    #delete_batch(model, ['batch_noFR9fgsU1yl76cgs5Ebqy6x', 'batch_gjgLUuxTrk6uisl58n0gqPGz'])
    #test()

    
    


