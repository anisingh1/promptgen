from datasets import load_dataset
import datasets
import os
import json
from sanitize import CodeVisitor, CodeProcessor
import ast
import sys
import time


code_datasets = [
    "openai_humaneval",
    "codeparrot/apps",
    "mbpp",
    #"deepmind/code_contests",
    #"BAAI/TACO",
    #"NTU-NLP-sg/xCodeEval"
]


class Dataset(object):
    def __init__(self, name, data_path = None, testfile_path = None, full = True, train_description_path = None, tmp_dir = "./tmp") -> None:
        self.name = name
        self.index = -1
        self.tmp_dir = tmp_dir
        self.data_path = data_path
        if not data_path:
            self.dataset = load_dataset(name, cache_dir = "./datasets")
        else:
            self.dataset = datasets.load_from_disk(data_path)

        if testfile_path and name == "NTU-NLP-sg/xCodeEval":
            self.testcases = json.load(open(testfile_path, "r"))
            if train_description_path != None:
                self.descriptions = json.load(open(train_description_path, "r"))
        elif testfile_path and name in ["mbpp", "openai_humaneval"]:
            raw_data = open(testfile_path, "r").read().splitlines()
            data = []
            for line in raw_data:
                data.append(json.loads(line))
            self.testcases = {}
            for instance in data:
                if self.name == "mbpp":
                    self.testcases[instance["task_id"].replace("HumanEval/", "").replace("Mbpp/", "")] = {"inputs": instance["base_input"], "outputs": None, "entry_point": instance["entry_point"]}
                else:
                    self.testcases[instance["task_id"].replace("HumanEval/", "").replace("Mbpp/", "")] = {"inputs": instance["base_input"], "outputs": None}
        else:
            self.testcases = {}

    
        self.code_keywords = {
            "openai_humaneval": "canonical_solution",
            "codeparrot/apps": "solutions",
            "mbpp": "code",
            "deepmind/code_contests": "solutions",
            "BAAI/TACO": "solutions",
            "NTU-NLP-sg/xCodeEval": None
        }

        self.testcase_keywords = {
            "openai_humaneval": "testcases",
            "codeparrot/apps": "input_output",
            "mbpp": "testcases",
            "deepmind/code_contests": ["private_tests", "generated_tests"],
            "BAAI/TACO": "input_output",
            "NTU-NLP-sg/xCodeEval": "testcases"
        }

        self.add_list = {
            "openai_humaneval": False,
            "codeparrot/apps": True,
            "mbpp": False,
            "deepmind/code_contests": True,
            "BAAI/TACO": True,
            "NTU-NLP-sg/xCodeEval": True,
        }

        if full:
            self.prompt2instance = self.get_prompt2instance()
            self.prompt2groundtruth = {}
            self.prompt2testcase = {}
            self.prompt2io = {}
            

        self.reset_index()

    def reset_index(self):
        self.index = -1

    def length(self):
        return len(self.dataset)

    def next(self):
        self.index += 1
        finish = False
        if self.index == len(self.dataset) - 1:
            finish = True

        instance = {}
        if self.name == "NTU-NLP-sg/xCodeEval" and len(self.testcases) > 0:
            for key in self.dataset[self.index]:
                instance[key] = self.dataset[self.index][key]
            instance["testcases"] = self.testcases[instance["src_uid"]]
        elif self.name in ["openai_humaneval", "mbpp"] and len(self.testcases) > 0:
            for key in self.dataset[self.index]:
                instance[key] = self.dataset[self.index][key]
            if str(instance["task_id"]).replace("HumanEval/", "") in self.testcases:
                instance["testcases"] = self.testcases[str(instance["task_id"]).replace("HumanEval/", "")]
                if "entry_point" in instance["testcases"]:
                    instance["entry_point"] = instance["testcases"]["entry_point"]
            else:
                instance["testcases"] = None
        else:
            instance = self.dataset[self.index]


        return instance, finish

    def get_function_signature(self, instance):
        if self.name == "mbpp":
            code = instance["code"]
            lines = code.splitlines()
            if "entry_point" in instance:
                for line in lines:
                    if line.startswith("def") and instance["entry_point"] in line:
                        return line.replace("def", "").replace(":", "").strip()
            else:
                visitor = CodeVisitor(code)
                visitor.run()
                for line in lines:
                    if line.startswith("def") and visitor.funcs[-1] in line:
                        return line.replace("def", "").replace(":", "").strip()
        elif self.name in ["codeparrot/apps", "BAAI/TACO"]:
            code = instance["starter_code"]
            lines = code.splitlines()
            for line in lines:
                if line.startswith("def"):
                    return line.replace("def", "").replace(":", "").split("->")[0].strip()
        return None
    
    def get_prompt(self, instance):
        prompt = ""
        if self.name == "openai_humaneval":
            prompt += instance["prompt"]
        if self.name in ["codeparrot/apps", "BAAI/TACO"]:
            prompt += "You are an expert Python programmer, and here is your task:\n"
            prompt += instance["question"]
            if len(instance["starter_code"]) > 0:
                prompt += "\nPlease write a Python function {} for the task.\n```python".format(self.get_function_signature(instance))
            else:
                prompt += "\nDo not give explanations, only give the Python code.\nPython Solution:\n```python\n"
        if self.name == "mbpp":
            signature = self.get_function_signature(instance)
            prompt += "You are an expert Python programmer, and here is your task: {} Please write a Python function {} for the task.\n```python".format(instance["prompt"], signature if signature else "")
        if self.name == "deepmind/code_contests":
            prompt += "You are an expert Python programmer, and here is your task:\n"
            prompt += instance["description"]
            prompt += "\nDo not give explanations, only give the Python code.\nPython Solution:\n```python\n"
        if self.name == "NTU-NLP-sg/xCodeEval":
            prompt += "You are an expert Python programmer, and here is your task:\n"
            prompt += instance["description"]
            prompt += "\n\nThe code you write should take inputs from {} and output to {}.\n".format(instance["input_from"], instance["output_to"])
            prompt += "\n\nInput: " + instance["input_spec"]
            prompt += "\n\nOutput: " + instance["output_spec"]
            if instance["notes"] != None:
                prompt += "\n\nNote: " + instance["notes"]
            if len(instance["sample_inputs"]) > 0:
                prompt += "\n\nExample:"
                for i, ip in enumerate(instance["sample_inputs"]):
                    prompt += "\nInput: {}\nOutput: {}".format(ip, instance["sample_outputs"][i])
            prompt += "\nDo not give explanations, only give the Python code.\nPython Solution:\n```python\n"
        
        prompt = prompt.strip()

        return prompt

    def get_chat(self, instance):
        return [{"role": "user", "content": self.get_prompt(instance)}]

    def get_prompt_for_current_instance(self):
        return self.get_prompt(self.dataset[self.index])
    
    def get_all_prompts(self, model = None, context_length = None):
        self.reset_index()
        prompts = []
        finish = False
        while(not finish):
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            prompts.append(prompt)

        new_prompts = []
        overlong_prompts = []

        if model != None and context_length != None:
            for p in prompts:
                if model.get_prompt_length(p) >= context_length:
                    overlong_prompts.append(p)
                else:
                    new_prompts.append(p)
        else:
            new_prompts += prompts
            
        self.reset_index()
        
        return new_prompts, overlong_prompts

    def get_all_chats(self):
        self.reset_index()
        dialogs = []
        finish = False
        while(not finish):
            instance, finish = self.next()
            dialog = self.get_chat(instance)
            dialogs.append(dialog)

        self.reset_index()
        
        return dialogs


    def get_prompt2instance(self):
        self.reset_index()
        finish = False
        prompt2instance = {}
        while(not finish):
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            prompt2instance[prompt] = instance

        self.reset_index()

        return prompt2instance

    def get_groundtruth(self, prompt, raw_solutions = None):
        if prompt not in self.prompt2instance:
            raise ValueError("Cannot find the prompt in the dataset.")

        
        
        solutions = []


        if len(self.prompt2testcase[prompt]) == 0:
            return solutions

        if self.code_keywords[self.name] != None:
            raw = self.prompt2instance[prompt][self.code_keywords[self.name]]
            if isinstance(raw, str) and self.name not in ["BAAI/TACO", "codeparrot/apps"]:
                solutions.append(raw if self.name != "openai_humaneval" else self.prompt2instance[prompt]["prompt"] + raw)
            elif isinstance(raw, str) and self.name in ["BAAI/TACO", "codeparrot/apps"]:
                if len(raw) > 0:
                    solutions = json.loads(raw)
                else:
                    solutions = []
            elif isinstance(raw, list) and isinstance(raw[0], str):
                solutions += raw
            elif isinstance(raw, list) and isinstance(raw[0], dict):
                for r in raw:
                    if "language" in r and r["language"] == "PYTHON":
                        solutions.append(r["solution"])
            elif isinstance(raw, dict) and "language" in raw and "solution" in raw:
                for index, lang in enumerate(raw["language"]):
                    if lang == 1:
                        solutions.append(raw["solution"][index])
            else:
                raise ValueError("Unknown solution format.")
        else:
            if self.prompt2instance[prompt]["src_uid"] in raw_solutions:
                solutions = raw_solutions[self.prompt2instance[prompt]["src_uid"]]

        new_solutions = []
        for solution in solutions:
            solution = solution.replace("\t", "    ")
            try:
                ast.parse(solution)
                if self.name in ["deepmind/code_contests"]:
                    raise
                new_solutions.append(solution)
            except Exception as e:
                #print(str([solution]).replace("\'", "\""))
                if self.name == "deepmind/code_contests":
                    if not os.path.exists(self.tmp_dir):
                        os.mkdir(self.tmp_dir)
                    with open(os.path.join(self.tmp_dir, "temp.py"), "w") as f:
                        f.write(solution)
                    os.system("2to3 -w -n {}".format(os.path.join(self.tmp_dir, "temp.py")))
                    solution = open(os.path.join(self.tmp_dir, "temp.py"), "r").read()
                    os.system("rm {}".format(os.path.join(self.tmp_dir, "temp.py")))
                    new_solutions.append(solution)
                else:
                    print("Solution skipped for parse error: {}".format(e))

        solutions = new_solutions
        new_solutions = []
        for solution in solutions:
            if self.name in ["codeparrot/apps", "BAAI/TACO"]:
                processor = CodeProcessor(solution, entry_point = self.prompt2testcase[prompt][0]["entry_point"], force_rename = True if self.prompt2testcase[prompt][0]["entry_point"] != None else False)
            else:
                processor = CodeProcessor(solution, entry_point = self.prompt2instance[prompt]["entry_point"] if "entry_point" in self.prompt2instance[prompt] else None, force_rename = True if self.name in ["openai_humaneval", "mbpp"] else False)
            new_solution = processor.run()
            if new_solution[0] == -1:
                continue
            new_solutions.append(new_solution)

        return new_solutions

    def get_prompt2groundtruth(self, solution_file = None):
        self.reset_index()
        if solution_file:
            solutions = {}
            with open(solution_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    solution = json.loads(line)
                    if not solution["lang"].startswith("Py"):
                        continue
                    if solution["src_uid"] not in solutions:
                        solutions[solution["src_uid"]] = []
                    solutions[solution["src_uid"]].append(solution["source_code"])
        else:
            solutions = {}


        finish = False
        prompt2groundtruth = {}
        prompt2io = {}
        count = 0
        while(not finish):
            print("Processing instance#{}".format(count), end = "\r", file = sys.stderr)
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            prompt2groundtruth[prompt] = self.get_groundtruth(prompt, raw_solutions = solutions)
            prompt2io[prompt] = False
            for solution in prompt2groundtruth[prompt]:
                visitor = CodeVisitor(solution[0])
                visitor.run()
                if visitor.has_input:
                    prompt2io[prompt] = True
                    break
            count += 1

        self.reset_index()

        return prompt2groundtruth, prompt2io


    def format_strs(self, lst):
        if isinstance(lst, str):
            return lst.replace("\r\n", "\n")
        elif isinstance(lst, list):
            new_lst = []
            for l in lst:
                new_lst.append(self.format_strs(l))
            return new_lst
        else:
            return lst
        


    def format_testcase(self, prompt, testcase, multiple = False):
        if not multiple and isinstance(testcase, dict) and "input" in testcase and "output" in testcase:
            new_testcase = {}
            inputs = self.format_strs(testcase["input"])
            outputs = self.format_strs(testcase["output"])
            if self.add_list[self.name]:
                new_testcase["input"] = [inputs]
                new_testcase["output"] = outputs
            else:
                new_testcase["input"] = inputs
                new_testcase["output"] = outputs[0]
        
            return new_testcase
        elif multiple and isinstance(testcase, dict):
            new_testcases = []
            if "inputs" in testcase:
                inputs = self.format_strs(testcase["inputs"])
                outputs = self.format_strs(testcase["outputs"])
            elif "input" in testcase:
                inputs = self.format_strs(testcase["input"])
                outputs = self.format_strs(testcase["output"])


            if self.name in ["codeparrot/apps", "BAAI/TACO"] and "fn_name" not in testcase:
                new_inputs = []
                for inp in inputs:
                    if isinstance(inp, list):
                        new_inputs.append("\n".join(inp))
                    else:
                        new_inputs.append(inp)
                inputs = new_inputs
                new_outputs = []
                for op in outputs:
                    if isinstance(op, list):
                        new_outputs.append("\n".join(op))
                    else:
                        new_outputs.append(op)
                outputs = new_outputs
            
            for index, inp in enumerate(inputs):
                new_testcase = {}
                if self.add_list[self.name]:
                    if self.name == "codeparrot/apps":
                        if "fn_name" not in testcase:
                            new_testcase["input"] = [inp]
                            new_testcase["output"] = [None] if outputs == None else [outputs[index]]
                            new_testcase["entry_point"] = None
                        else:
                            new_testcase["input"] = inp
                            new_testcase["output"] = [None] if outputs == None else outputs[index]
                            if not isinstance(new_testcase["output"], list):
                                new_testcase["output"] = [new_testcase["output"]]
                            new_testcase["entry_point"] = testcase["fn_name"]
                    elif self.name == "BAAI/TACO":
                        if "fn_name" not in testcase:
                            new_testcase["input"] = [inp]
                            if not isinstance(outputs[index], list):
                                new_testcase["output"] = [None] if outputs == None else [outputs[index]]
                            else:
                                new_testcase["output"] = [None] if outputs == None else outputs[index]
                            new_testcase["entry_point"] = None
                        else:
                            new_testcase["input"] = inp
                            new_testcase["output"] = [None] if outputs == None else outputs[index]
                            if not isinstance(new_testcase["output"], list):
                                new_testcase["output"] = [new_testcase["output"]]
                            new_testcase["entry_point"] = testcase["fn_name"]
                    else:
                        new_testcase["input"] = [inp]
                        new_testcase["output"] = [None] if outputs == None else [outputs[index]]
                else:
                    new_testcase["input"] = inp
                    new_testcase["output"] = [None] if outputs == None else outputs[index]
                new_testcases.append(new_testcase)
            return new_testcases
                
            
    def get_testcases(self, prompt):
        if prompt not in self.prompt2instance:
            raise ValueError("Cannot find the prompt in the dataset.")
        
        testcases = []

        key = self.testcase_keywords[self.name]
        if isinstance(key, str):
            if isinstance(self.prompt2instance[prompt][key], list):
                for testcase in self.prompt2instance[prompt][key]:
                    testcases.append(self.format_testcase(prompt, testcase, multiple = False))
            elif isinstance(self.prompt2instance[prompt][key], dict):
                testcases += self.format_testcase(prompt,self.prompt2instance[prompt][key], multiple = True)
            elif isinstance(self.prompt2instance[prompt][key], str):
                if len(self.prompt2instance[prompt][key].strip()) == 0:
                    return testcases
                try:
                    testcases += self.format_testcase(prompt, json.loads(self.prompt2instance[prompt][key]), multiple = True)
                except Exception as e:
                    print(e)
                    return testcases
            elif self.prompt2instance[prompt][key] == None:
                pass
            else:
                raise ValueError("Unsupported testcase format: {}.".format(type(self.prompt2instance[prompt][key])))
        elif isinstance(key, list):
            if len(self.prompt2instance[prompt][key[0]]) == 0 or ("input" in self.prompt2instance[prompt][key[0]] and len(self.prompt2instance[prompt][key[0]]["input"]) == 0):
                testcases += self.format_testcase(prompt,self.prompt2instance[prompt][key[1]], multiple = True)
            else:
                testcases += self.format_testcase(prompt,self.prompt2instance[prompt][key[0]], multiple = True)
                testcases += self.format_testcase(prompt,self.prompt2instance[prompt][key[1]], multiple = True)
                
        return testcases


    def save_testcases(self, file_path = None):
        if len(self.prompt2testcase) == 0:
            prompts, overlong_prompts = self.get_all_prompts()
            for i, p in enumerate(prompts + overlong_prompts):
                self.prompt2testcase[p] = self.get_testcases(p)
        
        if file_path != None:
            with open(file_path, "w") as f:
                f.write(json.dumps(self.prompt2testcase, sort_keys=True, indent=4, separators=(',', ': ')))
        elif self.data_path != None:
            with open(os.path.join(self.data_path, "testcases.json"), "w") as f:
                f.write(json.dumps(self.prompt2testcase, sort_keys=True, indent=4, separators=(',', ': ')))
        else:
            raise ValueError("Cannot find the file path for test cases.")


    def save_groundtruths(self, file_path  = None, solution_file = None):
        if len(self.prompt2groundtruth) == 0:
            self.prompt2groundtruth, self.prompt2io = self.get_prompt2groundtruth(solution_file = solution_file)

        data = {"prompt2groundtruth": self.prompt2groundtruth, "prompt2io": self.prompt2io}
        
        if file_path != None:
            with open(file_path, "w") as f:
                f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
        elif self.data_path != None:
            with open(os.path.join(self.data_path, "solutions.json"), "w") as f:
                f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
        else:
            raise ValueError("Cannot find the file path for ground truth solutions.")

    def save_prompt2id(self, file_path = None):
        self.reset_index()
        finish = False
        prompt2id = {}
        while(not finish):
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            if self.name == "BAAI/TACO":
                prompt2id[prompt] = instance["question"]
            elif self.name == "codeparrot/apps":
                prompt2id[prompt] = instance["problem_id"]
            elif self.name == "deepmind/code_contests":
                prompt2id[prompt] = instance["name"]
            elif self.name in ["mbpp", "openai_humaneval"]:
                prompt2id[prompt] = instance["task_id"]
            elif self.name == "NTU-NLP-sg/xCodeEval":
                prompt2id[prompt] = instance["description"]
        
        self.reset_index()

        if file_path != None:
            if os.path.exists(file_path):
                new_filepath = "{}_{}.json".format(file_path.replace(".json", ""), time.time())
                print("Warning! {} already exists and renamed to {} to avoid overwriting.".format(file_path, new_filepath))
                os.system("mv {} {}".format(file_path, new_filepath))
            with open(file_path, "w") as f:
                f.write(json.dumps(prompt2id, sort_keys=True, indent=4, separators=(',', ': ')))
        elif self.data_path != None:
            file_path = os.path.join(self.data_path, "prompt2id.json")
            if os.path.exists(file_path):
                new_filepath = "{}_{}.json".format(file_path.replace(".json", ""), time.time())
                print("Warning! {} already exists and renamed to {} to avoid overwriting.".format(file_path, new_filepath))
                os.system("mv {} {}".format(file_path, new_filepath))
            with open(file_path, "w") as f:
                f.write(json.dumps(prompt2id, sort_keys=True, indent=4, separators=(',', ': ')))
        else:
            raise ValueError("Cannot find the file path for prompt2id.")
        

    def load_testcases(self, file_path = None):
        if file_path != None:
            self.prompt2testcase = json.load(open(file_path, "r"))
        elif self.data_path != None:
            self.prompt2testcase = json.load(open(os.path.join(self.data_path, "testcases.json"), "r"))
        else:
            raise ValueError("Cannot find the file path for test cases.")

    def load_groundtruths(self, file_path = None):
        if file_path != None:
            data = json.load(open(file_path, "r"))
            self.prompt2groundtruth = data["prompt2groundtruth"]
            self.prompt2io = data["prompt2io"]
        elif self.data_path != None:
            data = json.load(open(os.path.join(self.data_path, "solutions.json"), "r"))
            self.prompt2groundtruth = data["prompt2groundtruth"]
            self.prompt2io = data["prompt2io"]
        else:
            raise ValueError("Cannot find the file path for test cases.")

    def remove_solutions_with_similar_time_costs(self, threshold = 0.1):
        self.load_groundtruths()
        time_costs = json.load(open(os.path.join(self.data_path, "time_costs.json"), "r"))
        prompts, overlong_prompts = self.get_all_prompts()

        for prompt in (prompts + overlong_prompts):
            if prompt in time_costs and prompt in self.prompt2groundtruth and len(time_costs[prompt]) == len(self.prompt2groundtruth[prompt]) and len(time_costs[prompt]) > 0:
                selected_index = [time_costs[prompt].index(min(time_costs[prompt]))]
                while True:
                    min_time = 9999999
                    min_index = -1
                    for i, t in enumerate(time_costs[prompt]):
                        if i in selected_index:
                            continue
                        if t > time_costs[prompt][selected_index[-1]] and (t - time_costs[prompt][selected_index[-1]]) / t > threshold and t < min_time:
                            min_time = t
                            min_index = i
                    if min_index != -1:
                        selected_index.append(min_index)
                    else:
                        break
                new_solutions = [self.prompt2groundtruth[prompt][i] for i in selected_index]
                self.prompt2groundtruth[prompt] = new_solutions
            else:
                self.prompt2groundtruth[prompt] = []
        
        data = {"prompt2groundtruth": self.prompt2groundtruth, "prompt2io": self.prompt2io}

        with open(os.path.join(self.data_path, "solutions_time_different.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))

    
    def get_solution_num_distribution(self, solution_file = None):
        self.load_groundtruths(file_path = solution_file)
        prompts, overlong_prompts = self.get_all_prompts()

        data = {}

        max_num = 0
        for prompt in (prompts + overlong_prompts):
            if prompt in self.prompt2groundtruth and len(self.prompt2groundtruth[prompt]) > 0:
                if len(self.prompt2groundtruth[prompt]) not in data:
                    data[len(self.prompt2groundtruth[prompt])] = 0
                data[len(self.prompt2groundtruth[prompt])] += 1
                if len(self.prompt2groundtruth[prompt]) > max_num:
                    max_num = len(self.prompt2groundtruth[prompt])
        
        lines = ["Solution_Num,#Problems"]
        for i in range(1, max_num + 1):
            if i in data:
                lines.append(f"{i},{data[i]}")
            else:
                lines.append(f"{i},0")
        
        with open(os.path.join(self.data_path, "solution_dist_time_different.csv"), "w", encoding = "utf-8") as f:
            f.write("\n".join(lines))
                

    def print_info(self):
        self.load_testcases()
        self.load_groundtruths()

        prompts, overlong_prompts = self.get_all_prompts()
        empty_solution = 0
        empty_testcase = 0
        empty_nofix_solution = 0
        total_solution = 0
        total_testcase = 0
        total_nofix_solution = 0
        total_num = len(prompts + overlong_prompts)

        if os.path.exists(os.path.join(self.data_path, "solutions_nofix.json")):
            nofix_solutions = json.load(open(os.path.join(self.data_path, "solutions_nofix.json"), "r"))["prompt2groundtruth"]
        else:
            nofix_solutions = {}


        for prompt in (prompts + overlong_prompts):
            if prompt not in nofix_solutions or len(nofix_solutions[prompt]) == 0:
                empty_nofix_solution += 1
            else:
                total_nofix_solution += len(nofix_solutions[prompt])
            if prompt not in self.prompt2groundtruth or len(self.prompt2groundtruth[prompt]) == 0:
                empty_solution += 1
            else:
                total_solution += len(self.prompt2groundtruth[prompt])
            if prompt not in self.prompt2testcase or len(self.prompt2testcase[prompt]) == 0 or prompt not in self.prompt2groundtruth or len(self.prompt2groundtruth[prompt]) == 0:
                empty_testcase += 1
            else:
                total_testcase += len(self.prompt2testcase[prompt])

        print("Total instance: {}\nBefore Fix:========\nInstance with solutions: {}\nAfter Fix:========\nInstance with solutions: {}\nInstance with testcases: {}\nAverage solutions: {}\nAverage testcases: {}".format(total_num, total_num - empty_nofix_solution, total_num - empty_solution, total_num - empty_testcase, total_solution / (total_num - empty_solution), total_testcase / (total_num - empty_testcase)))


    
    def get_best_groundtruth(self, time_cost_file):
        time_costs = json.load(open(time_cost_file, "r"))
        self.load_groundtruths()

        best_solutions = {}

        for prompt in self.prompt2groundtruth:
            if prompt not in time_costs:
                continue
            if len(time_costs[prompt]) == 0 or len(time_costs[prompt]) != len(self.prompt2groundtruth[prompt]) or len(self.prompt2groundtruth[prompt]) == 0:
                continue
            best_solutions[prompt] = self.prompt2groundtruth[prompt][time_costs[prompt].index(min(time_costs[prompt]))]
        
        with open(os.path.join(self.data_path, "best_solutions.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(best_solutions, sort_keys=True, indent=4, separators=(',', ': ')))
            
            

        
