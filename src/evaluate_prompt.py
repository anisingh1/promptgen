from sanitize import sanitize, CodeProcessor
from prompt import Prompt
from evaluate import Evaluator
import json
from dataset import Dataset
import os
import ast
import itertools
import numpy as np

class ChatPromptEvaluator(object):
    def __init__(self, sample_num = 20):
        self.prompt_generator = Prompt(chat=True)
        self.prompt_generator.load_templates()
        self.sample_num = sample_num

    def get_entrypoint(self, dataset, instance):
        entry_point = ""
        if dataset.name == "openai_humaneval":
            entry_point += instance["entry_point"]

        return entry_point

    def process_solution(self, solution, instance, dataset):
        if dataset.name == "openai_humaneval" and not solution.startswith("def"):
            lines = solution.splitlines()
            if len(lines) > 0:
                lines[0] = "    " + lines[0]
            solution = instance["prompt"] + "\n".join(lines)
        elif dataset.name == "openai_humaneval":
            try:
                ast.parse(solution)
            except Exception as e:
                lines = solution.splitlines()
                if len(lines) > 0:
                    lines[0] = "    " + lines[0]
                solution = instance["prompt"] + "\n".join(lines)

        return solution

    def build_new_status(self):
        return {
            "status": "Unchecked",
            "passed_solution": None,
            "exec_time": None,
            "failed_testcase": [],
            "large_testcase": None,
            "notes": {} 
        }

    def is_unchecked(self, lst):
        for status in lst:
            if isinstance(status, str) or status == None:
                return False
            if status["status"] not in ["Unchecked", "PARSE_ERROR"]:
                return False

        return True

    def find_result(self, dataset, prompt, name, rd, i, checked_outputs):
        cur_name = name
        if rd == 0:
            if name.startswith("correctness"):
                cur_name = "base"
                return checked_outputs[dataset][prompt][cur_name][rd][i], [cur_name, rd, i]
            elif name.startswith("time"):
                cur_name = "correct_solutions"
                return checked_outputs[dataset][prompt][cur_name][i], [cur_name, 0, i]
        elif rd > 0:
            if checked_outputs[dataset][prompt][cur_name][rd-1][i] == "<END>" or isinstance(checked_outputs[dataset][prompt][cur_name][rd-1][i], str):
                return self.find_result(dataset, prompt, name, rd-1, i, checked_outputs)
            else:
                return checked_outputs[dataset][prompt][cur_name][rd-1][i], [cur_name, rd, i]


    def write_batch_outputs(self, batch_output_file, output_file, hash_file):
        batch_outputs = json.load(open(batch_output_file, "r"))
        outputs = json.load(open(output_file, "r"))
        hash_map = json.load(open(hash_file, "r"))

        for dataset in outputs:
            for prompt in outputs[dataset]:
                for name in outputs[dataset][prompt]:
                    for j, res in enumerate(outputs[dataset][prompt][name]):
                        for i, r in enumerate(res):
                            if r == "<PENDING>":
                                if name != "base":
                                    sig = f"{dataset}@{name}@{hash_map[prompt]}@{j}@{i}"
                                    if sig not in batch_outputs:
                                        print("Cannot find signature {} pending query, skipping.".format(sig))
                                        outputs[dataset][prompt][name][j][i] = None
                                        continue
                                    if batch_outputs[sig] != None:
                                        outputs[dataset][prompt][name][j][i] = batch_outputs[sig]["content"]
                                    else:
                                        outputs[dataset][prompt][name][j][i] = None
                                else:
                                    sig = f"{dataset}@{name}@{hash_map[prompt]}@{j}"
                                    if sig not in batch_outputs:
                                        print("Cannot find signature {} pending query, skipping.".format(sig))
                                        outputs[dataset][prompt][name][j][i] = None
                                        continue
                                    if batch_outputs[sig] != None:
                                        outputs[dataset][prompt][name][j][i] = batch_outputs[sig]["content"][i]
                                    else:
                                        outputs[dataset][prompt][name][j][i] = None
        
        with open(output_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(outputs, sort_keys=True, indent=4, separators=(',', ': ')))
        


    def get_solutions(self, outputs, dataset, prompt, no_modify = False, chat = True, codegen = False):
        solutions = []
        if prompt not in dataset.prompt2instance:
            print('Cannot find the prompt of instance in dataset, skipped.')
            return solutions
        instance = dataset.prompt2instance[prompt]
        for out in outputs:
            if out == None:
                solutions.append([-1, False])
            else:
                code = sanitize(out, self.get_entrypoint(dataset, instance), codegen = codegen, global_code = True if dataset.name in ["codeparrot/apps"] and not codegen else False, chat = chat)
                code = self.process_solution(code, instance, dataset)
                processor = CodeProcessor(code,entry_point = dataset.prompt2instance[prompt]["entry_point"] if "entry_point" in dataset.prompt2instance[prompt] else None, force_rename = True if dataset.name in ["openai_humaneval", "mbpp"] else False)
                solutions.append(processor.run(no_modify = no_modify))
        return solutions

    def prepare_correctness_check(self, mode, rd, output_file, checked_file = None, chat = True, codegen = False):
        if checked_file:
            self.checked_outputs = json.load(open(checked_file, "r"))
        else:
            self.checked_outputs = {}

        self.outputs = json.load(open(output_file, "r"))

        indexes = {}

        solutions = {}
        
        for dataset in self.outputs:
            print("Processing dataset: {}".format(dataset))
            dataset_obj = Dataset(dataset, data_path = os.path.join("test_datasets/", dataset.replace("/", "_")), testfile_path = "datasets/MbppPlus.jsonl" if dataset == "mbpp" else None)
            solutions[dataset] = {}
            indexes[dataset] = {}
            num = 0
            failed_num = 0
            if dataset not in self.checked_outputs:
                self.checked_outputs[dataset] = {}
            for prompt in self.outputs[dataset]:
                solutions[dataset][prompt] = []
                indexes[dataset][prompt] = {"failed": []}
                if prompt not in self.checked_outputs[dataset] and checked_file == None:
                    self.checked_outputs[dataset][prompt] = {}
                elif prompt not in self.checked_outputs[dataset]:
                    continue
                for name in self.outputs[dataset][prompt]:
                    if mode != name and not name.startswith(mode):
                        if name in self.checked_outputs[dataset][prompt]:
                            continue
                        else:
                            raise ValueError("Checked outputs missing category: {}".format(name))
                    if name not in self.checked_outputs[dataset][prompt]:
                        self.checked_outputs[dataset][prompt][name] = []
                    if name == "time_original":
                        self.checked_outputs[dataset][prompt][name] = self.outputs[dataset][prompt][name]
                        processor = CodeProcessor(self.checked_outputs[dataset][prompt][name], force_rename = True if dataset in ["openai_humaneval", "mbpp"] else False)
                        ori = processor.run(no_modify = True)
                        if ori[0] == -1:
                            raise ValueError("Original solution parse error!")
                        solutions[dataset][prompt].append(ori)
                        indexes[dataset][prompt][ori[0]] = [[name, 0, 0]]
                        continue
                    for i, out in enumerate(self.outputs[dataset][prompt][name]):
                        if i < rd:
                            if len(self.checked_outputs[dataset][prompt][name]) <= i:
                                self.checked_outputs[dataset][prompt][name].append(out)
                            continue
                        if self.prompt_generator.gen_code(i, name):
                            if len(self.checked_outputs[dataset][prompt][name]) > i:
                                continue
                            temp_solutions = self.get_solutions(out, dataset_obj, prompt, chat = chat, codegen = codegen)
                            temp_checked = []
                            for j, ts in enumerate(temp_solutions):
                                if out[j] == None:
                                    status = self.build_new_status()
                                    status["status"] = "QUERY_ERROR"
                                    temp_checked.append(status)
                                    continue
                                elif (out[j].startswith("<") and out[j].endswith(">")):
                                    temp_checked.append(out[j])
                                    continue
                                num += 1
                                if ts[0] == -1:
                                    indexes[dataset][prompt]["failed"].append([name, i, j])
                                    failed_num += 1
                                    status = self.build_new_status()
                                    status["status"] = "PARSE_ERROR"
                                    temp_checked.append(status)
                                elif ts[0] not in indexes[dataset][prompt]:
                                    solutions[dataset][prompt].append(ts)
                                    indexes[dataset][prompt][ts[0]] = [[name, i, j]]
                                    status = self.build_new_status()
                                    temp_checked.append(status)
                                else:
                                    indexes[dataset][prompt][ts[0]].append([name, i, j])
                                    status = self.build_new_status()
                                    temp_checked.append(status)
                            self.checked_outputs[dataset][prompt][name].append(temp_checked)
                        else:
                            self.checked_outputs[dataset][prompt][name].append(out)
            print("Totally {}/{} ({}) invalid solutions found.".format(failed_num, num, failed_num/num if num > 0 else 0))


        self.checked_outputs = self.verify_natural_outputs(rd)
        
        with open(output_file.replace(".json", "_CHECKED.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(self.checked_outputs, sort_keys=True, indent=4, separators=(',', ': ')))
        
        with open(output_file.replace(".json", "_SOLUTIONS.json"), "w", encoding = "utf-8") as f:
                f.write(json.dumps(solutions, sort_keys=True, indent=4, separators=(',', ': ')))

        with open(output_file.replace(".json", "_INDEXES.json"), "w", encoding = "utf-8") as f:
                f.write(json.dumps(indexes, sort_keys=True, indent=4, separators=(',', ': ')))
        

    def prepare_time_measurement(self, result_dir, model, mode, rd):
        checked_file = os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json")
        index_file = os.path.join(result_dir, model, mode, f"rd{rd}_INDEXES.json")
        checked_outputs = json.load(open(checked_file, "r"))
        indexes = json.load(open(index_file, "r"))
        all_solutions = {}

        for i in range(0, 5):
            passed_solutions = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_PASSED_SOLUTIONS_p{i}.json"), "r"))
            for dataset in passed_solutions:
                if dataset not in all_solutions:
                    all_solutions[dataset] = {}
                for prompt in passed_solutions[dataset]:
                    all_solutions[dataset][prompt] = passed_solutions[dataset][prompt]

        for dataset in checked_outputs:
            for prompt in checked_outputs[dataset]:
                if "correct_solutions" not in checked_outputs[dataset][prompt] and len(checked_outputs[dataset][prompt]) == 0:
                    continue
                for i, status in enumerate(checked_outputs[dataset][prompt]["correct_solutions"]):
                    if isinstance(status, str):
                        continue
                    else:
                        processor = CodeProcessor(status["passed_solution"], force_rename = True if dataset in ["openai_humaneval", "mbpp"] else False)
                        solution = processor.run(no_modify = True)
                        if solution[0] == -1:
                            raise ValueError("Cannot parse correct solution!")
                        if prompt not in all_solutions[dataset]:
                            all_solutions[dataset][prompt] = []
                        all_solutions[dataset][prompt].append(solution)
                        if solution[0] not in indexes[dataset][prompt]:
                            indexes[dataset][prompt][solution[0]] = [["correct_solutions", 0, i]]
                        else:
                            indexes[dataset][prompt][solution[0]].append(["correct_solutions", 0, i])
                        
        
        if rd == 0 and mode == "time":
            for dataset in indexes:
                for prompt in indexes[dataset]:
                    for solution in indexes[dataset][prompt]:
                        if solution == "failed":
                            continue
                        new_indexes = []
                        for index in indexes[dataset][prompt][solution]:
                            if index[0] in ["time_simple_execution_feedback", "time_execution_feedback_with_testcase", "correct_solutions"]:
                                new_indexes.append(index)
                        if len(new_indexes) == 0:
                            prev_num = len(all_solutions[dataset][prompt])
                            new_solutions = []
                            for s in all_solutions[dataset][prompt]:
                                if s[0] != solution:
                                    new_solutions.append(s)
                            all_solutions[dataset][prompt] = new_solutions
                            cur_num = len(all_solutions[dataset][prompt])
        
        with open(os.path.join(result_dir, model, mode, f"rd{rd}_PASSED_SOLUTIONS.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(all_solutions, sort_keys=True, indent=4, separators=(',', ': ')))
        
        with open(index_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(indexes, sort_keys=True, indent=4, separators=(',', ': ')))
        

    def prepare_groundtruth_comparison(self, result_dir, model, mode, rd, best_solution_file):
        checked_file = os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json")
        checked_outputs = json.load(open(checked_file, "r"))

        indexes = {}

        best_solutions = json.load(open(best_solution_file, "r"))

        solutions = {}

        for dataset in checked_outputs:
            solutions[dataset] = {}
            indexes[dataset] = {}
            for prompt in checked_outputs[dataset]:
                solutions[dataset][prompt] = []
                indexes[dataset][prompt] = {}
                for name in checked_outputs[dataset][prompt]:
                    if not name.startswith(mode):
                        continue
                    for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                        if isinstance(status, str):
                            continue
                        elif "status" in status and status["status"] == "PASSED":
                            processor = CodeProcessor(status["passed_solution"], force_rename = True if dataset in ["openai_humaneval", "mbpp"] else False)
                            solution = processor.run(no_modify = True)
                            solutions[dataset][prompt].append(solution)
                            if solution[0] not in indexes[dataset][prompt]:
                                indexes[dataset][prompt][solution[0]] = []
                            indexes[dataset][prompt][solution[0]].append([name, len(checked_outputs[dataset][prompt][name])-1, i])
                if "groundtruth" not in checked_outputs[dataset][prompt]:
                    checked_outputs[dataset][prompt]["groundtruth"] = best_solutions[dataset][prompt]
                    solutions[dataset][prompt].append(best_solutions[dataset][prompt])
                    indexes[dataset][prompt][best_solutions[dataset][prompt][0]] = [["groundtruth", 0, 0]]

        with open(os.path.join(result_dir, model, mode, f"FINAL_CHECKED.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(checked_outputs, sort_keys=True, indent=4, separators=(',', ': ')))
        
        with open(os.path.join(result_dir, model, mode, f"FINAL_PASSED_SOLUTIONS.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(solutions, sort_keys=True, indent=4, separators=(',', ': ')))
        
        with open(os.path.join(result_dir, model, mode, f"FINAL_INDEXES.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(indexes, sort_keys=True, indent=4, separators=(',', ': ')))
                






    def handle_natural_strings(self, lst):
        ans = []
        for string in lst:
            if string == None:
                status = self.build_new_status()
                status["status"] = "QUERY_ERROR"
                ans.append(status)
                continue
            if isinstance(string, dict):
                ans.append(string)
                continue
            string = string.replace("\r\n", "\n").replace("\n\n", "\n")
            if "```python" in string:
                string = string.split("```python")[0]
            if "```" in string:
                string = string.split("```")[0]
            if string.startswith("\n"):
                string = string[1:]
            if string.endswith("\n"):
                string = string[:-1]
            string = string.strip()
            if string.split("\n")[-1].endswith(":"):
                string = "\n".join(string.split("\n")[:-1])
            if string.count("Comment:") > 1:
                string = "Comment:".join(string.split("Comment:")[:1])
            if len(string.replace("\n", "")) == 0:
                status = self.build_new_status()
                status["status"] = "QUERY_ERROR"
                ans.append(status)
            else:
                ans.append(string)
        return ans


    def verify_natural_outputs(self, rd):
        for dataset in self.checked_outputs:
            for prompt in self.checked_outputs[dataset]:
                for name in self.checked_outputs[dataset][prompt]:
                    if name == "correct_solutions":
                        continue
                    if name == "multiple_agents_with_reviewer":
                        if len(self.checked_outputs[dataset][prompt][name]) > 1 and rd == 1:
                            strings = []
                            for string in self.checked_outputs[dataset][prompt][name][1]:
                                if string == None:
                                    status = self.build_new_status()
                                    status["status"] = "QUERY_ERROR"
                                    strings.append(status)
                                elif "[Agree]" in string:
                                    comment = self.handle_natural_strings([string.split("[Agree]")[-1]])[0]
                                    strings.append("[Agree]" + comment if comment != None else " ")
                                elif "[Disagree]" in string:
                                    comment = self.handle_natural_strings([string.split("[Disagree]")[-1]])[0]
                                    strings.append("[Disagree]" + comment if comment != None else " ")
                                else:
                                    status = self.build_new_status()
                                    status["status"] = "QUERY_ERROR"
                                    strings.append(status)
                            self.checked_outputs[dataset][prompt][name][1] = self.handle_natural_strings(strings)
                    elif name == "multiple_agents_with_team":
                        if len(self.checked_outputs[dataset][prompt][name]) > 0 and rd == 0:
                            self.checked_outputs[dataset][prompt][name][0] = self.handle_natural_strings(self.checked_outputs[dataset][prompt][name][0])
                        if len(self.checked_outputs[dataset][prompt][name]) > 2 and rd == 2:
                            strings = []
                            for string in self.checked_outputs[dataset][prompt][name][2]:
                                if string == None:
                                    status = self.build_new_status()
                                    status["status"] = "QUERY_ERROR"
                                    strings.append(status)
                                elif "[Agree]" in string:
                                    comment = self.handle_natural_strings([string.split("[Agree]")[-1]])[0]
                                    strings.append("[Agree]" + comment if comment != None else " ")
                                elif "[Disagree]" in string:
                                    comment = self.handle_natural_strings([string.split("[Disagree]")[-1]])[0]
                                    strings.append("[Disagree]" + comment if comment != None else " ")
                                else:
                                    status = self.build_new_status()
                                    status["status"] = "QUERY_ERROR"
                                    strings.append(status)
                            self.checked_outputs[dataset][prompt][name][2] = self.handle_natural_strings(strings)
                        if len(self.checked_outputs[dataset][prompt][name]) > 3 and rd == 3:
                            self.checked_outputs[dataset][prompt][name][3] = self.handle_natural_strings(self.checked_outputs[dataset][prompt][name][3])
                    elif not self.prompt_generator.gen_code(rd, name) and len(self.checked_outputs[dataset][prompt][name]) > rd:
                        self.checked_outputs[dataset][prompt][name][rd] = self.handle_natural_strings(self.checked_outputs[dataset][prompt][name][rd])
        
        return self.checked_outputs

    def estimate_pass_at_k(
        self,
        num_samples,
        num_correct,
        k: int,
    ):
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array(
            [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
        )

    def evaluate_pass_rate(self, result_dir, model, mode, rd = None):
        if rd != None:
            checked_outputs = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json"), "r"))
        else:
            checked_outputs = json.load(open(os.path.join(result_dir, model, mode, f"FINAL_CHECKED.json"), "r"))
        prompt_generator = Prompt(chat=True)
        prompt_generator.load_templates()

        correct_nums = {}
        
        for dataset in checked_outputs:
            correct_nums[dataset] = {}
            for prompt in checked_outputs[dataset]:
                for name in checked_outputs[dataset][prompt]:
                    if name  == "correct_solutions":
                        if name not in correct_nums[dataset]:
                            correct_nums[dataset][name] = []
                        num = 0
                        for i, status in enumerate(checked_outputs[dataset][prompt][name]):
                            if isinstance(status, dict) and "status" in status and status["status"] in ["PASSED", "TIME_MEASURED"]:
                                num += 1
                        correct_nums[dataset][name].append(num)
                        continue
                    if rd != None and name not in ["correct_solutions", "groundtruth"] and not prompt_generator.gen_code(rd, name):
                        continue
                    if name == "groundtruth":
                        continue
                    if name not in correct_nums[dataset]:
                        correct_nums[dataset][name] = []
                    num = 0
                    for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                        if status == "<END>":
                            if rd != None:
                                status, _ = self.find_result(dataset, prompt, name, len(checked_outputs[dataset][prompt][name]) - 1, i, checked_outputs)
                            else:
                                continue
                        if isinstance(status, dict) and "status" in status and status["status"] in ["PASSED", "TIME_MEASURED"]:
                            num += 1
                    correct_nums[dataset][name].append(num)
        
        pass_rates = {}

        ks = [1, 5, 8, 10, 20]
        for dataset in correct_nums:
            pass_rates[dataset] = {}
            for name in correct_nums[dataset]:
                pass_rates[dataset][name] = {}
                for k in ks:
                    pass_rates[dataset][name][k] = sum(self.estimate_pass_at_k(20, correct_nums[dataset][name], k))/len(correct_nums[dataset][name])
        
        if rd != None:
            result_file = os.path.join(result_dir, model, mode, f"rd{rd}_pass_rate.json")
        else:
            result_file = os.path.join(result_dir, model, mode, f"FINAL_pass_rate.json")

        with open(result_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(pass_rates, sort_keys=True, indent=4, separators=(',', ': ')))

    def evaluate_correctness(self, result_dir):
        models = [
            "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-70B-Instruct_snapshots_e8cf5276ae3e97cfde8a058e64a636f2cde47820",
            "gpt-3.5-turbo"
        ]
        results = {}
        for m in models:
            results[m] = {}
            checked_outputs_base = json.load(open(os.path.join(result_dir, m, "base", f"rd0_CHECKED.json"), "r"))
            checked_outputs_rd0 = json.load(open(os.path.join(result_dir, m, "correctness", f"rd0_CHECKED.json"), "r"))
            checked_outputs_rd1 = json.load(open(os.path.join(result_dir, m, "correctness", f"rd1_CHECKED.json"), "r"))

            checked_outputs_final = json.load(open(os.path.join(result_dir, m, "time", f"FINAL_CHECKED.json"), "r"))

            correct_nums = {}
            ks = [1, 5, 8, 10, 20]
            key_map = {
                "base": checked_outputs_base,
                "correctness_testcase_feedback": checked_outputs_rd0,
                "correctness_reflection_and_feedback": checked_outputs_rd1
            }

            prompts = {}

            for dataset in checked_outputs_final:
                prompts[dataset] = []
                for prompt in checked_outputs_final[dataset]:
                    min_groundtruth_time = None
                    for status in checked_outputs_final[dataset][prompt]["groundtruth"]:
                        if status["status"] == "TIME_MEASURED":
                            if min_groundtruth_time == None:
                                min_groundtruth_time = status["exec_time"]
                            elif status["exec_time"] < min_groundtruth_time:
                                min_groundtruth_time = status["exec_time"]
                    if min_groundtruth_time == None:
                        continue
                    else:
                        prompts[dataset].append(prompt)
            nums = {}
            for key in key_map:
                for dataset in key_map[key]:
                    if dataset not in correct_nums:
                        correct_nums[dataset] = {}
                    for prompt in key_map[key][dataset]:
                        if prompt not in prompts[dataset]:
                            continue
                        for name in key_map[key][dataset][prompt]:
                            if name  == key and name == "base":
                                if name not in correct_nums[dataset]:
                                    correct_nums[dataset][name] = {}
                                best = {}
                                for k in ks:
                                    best[k] = 0
                                    if k not in correct_nums[dataset][name]:
                                        correct_nums[dataset][name][k] = 0
                                for i, status in enumerate(key_map[key][dataset][prompt][name][-1]):
                                    if isinstance(status, dict) and "status" in status and status["status"] in ["PASSED", "TIME_MEASURED"]:
                                        for k in ks:
                                            if i < k:
                                                best[k] += 1
                                for k in ks:
                                    if best[k] > 0:
                                        correct_nums[dataset][name][k] += 1
                            elif name == key:
                                if name not in correct_nums[dataset]:
                                    correct_nums[dataset][name] = {}
                                best = {}
                                for k in ks:
                                    best[k] = 0
                                    if k not in correct_nums[dataset][name]:
                                        correct_nums[dataset][name][k] = 0
                                for i, status in enumerate(key_map[key][dataset][prompt][name][-1]):
                                    if status == "<END>":
                                        status, _ = self.find_result(dataset, prompt, name, len(key_map[key][dataset][prompt][name]) - 1, i, key_map[key])
                                    if isinstance(status, dict) and "status" in status and status["status"] in ["PASSED", "TIME_MEASURED"]:
                                        for k in ks:
                                            if i < k:
                                                best[k] += 1
                                for k in ks:
                                    if best[k] > 0:
                                        correct_nums[dataset][name][k] += 1
            

            for dataset in correct_nums:
                results[m][dataset] = {}
                for name in correct_nums[dataset]:
                    results[m][dataset][name] = {}
                    for k in correct_nums[dataset][name]:
                        results[m][dataset][name][k] = correct_nums[dataset][name][k] / len(prompts[dataset])
        
        result_file = os.path.join(result_dir, f"correctness.json")
        csv_file = os.path.join(result_dir, f"correctness.csv")

        lines = []
        for m in results:
            lines.append(m + "+" * 20)
            for dataset in results[m]:
                lines.append(dataset + "="*10)
                lines.append("Prompt,Best@1,Best@8,Best@20")
                for name in results[m][dataset]:
                    lines.append("{},{},{},{}".format(
                        name,
                        format(results[m][dataset][name][1] * 100, ".2f"),
                        format(results[m][dataset][name][8] * 100, ".2f"),
                        format(results[m][dataset][name][20] * 100, ".2f")
                    ))


        with open(result_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))

        with open(csv_file, "w", encoding = "utf-8") as f:
            f.write("\n".join(lines))
        


    def evaluate_execution_time(self, result_dir, model, mode, fallback = True):
        checked_file = os.path.join(result_dir, model, mode, f"FINAL_CHECKED.json")
        checked_outputs = json.load(open(checked_file, "r"))


        ks = [1, 5, 8, 10, 20]

        nums = {}
        gt_speedup = {}
        gt_opt = {}
        total_num = {}
        correctness = {}
        overall_num = {}


        for dataset in checked_outputs:
            nums[dataset] = {}
            gt_speedup[dataset] = {}
            gt_opt[dataset] = {}
            total_num[dataset] = {}
            correctness[dataset] = {}
            overall_num[dataset] = {}
            for prompt in checked_outputs[dataset]:
                min_groundtruth_time = None
                for status in checked_outputs[dataset][prompt]["groundtruth"]:
                    if status["status"] == "TIME_MEASURED":
                        if min_groundtruth_time == None:
                            min_groundtruth_time = status["exec_time"]
                        elif status["exec_time"] < min_groundtruth_time:
                            min_groundtruth_time = status["exec_time"]
                if min_groundtruth_time == None:
                    continue
                for name in checked_outputs[dataset][prompt]:
                    if not name.startswith("time") and name != "base":
                        continue
                    if name not in nums[dataset]:
                        nums[dataset][name] = {}
                        gt_speedup[dataset][name] = {}
                        gt_opt[dataset][name] = {}
                        correctness[dataset][name] = {}
                        total_num[dataset][name] = 0
                        for k in ks:
                            gt_speedup[dataset][name][k] = 0
                            gt_opt[dataset][name][k] = 0
                            nums[dataset][name][k] = 0
                            correctness[dataset][name][k] = 0
                        overall_num[dataset][name] = 0
                        gt_speedup[dataset][name]["avg"] = 0
                        gt_opt[dataset][name]["avg"] = 0
                        correctness[dataset][name]["avg"] = 0
                    best = {}
                    correct_num = {}
                    for k in ks:
                        best[k] = 0
                        correct_num[k] = 0
                    num = 0
                    temp_gt_speedup = 0
                    temp_gt_opt = 0
                    overall_num[dataset][name] += 1
                    for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                        if isinstance(status, dict) and status["status"] == "TIME_MEASURED":
                            cur_status = status
                        elif isinstance(checked_outputs[dataset][prompt]["correct_solutions"][i], dict) and checked_outputs[dataset][prompt]["correct_solutions"][i]["status"] == "TIME_MEASURED" and fallback and name != "base":
                            cur_status = checked_outputs[dataset][prompt]["correct_solutions"][i]
                        else:
                            continue
                        if min_groundtruth_time / cur_status["exec_time"] > 100 and dataset in ["openai_humaneval", "mbpp"]:
                            continue
                        for k in ks:
                            if i < k:
                                if best[k] == 0:
                                    best[k] = cur_status["exec_time"]
                                elif best[k] > 0 and cur_status["exec_time"] < best[k]:
                                    best[k] = cur_status["exec_time"]
                                correct_num[k] += 1
                        num += 1
                        temp_gt_speedup += min_groundtruth_time / cur_status["exec_time"]
                        if (min_groundtruth_time - cur_status["exec_time"]) / min_groundtruth_time > 0.1:
                            temp_gt_opt += 1
                    if num > 0:
                        total_num[dataset][name] += 1
                        gt_speedup[dataset][name]["avg"] += temp_gt_speedup / num
                        gt_opt[dataset][name]["avg"] += temp_gt_opt / num
                        correctness[dataset][name]["avg"] += num / 20
                        for k in ks:
                            if correct_num[k] > 0:
                                correctness[dataset][name][k] += 1
                            if best[k] > 0:
                                nums[dataset][name][k] += 1
                                gt_speedup[dataset][name][k] += min_groundtruth_time / best[k]
                                if (min_groundtruth_time - best[k]) / min_groundtruth_time > 0.1:
                                    gt_opt[dataset][name][k] += 1
            for name in gt_speedup[dataset]:
                for k in ks:
                    gt_speedup[dataset][name][k] = gt_speedup[dataset][name][k] / nums[dataset][name][k]
                    gt_opt[dataset][name][k] = gt_opt[dataset][name][k] / overall_num[dataset][name]
                    #gt_opt[dataset][name][k] = gt_opt[dataset][name][k] / nums[dataset][name][k]
                    correctness[dataset][name][k] = correctness[dataset][name][k] / overall_num[dataset][name]

                gt_speedup[dataset][name]["avg"] = gt_speedup[dataset][name]["avg"] / total_num[dataset][name]
                gt_opt[dataset][name]["avg"] = gt_opt[dataset][name]["avg"] / overall_num[dataset][name]
                correctness[dataset][name]["avg"] = correctness[dataset][name]["avg"] / overall_num[dataset][name]

        data = {}
        for dataset in gt_speedup:
            data[dataset] = {
                "gt_speedup": gt_speedup[dataset],
                "gt_opt": gt_opt[dataset],
                "correctness": correctness[dataset],
                "num": overall_num[dataset]
            }

        if fallback:
            result_file = os.path.join(result_dir, model, mode, f"FINAL_exec_time.json")
        else:
            result_file = os.path.join(result_dir, model, mode, f"FINAL_exec_time_nofallback.json")

        with open(result_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(data, indent=4, separators=(',', ': ')))


    def evaluate_execution_time_old(self, result_dir, model, mode):
        checked_file = os.path.join(result_dir, model, mode, f"FINAL_CHECKED.json")
        checked_outputs = json.load(open(checked_file, "r"))


        ks = [1, 5, 8, 10, 20]

        nums = {}
        gt_speedup = {}
        gt_opt = {}
        ori_speedup = {}
        ori_opt = {}
        total_num = {}
        correctness = {}
        overall_num = {}


        for dataset in checked_outputs:
            nums[dataset] = {}
            gt_speedup[dataset] = {}
            gt_opt[dataset] = {}
            ori_speedup[dataset] = {}
            ori_opt[dataset] = {}
            total_num[dataset] = {}
            correctness[dataset] = {}
            overall_num[dataset] = {}
            for prompt in checked_outputs[dataset]:
                min_groundtruth_time = None
                for status in checked_outputs[dataset][prompt]["groundtruth"]:
                    if status["status"] == "TIME_MEASURED":
                        if min_groundtruth_time == None:
                            min_groundtruth_time = status["exec_time"]
                        elif status["exec_time"] < min_groundtruth_time:
                            min_groundtruth_time = status["exec_time"]
                if min_groundtruth_time == None:
                    continue
                for name in checked_outputs[dataset][prompt]:
                    if not name.startswith("time") or name == "time_multiple_agents_with_reviewer":
                        continue
                    if name not in nums[dataset]:
                        nums[dataset][name] = {}
                        gt_speedup[dataset][name] = {}
                        gt_opt[dataset][name] = {}
                        correctness[dataset][name] = 0
                        total_num[dataset][name] = 0
                        for k in ks:
                            gt_speedup[dataset][name][k] = 0
                            gt_opt[dataset][name][k] = 0
                            nums[dataset][name][k] = 0
                        gt_speedup[dataset][name]["avg"] = 0
                        gt_opt[dataset][name]["avg"] = 0
                        ori_speedup[dataset][name] = 0
                        ori_opt[dataset][name] = 0
                        overall_num[dataset][name] = 0
                    best = {}
                    for k in ks:
                        best[k] = 0
                    num = 0
                    temp_ori_speedup = 0
                    temp_ori_opt = 0
                    temp_gt_speedup = 0
                    temp_gt_opt = 0
                    correct_num = 0
                    for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                        if isinstance(status, dict) and status["status"] == "TIME_MEASURED":
                            if min_groundtruth_time / status["exec_time"] > 100 and dataset in ["openai_humaneval", "mbpp"]:
                                continue
                            for k in ks:
                                if i < k:
                                    if best[k] == 0:
                                        best[k] = status["exec_time"]
                                    elif best[k] > 0 and status["exec_time"] < best[k]:
                                        best[k] = status["exec_time"]
                            num += 1
                            temp_ori_speedup += checked_outputs[dataset][prompt]["correct_solutions"][i]["exec_time"] / status["exec_time"]
                            temp_gt_speedup += min_groundtruth_time / status["exec_time"]
                            if (checked_outputs[dataset][prompt]["correct_solutions"][i]["exec_time"] - status["exec_time"]) / checked_outputs[dataset][prompt]["correct_solutions"][i]["exec_time"] >= 0.1:
                                temp_ori_opt += 1
                            if (min_groundtruth_time - status["exec_time"]) / min_groundtruth_time > 0.1:
                                temp_gt_opt += 1
                        if isinstance(checked_outputs[dataset][prompt]["correct_solutions"][i], dict) and checked_outputs[dataset][prompt]["correct_solutions"][i]["status"] == "TIME_MEASURED":
                            correct_num += 1
                    if correct_num > 0:
                        overall_num[dataset][name] += 1
                    if num > 0:
                        ori_speedup[dataset][name] += temp_ori_speedup / num
                        ori_opt[dataset][name] += temp_ori_opt / num
                        gt_speedup[dataset][name]["avg"] += temp_gt_speedup / num
                        gt_opt[dataset][name]["avg"] += temp_gt_opt / num
                        total_num[dataset][name] += 1
                        correctness[dataset][name] += num / correct_num
                        for k in ks:
                            if best[k] > 0:
                                nums[dataset][name][k] += 1
                                gt_speedup[dataset][name][k] += min_groundtruth_time / best[k]
                                if (min_groundtruth_time - best[k]) / min_groundtruth_time > 0.1:
                                    gt_opt[dataset][name][k] += 1
            for name in gt_speedup[dataset]:
                for k in ks:
                    gt_speedup[dataset][name][k] = gt_speedup[dataset][name][k] / nums[dataset][name][k]
                    gt_opt[dataset][name][k] = gt_opt[dataset][name][k] / nums[dataset][name][k]
                gt_speedup[dataset][name]["avg"] = gt_speedup[dataset][name]["avg"] / total_num[dataset][name]
                gt_opt[dataset][name]["avg"] = gt_opt[dataset][name]["avg"] / total_num[dataset][name]
                ori_speedup[dataset][name] = ori_speedup[dataset][name] / total_num[dataset][name]
                ori_opt[dataset][name] = ori_opt[dataset][name] / total_num[dataset][name]
                correctness[dataset][name] = correctness[dataset][name] / overall_num[dataset][name]

        data = {}
        for dataset in gt_speedup:
            data[dataset] = {
                "gt_speedup": gt_speedup[dataset],
                "gt_opt": gt_opt[dataset],
                "ori_speedup": ori_speedup[dataset],
                "ori_opt": ori_opt[dataset],
                "correctness": correctness[dataset]
            }


        result_file = os.path.join(result_dir, model, mode, f"FINAL_exec_time_test.json")

        with open(result_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(data, indent=4, separators=(',', ': ')))

    def get_better_solutions(self, result_dir, model, mode, name):
        checked_file = os.path.join(result_dir, model, mode, f"FINAL_CHECKED.json")

        checked_outputs = json.load(open(checked_file, "r"))

        best_solutions = {}

        for dataset in checked_outputs:
            best_solutions[dataset] = {}
            for prompt in checked_outputs[dataset]:
                min_groundtruth_time = None
                min_groundtruth = None
                for status in checked_outputs[dataset][prompt]["groundtruth"]:
                    if status["status"] == "TIME_MEASURED":
                        if min_groundtruth_time == None:
                            min_groundtruth_time = status["exec_time"]
                            min_groundtruth = status["passed_solution"]
                        elif status["exec_time"] < min_groundtruth_time:
                            min_groundtruth_time = status["exec_time"]
                            min_groundtruth = status["passed_solution"]
                if min_groundtruth_time == None:
                    continue
                min_time = None
                min_solution = None
                for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                    if isinstance(status, dict) and status["status"] == "TIME_MEASURED":
                        cur_status = status
                    else:
                        continue
                    if min_time == None:
                        min_time = cur_status["exec_time"]
                        min_solution = status["passed_solution"]
                    elif cur_status["exec_time"] < min_time:
                        min_time = cur_status["exec_time"]
                        min_solution = status["passed_solution"]
                    if min_time == None or min_solution == None:
                        continue
                    speedup = min_groundtruth_time / min_time
                    if  speedup < 2 :
                        continue
                    if prompt not in best_solutions[dataset]:
                        best_solutions[dataset][prompt] = [min_solution, min_groundtruth, speedup]
        with open("better_solutions.json", "w", encoding = "utf-8") as f:
            f.write(json.dumps(best_solutions, sort_keys=True, indent=4, separators=(',', ': ')))

                            


    def finalize_correctness_check(self, result_dir, model, mode, rd, task_num = 5, clean = False):
        indexes = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_INDEXES.json"), "r"))
        checked_outputs = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json"), "r"))
        solutions = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_SOLUTIONS.json"), "r"))
        for i in range(0, task_num):
            passed_solutions = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_PASSED_SOLUTIONS_p{i}.json"), "r"))
            failed_cases = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_FAILED_TESTCASES_p{i}.json"), "r"))
            for dataset in failed_cases:
                for prompt in failed_cases[dataset]:
                    for i, testcases in enumerate(failed_cases[dataset][prompt]):
                        if len(testcases) > 0:
                            for index in indexes[dataset][prompt][solutions[dataset][prompt][i][0]]:
                                checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"] = "TESTCASE_FAILED"
                                checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["failed_testcase"] = testcases
            for dataset in passed_solutions:
                for prompt in passed_solutions[dataset]:
                    for solution in passed_solutions[dataset][prompt]:
                        for index in indexes[dataset][prompt][solution[0]]:
                            checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["passed_solution"] = solution[0]
                            checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"] = "PASSED"
        if clean:
            for dataset in checked_outputs:
                deleted_prompts = []
                for prompt in checked_outputs[dataset]:
                    for name in checked_outputs[dataset][prompt]:
                        if name.startswith(mode) and self.prompt_generator.round_exist(rd, name):
                            try:
                                if self.is_unchecked(checked_outputs[dataset][prompt][name][rd]):
                                    deleted_prompts.append(prompt)
                                    continue
                            except Exception as e:
                                print(name, prompt, rd)
                                exit()
                            for i, status in enumerate(checked_outputs[dataset][prompt][name][rd]):
                                if isinstance(status, str):
                                    continue
                                elif status == None:
                                    checked_outputs[dataset][prompt][name][rd][i] = self.build_new_status()
                                    checked_outputs[dataset][prompt][name][rd][i]["status"] = "QUERY_ERROR"
                                    continue
                                if status["status"] == "Unchecked":
                                    status["status"] = "TIMEOUT"
                '''
                for prompt in deleted_prompts:
                    del checked_outputs[dataset][prompt]
                '''

        with open(os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json"), "w") as f:
            f.write(json.dumps(checked_outputs, sort_keys=True, indent=4, separators=(',', ': ')))


    def finalize_time_measurement(self, result_dir, model, mode, task_num = 10, rd = None):
        if rd != None:
            checked_file = os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json")
            checked_outputs = json.load(open(checked_file, "r"))
            indexes = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_INDEXES.json"), "r"))
            solutions = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_PASSED_SOLUTIONS.json"), "r"))
        else:
            checked_file = os.path.join(result_dir, model, mode, f"FINAL_CHECKED.json")
            checked_outputs = json.load(open(checked_file, "r"))
            indexes = json.load(open(os.path.join(result_dir, model, mode, f"FINAL_INDEXES.json"), "r"))
            solutions = json.load(open(os.path.join(result_dir, model, mode, f"FINAL_PASSED_SOLUTIONS.json"), "r"))


        for i in range(0, task_num):
            if rd != None:
                time_costs = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_TIME_COSTS_p{i}.json"), "r"))
                large_cases = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_LARGE_TESTCASES_p{i}.json"), "r"))
            else:
                if not os.path.exists(os.path.join(result_dir, model, mode, f"FINAL_TIME_COSTS_p{i}.json")):
                    print(f"Cannot find test cost files for split #{i}.")
                    continue
                time_costs = json.load(open(os.path.join(result_dir, model, mode, f"FINAL_TIME_COSTS_p{i}.json"), "r"))
                large_cases = json.load(open(os.path.join(result_dir, model, mode, f"FINAL_LARGE_TESTCASES_p{i}.json"), "r"))

            for dataset in time_costs:
                for prompt in time_costs[dataset]:
                    for i, cost in enumerate(time_costs[dataset][prompt]):
                        for index in indexes[dataset][prompt][solutions[dataset][prompt][i][0]]:
                            if index[0] in ["correct_solutions", "groundtruth"]:
                                if checked_outputs[dataset][prompt][index[0]][index[2]]["status"] not in ["PASSED", "TIME_MEASURED"]:
                                    print("Inconsistent data occurred, status: {}".format(checked_outputs[dataset][prompt][index[0]][index[2]]["status"]))
                                    continue
                                checked_outputs[dataset][prompt][index[0]][index[2]]["status"] = "TIME_MEASURED"
                                checked_outputs[dataset][prompt][index[0]][index[2]]["exec_time"] = cost
                                checked_outputs[dataset][prompt][index[0]][index[2]]["large_testcase"] = [large_cases[dataset][prompt][i]]
                            else:
                                if isinstance(checked_outputs[dataset][prompt][index[0]][index[1]][index[2]], str):
                                    print(checked_outputs[dataset][prompt][index[0]][index[1]][index[2]])
                                    print(index)
                                    continue
                                if checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"] not in ["PASSED", "TIME_MEASURED"]:
                                    print("Inconsistent data occurred, status: {}".format(checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"]))
                                    continue
                                try:
                                    checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"] = "TIME_MEASURED"
                                    checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["exec_time"] = cost
                                    checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["large_testcase"] = [large_cases[dataset][prompt][i]]
                                except Exception as e:
                                    print(e)
                                    print(index)
                                    continue
        
        with open(checked_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(checked_outputs, sort_keys=True, indent=4, separators=(',', ': ')))
        

    def gather_correct_solutions_and_history(self, result_dir, model, mode, name, new_mode):
        for i in range(10):
            checked_file = os.path.join(result_dir, model, mode, f"rd{i+1}_CHECKED.json")
            if not os.path.exists(checked_file):
                checked_file = os.path.join(result_dir, model, mode, f"rd{i}_CHECKED.json")
                history_file = os.path.join(result_dir, model, mode, f"rd{i+1}_HISTORY.json")
                if not os.path.exists(checked_file) or not os.path.exists(history_file):
                    raise ValueError("Cannot find the checked file or history file of any round.")
                print(f"Selected the last round: {i}")
                break
        
        checked_outputs = json.load(open(checked_file, "r"))
        for dataset in checked_outputs:
            for prompt in checked_outputs[dataset]:
                if name not in checked_outputs[dataset][prompt]:
                    continue
                checked_outputs[dataset][prompt]["correct_solutions"] = []
                for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                    if status == "<END>":
                        status,_ = self.find_result(dataset, prompt, name, len(checked_outputs[dataset][prompt][name]) - 1, i, checked_outputs)
                    if "status" in status and status["status"] == "PASSED":
                        checked_outputs[dataset][prompt]["correct_solutions"].append(status)
                    else:
                        checked_outputs[dataset][prompt]["correct_solutions"].append("<FAILED>")
        
        time_categories = [
            "time_simple_instruction",
            "time_in_context_learning",
            "time_pre_defined_strategy",
            "time_chain_of_thought",
            "time_time_complexity_reduction",
            "time_simple_execution_feedback",
            "time_execution_feedback_with_testcase",
            "time_multiple_agents_with_reviewer",
            "time_multiple_agents_with_team"
        ]
        history = json.load(open(history_file, "r"))
        new_history = {}
        for dataset in history:
            new_history[dataset] = {}
            for n in time_categories:
                new_history[dataset][n] = history[dataset][name]
        
        new_history_file = os.path.join(result_dir, model, new_mode, "rd0_HISTORY.json")
        with open(new_history_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(new_history, sort_keys=True, indent=4, separators=(',', ': ')))
    
        final_file = os.path.join(result_dir, model, mode, "FINAL_CHECKED.json")
        with open(final_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(checked_outputs, sort_keys=True, indent=4, separators=(',', ': ')))
        

    def gather_all_solutions_for_time(self, result_dir, model):
        for i in range(10):
            checked_file = os.path.join(result_dir, model, "time", f"rd{i+1}_CHECKED.json")
            if not os.path.exists(checked_file):
                checked_file = os.path.join(result_dir, model, "time", f"rd{i}_CHECKED.json")
                if not os.path.exists(checked_file):
                    raise ValueError("Cannot find the checked file of any round.")
                print(f"Selected the last round: {i}")
                break

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


        base_checked_file =  os.path.join(result_dir, model, "base", f"rd0_CHECKED.json")
        base_checked_outputs = json.load(open(base_checked_file, "r"))
        checked_outputs = json.load(open(checked_file, "r"))

        
        for dataset in checked_outputs:
            deleted_prompts = []
            for prompt in checked_outputs[dataset]:
                if len(checked_outputs[dataset][prompt]) == 0:
                    deleted_prompts.append(prompt)
            for prompt in deleted_prompts:
                del checked_outputs[dataset][prompt]
            

        solutions = {}
        indexes = {}

        for dataset in base_checked_outputs:
            for prompt in base_checked_outputs[dataset]:
                if prompt in checked_outputs[dataset]:
                    checked_outputs[dataset][prompt]["base"] = base_checked_outputs[dataset][prompt]["base"]

        
        for dataset in checked_outputs:
            best_solutions = json.load(open(os.path.join("test_datasets", dataset.replace("/", "_"), "best_solutions.json"), "r"))
            deleted_prompts = []
            for prompt in checked_outputs[dataset]:
                if prompt not in best_solutions:
                    deleted_prompts.append(prompt)
                    continue
                status = self.build_new_status()
                status["status"] = "PASSED"
                status["passed_solution"] = best_solutions[prompt][0]
                checked_outputs[dataset][prompt]["groundtruth"] = [status]
            print("Delete {} prompts in dataset {}".format(len(deleted_prompts), dataset))
            for prompt in deleted_prompts:
                del checked_outputs[dataset][prompt]
        
        num = 0
        for dataset in checked_outputs:
            if dataset not in solutions:
                solutions[dataset] = {}
                indexes[dataset] = {}
            for prompt in checked_outputs[dataset]:
                if prompt not in solutions[dataset]:
                    solutions[dataset][prompt] = []
                if prompt not in indexes[dataset]:
                    indexes[dataset][prompt] = {}
                for name in time_categories + ["groundtruth", "correct_solutions", "base"]:
                    if name not in checked_outputs[dataset][prompt]:
                        print(prompt)
                        raise ValueError("Cannot find category {}".format(name))
                    if name in ["correct_solutions", "groundtruth"]:
                        for i, status in enumerate(checked_outputs[dataset][prompt][name]):
                            if status == "FAILED":
                                continue
                            index = [name, 0, i]
                            if "status" in status and status["status"] in ["PASSED", "TIME_MEASURED"]:
                                processor = CodeProcessor(status["passed_solution"], force_rename = True if dataset in ["openai_humaneval", "mbpp"] else False)
                                solution = processor.run(no_modify = True)
                                if solution[0] != -1 and solution[0] not in indexes[dataset][prompt]:
                                    indexes[dataset][prompt][solution[0]] = []
                                    num += 1
                                    solutions[dataset][prompt].append(solution)
                                    indexes[dataset][prompt][solution[0]].append(index)
                                elif solution[0] != -1:
                                    indexes[dataset][prompt][solution[0]].append(index)
                    else:
                        for i, status in enumerate(checked_outputs[dataset][prompt][name][-1]):
                            if status == "<END>":
                                status, index = self.find_result(dataset, prompt, name, len(checked_outputs[dataset][prompt][name]) - 1, i, checked_outputs)
                                if name == "correctness_reflection_and_feedback" and index[0] == "base":
                                    continue
                                if name.startswith("time") and index[0] == "correct_solutions":
                                    continue
                            else:
                                index = [name, len(checked_outputs[dataset][prompt][name]) - 1, i]
                            if isinstance(status, str):
                                print(status, index)
                                continue
                            if isinstance(status, dict) and "status" in status and status["status"] in ["PASSED", "TIME_MEASURED"]:
                                processor = CodeProcessor(status["passed_solution"], force_rename = True if dataset in ["openai_humaneval", "mbpp"] else False)
                                solution = processor.run(no_modify = True)
                                if solution[0] != -1 and solution[0] not in indexes[dataset][prompt]:
                                    indexes[dataset][prompt][solution[0]] = []
                                    num += 1
                                    solutions[dataset][prompt].append(solution)
                                    indexes[dataset][prompt][solution[0]].append(index)
                                elif solution[0] != -1:
                                    indexes[dataset][prompt][solution[0]].append(index)

        print(f"Totally {num} solutions.")

        solution_file = os.path.join(result_dir, model, "time", "FINAL_PASSED_SOLUTIONS.json")
        index_file = os.path.join(result_dir, model, "time", "FINAL_INDEXES.json")
        final_checked_file = os.path.join(result_dir, model, "time", "FINAL_CHECKED.json")
        with open(solution_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(solutions, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(index_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(indexes, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(final_checked_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(checked_outputs, sort_keys=True, indent=4, separators=(',', ': ')))


                
                    


        

    
    def update_history(self, result_dir, model, mode, rd):
        ori_outputs = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}.json"), "r"))
        checked_outputs = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json"), "r"))
        history = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_HISTORY.json"), "r"))
        leader_history_file = os.path.join(result_dir, model, mode, f"rd{rd}_HISTORY_LEADER.json")
        if os.path.exists(leader_history_file):
            leader_history = json.load(open(leader_history_file, "r"))
        else:
            leader_history = None
        reviewer_history_file = os.path.join(result_dir, model, mode, f"rd{rd}_HISTORY_REVIEWER.json")
        if os.path.exists(reviewer_history_file):
            reviewer_history = json.load(open(reviewer_history_file, "r"))
        else:
            reviewer_history = None

        deleted_prompts = []
        for dataset in history:
            for name in history[dataset]:
                if not name.startswith(mode):
                    continue
                for prompt in history[dataset][name]:
                    if prompt not in checked_outputs[dataset]:
                        deleted_prompts.append([dataset, name, prompt])
                        continue
                    for i, h in enumerate(history[dataset][name][prompt]):
                        if name not in ori_outputs[dataset][prompt]:
                            continue
                        if not self.prompt_generator.round_exist(rd, name):
                            continue
                        if ori_outputs[dataset][prompt][name][-1][i] != "<END>":
                            if name == "time_multiple_agents_with_reviewer" and rd == 1:
                                reviewer_history[dataset][name][prompt][i].append({"role": "assistant", "content": ori_outputs[dataset][prompt][name][-1][i]})
                            elif name == "time_multiple_agents_with_team" and rd in [0, 3]:
                                leader_history[dataset][name][prompt][i].append({"role": "assistant", "content": ori_outputs[dataset][prompt][name][-1][i]})
                            elif name == "time_multiple_agents_with_team" and rd == 2:
                                reviewer_history[dataset][name][prompt][i].append({"role": "assistant", "content": ori_outputs[dataset][prompt][name][-1][i]})
                            else:
                                history[dataset][name][prompt][i].append({"role": "assistant", "content": ori_outputs[dataset][prompt][name][-1][i]})
        
        for i in deleted_prompts:
            try:
                del history[i[0]][i[1]][i[2]]
            except Exception:
                pass
            try:
                del leader_history[i[0]][i[1]][i[2]]
            except Exception:
                pass
            try:
                del reviewer_history[i[0]][i[1]][i[2]]
            except Exception:
                pass
            
        
        
        with open(os.path.join(result_dir, model, mode, f"rd{rd+1}_HISTORY.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(history, sort_keys=True, indent=4, separators=(',', ': ')))

        if leader_history != None:
            with open(os.path.join(result_dir, model, mode, f"rd{rd+1}_HISTORY_LEADER.json"), "w", encoding = "utf-8") as f:
                f.write(json.dumps(leader_history, sort_keys=True, indent=4, separators=(',', ': ')))

        if reviewer_history != None:
            with open(os.path.join(result_dir, model, mode, f"rd{rd+1}_HISTORY_REVIEWER.json"), "w", encoding = "utf-8") as f:
                f.write(json.dumps(reviewer_history, sort_keys=True, indent=4, separators=(',', ': ')))


    def check_history(self, result_dir, model, mode, rd):
        history = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_HISTORY.json"), "r"))
        leader_history_file = os.path.join(result_dir, model, mode, f"rd{rd}_HISTORY_LEADER.json")
        if os.path.exists(leader_history_file):
            leader_history = json.load(open(leader_history_file, "r"))
        else:
            leader_history = None
        reviewer_history_file = os.path.join(result_dir, model, mode, f"rd{rd}_HISTORY_REVIEWER.json")
        if os.path.exists(reviewer_history_file):
            reviewer_history = json.load(open(reviewer_history_file, "r"))
        else:
            reviewer_history = None

        for dataset in history:
            for name in history[dataset]:
                for prompt in history[dataset][name]:
                    for i, h in enumerate(history[dataset][name][prompt]):
                        print(i, end = "\r")
                        if len(h) > 1 and h[-1]["role"] != "assistant":
                            raise ValueError("Check failed! Incorrect last conversation.")
                        for i in range(1, len(h)):
                            if i % 2 == 1 and h[i]["role"] != "user":
                                raise ValueError("Check failed! Incorrect user conversation.")
                            if i % 2 == 0 and h[-1]["role"] != "assistant":
                                raise ValueError("Check failed! Incorrect model conversation.")

        if leader_history:
            for dataset in leader_history:
                for name in leader_history[dataset]:
                    for prompt in leader_history[dataset][name]:
                        for i, h in enumerate(leader_history[dataset][name][prompt]):
                            print(i, end = "\r")
                            if len(h) > 1 and h[-1]["role"] != "assistant":
                                raise ValueError("Check failed! Incorrect last conversation.")
                            for i in range(1, len(h)):
                                if i % 2 == 1 and h[i]["role"] != "user":
                                    raise ValueError("Check failed! Incorrect user conversation.")
                                if i % 2 == 0 and h[-1]["role"] != "assistant":
                                    raise ValueError("Check failed! Incorrect model conversation.")

        if reviewer_history:
            for dataset in reviewer_history:
                for name in reviewer_history[dataset]:
                    for prompt in reviewer_history[dataset][name]:
                        for i, h in enumerate(reviewer_history[dataset][name][prompt]):
                            print(i, end = "\r")
                            if len(h) > 1 and h[-1]["role"] != "assistant":
                                raise ValueError("Check failed! Incorrect last conversation.")
                            for i in range(1, len(h)):
                                if i % 2 == 1 and h[i]["role"] != "user":
                                    raise ValueError("Check failed! Incorrect user conversation.")
                                if i % 2 == 0 and h[-1]["role"] != "assistant":
                                    raise ValueError("Check failed! Incorrect model conversation.")
                            

    def run_correctness_check(self, solution_file, task_index = 0, task_num = 5):
        solutions = json.load(open(solution_file, "r"))
        passed_solutions = {}
        failed_cases = {}
        for dataset in solutions:
            print(f"Processing dataset: {dataset}")
            evaluator = Evaluator(dataset, mbpp_helpfile = "datasets/MbppPlus.jsonl" if dataset == "mbpp" else None)
            evaluator.solutions = solutions[dataset]
            prompts, overlong_prompts = evaluator.dataset.get_all_prompts()
            total_num = len(prompts + overlong_prompts)
            verify_num = (total_num // task_num) + 1
            #verify_num = 5
            start_index = verify_num * task_index
            passed_solutions[dataset], failed_cases[dataset] = evaluator.verify_predictions(start_index = start_index, verify_num = verify_num, failed_case = True)
        
        with open(solution_file.replace("SOLUTIONS.json", f"PASSED_SOLUTIONS_p{task_index}.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(passed_solutions, sort_keys=True, indent=4, separators=(',', ': ')))

        
        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            print("NON_JSON_SERIALIZABLE object {} encountered.".format(type(obj)))
            return "<NON_JSON_SERIALIZABLE>"
            raise TypeError
        
        with open(solution_file.replace("SOLUTIONS.json", f"FAILED_TESTCASES_p{task_index}.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(failed_cases, default=set_default, indent=4, separators=(',', ': ')))

        

    def run_time_measurement(self, solution_file, task_index = 0, task_num = 7):
        passed_solutions = json.load(open(solution_file, "r"))
        time_costs = {}
        large_testcases = {}
        for dataset in passed_solutions:
            print(f"Processing dataset: {dataset}")
            evaluator = Evaluator(dataset, mbpp_helpfile = "datasets/MbppPlus.jsonl" if dataset == "mbpp" else None)
            evaluator.solutions = passed_solutions[dataset]
            subset = list(passed_solutions[dataset].keys())
            total_num = len(subset)
            verify_num = (total_num // task_num) + 1
            #verify_num = 1
            start_index = verify_num * task_index
            time_costs[dataset], large_testcases[dataset] = evaluator.measure_runtime_for_predictions(subset = subset, large_testcase = True, start_index = start_index, verify_num = verify_num)
        
        with open(solution_file.replace("PASSED_SOLUTIONS.json", f"TIME_COSTS_p{task_index}.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(time_costs, sort_keys=True, indent=4, separators=(',', ': ')))
        
        with open(solution_file.replace("PASSED_SOLUTIONS.json", f"LARGE_TESTCASES_p{task_index}.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(large_testcases, sort_keys=True, indent=4, separators=(',', ': ')))

    def check_passed_solutions(self, result_dir, model, mode, rd):
        passed_solutions = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_PASSED_SOLUTIONS.json"), "r"))
        indexes = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_INDEXES.json"), "r"))
        checked_outputs = json.load(open(os.path.join(result_dir, model, mode, f"rd{rd}_CHECKED.json"), "r"))

        for dataset in checked_outputs:
            for prompt in checked_outputs[dataset]:
                if prompt not in passed_solutions[dataset]:
                    continue
                for solution in passed_solutions[dataset][prompt]:
                    for index in indexes[dataset][prompt][solution[0]]:
                        if index[0] in ["correct_solutions", "groundtruth"]:
                            if checked_outputs[dataset][prompt][index[0]][index[2]]["status"] != "PASSED":
                                print("Inconsistent status for passed solution: {}".format(checked_outputs[dataset][prompt][index[0]][index[2]]["status"]))
                        else:
                            if checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"] != "PASSED":
                                print("Inconsistent status for passed solution: {}".format(checked_outputs[dataset][prompt][index[0]][index[1]][index[2]]["status"]))

    def update_best_groundtruth(self, checked_file):
        checked_outputs = json.load(open(checked_file, "r"))

        for dataset in checked_outputs:
            best_solution_file = os.path.join("test_datasets", dataset.replace("/", "_"), "best_solutions.json")
            best_solutions = json.load(open(best_solution_file, "r"))
            ori_num = len(best_solutions)
            for prompt in checked_outputs[dataset]:
                if prompt in best_solutions:
                    continue
                min_time = None
                min_solution = None
                for status in checked_outputs[dataset][prompt]["groundtruth"]:
                    if not isinstance(status, dict):
                        continue
                    if status["status"] == "TIME_MEASURED":
                        if min_time == None:
                            min_time = status["exec_time"]
                            min_solution = status["passed_solution"]
                        elif status["exec_time"] < min_time:
                            min_time = status["exec_time"]
                            min_solution = status["passed_solution"]
                if min_solution != None:
                    processor = CodeProcessor(min_solution, force_rename = True if dataset in ["openai_humaneval", "mbpp"] else False)
                    solution = processor.run(no_modify = True)
                    if solution[0] != -1:
                        best_solutions[prompt] = solution
            if dataset == "mbpp":
                solutions = json.load(open("test_datasets/mbpp/solutions.json", "r"))["prompt2groundtruth"]
                for prompt in solutions:
                    if prompt not in best_solutions:
                        best_solutions[prompt] = solutions[prompt]
            cur_num = len(best_solutions)
            print("Dataset: {}, ori_num: {}, cur_num: {}".format(dataset, ori_num, cur_num))
            
            with open(best_solution_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(best_solutions, sort_keys=True, indent=4, separators=(',', ': ')))
            

    
    
    def combine_results(self, result_dir, models, metrics):
        results = {}
        for metric in metrics:
            results[metric] = {}
            if metric == "pass_rate":
                for m in models:
                    base_results = json.load(open(os.path.join(result_dir, m, "base", "rd0_pass_rate.json"), "r"))
                    correctness_results = json.load(open(os.path.join(result_dir, m, "correctness", "rd1_pass_rate.json"), "r"))
                    for dataset in base_results:
                        if dataset not in results[metric]:
                            results[metric][dataset] = {}
                        if m not in results[metric][dataset]:
                            results[metric][dataset][m] = {}
                        for name in base_results[dataset]:
                            results[metric][dataset][m][name] = base_results[dataset][name]
                    for dataset in correctness_results:
                        if dataset not in results[metric]:
                            results[metric][dataset] = {}
                        if m not in results[metric][dataset]:
                            results[metric][dataset][m] = {}
                        for name in correctness_results[dataset]:
                            results[metric][dataset][m][name] = correctness_results[dataset][name]
        
        with open(os.path.join(result_dir, "results.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))


    def evaluate_execution_time_for_all(self):
        models = [
            "_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62",
            "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-8B-Instruct_snapshots_e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
            "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-70B-Instruct_snapshots_e8cf5276ae3e97cfde8a058e64a636f2cde47820",
            "_export_code-pretrain_yun_transformers_models--microsoft--Phi-3-mini-128k-instruct_snapshots_f10fb29b79f038c78229ab4dcd9234a9666a770f",
            "_export_code-pretrain_yun_transformers_models--mistralai--Mixtral-8x7B-Instruct-v0.1_snapshots_1e637f2d7cb0a9d6fb1922f305cb784995190a83",
            "gpt-3.5-turbo",
            #"gpt-4"
        ]

        evaluator = ChatPromptEvaluator()
        for m in models:
            evaluator.evaluate_execution_time("prompt_chat_results", m, "time", fallback = True)
            evaluator.evaluate_execution_time("prompt_chat_results", m, "time", fallback = False)

    def combine_execution_time_results_for_models(self, fallback = True, best = "20"):
        results = {}
        models = [
            "_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62",
            "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-8B-Instruct_snapshots_e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
            "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-70B-Instruct_snapshots_e8cf5276ae3e97cfde8a058e64a636f2cde47820",
            "_export_code-pretrain_yun_transformers_models--microsoft--Phi-3-mini-128k-instruct_snapshots_f10fb29b79f038c78229ab4dcd9234a9666a770f",
            "_export_code-pretrain_yun_transformers_models--mistralai--Mixtral-8x7B-Instruct-v0.1_snapshots_1e637f2d7cb0a9d6fb1922f305cb784995190a83",
            "gpt-3.5-turbo",
            "gpt-4"
        ]

        for m in models:
            if fallback:
                filename = os.path.join("prompt_chat_results", m, "time", "FINAL_exec_time.json")
            else:
                filename = os.path.join("prompt_chat_results", m, "time", "FINAL_exec_time_nofallback.json")
            data = json.load(open(filename, "r"))

            for dataset in data:
                if dataset not in results:
                    results[dataset] = {}
                if m not in results[dataset]:
                    results[dataset][m] = {}
                for metric in data[dataset]:
                    if metric == "num":
                        continue
                    results[dataset][m][metric] = data[dataset][metric]["time_execution_feedback_with_testcase"][best]
                    if metric == "correctness":
                        results[dataset][m]["base_" + metric] = data[dataset][metric]["base"][best]
                    else:
                        results[dataset][m][metric.replace("gt", "base")] = data[dataset][metric]["base"][best]


        lines = []
        for dataset in results:
            lines.append(dataset + "+"*20)
            lines.append("Model,Speedup,Opt%,Correct%,Base Speedup,Base Opt%,Base Correct%")
            for m in results[dataset]:
                lines.append("{},{},{},{},{},{},{}".format(m, 
                results[dataset][m]["gt_speedup"], 
                format(results[dataset][m]["gt_opt"] * 100, ".2f"), 
                format(results[dataset][m]["correctness"] *100, ".2f"),
                results[dataset][m]["base_speedup"],
                format(results[dataset][m]["base_opt"] * 100, ".2f"), 
                format(results[dataset][m]["base_correctness"] *100, ".2f"),
                ))

        
        with open(f"prompt_chat_results/exec_time_{best}.json", "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))

        with open(f"prompt_chat_results/exec_time_{best}.csv", "w", encoding = "utf-8") as f:
            f.write("\n".join(lines))


    def combine_execution_time_results_for_prompts(self, fallback = True, best = "20"):
        models = [
            "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-70B-Instruct_snapshots_e8cf5276ae3e97cfde8a058e64a636f2cde47820",
            "gpt-3.5-turbo"
        ]

        results = {}

        for m in models:
            if fallback:
                filename = os.path.join("prompt_chat_results", m, "time", "FINAL_exec_time.json")
            else:
                filename = os.path.join("prompt_chat_results", m, "time", "FINAL_exec_time_nofallback.json")
            data = json.load(open(filename, "r"))
            results[m] = data

        
        lines = []
        for m in results:
            lines.append(m + "+"*20)
            for dataset in results[m]:
                lines.append(dataset + "="*10)
                lines.append("Prompt,Speedup,Opt%,Correct%")
                for prompt in results[m][dataset]["gt_speedup"]:
                    lines.append("{},{},{},{}".format(
                        prompt,
                        results[m][dataset]["gt_speedup"][prompt][best],
                        format(results[m][dataset]["gt_opt"][prompt][best] * 100, ".2f"),
                        format(results[m][dataset]["correctness"][prompt][best] * 100, ".2f")
                    ))

        if fallback:
            with open(f"prompt_chat_results/exec_time_prompts_{best}.json", "w", encoding = "utf-8") as f:
                f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))

            with open(f"prompt_chat_results/exec_time_prompts_{best}.csv", "w", encoding = "utf-8") as f:
                f.write("\n".join(lines))
        else:
            with open(f"prompt_chat_results/exec_time_prompts_nofallback_{best}.json", "w", encoding = "utf-8") as f:
                f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))

            with open(f"prompt_chat_results/exec_time_prompts_nofallback_{best}.csv", "w", encoding = "utf-8") as f:
                f.write("\n".join(lines))





    def json2csv(self, jsonfile):
        results = json.load(open(jsonfile, "r"))
        lines = []
        for metric in results:
            lines.append(metric+ "+"*20)
            for dataset in results[metric]:
                lines.append(dataset + "="*10)
                for m in results[metric][dataset]:
                    lines.append(m)
                    lines.append("Strategy,Pass@1,Pass@5,Pass@8,Pass@10,Pass@20")
                    for name in results[metric][dataset][m]:
                        line = "{},{},{},{},{},{}".format(name,
                        format(results[metric][dataset][m][name]["1"] * 100, ".2f"),
                        format(results[metric][dataset][m][name]["5"] * 100, ".2f"),
                        format(results[metric][dataset][m][name]["8"] * 100, ".2f"),
                        format(results[metric][dataset][m][name]["10"] * 100, ".2f"),
                        format(results[metric][dataset][m][name]["20"] * 100, ".2f"))
                        lines.append(line)
        
        with open(jsonfile.replace(".json", ".csv"), "w", encoding = "utf-8") as f:
            f.write("\n".join(lines))


    def json2csv2(self, jsonfile):
        results = json.load(open(jsonfile, "r"))
        lines = []
        for dataset in results:
            lines.append(dataset + "="*20)
            for metric in results[dataset]:
                lines.append(metric+ "+"*10)
                if metric in ["gt_speedup", "gt_opt"]:
                    lines.append("Strategy,Best@1,Best@5,Best@8,Best@10,Best@20,Avg")
                    for name in results[dataset][metric]:
                        line = "{},{},{},{},{},{},{}".format(name,
                        format(results[dataset][metric][name]["1"] * (100 if "speedup" not in metric else 1), ".2f"),
                        format(results[dataset][metric][name]["5"] * (100 if "speedup" not in metric else 1), ".2f"),
                        format(results[dataset][metric][name]["8"] * (100 if "speedup" not in metric else 1), ".2f"),
                        format(results[dataset][metric][name]["10"] * (100 if "speedup" not in metric else 1), ".2f"),
                        format(results[dataset][metric][name]["20"] * (100 if "speedup" not in metric else 1), ".2f"),
                        format(results[dataset][metric][name]["avg"] * (100 if "speedup" not in metric else 1), ".2f"))
                        lines.append(line)
                else:
                    lines.append("Strategy,Avg")
                    for name in results[dataset][metric]:
                        line = "{},{}".format(name,
                        format(results[dataset][metric][name] * (100 if "speedup" not in metric else 1), ".2f"))
                        lines.append(line)

        
        with open(jsonfile.replace(".json", ".csv"), "w", encoding = "utf-8") as f:
            f.write("\n".join(lines))




if __name__ == "__main__":
    evaluator = ChatPromptEvaluator()
    '''
    evaluator.write_batch_outputs(
    "prompt_chat_results/gpt-4/time/rd1_batch_outputs.json",\
    "prompt_chat_results/gpt-4/time/rd1.json",\
    "prompt_chat_results/hash_map.json"
    )
    '''
    #evaluator.prepare_correctness_check("time", 1, "prompt_chat_results/gpt-4/time/rd1.json", chat = True, codegen = False, checked_file = "prompt_chat_results/gpt-4/time/rd0_CHECKED.json")
    #evaluator.run_correctness_check("prompt_chat_results/gpt-4/time/rd1_SOLUTIONS.json", task_index = 4, task_num = 5)
    #evaluator.finalize_correctness_check("prompt_chat_results", "gpt-4", "time", 1, clean = True)
    #evaluator.finalize_time_measurement("prompt_chat_results", "gpt-4", "time", rd = None)
    #evaluator.prepare_time_measurement("prompt_chat_results", "_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62", "time", 0)
    #evaluator.run_time_measurement("prompt_chat_results/gpt-4/time/FINAL_PASSED_SOLUTIONS.json", task_index = 39, task_num = 40)
    #evaluator.update_history("prompt_chat_results", "gpt-4", "time", 0)
    #evaluator.check_history("prompt_chat_results", "gpt-3.5-turbo", "correctness", 1)
    #evaluator.gather_all_solutions_for_time("prompt_chat_results", "_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62")
    #evaluator.gather_correct_solutions_and_history("prompt_chat_results", "_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62", "correctness", "correctness_reflection_and_feedback", "time")
    #evaluator.evaluate_pass_rate("prompt_chat_results", "_export_code-pretrain_yun_transformers_models--mistralai--Mixtral-8x7B-Instruct-v0.1_snapshots_1e637f2d7cb0a9d6fb1922f305cb784995190a83", "time", rd = 1)
    #evaluator.evaluate_execution_time("prompt_chat_results", "gpt-4", "time", fallback = True)
    #evaluator.evaluate_execution_time("prompt_chat_results", "_export_code-pretrain_yun_transformers_models--CohereForAI--c4ai-command-r-plus_snapshots_ba7f1d954c9d1609013677d87e4142ab95c34e62", "time", fallback = True)
    #evaluator.evaluate_execution_time_for_all()
    #evaluator.combine_execution_time_results_for_models(fallback = True, best = "20")
    #evaluator.combine_execution_time_results_for_prompts(fallback = True, best = "1")
    #evaluator.evaluate_correctness("prompt_chat_results")
    #evaluator.get_better_solutions("prompt_chat_results", "gpt-4", "time", "time_execution_feedback_with_testcase")
    '''
    evaluator.combine_results("prompt_chat_results", 
    [
        "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-8B-Instruct_snapshots_e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
        "_export_code-pretrain_yun_transformers_models--meta-llama--Meta-Llama-3-70B-Instruct_snapshots_e8cf5276ae3e97cfde8a058e64a636f2cde47820",
        "_export_code-pretrain_yun_transformers_models--microsoft--Phi-3-mini-128k-instruct_snapshots_f10fb29b79f038c78229ab4dcd9234a9666a770f",
        "_export_code-pretrain_yun_transformers_models--mistralai--Mixtral-8x7B-Instruct-v0.1_snapshots_1e637f2d7cb0a9d6fb1922f305cb784995190a83",
        "gpt-3.5-turbo"
    ],
    ["pass_rate"]
    )'''
    #evaluator.check_passed_solutions("prompt_chat_results", "gpt-3.5-turbo", "time", 0)
    #evaluator.update_best_groundtruth("prompt_chat_results/gpt-3.5-turbo/time_old/FINAL_CHECKED.json")
    #evaluator.json2csv("prompt_chat_results/results.json")
    #evaluator.json2csv2("prompt_chat_results/gpt-3.5-turbo/time_old/FINAL_exec_time.json")

