
import json, os
import ast
from sanitize import sanitize, CodeProcessor
from dataset import Dataset
from code_execution import untrusted_check, untrusted_runtime_measure, check_success, is_all_equal, FAILED
import sys
import numpy as np
import itertools
from multiprocessing import Pool
import time
from datetime import datetime
from prompt import Prompt
from copy import deepcopy



FAILED = -1
SUCCEED = 1


class Extractor(object):
    def __init__(self, dataset, output_file) -> None:
        self.dataset = Dataset(dataset, data_path = os.path.join("test_datasets/", dataset.replace("/", "_")))
        self.output_file = output_file
        self.outputs = json.load(open(self.output_file, "r"))

        self.solutions = {}


    def get_entrypoint(self, instance):
        entry_point = ""
        if self.dataset.name == "openai_humaneval":
            entry_point += instance["entry_point"]

        return entry_point

    def process_solution(self, solution, instance):
        if self.dataset.name == "openai_humaneval" and not solution.startswith("def"):
            lines = solution.splitlines()
            if len(lines) > 0:
                lines[0] = "    " + lines[0]
            solution = instance["prompt"] + "\n".join(lines)
        elif self.dataset.name == "openai_humaneval":
            try:
                ast.parse(solution)
            except:
                lines = solution.splitlines()
                if len(lines) > 0:
                    lines[0] = "    " + lines[0]
                solution = instance["prompt"] + "\n".join(lines)

        return solution


    def get_solutions(self, codegen = False, chat = False):
        for index, prompt in enumerate(self.outputs):
            if prompt not in self.dataset.prompt2instance:
                print('Cannot find the prompt of instance #{} in dataset, skipped.'.format(index))
                continue
            instance = self.dataset.prompt2instance[prompt]
            solutions = []
            if not self.outputs[prompt][1]:
                continue
            if not isinstance(self.outputs[prompt][0], list):
                print(self.outputs[prompt][0])
                continue
            for code in self.outputs[prompt][0]:
                solution = sanitize(code, self.get_entrypoint(instance), codegen = codegen, global_code = True if self.dataset.name in ["NTU-NLP-sg/xCodeEval", "codeparrot/apps"] else False, chat = chat)
                solution = self.process_solution(solution, instance)
                if solution not in solutions:
                    solutions.append(solution)
            
            self.solutions[prompt] = solutions


    def save_solutions(self):
        filename = self.output_file.replace(".json", "_SOLUTIONS.json")
        with open(filename, "w", encoding = "utf-8") as f:
            f.write(json.dumps(self.solutions, sort_keys=True, indent=4, separators=(',', ': ')))



    def process_solutions(self):
        count = 0
        total = 0
        for index, prompt in enumerate(self.solutions):
            print("Processing solutions for instance #{}".format(index), end = "\r", file = sys.stderr)
            new_solutions = []
            for i, solution in enumerate(self.solutions[prompt]):
                total += 1
                processor = CodeProcessor(solution, entry_point = self.dataset.prompt2instance[prompt]["entry_point"] if "entry_point" in self.dataset.prompt2instance[prompt] else None, force_rename = True if self.dataset.name in ["mbpp", "openai_humaneval"] else False)
                res = processor.run()
                if FAILED == res[0]:
                    count += 1
                new_solutions.append(res)
                
            self.solutions[prompt] = new_solutions
        
        print(f"Processing completed for dataset {self.dataset.name}. {count}/{total} ({count/total}) invalid solutions found.")


class Evaluator(object):
    def __init__(self, dataset, dataset_repo = "test_datasets", mbpp_helpfile = None):
        self.dataset = Dataset(dataset, data_path = os.path.join(dataset_repo, dataset.replace("/", "_")), testfile_path = mbpp_helpfile)
        self.dataset.load_testcases()
        self.dataset.load_groundtruths()


    def load_solutions(self, solution_file):
        self.solution_file = solution_file
        self.solutions = json.load(open(self.solution_file, "r"))

    def check_element_type(lst, t):
        for l in lst:
            if not isinstance(l, t):
                if t == float and isinstance(l, int):
                    continue
                return False

        return True

    def transform_element_type(lst, t):
        new_lst = []
        for l in lst:
            new_lst.append(t(l))
        return new_lst

    def prepare_testcases(self, solution_io, gt_io, testcases):
        if solution_io == gt_io:
            return testcases
        elif solution_io:
            new_testcases = []
            for testcase in testcases:
                new_testcase = {}
                new_testcase["output"] = testcase["output"]
                if check_element_type(testcase["input"], str):
                    new_testcase["input"] = ["\n".join(testcase["input"]) + "\n"]
                elif check_element_type(testcase["input"], float):
                    new_testcase["input"] = ["\n".join(self.transform_element_type(testcase["input"], str)) + "\n"]
                else:
                    new_testcase["input"] = testcase["input"]
                new_testcases.append(new_testcase)
            return new_testcases
        else:
            new_testcases = []
            for testcase in testcases:
                new_testcase = {}
                new_testcase["output"] = testcase["output"]
                if len(testcase["input"]) == 1 and isinstance(testcase["input"][0], str):
                    new_testcase["input"] = [testcase["input"][0].split("\n")]
                else:
                    new_testcase["input"] = testcase["input"]
                new_testcases.append(new_testcase)
            return new_testcases


    def get_expected_outputs(self):
        prompts, overlong_prompts = self.dataset.get_all_prompts()

        max_count = 1
        count = 0

        for index, prompt in enumerate(prompts + overlong_prompts):
            #if "Write a function to find the nth newman" not in prompt:
            #    continue
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            gt, io = self.dataset.prompt2groundtruth[prompt][0]
            print(index)
            stat, results = self.execute_code(gt, io, self.dataset.prompt2testcase[prompt], check = False)
            
            if stat == "fail":
                print(results)
                print()
                print(self.dataset.prompt2testcase[prompt])
                print()
                print(gt)
                print()
                print(prompt)
                self.dataset.prompt2testcase[prompt] = []
                count += 1
                if count > max_count:
                    raise ValueError("Groundtruth solution verification failed!")
                continue
            if len(results) != len(self.dataset.prompt2testcase[prompt]):
                raise ValueError("Num of returned results is inconsistent with original testcase inputs.")
            for i, res in enumerate(results):
                self.dataset.prompt2testcase[prompt][i]["output"] = res["model_output"]
            try:
                json.dumps(self.dataset.prompt2testcase[prompt])
            except:
                self.dataset.prompt2testcase[prompt] = []
        
        self.dataset.save_testcases()
        self.dataset.load_testcases()
        self.verify_groundtruth(remove_instance=True)
        self.dataset.save_testcases()


    def fix_testcases(self, old_solution_file):
        prompts, overlong_prompts = self.dataset.get_all_prompts()

        data = json.load(open(old_solution_file, "r"))

        old_solutions = data["prompt2groundtruth"]
        prompt2ios = data["prompt2io"]

        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2groundtruth[prompt]) > 0:
                continue
            elif prompt not in self.dataset.prompt2groundtruth:
                continue
            elif prompt not in old_solutions:
                continue

            print(f"Instance # {index}")

            candidates = []
            candidate_gts = []
            for i, gts in enumerate(old_solutions[prompt]):
                print("Verifying instance #{} solution #{}     ".format(index, i), end = '\r', file = sys.stderr)
                gt, io = gts
                try:
                    stat, results = self.execute_code(gt, io, self.dataset.prompt2testcase[prompt], check = False)
                except Exception as e:
                    print(e)
                    continue
                if check_success(results) and len(results) == len(self.dataset.prompt2testcase[prompt]):
                    candidates.append(results)
                    candidate_gts.append(gts)


            equal_groups = []
            for i, c1 in enumerate(candidates):
                print(i, end = '\r', file = sys.stderr)
                if len(equal_groups) == 0:
                    equal_groups.append([i])
                    continue
                equal = False
                for group in equal_groups:
                    if is_all_equal(candidates[group[0]], c1):
                        equal = True
                        group.append(i)
                        break
                if not equal:
                    equal_groups.append([i])
            scores = [len(group) for group in equal_groups]

            if len(scores) == 0:
                continue
            
            max_score = max(scores)
            if max_score > len(candidates) * 0.8:
                candidate = candidates[equal_groups[scores.index(max_score)][0]]
            else:
                candidate = None
            
            if candidate != None:
                testcases = []
                for i, testcase in enumerate(self.dataset.prompt2testcase[prompt]):
                    testcase["output"] = candidate[i]["model_output"]
                    testcases.append(testcase)
                print(equal_groups)
                print("Fix the incorrect expected outputs in testcase with confidence rate: {}".format(max_score / len(candidates)))
                self.dataset.prompt2testcase[prompt] = testcases
                self.dataset.prompt2groundtruth[prompt] = [candidate_gts[i] for i in equal_groups[scores.index(max_score)]]
                self.dataset.prompt2io[prompt] = prompt2ios[prompt]

        
        self.dataset.save_testcases()
        self.dataset.save_groundtruths()
                

            

            


    def verify_groundtruth(self, remove_instance = False, update_solution = False, debug = False, start_index = None, verify_num = None):
        prompts, overlong_prompts = self.dataset.get_all_prompts()

        max_count = 2
        count = 0

        success = 0

        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if index in [40, 264, 2156, 2200, 3799, 4812] and self.dataset.name == "codeparrot/apps":
                self.dataset.prompt2groundtruth[prompt] = []
                continue
            succeed = False
            if update_solution:
                new_solutions = []
            for i, gts in enumerate(self.dataset.prompt2groundtruth[prompt]):
                print("Verifying instance #{} solution #{}     ".format(index, i), end = '\r', file = sys.stderr)
                gt, io = gts
                stat, results = self.execute_code(gt, io, self.dataset.prompt2testcase[prompt], check = True, fast_check = True)
                
                if stat == "fail":
                    failed_cases = [res for res in results if res["status"] == -1]
                    if debug:
                        pass
                        '''
                        print(results)
                        print("-"*20 + f"Instance#{index},Solution#{i}" + "-"*20)
                        print(failed_cases)
                        print(gt)
                        print([self.dataset.prompt2testcase[prompt][i] for i, res in enumerate(results) if res["status"] == -1])
                        print("Groundtruth solution verification failed for instance #{}!".format(index))
                        print("Original Solutions:")
                        if self.dataset.name == "deepmind/code_contests":
                            raw = self.dataset.prompt2instance[prompt]["solutions"]
                            solutions = []
                            for index, lang in enumerate(raw["language"]):
                                if lang == 1:
                                    solutions.append(raw["solution"][index])
                            for s in solutions:
                                print(s)
                        elif self.dataset.name == "codeparrot/apps":
                            raw = self.dataset.prompt2instance[prompt]["solutions"]
                            solutions = json.loads(raw)
                            for solution in solutions:
                                print(solution)
                        '''
                    if remove_instance:
                        self.dataset.prompt2testcase[prompt] = []
                        print("Instance Removed.")
                else:
                    if update_solution:
                        new_solutions.append(gts)
                    succeed = True
            if succeed:
                success += 1
            else:
                if debug:
                    print("-"*20 + f"Instance#{index}" + "-"*20)
                count += 1
            if update_solution:
                print("Updated solutions for instance #{}, originally {} solutions, now {} solutions.".format(index, len(self.dataset.prompt2groundtruth[prompt]), len(new_solutions)))
                self.dataset.prompt2groundtruth[prompt] = new_solutions
            '''
            if count > max_count:
                raise ValueError("Groundtruth solution verification failed!")
            
            if index % 200 == 0 and update_solution and self.dataset.name == "codeparrot/apps":
                print("Save ground verified ground truths at instance #{}".format(index))
                self.dataset.save_groundtruths()
            '''


        #if update_solution:
        #    self.dataset.save_groundtruths()
        
        print("Verification completed, Pass: {}, Fail: {}.".format(success, len(self.dataset.prompt2instance) - success))
        

    def execute_code(self, code, io, testcases, check = True, fast_check = False, total_timeout = 10):
        if len(testcases) == 0:
            raise ValueError("No testcase to be executed.")
        stat, testcases = untrusted_check(io, code, testcases, 0, [30 for t in testcases], check = check, fast_check = True, total_timeout = total_timeout)
        return stat, testcases


    # unbiased estimator from https://github.com/openai/human-eval
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


    def save_temp_data(self, passed_solutions, results):
        with open(self.solution_file.replace("SOLUTIONS.json", "PASSED_SOLUTIONS.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(passed_solutions, sort_keys=True, indent=4, separators=(',', ': ')))
        
        with open(self.solution_file.replace("SOLUTIONS.json", "RESULTS.json"), "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))


    def load_temp_data(self):
        passed_solutions = json.load(open(self.solution_file.replace("SOLUTIONS.json", "PASSED_SOLUTIONS.json"), "r"))

        return passed_solutions




    def verify_predictions(self, debug = False, load_temp_data = False, start_index = None, verify_num = None, failed_case = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        if not load_temp_data:
            passed_solutions = {}
            failed_cases = {}
        else:
            passed_solutions = self.load_temp_data()
            failed_cases = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            if len(self.dataset.prompt2groundtruth[prompt]) == 0 and self.dataset.name != "NTU-NLP-sg/xCodeEval":
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in passed_solutions:
                continue
            if prompt not in self.solutions:
                continue
            if len(self.solutions[prompt]) == 0:
                continue
            
            correct_count = 0
            passed_solutions[prompt] = []
            failed_cases[prompt] = []
            total_timeout = 10
            if self.dataset.name == "NTU-NLP-sg/xCodeEval":
                instance = self.dataset.prompt2instance[prompt]
                try:
                    total_timeout = len(self.dataset.prompt2testcase[prompt]) * float(instance["time_limit"].replace(" seconds", "").replace(" second", ""))
                except:
                    total_timeout = len(self.dataset.prompt2testcase[prompt])
            for i, solution in enumerate(self.solutions[prompt]):
                print("Verifying prediction #{} for instance #{}         ".format(i, index), end = "\r", file = sys.stderr)
                cur_failed_cases = []
                s, io = solution
                if s == -1:
                    continue
                stat, r = self.execute_code(s, io, self.dataset.prompt2testcase[prompt], check = True, fast_check = True, total_timeout = total_timeout)
                if stat == "pass":
                    correct_count += 1
                    passed_solutions[prompt].append(solution)
                else:
                    if failed_case:
                        for result in r:
                            if result["status"] == FAILED:
                                try:
                                    json.dumps(result)
                                except Exception as e:
                                    del result["model_output"]
                                cur_failed_cases.append(result)
                failed_cases[prompt].append(cur_failed_cases)
                            

            #if index % 200 == 0 and self.dataset.name == "codeparrot/apps" and start_index == None and verify_num == None:
            #    print("Save passed solutions at instance #{}".format(index))
            #    self.save_temp_data(passed_solutions, results)

        if failed_case:
            return passed_solutions, failed_cases
        else:
            return passed_solutions

    def measure_runtime_for_groundtruths(self, load_data = False, subset = None, start_index = None, verify_num = None):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        if load_data:
            time_costs = json.load(open(os.path.join(self.dataset.data_path, "time_costs.json"), "r"))
        else:
            time_costs = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in time_costs:
                continue
            if subset != None and prompt not in subset:
                continue
            time_costs[prompt] = []
            for i, gts in enumerate(self.dataset.prompt2groundtruth[prompt]):
                print("Verifying instance #{} solution #{}          ".format(index, i), end = '\r', file = sys.stderr)
                gt, io = gts
                try:
                    results = self.execute_code_for_runtime(gt, io, self.dataset.prompt2testcase[prompt])
                except Exception as e:
                    continue
                time_cost = sum(results)
                if time_cost == 0:
                    continue
                time_costs[prompt].append(time_cost)

            if index % 200 == 0 and self.dataset.name == "codeparrot/apps" and start_index == None and verify_num == None and subset == None:
                print("Save time costs at instance #{}".format(index))
                with open(os.path.join(self.dataset.data_path, "time_costs.json"), "w", encoding = "utf-8") as f:
                    f.write(json.dumps(time_costs, sort_keys=True, indent=4, separators=(',', ': ')))
                
        if start_index == None and verify_num == None and subset == None:
            filename = os.path.join(self.dataset.data_path, "time_costs.json")
            with open(filename, "w", encoding = "utf-8") as f:
                f.write(json.dumps(time_costs, sort_keys=True, indent=4, separators=(',', ': ')))
        else:
            return time_costs
        



    def measure_runtime_for_predictions(self, subset = None, start_index = None, verify_num = None, large_testcase = False):
        if large_testcase:
            large_testcases = {}
        if subset == None:
            prompts, overlong_prompts = self.dataset.get_all_prompts()
            time_costs = {}
            for index, prompt in enumerate(prompts + overlong_prompts):
                if start_index != None and index < start_index:
                    continue
                if start_index != None and verify_num != None and index >= start_index + verify_num:
                    break
                if prompt not in self.solutions:
                    continue
                if len(self.solutions[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0 and self.dataset.name != "NTU-NLP-sg/xCodeEval":
                    continue
                if subset != None and prompt not in subset:
                    continue
                time_costs[prompt] = []
                if large_testcase:
                    large_testcases[prompt] = []
                for i, solution in enumerate(self.solutions[prompt]):
                    print("[Time: {}] Verifying prediction #{} for instance #{}/{}         ".format(datetime.now(), i, index, start_index + verify_num), end = "\r", file = sys.stderr)
                    s, io = solution
                    results = self.execute_code_for_runtime(s, io, self.dataset.prompt2testcase[prompt])
                    time_cost = sum(results)
                    if time_cost == 0:
                        continue
                    if large_testcase:
                        large_testcases[prompt].append(self.dataset.prompt2testcase[prompt][results.index(max(results))])
                    time_costs[prompt].append(time_cost)
        else:
            time_costs = {}
            for index, prompt in enumerate(subset):
                if start_index != None and index < start_index:
                    continue
                if start_index != None and verify_num != None and index >= start_index + verify_num:
                    break
                if prompt not in self.solutions:
                    continue
                if len(self.solutions[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0 and self.dataset.name != "NTU-NLP-sg/xCodeEval":
                    continue
                time_costs[prompt] = []
                if large_testcase:
                    large_testcases[prompt] = []
                for i, solution in enumerate(self.solutions[prompt]):
                    print("[Time: {}] Verifying prediction #{} for instance #{}/{}         ".format(datetime.now(), i, index, start_index + verify_num), end = "\r", file = sys.stderr)
                    s, io = solution
                    results = self.execute_code_for_runtime(s, io, self.dataset.prompt2testcase[prompt])
                    time_cost = sum(results)
                    if time_cost == 0:
                        continue
                    if large_testcase:
                        large_case = self.dataset.prompt2testcase[prompt][results.index(max(results))]
                        large_case["global"] = io
                        large_testcases[prompt].append(large_case)
                    time_costs[prompt].append(time_cost)

        if large_testcase:
            return time_costs, large_testcases
        else:
            return time_costs


    def measure_all_runtime_for_predictions(self, models, solutions, subset = None, start_index = None, verify_num = None):
        if subset == None:
            prompts, overlong_prompts = self.dataset.get_all_prompts()
            time_costs = {}
            for index, prompt in enumerate(prompts + overlong_prompts):
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0 and self.dataset.name != "NTU-NLP-sg/xCodeEval":
                    continue
                if subset != None and prompt not in subset:
                    continue
                time_costs[prompt] = {}
                for model in models:
                    time_costs[prompt][model] = []
                    if prompt not in solutions[model] or len(solutions[model][prompt]) == 0:
                        continue
                    for i, solution in enumerate(solutions[model][prompt]):
                        print("[Time: {}] Verifying prediction #{} for instance #{}         ".format(datetime.now(), i, index), end = "\r", file = sys.stderr)
                        s, io = solution
                        results = self.execute_code_for_runtime(s, io, self.dataset.prompt2testcase[prompt])
                        time_cost = sum(results)
                        if time_cost == 0:
                            continue
                        time_costs[prompt][model].append(time_cost)
        else:
            time_costs = {}
            for index, prompt in enumerate(subset):
                if start_index != None and index < start_index:
                    continue
                if start_index != None and verify_num != None and index >= start_index + verify_num:
                    break
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0 and self.dataset.name != "NTU-NLP-sg/xCodeEval":
                    continue
                time_costs[prompt] = {}
                for model in models:
                    time_costs[prompt][model] = []
                    if prompt not in solutions[model] or len(solutions[model][prompt]) == 0:
                        continue
                    for i, solution in enumerate(solutions[model][prompt]):
                        print("[Time: {}] Verifying prediction #{} for instance #{}/{} for model {}".format(datetime.now(), i, index, start_index + verify_num, model), end = "\r", file = sys.stderr)
                        s, io = solution
                        results = self.execute_code_for_runtime(s, io, self.dataset.prompt2testcase[prompt])
                        time_cost = sum(results)
                        if time_cost == 0:
                            continue
                        time_costs[prompt][model].append(time_cost)


        return time_costs

        

    def execute_code_for_runtime(self, code, io, testcases):
        if len(testcases) == 0:
            raise ValueError("No testcase to be executed.")
        results = untrusted_runtime_measure(io, code, testcases, [300 for t in testcases])
        return results

        