import json



class Prompt(object):
    def __init__(self, chat = False):
        self.templates = {}
        if chat:
            self.template_file_path = "src/templates_chat.json"
        else:
            self.template_file_path = "src/templates.json"
        self.example_file_path = "src/fixed_examples.json"


    def build_templates(self, template):
        pass

    def load_templates(self, file_path = None):
        if file_path:
            self.template_file_path = file_path
        self.templates = json.load(open(self.template_file_path, "r"))
    
    def load_examples(self, file_path = None):
        if not file_path:
            self.example_file_path = "src/fixed_examples.json"
        else:
            self.example_file_path = file_path
        self.examples = json.load(open(self.example_file_path, "r"))

    def save_templates(self):
        with open(self.template_file_path, "w", encoding = "utf-8") as f:
            f.write(json.dumps(self.templates, sort_keys=True, indent=4, separators=(',', ': ')))

    def print_templates(self):
        for name in self.templates:
            print(f"=============================Template Name: {name}============================")
            for i, r in enumerate(self.templates[name]):
                print(f"===========Round: {i}==========")
                print("demo:")
                print(r["demo"])
                print("instruction:")
                print(r["instruction"])

    def round_exist(self, rd, name):
        if name == "base":
            return True
        for t in self.templates[name]:
            if t["rd"] == rd:
                return True
        
        if self.templates[name][-1]["repeat"] and rd > self.templates[name][-1]["rd"]:
            return True

        return False

    def gen_code(self, rd, name):
        if name == "base":
            if rd % 2 == 0:
                return True
            else:
                return False
        for t in self.templates[name]:
            if t["rd"] == rd:
                return t["code"]
        
        if self.templates[name][-1]["repeat"] and rd > self.templates[name][-1]["rd"]:
            return self.templates[name][-1]["code"]
        
        return False

    def does_sample(self, rd, name):
        for t in self.templates[name]:
            if t["rd"] == rd:
                template = t
                break
        return template["samples"]


    def apply_template(self, name, rd, kwargs, condition = None):
        for t in self.templates[name]:
            if t["rd"] == rd and t["condition"] == condition:
                template = t
                break
        prompt = ""
        if "demo" in template and template["demo"] != "":
            index = 0
            while True:
                prompt += template["demo"]
                for parameter in template["demo_parameters"]:
                    if isinstance(kwargs[parameter], str):
                        prompt = prompt.replace("{" + parameter + "}", kwargs[parameter])
                        break
                    elif isinstance(kwargs[parameter], list):
                        prompt = prompt.replace("{" + parameter + "}", kwargs[parameter][index])
                index += 1
                if index >= len(kwargs[list(kwargs.keys())[0]]):
                    break
        prompt += template["instruction"]
        for parameter in template["instruction_parameters"]:
            prompt = prompt.replace("{" + parameter + "}", kwargs[parameter])
        prompt = prompt.replace("[trigger]", self.templates["trigger"]).replace("[regular_requirement]", self.templates["regular_requirement"])
        return prompt

    def build_chat_message(self, message, history):
        return history + [{"role": "user", "content": message}]
        

    def get_correctness_testcase_feedback(self, history, testcases = None):
        if testcases == None:
            raise ValueError("Missing testcases.")
        prompts = {}
        for prompt in testcases:
            prompts[prompt] = []
            for i, testcase in enumerate(testcases[prompt]):
                if testcase == None:
                    prompts[prompt].append(None)
                elif testcase in ["PARSE_ERROR", "TIMEOUT"]:
                    message = self.apply_template("correctness_testcase_feedback", 0, {}, condition = "parse_error")
                    prompts[prompt].append(self.build_chat_message(message, history[prompt][i]))
                elif "model_output" in testcase[0]:
                    if testcase[0]["global"]:
                        testcase_str = "Input: {}\nYour Outputs: {}\nExpected Outputs: {}\n".format(testcase[0]["input"], testcase[0]["model_output"], testcase[0]["output"])
                    else:
                        inputs = [str(i) for i in testcase[0]["input"]]
                        testcase_str = "Input: {}\nYour Outputs: {}\nExpected Outputs: {}\n".format(",".join(inputs), testcase[0]["model_output"], testcase[0]["output"])
                    message = self.apply_template("correctness_testcase_feedback", 0, {"testcase": testcase_str}, condition = "failed_testcase")
                    prompts[prompt].append(self.build_chat_message(message, history[prompt][i]))
                else:
                    if testcase[0]["global"]:
                        testcase_str = "Input: {}\nExpected Outputs: {}\n".format(testcase[0]["input"], testcase[0]["output"])
                    else:
                        inputs = [str(i) for i in testcase[0]["input"]]
                        testcase_str = "Input: {}\nExpected Outputs: {}\n".format(",".join(inputs), testcase[0]["output"])
                    error = testcase[0]["status_reason"]
                    message = self.apply_template("correctness_testcase_feedback", 0, {"error": error, "testcase": testcase_str}, condition = "runtime_error")
                    prompts[prompt].append(self.build_chat_message(message, history[prompt][i]))
        
        return prompts

    def get_correctness_reflection_and_feedback(self, rd, history, testcases = None, indicators = None):
        if testcases == None:
            raise ValueError("Missing testcases.")
        prompts = {}
        if rd % 2 == 0:
            for prompt in testcases:
                prompts[prompt] = []
                for i, testcase in enumerate(testcases[prompt]):
                    if testcase == None:
                        prompts[prompt].append(None)
                    elif testcase in ["PARSE_ERROR", "TIMEOUT"]:
                        message = self.apply_template("correctness_reflection_and_feedback", 0, {}, condition = "parse_error")
                        prompts[prompt].append(self.build_chat_message(message, history[prompt][i]))
                    elif "model_output" in testcase[0]:
                        if testcase[0]["global"]:
                            testcase_str = "Input: {}\nYour Outputs: {}\nExpected Outputs: {}\n".format(testcase[0]["input"], testcase[0]["model_output"], testcase[0]["output"])
                        else:
                            inputs = [str(i) for i in testcase[0]["input"]]
                            testcase_str = "Input: {}\nYour Outputs: {}\nExpected Outputs: {}\n".format(",".join(inputs), testcase[0]["model_output"], testcase[0]["output"])
                        message = self.apply_template("correctness_reflection_and_feedback", rd, {"testcase": testcase_str}, condition = "failed_testcase")
                        prompts[prompt].append(self.build_chat_message(message, history[prompt][i]))
                    else:
                        if testcase[0]["global"]:
                            testcase_str = "Input: {}\nExpected Outputs: {}\n".format(testcase[0]["input"], testcase[0]["output"])
                        else:
                            inputs = [str(i) for i in testcase[0]["input"]]
                            testcase_str = "Input: {}\nExpected Outputs: {}\n".format(",".join(inputs), testcase[0]["output"])
                        error = testcase[0]["status_reason"]
                        message = self.apply_template("correctness_reflection_and_feedback", rd, {"error": error, "testcase": testcase_str}, condition = "runtime_error")
                        prompts[prompt].append(self.build_chat_message(message, history[prompt][i]))
        elif rd % 2 == 1:
            if indicators == None:
                raise ValueError("Missing indicators.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i]:
                        message = self.apply_template("correctness_reflection_and_feedback", rd, {})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        
        return prompts

    def get_time_simple_instruction(self, history, indicators = None):
        prompts = {}
        if indicators == None:
            raise ValueError("Missing indicators.")
        for prompt in history:
            if prompt not in indicators:
                continue
            prompts[prompt] = []
            for i, h in enumerate(history[prompt]):
                if indicators[prompt][i]:
                    message = self.apply_template("time_simple_instruction", 0, {})
                    prompts[prompt].append(self.build_chat_message(message, h))
                else:
                    prompts[prompt].append(None)

        return prompts

    def get_time_in_context_learning(self, history, demos = None, demo_num = 1, indicators = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        if demos == None:
            raise ValueError("Missing demos.")
        prompts = {}
        example = "\nOriginal Program:\n{}\n\nOptimized Program:\n{}\n"
        demo_str = ""
        for i in range(demo_num):
            demo_str += example.format(demos[i][0], demos[i][1])
        
        for prompt in history:
            if prompt not in indicators:
                continue
            prompts[prompt] = []
            for i, h in enumerate(history[prompt]):
                if indicators[prompt][i]:
                    message = self.apply_template("time_in_context_learning", 0, {"demo": demo_str})
                    prompts[prompt].append(self.build_chat_message(message, h))
                else:
                    prompts[prompt].append(None)
        
        return prompts

    def get_time_pre_defined_strategy(self, history, indicators = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        for prompt in history:
            if prompt not in indicators:
                continue
            prompts[prompt] = []
            for i, h in enumerate(history[prompt]):
                if indicators[prompt][i]:
                    message = self.apply_template("time_pre_defined_strategy", 0, {})
                    prompts[prompt].append(self.build_chat_message(message, h))
                else:
                    prompts[prompt].append(None)

        return prompts

    def get_time_chain_of_thought(self, rd, history, indicators = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        for prompt in history:
            if prompt not in indicators:
                continue
            prompts[prompt] = []
            for i, h in enumerate(history[prompt]):
                if indicators[prompt][i]:
                    message = self.apply_template("time_chain_of_thought", rd, {})
                    prompts[prompt].append(self.build_chat_message(message, h))
                else:
                    prompts[prompt].append(None)

        return prompts

    def get_time_time_complexity_reduction(self, rd, history, indicators = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        for prompt in history:
            if prompt not in indicators:
                continue
            prompts[prompt] = []
            for i, h in enumerate(history[prompt]):
                if indicators[prompt][i]:
                    message = self.apply_template("time_time_complexity_reduction", rd, {})
                    prompts[prompt].append(self.build_chat_message(message, h))
                else:
                    prompts[prompt].append(None)

        return prompts


    def get_time_simple_execution_feedback(self, rd, history, times = None, indicators = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        if rd == 0:
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i]:
                        message = self.apply_template("time_simple_execution_feedback", rd, {})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 1:
            if times == None:
                raise ValueError("Missing required data times.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i] and times[prompt][i] != None:
                        if times[prompt][i][0] > times[prompt][i][1]:
                            message = self.apply_template("time_simple_execution_feedback", rd, {"ori_time": str(times[prompt][i][0]), "opt_time": str(times[prompt][i][1])}, condition = "positive")
                        else:
                            message = self.apply_template("time_simple_execution_feedback", rd, {"ori_time": str(times[prompt][i][0]), "opt_time": str(times[prompt][i][1])}, condition = "negative")
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        
        return prompts
    
    def get_time_execution_feedback_with_testcase(self, rd, history, testcases = None, indicators = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        if rd == 0:
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i]:
                        message = self.apply_template("time_execution_feedback_with_testcase", rd, {})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 1:
            if testcases == None:
                raise ValueError("Missing required data times.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i]:
                        if testcases[prompt][i][0]["global"]:
                            testcase_str = "Input: {}\nExpected Outputs: {}\n".format(testcases[prompt][i][0]["input"], testcases[prompt][i][0]["output"])
                        else:
                            inputs = [str(i) for i in testcases[prompt][i][0]["input"]]
                            testcase_str = "Input: {}\nExpected Outputs: {}\n".format(",".join(inputs), testcases[prompt][i][0]["output"])
                        message = self.apply_template("time_execution_feedback_with_testcase", rd, {"testcase": testcase_str})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        
        return prompts

    def get_time_multiple_agents_with_reviewer(self, rd, history, reviewer_history = None, indicators = None, decisions = None, comments = None, programs = None, opt_programs = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        if rd == 0:
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_reviewer", rd, {})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 1:
            if reviewer_history == None or programs == None or opt_programs == None:
                raise ValueError("Missing reviewer history or programs or opt_programs.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(reviewer_history[prompt]):
                    if indicators[prompt][i] and programs[prompt][i] and opt_programs[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_reviewer", rd, {"program": programs[prompt][i], "opt_program": opt_programs[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 2:
            if decisions == None or comments == None:
                raise ValueError("Missing decisions or comments.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i] and decisions[prompt][i] and comments[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_reviewer", rd, {"decision": decisions[prompt][i], "comment": comments[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        
        return prompts


    def get_time_multiple_agents_with_team(self, rd, history, reviewer_history = None, leader_history = None, indicators = None, problems = None, programs = None, plans = None, opt_programs = None, decisions = None, comments = None):
        if indicators == None:
            raise ValueError("Missing indicators.")
        prompts = {}
        if rd == 0:
            if leader_history == None or problems == None or programs == None:
                raise ValueError("Missing leader_history, problems or programs.")
            for prompt in leader_history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(leader_history[prompt]):
                    if indicators[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_team", rd, {"problem": problems[prompt], "program": programs[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 1:
            if plans == None:
                raise ValueError("Missing programs or plans.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i] and plans[prompt][i] and programs[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_team", rd, {"plan": plans[prompt][i], "program": programs[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 2:
            if reviewer_history == None or programs == None or opt_programs == None:
                raise ValueError("Missing reviewer history or programs or opt_programs.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(reviewer_history[prompt]):
                    if indicators[prompt][i] and plans[prompt][i] and programs[prompt][i] and opt_programs[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_team", rd, {"program": programs[prompt][i], "plan": plans[prompt][i], "opt_program": opt_programs[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 3:
            if leader_history == None or opt_programs == None or decisions == None or comments == None:
                raise ValueError("Missing leader_history, opt_programs, decisions or comments.")
            for prompt in leader_history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(leader_history[prompt]):
                    if indicators[prompt][i] and opt_programs[prompt][i] and decisions[prompt][i] and comments[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_team", rd, {"opt_program": opt_programs[prompt][i], "decision": decisions[prompt][i], "comment": comments[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        elif rd == 4:
            if plans == None:
                raise ValueError("Missing plans.")
            for prompt in history:
                if prompt not in indicators:
                    continue
                prompts[prompt] = []
                for i, h in enumerate(history[prompt]):
                    if indicators[prompt][i] and plans[prompt][i]:
                        message = self.apply_template("time_multiple_agents_with_team", rd, {"plan": plans[prompt][i]})
                        prompts[prompt].append(self.build_chat_message(message, h))
                    else:
                        prompts[prompt].append(None)
        
        return prompts  


    def get_chat_prompts(self, name, rd, history, **kwargs):
        if name == "correctness_testcase_feedback":
            return self.get_correctness_testcase_feedback(history, **kwargs)
        elif name == "correctness_reflection_and_feedback":
            return self.get_correctness_reflection_and_feedback(rd, history, **kwargs)
        elif name == "time_simple_instruction":
            return self.get_time_simple_instruction(history, **kwargs)
        elif name == "time_in_context_learning":
            return self.get_time_in_context_learning(history, **kwargs)
        elif name == "time_pre_defined_strategy":
            return self.get_time_pre_defined_strategy(history, **kwargs)
        elif name == "time_chain_of_thought":
            return self.get_time_chain_of_thought(rd, history, **kwargs)
        elif name == "time_time_complexity_reduction":
            return self.get_time_time_complexity_reduction(rd, history, **kwargs)
        elif name == "time_simple_execution_feedback":
            return self.get_time_simple_execution_feedback(rd, history, **kwargs)
        elif name == "time_execution_feedback_with_testcase":
            return self.get_time_execution_feedback_with_testcase(rd, history, **kwargs)
        elif name == "time_multiple_agents_with_reviewer":
            return self.get_time_multiple_agents_with_reviewer(rd, history, **kwargs)
        elif name == "time_multiple_agents_with_team":
            return self.get_time_multiple_agents_with_team(rd, history, **kwargs)
        else:
            raise ValueError(f"Unknown prompt name: {name}")

