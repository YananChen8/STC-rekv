import os
class OVOBenchOnlineScore():
    def __init__(self) -> None:
        pass

    def eval():
        pass

class OVOBenchOfflineScore():
    def __init__(self, args, results):
        self.args = args
        self.results = results

    def calculate_score_backward_realtime(self, results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        # Calculate Score for Every Result
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(self, results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)
        
        def get_score_SSR_CRR(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        
        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            # Calculate score for REC
            if result["task"] == "REC":
                for j, test_info_ in enumerate(result["test_info"]):
                    scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
            # Calculate score for SSR
            if result["task"] == "SSR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["SSR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
            # Calculate score for CRR
            if result["task"] == "CRR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["CRR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
        return results, scores
    
    def score(self):
        print(f"Offline Model: {self.args.model}")
        backward_results = self.results["backward"]
        realtime_results = self.results["realtime"]
        forward_results = self.results["forward"]

        avg_scores = {
            "backward": [],
            "realtime": [],
            "forward": []
        }

        # 默认值，防止某一部分没有结果时变量未初始化
        backward_score = 0.0
        realtime_score = 0.0
        forward_score = 0.0
        num_parts = 0

        if len(backward_results) > 0:
            print("Evaluate Backward Tracing...")
            backward_results, backward_scores = self.calculate_score_backward_realtime(backward_results)
            for k, v in backward_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}")
                avg_scores["backward"].append(sum(v) / len(v))
            if len(avg_scores["backward"]) > 0:
                backward_score = 100 * sum(avg_scores["backward"]) / len(avg_scores["backward"])
                num_parts += 1
            print(f"Backward Avg.: {backward_score:.2f}\n")

        if len(realtime_results) > 0:
            print("Evaluate Real-time Visual Perception...")
            realtime_results, realtime_scores = self.calculate_score_backward_realtime(realtime_results)
            for k, v in realtime_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}")
                avg_scores["realtime"].append(sum(v) / len(v))
            if len(avg_scores["realtime"]) > 0:
                realtime_score = 100 * sum(avg_scores["realtime"]) / len(avg_scores["realtime"])
                num_parts += 1
            print(f"Realtime Avg.: {realtime_score:.2f}\n")

        if len(forward_results) > 0:
            print("Evaluate Forward Active Responding...")
            forward_results, forward_scores = self.calculate_score_forward(forward_results)
            for k, v in forward_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}")
                avg_scores["forward"].append(sum(v) / len(v))
            if len(avg_scores["forward"]) > 0:
                forward_score = 100 * sum(avg_scores["forward"]) / len(avg_scores["forward"])
                num_parts += 1
            print(f"Forward Avg.: {forward_score:.2f}\n")

        if num_parts > 0:
            total_avg = (backward_score + realtime_score + forward_score) / num_parts
            print(f"Total Avg.: {total_avg:.2f}")

