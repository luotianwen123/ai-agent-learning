"""SimpleAgent类记忆体"""
import json
import requests
import os
class SimpleAgent:
    def __init__(self, agent_name, system_prompt, api_key, api_url, max_history_len=10):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.max_history_len = max_history_len
        self.api_key = api_key
        self.api_url = api_url
        self.history = []
    @staticmethod
    def load_config_from_json(file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            agent = SimpleAgent(
                agent_name=config["agent_name"],
                system_prompt=config["system_prompt"],
                api_key=config["api_key"],
                api_url=config["api_url"],
                max_history_len=config.get("max_history_len", 10),
            )
            return agent, "配置加载成功"
        except FileNotFoundError:
            return None, f"配置文件不存在: {file_path}"
        except json.JSONDecodeError:
            return None, f"配置文件 JSON 格式错误: {file_path}"
        except KeyError as e:
            return None, f"配置缺少必要字段: {e}"
    def dump_history_to_json(self, file_path):
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                return False, f"创建目录失败: {e}"
        try:
            with open(file_path, 'w', encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            return True, "保存成功"
        except (OSError, TypeError) as e:
            return False, f"保存历史记录失败: {e}"
    def load_history_from_json(self, file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                self.history = json.load(f)
            self.history = self.history[-self.max_history_len:]
            return True, "加载成功"
        except FileNotFoundError:
            return False, f"文件不存在: {file_path}"
        except json.JSONDecodeError:
            return False, "JSON 解析失败"
        except OSError as e:
            return False, f"加载历史记录失败: {e}"
    def chat(self, user_input):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": "gpt-3.5-turbo", "messages": messages}
        try:
            resp = requests.post(url=self.api_url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                try:
                    res_data = resp.json()
                    ai_reply = res_data["choices"][0]["message"]["content"]
                    self.history.append({"role": "user", "content": user_input})
                    self.history.append({"role": "assistant", "content": ai_reply})
                    self.history = self.history[-self.max_history_len:]
                    return True, ai_reply
                except (json.JSONDecodeError, KeyError, IndexError):
                    return False, "响应解析失败"
            else:
                return False, f"接口错误，状态码{resp.status_code}"
        except (requests.ConnectionError, requests.Timeout):
            return False, "网络请求异常"
    def get_history(self):
        return self.history
    def reset_prompt(self, new_prompt):
        self.system_prompt = new_prompt
def main():
    agent, msg = SimpleAgent.load_config_from_json("config.json")
    if agent is None:
        print("创建 Agent 失败:", msg)
        return
    save_path = "history.json"
    while True:
        user_input = input("请输入您的问题：")
        clean_input = user_input.strip()
        if not clean_input:
            continue
        if clean_input.lower() == "exit":
            break
        ok, reply = agent.chat(user_input)
        if ok:
            print("AI 回复:", reply)

            success, save_msg = agent.dump_history_to_json(save_path)
            if success:
                print(f"历史记录已保存到 {save_path}")
            else:
                print(f"保存历史记录失败: {save_msg}")
        else:
            print("调用出错:", reply)
if __name__ == "__main__":
    main()