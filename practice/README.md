# practice/ · LangGraph 四练

按「专题_序号_名称」命名。**这不是四个独立 demo，而是一条"从手写状态机走到框架"的阶梯。**

| 文件 | 这一步在练什么 | 上框架了吗 |
|---|---|---|
| `langgraph_01_初次练习.py` | `TypedDict` 基础：定义 `Comment` 类型 + 一个取值/格式化函数 | ❌ 纯 Python |
| `langgraph_02_第二次练习.py` | **手写状态机**：`take_step(state) -> state`，每轮**自己** `messages + [新消息]`、**自己**判 `done` | ❌ 纯 Python（手动流转） |
| `langgraph_03_第三次练习.py` | **第一次上框架**：`StateGraph` + 节点 + 条件边（`route_func` + `path_map`）；`messages` 交给 `Annotated[list, add_messages]` | ✅ LangGraph |
| `langgraph_03_Annotated.py` | 在 `03` 基础上加 **`SqliteSaver` 持久化**（`thread_id="user_1"`），并做「传 `init_state` vs 传 `{}`」对照实验 | ✅ LangGraph + checkpointer |

---

## ⭐ `03` 顶部的注释（原文照录 —— 这四个文件里最值钱的一段）

```python
#messages 是追加；
#但这不是默认行为，是因为加了 Annotated[list, add_messages] 才变成追加。
#如果去掉这个注解，LangGraph 会默认直接覆盖。
#tool_agent.py 里不用是因为我手动在每一轮更新对话后加入messages信息
```

**它说的是三件事：**

1. `messages` 会**累加**，但**这不是默认行为** —— 是 `Annotated[list, add_messages]` 把它变成了"**合并型通道**"。
2. **去掉这个注解**（写成 `messages: list`），LangGraph 就按默认的"**覆盖型**"处理：只留最后一次写入。
3. `tool_agent.py`（手写版）**不需要**这个注解 —— 因为它在 `while` 循环里**自己每轮 `append`**，累积是它自己做的。

**对照 `02` 看更清楚：**

```python
# 02（手写）：人手动追加
updated_messages = state["messages"] + [new_message]

# 03（框架）：节点只 return 增量，追加由注解声明
messages: Annotated[list, add_messages]     # ← 声明一次
return {"messages": [new_msg], ...}         # ← 只报这一笔
```

> 同一件事，从"**人写**"变成"**注解声明**"。
> **注解不是注释，是给框架看的配置。**

---

## `03_Annotated` 的对照实验（原文照录）

```
                        传 init_state                       传 {}
① 读出记忆              step=3, messages=3 条, done=True     同左
② 叠加你的输入          step=0 → 替换成 0                   不覆盖
                        messages=[] → 合并，仍 3 条           不合并
                        done=False → 替换成 False            不覆盖 → True
③ 节点跑几次            done=False → 跑 3 次                 done=True → 跑 1 次
④ 最终                  step=3 · messages=6                 step=4 · messages=10
```

**怎么读这张表**（同一个 `thread_id` 第二次 invoke 时）：

- **传 `init_state`**：`step` / `done` 是**覆盖型** → 你传 `0` / `False`，就把记忆里的 `3` / `True` **盖掉**；而 `messages` 是**合并型** → 传 `[]` 等于"再叠一笔空"，**既不合并也不覆盖**，还是 3 条 → `done` 被盖成 `False`，于是**又跑了 3 次**，最终 `messages=6`。
- **传 `{}`**：**什么都不覆盖** → 记忆里的 `done=True` 保留 → 条件边直接走 `END` → **只跑 1 次** → 最终 `step=4`、`messages=10`。

> 一句话：**`invoke` 的入参不是"初始化"，是"再叠一笔写入"** —— 覆盖型字段会被它盖掉，合并型字段会被它追加。

---

## 怎么跑

```powershell
# 在仓库根目录
python practice/langgraph_01_初次练习.py
python practice/langgraph_02_第二次练习.py
python practice/langgraph_03_第三次练习.py
python practice/langgraph_03_Annotated.py
```

⚠️ `03_Annotated.py` 里写的是**相对路径** `checkpoints.db` → **从哪个目录启动，存档库就建在哪个目录**（"相对路径的基准是运行时工作目录"的一个现场）。

---

## 和主项目的关系

```
practice/（阶梯）                                    →   主项目（终点）
02 手写状态流转（自己 append / 自己判 done）
   └→ 03 StateGraph + Annotated[list, add_messages]
        └→ 03_Annotated 加 SqliteSaver（跨进程记忆）
             └→ tool_agent_langgraph.py：
                五个节点 + 两条条件边 + 三条终止路径（正常 / 超限 / 重复调用熔断）+ SqliteSaver
```

`practice/` 里每个文件都只练**一件事**；主项目把它们合起来用。想不起来某个机制在哪练过，就回这里找。
