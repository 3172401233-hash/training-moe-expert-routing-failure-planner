# training-moe-expert-routing-failure-planner
规划 MoE 专家路由失败、容量漂移和重路由

## 概述

MoE 训练中，当部分 expert 所在 worker 失联时，token 已经按旧路由发出。
本规划器根据当前状态决定每个受影响 token 的去向：**等待** worker 恢复、
**重路由**到其他 expert，或**标记失败**。

安全不变量：

- 同一 token 的同一路由 slot 不会同时由两个 expert 计算（检测接收侧与路由表两侧的重复）。
- 重路由目标 expert 的负载（已完成 + 在途 + 新规划）不超过当前 batch 容量。
- 已在当前梯度版本完成计算的 token 不被打扰；旧版本的已完成结果计入回滚集合。

## 输入 / 输出

`planRecovery(input)`（见 `src/planner.js`）接收：

- `tokens`：token 及其 top-k 路由（expert、slot、weight）
- `capacities`：每个 expert 在当前 batch 的容量上限
- `workers`：worker 状态（`alive` / `dead` + `deadSince`）及其承载的 expert
- `expertToWorker`：expert 到 worker 的映射
- `received`：各 expert 已接收的 token（`queued` / `computing` / `done` + 梯度版本）
- `currentGradientVersion`：当前梯度版本
- `retryWindow`：`{ now, maxWaitMs }`，失联后允许等待的窗口

输出：

- `reroutes`：安全重路由集合 `{ tokenId, slot, fromExpert, toExpert }`
- `waits`：决定等待的 token 及剩余等待时间
- `rollbackSteps`：需要回滚的梯度版本（旧版本结果、重复计算）
- `unrecoverable`：无法恢复的 token 及原因
- `plannedLoad`：规划后每个 expert 的预计负载

## 决策顺序

1. 重复路由检测：同一 token+slot 被两个 expert 完成 → 保留首个，其余回滚。
2. 已在当前版本完成 → 不动。
3. 旧版本完成 → 记录回滚 step，按未完成处理。
4. worker 失联但在重试窗口内 → 等待并预留容量。
5. 超过重试窗口 → 重路由：优先 token 自身 top-k 中的其他 expert，
   其次任意存活且有剩余容量的 expert。
6. 无可用 expert → 标记为不可恢复。

## 测试

```bash
node --test
```

覆盖：单 worker 失联、重试窗口内等待、容量溢出、部分计算完成（含旧版本回滚）、
接收侧与路由表的路由重复、重路由目标安全性。

## 与训练框架的关系

本模块是**纯函数规划器，不连接任何训练框架**。它不发送 token、不执行
all-to-all 通信、不触碰梯度，也不与 PyTorch / DeepSpeed / Megatron 等运行时
交互。调用方（训练框架侧的容错组件）负责采集输入快照、执行规划结果中的
等待 / 重路由 / 回滚动作。Node.js 版本要求 ≥ 24.16.0。
