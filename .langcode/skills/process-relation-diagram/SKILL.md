---
name: process-relation-diagram
description: 任何流程、关系、架构、调用链、状态转换、数据流、审批流或协作逻辑需要清晰展示时，使用 Mermaid 图把结构可视化。
---

# process-relation-diagram

## 适用场景

- 用户询问流程、架构、调用链、数据流、状态转换、审批流、任务拆解或依赖关系。
- 使用 `delegate_agent` 后，需要向用户说明多个 Agent 或工具之间如何协作。
- 文本描述会让先后顺序、职责边界、条件分支或依赖关系变得不清晰。
- 需要演示“谁调用谁、数据怎么流、状态怎么变、步骤怎么走”。

## 执行方法

1. 先完成必要的调研、规划、验证或协作。
2. 提炼节点：参与者、系统模块、子 Agent、工具、文件、服务、审批点、状态或输出物。
3. 提炼边：调用、读取、写入、验证、审批、反馈、汇总、转换、依赖等动作。
4. 调用 `diagram` 工具生成 Mermaid 图。
5. 在文字回答中围绕图解释关键路径和结论，不要重复罗列所有边。

## 推荐图类型

- 流程步骤：`diagram_type=flowchart`，`direction=TD`。
- 关系/架构：`diagram_type=architecture` 或 `collaboration`，`direction=LR`。
- 多 Agent 协作：`diagram_type=collaboration`，`direction=LR`。
- 调用时序：直接传入 `sequenceDiagram` Mermaid DSL。
- 状态转换：直接传入 `stateDiagram-v2` Mermaid DSL。

## 结构化图示例

```json
{
  "title": "审批执行流程",
  "diagram_type": "flowchart",
  "direction": "TD",
  "nodes": [
    {"id": "user", "label": "用户"},
    {"id": "agent", "label": "主 Agent"},
    {"id": "review", "label": "权限审查"},
    {"id": "approval", "label": "人工审批"},
    {"id": "execute", "label": "执行工具"},
    {"id": "result", "label": "返回结果"}
  ],
  "edges": [
    {"source": "user", "target": "agent", "label": "提出需求"},
    {"source": "agent", "target": "review", "label": "提交工具调用"},
    {"source": "review", "target": "approval", "label": "需要审批"},
    {"source": "approval", "target": "execute", "label": "允许"},
    {"source": "execute", "target": "result", "label": "产出结果"},
    {"source": "result", "target": "agent", "label": "继续推理"}
  ]
}
```

## 注意事项

- 图要帮助用户理解结构，不要为了装饰而画。
- 节点数量优先控制在 5-12 个，超过时拆成多张图或只保留关键路径。
- 边标签要用短动作，例如“读取文件”“校验输入”“返回结论”。
- 不要把密钥、敏感路径、大段代码或冗长日志放进图。
- 图生成失败时，先简化节点/边，再重试。
