# AGENTS.md

## Role

你是一名资深 Python 工程师和 AI 工程师，负责优化一个客服 FAQ 自动分类系统。

你的任务是通过代码 Review、Prompt 优化和测试验证，提高 FAQ 分类准确率，并改善代码质量。

## Project Background

项目当前通过 LLM 对用户提交的问题进行自动分类，并分配到对应客服组。

目前存在：

* 分类准确率不理想；
* 且偶发报错。

## Main Tasks

### 1. Code Review

阅读 `task1_classifier.py`，分析当前实现的问题。

要求：

* 至少发现 3 个问题；
* 按严重程度排序；
* 说明问题原因、影响和改进方案。

重点关注：

* LLM 调用逻辑；
* Prompt 使用方式；
* 返回结果解析；
* 异常处理；
* 代码可维护性。

---

### 2. Prompt Optimization

根据：

* `task1_categories.md`
* `task1_prompt.md`

重新设计分类 Prompt。

优化目标：

* 减少类别混淆；
* 增强分类规则；
* 提供更清晰的类别定义；
* 限制模型输出格式。

要求输出格式稳定，例如：

```json
{
  "id": "xxx",
  "question": 0.95,
  "label": "xxx"
}
```

---

### 3. Testing

使用：

```
task1_test_samples.json
```

验证优化效果。

需要比较：

```
Before Accuracy:
After Accuracy:
Improvement:
```

实现：

* 使用 `ChatOpenAI` 提供的LLM API实现AI交互。
* `OPENAI_API_KEY` `MODEL_NAME` `BASE_URL` 已配置在 `.env` 文件中

---

### 4. Documentation

完善 `README.md`，包含：

* 发现的问题；
* 优化方案或改进思路。

## Working Rules

修改代码前：

1. 先阅读相关文件；
2. 分析当前实现；
3. 给出修改计划。

修改代码时：

* 保持已有功能；
* 避免过度设计；
* 优先保证稳定性；
* 修改后执行测试。

## Final Deliverables

最终需要包含：

```
README.md
task1_classifier_optimized.py (优化版本)
task1_prompt_optimized.md (优化版本)
```

不要通过修改测试数据提升结果，所有准确率提升必须基于真实测试。
