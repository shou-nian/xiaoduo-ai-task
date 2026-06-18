#!/usr/bin/env python3
"""
客服 FAQ 自动分类优化版本。

功能：
1. 使用 .env 中的 OPENAI_API_KEY、MODEL_NAME、BASE_URL 初始化 ChatOpenAI。
2. 使用 task1_prompt_optimized.md 中的优化 Prompt 进行分类。
3. 对模型 JSON 输出进行解析、标签校验和兜底处理。
4. 提供批量分类和 Before/After 准确率评估入口。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = BASE_DIR / "task1_prompt_optimized.md"

VALID_LABELS = (
    "退款退货",
    "物流查询",
    "账号问题",
    "商品咨询",
    "投诉建议",
    "其他",
)

LEGACY_SYSTEM_PROMPT = "你是一个客服分类助手。请根据用户问题选择最合适的分类。"

LEGACY_USER_TEMPLATE = """请对以下用户问题进行分类。
分类类别：退款退货、物流查询、账号问题、商品咨询、投诉建议、其他
用户问题：{question}

请直接回复分类结果，只回复类别名称。"""

FALLBACK_LABEL = "其他"


@dataclass(frozen=True)
class ClassificationResult:
    """单条分类结果。"""

    id: str
    question: str
    label: str
    confidence: float
    raw_response: str = ""
    error: str = ""


def build_llm() -> ChatOpenAI:
    """从环境变量创建 ChatOpenAI 客户端。"""
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    base_url = os.getenv("BASE_URL")

    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", api_key),
            ("MODEL_NAME", model_name),
            ("BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"缺少必要环境变量：{', '.join(missing)}")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=30,
        max_retries=0,
    )


def load_optimized_system_prompt(prompt_file: Path = DEFAULT_PROMPT_FILE) -> str:
    """加载优化后的 Prompt，并截取 System Prompt 主体。"""
    if not prompt_file.exists():
        raise FileNotFoundError(f"找不到 Prompt 文件：{prompt_file}")

    prompt_text = prompt_file.read_text(encoding="utf-8")
    match = re.search(
        r"## System Prompt\s*(?P<system>.*?)(?:\n## User Message 模板|\Z)",
        prompt_text,
        flags=re.S,
    )
    if match:
        return match.group("system").strip()

    return prompt_text.strip()


def build_user_message(item_id: Any, question: str) -> str:
    """构造优化 Prompt 的用户输入。"""
    payload = {
        "id": str(item_id),
        "question": question,
    }
    return (
        "请对下面的用户问题进行分类。\n\n"
        "输入：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "只输出符合要求的 JSON 对象。"
    )


def invoke_with_retry(
        llm: ChatOpenAI,
        messages: list[SystemMessage | HumanMessage],
        max_attempts: int = 3,
        retry_delay: float = 1.0,
) -> str:
    """调用 LLM，并对临时异常做有限重试。"""
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            raise ValueError("模型返回内容为空")
        except Exception as exc:  # noqa: BLE001 - 需要兜底单条样本失败
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"LLM 调用失败：{last_error}") from last_error


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型响应中解析 JSON 对象。"""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return data


def normalize_label(label: Any) -> str:
    """将模型标签规范化到合法标签集合。"""
    if not isinstance(label, str):
        return FALLBACK_LABEL

    normalized = label.strip()
    if normalized in VALID_LABELS:
        return normalized

    for valid_label in VALID_LABELS:
        if valid_label in normalized:
            return valid_label

    return FALLBACK_LABEL


def parse_optimized_response(raw_response: str, item_id: Any, question: str) -> ClassificationResult:
    """解析优化 Prompt 的 JSON 响应。"""
    data = extract_json_object(raw_response)

    label = normalize_label(data.get("label"))
    confidence = data.get("confidence", 0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    confidence_value = max(0.0, min(1.0, confidence_value))

    return ClassificationResult(
        id=str(data.get("id", item_id)),
        question=question,
        label=label,
        confidence=confidence_value,
        raw_response=raw_response,
    )


def classify_question_optimized(
        question: str,
        item_id: Any = "",
        llm: ChatOpenAI | None = None,
        system_prompt: str | None = None,
) -> ClassificationResult:
    """使用优化 Prompt 对单条问题分类。"""
    if not isinstance(question, str) or not question.strip():
        return ClassificationResult(
            id=str(item_id),
            question="" if question is None else str(question),
            label=FALLBACK_LABEL,
            confidence=1.0,
            error="空问题，使用兜底分类",
        )

    llm = llm or build_llm()
    system_prompt = system_prompt or load_optimized_system_prompt()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=build_user_message(item_id, question)),
    ]

    try:
        raw_response = invoke_with_retry(llm, messages)
        return parse_optimized_response(raw_response, item_id, question)
    except Exception as exc:  # noqa: BLE001 - 批量任务中单条失败不应中断全批次
        return ClassificationResult(
            id=str(item_id),
            question=question,
            label=FALLBACK_LABEL,
            confidence=0.0,
            error=str(exc),
        )


def classify_question_before(
        question: str,
        item_id: Any = "",
        llm: ChatOpenAI | None = None,
) -> ClassificationResult:
    """使用原始简化 Prompt 对单条问题分类，用于 Before Accuracy 对比。"""
    if not isinstance(question, str) or not question.strip():
        return ClassificationResult(
            id=str(item_id),
            question="" if question is None else str(question),
            label=FALLBACK_LABEL,
            confidence=1.0,
            error="空问题，使用兜底分类",
        )

    llm = llm or build_llm()
    messages = [
        SystemMessage(content=LEGACY_SYSTEM_PROMPT),
        HumanMessage(content=LEGACY_USER_TEMPLATE.format(question=question)),
    ]

    try:
        raw_response = invoke_with_retry(llm, messages)
        return ClassificationResult(
            id=str(item_id),
            question=question,
            label=normalize_label(raw_response),
            confidence=0.0,
            raw_response=raw_response,
        )
    except Exception as exc:  # noqa: BLE001
        return ClassificationResult(
            id=str(item_id),
            question=question,
            label=FALLBACK_LABEL,
            confidence=0.0,
            error=str(exc),
        )


def load_samples(input_file: str | Path) -> list[dict[str, Any]]:
    """读取并校验测试样本。"""
    path = Path(input_file)
    with path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError("输入 JSON 顶层结构必须是列表")

    valid_samples: list[dict[str, Any]] = []
    for index, item in enumerate(samples, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条样本不是对象")
        if "id" not in item or "question" not in item:
            raise ValueError(f"第 {index} 条样本缺少 id 或 question 字段")
        valid_samples.append(item)

    return valid_samples


def batch_classify(
        input_file: str | Path,
        output_file: str | Path,
        classifier: Callable[[str, Any, ChatOpenAI | None], ClassificationResult] = classify_question_optimized,
) -> list[dict[str, Any]]:
    """批量分类并写入输出文件。"""
    samples = load_samples(input_file)
    llm = build_llm()
    system_prompt = load_optimized_system_prompt()

    results: list[dict[str, Any]] = []
    for item in samples:
        question = item["question"]
        if classifier is classify_question_optimized:
            result = classify_question_optimized(
                question=question,
                item_id=item["id"],
                llm=llm,
                system_prompt=system_prompt,
            )
        else:
            result = classifier(question, item["id"], llm)

        results.append(
            {
                "id": result.id,
                "question": result.question,
                "predicted_category": result.label,
                "confidence": result.confidence,
                "raw_response": result.raw_response,
                "error": result.error,
            }
        )

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    return results


def calculate_accuracy(samples: list[dict[str, Any]], predictions: list[ClassificationResult]) -> float:
    """计算分类准确率。"""
    if len(samples) != len(predictions):
        raise ValueError("样本数量与预测结果数量不一致")
    if not samples:
        return 0.0

    correct = 0
    for sample, prediction in zip(samples, predictions, strict=True):
        expected_label = normalize_label(sample.get("label"))
        if prediction.label == expected_label:
            correct += 1

    return correct / len(samples)


def evaluate_before_after(input_file: str | Path) -> dict[str, Any]:
    """对测试集计算 Before/After Accuracy。"""
    samples = load_samples(input_file)
    llm = build_llm()
    system_prompt = load_optimized_system_prompt()

    before_predictions = [
        classify_question_before(item["question"], item["id"], llm)
        for item in samples
    ]
    after_predictions = [
        classify_question_optimized(item["question"], item["id"], llm, system_prompt)
        for item in samples
    ]

    before_accuracy = calculate_accuracy(samples, before_predictions)
    after_accuracy = calculate_accuracy(samples, after_predictions)

    def build_prediction_records(predictions: list[ClassificationResult]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for sample, prediction in zip(samples, predictions, strict=True):
            expected_label = sample.get("label", "")
            records.append(
                {
                    "id": str(sample.get("id", prediction.id)),
                    "question": sample.get("question", prediction.question),
                    "expected_label": expected_label,
                    "predicted_label": prediction.label,
                    "is_correct": prediction.label == normalize_label(expected_label),
                    "confidence": prediction.confidence,
                    "raw_response": prediction.raw_response,
                    "error": prediction.error,
                }
            )
        return records

    return {
        "before_accuracy": before_accuracy,
        "after_accuracy": after_accuracy,
        "improvement": after_accuracy - before_accuracy,
        "total": len(samples),
        "before_predictions": build_prediction_records(before_predictions),
        "after_predictions": build_prediction_records(after_predictions),
    }


def save_evaluation_report(report: dict[str, Any], output_file: str | Path) -> None:
    """保存评估报告。"""
    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="客服 FAQ 自动分类优化版本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify_parser = subparsers.add_parser("classify", help="批量分类")
    classify_parser.add_argument("input_file", help="输入 JSON 文件")
    classify_parser.add_argument("output_file", help="输出 JSON 文件")

    evaluate_parser = subparsers.add_parser("evaluate", help="评估 Before/After Accuracy")
    evaluate_parser.add_argument("input_file", help="测试样本 JSON 文件")
    evaluate_parser.add_argument("output_file", help="评估报告 JSON 文件")

    args = parser.parse_args()

    if args.command == "classify":
        results = batch_classify(args.input_file, args.output_file)
        print(f"分类完成，共处理 {len(results)} 条问题")
        return

    if args.command == "evaluate":
        report = evaluate_before_after(args.input_file)
        save_evaluation_report(report, args.output_file)
        print(f"Before Accuracy: {report['before_accuracy']:.2%}")
        print(f"After Accuracy: {report['after_accuracy']:.2%}")
        print(f"Improvement: {report['improvement']:.2%}")


if __name__ == "__main__":
    main()
