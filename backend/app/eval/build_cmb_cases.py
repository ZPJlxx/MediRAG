"""CMB-Clin 真实病例转换：为 A/B 评估生成真实病例集。

来源：FreedomIntelligence/CMB 的 CMB-Clin-qa.json（Apache-2.0，74 例真实问诊病例）。
每例含 description（现病史/主诉/体格检查）+ QA_pairs（参考诊断与依据）。

转换产物：app/eval/cmb_cases.json，供 ab_compare.py 加载——
- symptom_description <- 病史摘要+主诉+体检异常项（压缩到工作流入参上限内）
- expected_diagnosis_terms <- 从 QA 参考答案的「诊断：」行提取关键词，
  用于统计主诊断/鉴别诊断的覆盖一致率（替代原 expected_drugs 的口径）

运行：cd backend && python -m app.eval.build_cmb_cases [--n 12]
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MAX_SYMPTOM_LEN = 1500  # StartWorkflowRequest.symptom_description 上限 2000，留余量


def _extract_age_gender(description: str) -> tuple[int | None, str]:
    age_match = re.search(r"(\d{1,3})\s*岁", description)
    age = int(age_match.group(1)) if age_match else None
    # 性别：取「XX岁」前的男/字或描述中的男性/女性字样
    seg = re.search(r"([男女])[，,、\s]*\d{1,3}\s*岁", description)
    if seg:
        return age, seg.group(1)
    if "男性" in description or re.search(r"病人，男", description):
        return age, "男"
    if "女性" in description or re.search(r"病人，女", description):
        return age, "女"
    return age, ""


def _extract_diagnosis_terms(qa_pairs: list[dict]) -> list[str]:
    """从参考答案里提取诊断关键词（按优先级：诊断行 > 首句）。"""
    terms: list[str] = []
    for qa in qa_pairs or []:
        answer = str(qa.get("answer") or "")
        question = str(qa.get("question") or "")
        if "诊断" not in question:
            continue
        for line in answer.splitlines():
            line = line.strip()
            match = re.match(r"诊断[：:]\s*(.+)", line)
            if not match:
                continue
            # 拆出诊断名（去掉编号/顿号分隔的多个诊断各自成词）
            for part in re.split(r"[；;，,、①②③④⑤]", match.group(1)):
                part = re.sub(r"[（(].*?[)）]", "", part).strip()
                # 去掉「合并/伴」后仍保留主病名——直接整体保留也行，这里拆分主次
                for sub in re.split(r"合并|伴有?|及", part):
                    sub = sub.strip(" 。.．")
                    if 2 <= len(sub) <= 20:
                        terms.append(sub)
            break  # 只取第一个「诊断：」行
    # 去重保序
    seen, unique = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:6]


def build(n: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id="FreedomIntelligence/CMB",
        filename="CMB-Clin/CMB-Clin-qa.json",
        repo_type="dataset",
    )
    data = json.load(open(local, encoding="utf-8"))

    cases = []
    for item in data:
        description = str(item.get("description") or "")
        qa_pairs = item.get("QA_pairs") or []
        if len(description) < 80:
            continue

        age, gender = _extract_age_gender(description)
        terms = _extract_diagnosis_terms(qa_pairs)
        if not terms:
            # 回退：标题即病名（如「案例分析-腹外疝」-> 腹外疝）
            title = str(item.get("title") or "")
            fallback = re.sub(r"案例分析[-—]", "", title).strip()
            fallback = re.sub(r"^[\u4e00-\u9fa5a-zA-Z、\s]*?[-—]", "", fallback).strip()
            if 2 <= len(fallback) <= 20:
                terms = [fallback]
        if not terms:
            continue

        symptom = re.sub(r"\s+", " ", description).strip()[:MAX_SYMPTOM_LEN]
        cases.append({
            "cmb_id": item.get("id"),
            "title": str(item.get("title") or "")[:40],
            "patient_name": f"CMB病例{item.get('id', len(cases))}",
            "patient_age": age,
            "patient_gender": gender,
            "allergy_history": [],
            "symptom_description": symptom,
            "expected_diagnosis_terms": terms,
        })
        if len(cases) >= n:
            break

    out_path = os.path.join(os.path.dirname(__file__), "cmb_cases.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"已生成 {len(cases)} 例 -> {out_path}")
    for c in cases[:3]:
        print(f"  [{c['cmb_id']}] {c['title']} | {c['patient_gender']}{c['patient_age']}岁 |"
              f" 期望诊断: {c['expected_diagnosis_terms'][:3]}")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()
    build(args.n)
