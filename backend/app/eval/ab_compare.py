"""A/B 对比评估：RAG grounding 开启 vs 关闭。

同一批病例跑两遍完整诊疗工作流（use_rag=True / False），对比：
- 证据引用数（RAG 组应显著非零）
- 诊断确定性
- 用药建议与知识库标准方案的一致率（程序化关键词匹配）

说明：需要已配置可用的 LLM 网关；病例数 x 2 次完整工作流，耗时与费用
取决于模型速度。运行：cd backend && python -m app.eval.ab_compare
"""

import json
from pathlib import Path

from loguru import logger

from app.core.database import SessionLocal, create_tables
from app.dao.medical import InquiryRecordDAO, ResultDAO, safe_json_loads
from app.medical.workflow import medical_workflow

# A/B 病例集（与样例知识库对应，便于核对标准答案）
CASES = [
    {
        "patient_name": "AB-高血压病例",
        "patient_age": 58,
        "patient_gender": "男",
        "allergy_history": [],
        "symptom_description": "男性58岁，反复头晕头痛两个月，诊室血压162/98mmHg，无胸痛，无双下肢水肿",
        "use_rag": True,
        # 知识库标准方案关键词（用于一致率统计）
        "expected_drugs": ["氨氯地平", "缬沙坦", "美托洛尔"],
        "expected_lab": [],
    },
    {
        "patient_name": "AB-肺炎病例",
        "patient_age": 34,
        "patient_gender": "女",
        "allergy_history": ["青霉素"],
        "symptom_description": "女性34岁，咳嗽咳黄痰伴发热3天，体温38.6度，左下肺湿啰音，血白细胞升高",
        "use_rag": True,
        "expected_drugs": ["阿奇霉素"],
        "expected_lab": [],
    },
    {
        "patient_name": "AB-糖尿病病例",
        "patient_age": 47,
        "patient_gender": "女",
        "allergy_history": [],
        "symptom_description": "女性47岁，多饮多尿伴体重下降三个月，空腹血糖8.2mmol/L，无明显腹痛",
        "use_rag": True,
        "expected_drugs": ["二甲双胍"],
        "expected_lab": [],
    },
]


def run_case(case: dict, doctor_id: int) -> dict:
    """跑一个病例完整工作流（必要时自动回填化验结果），返回分析产物。"""
    inquiry_id = medical_workflow.start_workflow(
        doctor_id=doctor_id,
        patient_name=case["patient_name"],
        patient_age=case["patient_age"],
        patient_gender=case["patient_gender"],
        allergy_history=case["allergy_history"],
        symptom_description=case["symptom_description"],
        use_rag=case["use_rag"],
    )
    status = medical_workflow.get_status(inquiry_id)

    if status.get("waiting_lab"):
        lab_tests = status.get("lab_tests_needed", [])
        lab_results = [
            {"test_name": t, "result_value": "见备注", "unit": "", "reference_range": "-",
             "is_abnormal": False}
            for t in lab_tests
        ]
        # 糖尿病病例补充关键阳性结果
        if "糖尿病" in case["patient_name"]:
            lab_results.append(
                {"test_name": "空腹血糖", "result_value": "8.2", "unit": "mmol/L",
                 "reference_range": "3.9-6.1", "is_abnormal": True}
            )
            lab_results.append(
                {"test_name": "糖化血红蛋白", "result_value": "8.0", "unit": "%",
                 "reference_range": "<6.5", "is_abnormal": True}
            )
        medical_workflow.handle_lab_results(inquiry_id, lab_results)
        status = medical_workflow.get_status(inquiry_id)

    return {"inquiry_id": inquiry_id, "status": status}


def summarize(status: dict, expected_drugs: list[str], expected_diagnosis_terms: list[str] | None = None) -> dict:
    treatment = status.get("treatment") or {}
    medication_text = json.dumps(treatment.get("medication_advice", []), ensure_ascii=False)
    drug_matches = sum(1 for d in expected_drugs if d in medication_text)

    # 诊断一致率：期望诊断关键词出现在主诊断或鉴别诊断中即算命中（CMB 真实病例口径）
    diagnosis = status.get("diagnosis") or {}
    diagnosis_text = json.dumps(
        {"main": diagnosis.get("main_diagnosis", ""),
         "diff": diagnosis.get("differential_diagnoses", [])},
        ensure_ascii=False,
    )
    term_matches = 0
    if expected_diagnosis_terms:
        term_matches = sum(1 for t in expected_diagnosis_terms if t in diagnosis_text)

    all_citations = (
        (status.get("symptom_citations") or [])
        + (status.get("diagnosis_citations") or [])
        + (status.get("treatment_citations") or [])
    )
    return {
        "completed": status.get("main_status") == 5,
        "main_diagnosis": diagnosis.get("main_diagnosis", ""),
        "certainty": diagnosis.get("certainty"),
        "citations_total": len(all_citations),
        "drug_match": drug_matches,
        "drug_total": len(expected_drugs),
        "diagnosis_match": term_matches,
        "diagnosis_total": len(expected_diagnosis_terms) if expected_diagnosis_terms else 0,
    }


def load_cases() -> list[dict]:
    """自制病例（用药一致率口径）+ CMB 真实病例（诊断一致率口径，如已生成）。"""
    cases = [dict(c, expected_diagnosis_terms=None) for c in CASES]
    cmb_path = Path(__file__).parent / "cmb_cases.json"
    if cmb_path.exists():
        cmb_cases = json.loads(cmb_path.read_text(encoding="utf-8"))
        for c in cmb_cases:
            cases.append(dict(c, expected_drugs=[]))
        logger.info(f"已加载 {len(cmb_cases)} 例 CMB 真实病例（诊断一致率口径）")
    return cases


def main(doctor_id: int = 2, repeat_cases: bool = True):
    create_tables()
    SessionLocal()

    all_cases = load_cases()
    groups = {"rag_on": [], "rag_off": []}
    for group_name, use_rag in (("rag_on", True), ("rag_off", False)):
        for case in all_cases:
            run_case_data = dict(case)
            run_case_data["use_rag"] = use_rag
            run_case_data["patient_name"] = case["patient_name"] + ("(RAG)" if use_rag else "(无RAG)")
            try:
                outcome = run_case(run_case_data, doctor_id)
                summary = summarize(
                    outcome["status"],
                    case.get("expected_drugs", []),
                    case.get("expected_diagnosis_terms"),
                )
                summary["case"] = case["patient_name"]
                groups[group_name].append(summary)
                logger.info(
                    f"[{group_name}] {case['patient_name']}: 完成={summary['completed']}, "
                    f"引用={summary['citations_total']}, 用药一致={summary['drug_match']}/{summary['drug_total']}, "
                    f"诊断一致={summary['diagnosis_match']}/{summary['diagnosis_total']}"
                )
            except Exception as exc:
                logger.error(f"[{group_name}] {case['patient_name']} 失败: {exc}")

    # 汇总
    report = {}
    for group_name, results in groups.items():
        if not results:
            continue
        completed = sum(1 for r in results if r["completed"])
        total_citations = sum(r["citations_total"] for r in results)
        drug_match = sum(r["drug_match"] for r in results)
        drug_total = sum(r["drug_total"] for r in results)
        diag_match = sum(r["diagnosis_match"] for r in results)
        diag_total = sum(r["diagnosis_total"] for r in results)
        certainties = [r["certainty"] for r in results if isinstance(r.get("certainty"), (int, float))]
        report[group_name] = {
            "cases": len(results),
            "completed": completed,
            "citations_total": total_citations,
            "avg_citations_per_case": total_citations / len(results),
            "drug_alignment": (drug_match / drug_total) if drug_total else None,
            "diagnosis_alignment": (diag_match / diag_total) if diag_total else None,
            "avg_certainty": sum(certainties) / len(certainties) if certainties else None,
            "details": results,
        }

    print("\n" + "=" * 64)
    print("A/B 对比：RAG grounding 开启 vs 关闭（自制+CMB真实病例）")
    print("=" * 64)
    for group_name, stats in report.items():
        print(f"\n[{group_name}]")
        for key in ("cases", "completed", "citations_total", "avg_citations_per_case",
                    "drug_alignment", "diagnosis_alignment", "avg_certainty"):
            value = stats.get(key)
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    return report


if __name__ == "__main__":
    main()
