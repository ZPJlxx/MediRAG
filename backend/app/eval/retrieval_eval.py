"""检索质量评估：recall@k / MRR / 命中率。

复用 general-doc-qa-assistant 的评估思路，针对内置医学知识库设计标注集：
每条 = (查询, 期望命中的来源文件, 期望覆盖的关键词)。
运行：cd backend && python -m app.eval.retrieval_eval
"""

import re
import sys

# 标注集：查询 -> (可接受来源集合[扩容后多来源白名单], 期望关键词)
# 说明：知识库扩容（真实指南+Huatuo）后，同一查询命中权威指南 PDF 或真实问答
# 与命中自编演示文档同等正确，白名单放宽容忍度；关键词仍验证内容质量。
HT = "huatuo/huatuo26m_lite_corpus_s20000.md"
EVAL_SET = [
    ("高血压的诊断标准是什么", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", "guidelines/国家基层高血压防治管理指南2025版.pdf", HT}, ["140", "90"]),
    ("高血压如何分级", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", "guidelines/国家基层高血压防治管理指南2025版.pdf", HT}, ["1级", "2级", "3级"]),
    ("氨氯地平的用法用量", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", HT}, ["5mg"]),
    ("缬沙坦有哪些禁忌", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", HT}, ["妊娠", "肾动脉"]),
    ("美托洛尔不能用于哪些人", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", HT}, ["心动过缓", "哮喘"]),
    ("高血压患者需要警惕什么信号", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", HT}, ["头痛", "呕吐"]),
    ("肺炎如何判断严重程度", {"社区获得性肺炎诊疗摘要.md", HT}, ["CURB-65"]),
    ("社区获得性肺炎常用抗生素", {"社区获得性肺炎诊疗摘要.md", HT}, ["青霉素", "阿奇霉素"]),
    ("阿奇霉素怎么吃", {"社区获得性肺炎诊疗摘要.md", HT}, ["500mg"]),
    ("糖尿病的诊断标准", {"2型糖尿病诊疗速查.md", "guidelines/中国糖尿病防治指南（2024版）.pdf", "guidelines/国家基层糖尿病防治管理指南（2025）.pdf", HT}, ["11.1", "7.0"]),
    ("二甲双胍怎么用", {"2型糖尿病诊疗速查.md", "guidelines/中国糖尿病防治指南（2024版）.pdf", "guidelines/国家基层糖尿病防治管理指南（2025）.pdf", HT}, ["500mg"]),
    ("低血糖有什么表现", {"2型糖尿病诊疗速查.md", "guidelines/中国糖尿病防治指南（2024版）.pdf", "guidelines/国家基层糖尿病防治管理指南（2025）.pdf", HT}, ["心慌", "出汗"]),
    ("糖尿病多久复查一次糖化血红蛋白", {"2型糖尿病诊疗速查.md", "guidelines/中国糖尿病防治指南（2024版）.pdf", "guidelines/国家基层糖尿病防治管理指南（2025）.pdf", HT}, ["3个月"]),
    ("尿微量白蛋白对高血压意味着什么", {"高血压诊疗要点.md", "guidelines/中国高血压临床实践指南.pdf", HT}, ["靶器官"]),
]


def run_eval(top_k: int = 5) -> dict:
    from app.rag.knowledge_manager import knowledge_manager

    recalls, mrrs, keyword_hits, keyword_total = [], [], 0, 0
    failures = []

    for rank, (query, expected_files, expected_keywords) in enumerate(EVAL_SET, 1):
        try:
            citations = knowledge_manager.search_citations(query, top_k=top_k)
        except Exception as exc:
            failures.append((query, f"检索异常: {exc}"))
            recalls.append(0.0)
            mrrs.append(0.0)
            continue

        sources = [c.source for c in citations]
        # citation.source 是纯文件名（loader 用 Path.name），白名单含子目录相对路径——按 basename 对齐
        expected_names = {f.rsplit("/", 1)[-1] for f in expected_files}
        hit_flags = [1.0 if src in expected_names else 0.0 for src in sources]
        recall = min(1.0, sum(hit_flags)) if expected_files else 1.0
        recalls.append(recall)

        mrr = 0.0
        for position, flag in enumerate(hit_flags, 1):
            if flag:
                mrr = 1.0 / position
                break
        mrrs.append(mrr)

        merged_text = " ".join(c.snippet for c in citations)
        for keyword in expected_keywords:
            keyword_total += 1
            if re.search(re.escape(keyword), merged_text):
                keyword_hits += 1

        if recall == 0.0:
            failures.append((query, f"未命中 {expected_files}，实际: {sources[:3]}"))

    return {
        "queries": len(EVAL_SET),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "keyword_coverage": keyword_hits / keyword_total if keyword_total else 0.0,
        "failures": failures,
    }


def main():
    print("=" * 60)
    print("MediRAG 医学知识库检索质量评估")
    print("=" * 60)
    result = run_eval(top_k=5)
    print(f"查询数:        {result['queries']}")
    print(f"recall@5:      {result['recall_at_k']:.1%}")
    print(f"MRR:           {result['mrr']:.3f}")
    print(f"关键词覆盖率:  {result['keyword_coverage']:.1%}")
    if result["failures"]:
        print("\n未命中查询:")
        for query, reason in result["failures"]:
            print(f"  - {query}: {reason}")
    return result


if __name__ == "__main__":
    main()
