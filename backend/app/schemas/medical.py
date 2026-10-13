"""医疗领域 Pydantic v2 模型。

从 MediAgent-LLM 的 data_models/medical_data.py（Pydantic v1 写法）迁移：
- @validator -> @field_validator，.dict() -> .model_dump()
- 新增 Citation：诊断/治疗建议的证据引用（本项目的核心创新点）
- 放宽"至少5种疾病/3种鉴别诊断"的硬校验为">=1"：原项目因模型少返回一项
  就整体失败的脆弱性问题在此修复，数量要求改由提示词约束
"""

import json
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AgentType(int, Enum):
    SYMPTOM_ANALYSIS = 1
    DIAGNOSIS_ANALYSIS = 2
    TREATMENT_ADVICE = 3
    HYPOTHESIS_VERIFICATION = 4


class InquiryStatus(int, Enum):
    INITIAL = 0
    SYMPTOM_ANALYSIS = 1
    WAITING_LAB = 2
    DIAGNOSIS_ANALYSIS = 3
    TREATMENT_ADVICE = 4
    COMPLETED = 5
    FAILED = 99


class AgentStatus(int, Enum):
    ANALYZING = 1
    SUCCESS = 2
    FAILED = 3


STAGE_LABELS = {
    InquiryStatus.INITIAL: "已创建",
    InquiryStatus.SYMPTOM_ANALYSIS: "症状分析",
    InquiryStatus.WAITING_LAB: "等待化验",
    InquiryStatus.DIAGNOSIS_ANALYSIS: "诊断分析",
    InquiryStatus.TREATMENT_ADVICE: "治疗建议",
    InquiryStatus.COMPLETED: "已完成",
    InquiryStatus.FAILED: "失败",
}


class Citation(BaseModel):
    """证据引用：一条结论对应知识库中的一处原文。"""

    source: str = Field(..., description="来源文档文件名")
    location: str = Field(default="", description="位置标签，如 '第 3 页' / '行 12-18'")
    snippet: str = Field(default="", description="命中的原文片段（截取窗口）")


class SymptomAnalysisResult(BaseModel):
    key_symptoms: list[str] = Field(default_factory=list, description="关键症状")
    possible_causes: list[str] = Field(default_factory=list, description="可能原因")
    possible_diseases: list[dict] = Field(default_factory=list, description="可能疾病 [{disease, probability}]")
    further_information_needed: list[str] = Field(default_factory=list, description="待确认信息")
    lab_tests_needed: list[str] = Field(default_factory=list, description="建议化验项目")
    citations: list[Citation] = Field(default_factory=list, description="证据引用")

    @field_validator("possible_diseases")
    @classmethod
    def at_least_one_disease(cls, v):
        if not v:
            raise ValueError("possible_diseases 不能为空")
        return v

    def to_json(self) -> str:
        return self.model_dump_json()


class HypothesisVerificationResult(BaseModel):
    """单个鉴别假设的并行核验结果（Send fan-out 分支的产物）。"""

    disease: str = Field(..., description="被核验的假设疾病")
    support_score: float = Field(..., ge=0, le=100, description="症状+化验+证据对该假设的支持度")
    matched_criteria: list[str] = Field(default_factory=list, description="符合的诊断标准/依据")
    missing_information: list[str] = Field(default_factory=list, description="尚缺的支撑信息")
    rationale: str = Field(default="", description="一句话核验结论")
    citations: list[Citation] = Field(default_factory=list, description="证据引用")

    def to_json(self) -> str:
        return self.model_dump_json()


class DiagnosisAnalysisResult(BaseModel):
    main_diagnosis: str = Field(..., description="主诊断")
    differential_diagnoses: list[str] = Field(default_factory=list, description="鉴别诊断")
    diagnosis_basis: list[str] = Field(default_factory=list, description="诊断依据")
    certainty: float = Field(..., ge=0, le=100, description="诊断确定性百分比")
    citations: list[Citation] = Field(default_factory=list, description="证据引用")

    def to_json(self) -> str:
        return self.model_dump_json()


class MedicationAdvice(BaseModel):
    drug_name: str = Field(default="", description="药物名称")
    dosage: str = Field(default="", description="用法用量")
    duration: str = Field(default="", description="用药时长")
    precautions: str = Field(default="", description="注意事项")


class TreatmentAdviceResult(BaseModel):
    treatment_plan: dict = Field(default_factory=dict, description="治疗方案 {medication[], non_medication[]}")
    medication_advice: list[dict] = Field(default_factory=list, description="用药建议")
    lifestyle_advice: list[str] = Field(default_factory=list, description="生活方式建议")
    follow_up_advice: list[str] = Field(default_factory=list, description="随访建议")
    warning_signals: list[str] = Field(default_factory=list, description="警示信号")
    citations: list[Citation] = Field(default_factory=list, description="证据引用")

    def to_json(self) -> str:
        return self.model_dump_json()


class LabResultItem(BaseModel):
    """单项化验结果（医生回填）。"""

    test_name: str = Field(..., description="检查名称")
    result_value: str = Field(default="", description="结果值")
    unit: str = Field(default="", description="单位")
    reference_range: str = Field(default="", description="参考范围")
    is_abnormal: bool = Field(default=False, description="是否异常")
