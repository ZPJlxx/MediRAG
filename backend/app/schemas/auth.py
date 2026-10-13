"""认证与用户管理相关的请求/响应模型（替代 Enterprise-RAG 的明文登录表单）。"""

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)


class UserInfo(BaseModel):
    id: int
    username: str
    nickname: str
    role: str

    model_config = {"from_attributes": True}


class LoginResponse(BaseModel):
    token: str
    user: UserInfo


class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    nickname: str = Field(default="", max_length=50)
    role: str = Field(default="doctor", pattern="^(admin|doctor)$")


class UserUpdate(BaseModel):
    nickname: str = Field(default="", max_length=50)
    role: str = Field(default="doctor", pattern="^(admin|doctor)$")
    is_active: bool = True
    password: str = Field(default="", max_length=100, description="留空则不修改密码")


class StartWorkflowRequest(BaseModel):
    patient_name: str = Field(..., min_length=1, max_length=50)
    patient_age: int | None = Field(default=None, ge=0, le=150)
    patient_gender: str = Field(default="", max_length=10)
    allergy_history: list[str] = Field(default_factory=list, description="过敏史")
    symptom_description: str = Field(..., min_length=5, max_length=2000)
    use_rag: bool = Field(default=True, description="是否启用 RAG grounding（A/B 对比用）")


class LabResultRequest(BaseModel):
    lab_results: list[dict] = Field(..., min_length=1, description="化验结果列表")
