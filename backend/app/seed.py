"""初始化脚本：建表 + 种子用户 + 导入样例医学知识库。

用法（backend 目录下）：
    python -m app.seed
种子账号：admin/admin123（管理员）、doctor/doctor123（医生）
"""

import hashlib
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.core.database import SessionLocal, create_tables
from app.core.security import hash_password
from app.models.entities import KnowledgeFile, SysUser
from app.rag.knowledge_manager import knowledge_manager

SAMPLE_DOCS = {
    "高血压诊疗要点.md": """# 高血压诊疗要点（演示用样例知识库）

## 一、诊断标准
依据《中国高血压防治指南》：
- 诊室血压：收缩压≥140mmHg 和/或 舒张压≥90mmHg，非同日三次测量达标即可诊断
- 家庭自测血压：≥135/85mmHg
- 24小时动态血压平均值：≥130/80mmHg

## 二、分级标准
- 1级（轻度）：收缩压140-159 或 舒张压90-99
- 2级（中度）：收缩压160-179 或 舒张压100-109
- 3级（重度）：收缩压≥180 或 舒张压≥110

## 三、危险分层
高血压合并以下任一项属高危：糖尿病、临床心血管病、慢性肾病（CKD≥3期）、
靶器官损害（左心室肥厚、颈动脉斑块、微量白蛋白尿）。

## 四、常用降压药物
1. 苯磺酸氨氯地平片（二氢吡啶类CCB）
   - 用法用量：口服，起始5mg每日一次，最大10mg每日一次
   - 注意事项：常见踝部水肿、面部潮红；肝功能不全者减量
2. 缬沙坦胶囊（ARB）
   - 用法用量：口服，80mg每日一次，可增至160mg每日一次
   - 禁忌：妊娠期禁用；双侧肾动脉狭窄禁用；高钾血症慎用
3. 美托洛尔缓释片（β受体阻滞剂）
   - 用法用量：口服，47.5mg每日一次，依心率调整
   - 禁忌：严重心动过缓（心率<60）、二度及以上房室传导阻滞、支气管哮喘急性期

## 五、一般治疗
- 生活方式干预：限盐（每日食盐<5g）、减重（BMI<24）、戒烟限酒、每周5次30分钟有氧运动
- 随访：血压达标前每2-4周随访一次；达标后每3个月随访一次
- 警示信号：剧烈头痛、喷射性呕吐、视物模糊、胸痛、肢体活动障碍——需警惕高血压急症，立即就医

## 六、青霉素类药物过敏提示
对青霉素过敏者禁用阿莫西林等青霉素类抗菌药；头孢类与青霉素存在约10%交叉过敏风险，
用药前需详细询问过敏史并进行皮试（说明书要求时）。
""",
    "社区获得性肺炎诊疗摘要.md": """# 社区获得性肺炎（CAP）诊疗摘要（演示用样例知识库）

## 一、临床诊断线索
- 新出现的咳嗽、咳痰（可为脓性痰）、发热（多为中高热）
- 肺部实变体征或湿啰音
- 血常规：白细胞>10×10^9/L 或 <4×10^9/L，中性粒细胞比例升高
- 胸部影像学：新出现的斑片状浸润影、磨玻璃影是确诊依据

## 二、病情评估（CURB-65）
意识障碍、尿素氮>7mmol/L、呼吸频率≥30次/分、收缩压<90mmHg或舒张压≤60mmHg、年龄≥65岁。
每项1分：0-1分门诊治疗；2分住院；≥3分考虑重症监护。

## 三、常见病原体与经验性用药
- 肺炎链球菌（最常见）：首选青霉素类（阿莫西林）或头孢类
- 非典型病原体（支原体/衣原体）：首选大环内酯类（阿奇霉素）或多西环素
- 经验性门诊方案：阿莫西林克拉维酸 或 阿奇霉素
- 住院方案：β-内酰胺类联合大环内酯类，或单用呼吸喹诺酮（左氧氟沙星、莫西沙星）

## 四、疗程与随访
- 一般疗程：5-7天，体温正常48-72小时且症状缓解可停药
- 随访：治疗后4-6周复查胸片，确认病灶吸收

## 五、警示信号
呼吸困难加重、意识改变、血压下降、氧饱和度<90%——提示重症肺炎，需立即住院评估。

## 六、阿奇霉素用药要点
- 用法：口服，首日500mg顿服，第2-5日250mg每日一次
- 注意：QT间期延长者慎用；与含铝/镁抗酸药同服需间隔2小时；肝功能不全慎用
""",
    "2型糖尿病诊疗速查.md": """# 2型糖尿病诊疗速查（演示用样例知识库）

## 一、诊断标准（静脉血浆血糖）
- 典型糖尿病症状（多饮、多食、多尿、体重下降）+ 随机血糖≥11.1mmol/L
- 或 空腹血糖≥7.0mmol/L
- 或 OGTT 2小时血糖≥11.1mmol/L
- 或 HbA1c≥6.5%
无典型症状者需改日复测确认。

## 二、综合控制目标
- 空腹血糖：4.4-7.0mmol/L；非空腹血糖<10mmol/L
- HbA1c<7.0%（年轻、病程短者可<6.5%）
- 血压<130/80mmHg；LDL-C<2.6mmol/L（合并心血管病者<1.8）

## 三、常用降糖药物
1. 二甲双胍（一线首选）
   - 用法：口服，起始500mg每日2次，最大剂量2550mg/日
   - 注意事项：餐中或餐后服用减少胃肠反应；肾功能不全（eGFR<30）禁用；造影检查前后暂停
2. 格列美脲（磺脲类）
   - 用法：口服，起始1mg每日一次早餐前，最大6mg/日
   - 注意：主要不良反应为低血糖，老年人慎用
3. 达格列净（SGLT2抑制剂）
   - 用法：口服，10mg每日一次
   - 注意：泌尿生殖感染风险增加；脱水者慎用；心衰患者有额外获益

## 四、随访计划
- 每3个月复查HbA1c；每半年查眼底与尿微量白蛋白；每年评估肝肾功能与足部情况

## 五、警示信号
血糖>16.7mmol/L伴恶心呕吐、呼吸深快、意识模糊——警惕糖尿病酮症酸中毒，立即就医；
心慌、出汗、手抖、饥饿感——低血糖表现，立即进食15g快速碳水。
""",
}


def seed_users(session):
    if session.query(SysUser).count() > 0:
        logger.info("用户已存在，跳过种子用户")
        return
    session.add_all([
        SysUser(username="admin", password_hash=hash_password("admin123"),
                nickname="系统管理员", role="admin"),
        SysUser(username="doctor", password_hash=hash_password("doctor123"),
                nickname="张医生", role="doctor"),
    ])
    session.commit()
    logger.info("种子用户已创建：admin/admin123, doctor/doctor123")


def seed_knowledge(session):
    """写入样例医学知识文档（存在则跳过）。"""
    settings.ensure_runtime_directories()
    created = 0
    for filename, content in SAMPLE_DOCS.items():
        path = settings.knowledge_dir / filename
        if path.exists():
            continue
        path.write_text(content, encoding="utf-8")
        db_record = KnowledgeFile(
            filename=filename,
            collection_name=hashlib.md5(filename.encode("utf-8")).hexdigest(),
            size_bytes=len(content.encode("utf-8")),
            uploader_id=0,
        )
        session.add(db_record)
        created += 1
    session.commit()
    if created:
        logger.info(f"已导入 {created} 份样例医学知识文档")
        knowledge_manager._retrievers.clear() if hasattr(knowledge_manager, "_retrievers") else None
    else:
        logger.info("样例知识文档已存在，跳过")


def main():
    logger.info("开始初始化 MediRAG 数据库...")
    settings.ensure_runtime_directories()
    create_tables()
    session = SessionLocal()
    try:
        seed_users(session)
        seed_knowledge(session)
    finally:
        session.close()
    logger.info("初始化完成")


if __name__ == "__main__":
    main()
