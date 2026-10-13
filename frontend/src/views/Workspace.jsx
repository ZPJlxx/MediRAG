import { useEffect, useRef, useState } from 'react'
import {
  apiStartWorkflow,
  apiWorkflowStatus,
  apiSubmitLabResults,
  apiListHistory,
} from '../api.js'
import Citations from '../components/Citations.jsx'

const STEPS = [
  { key: 'created', label: '创建问诊' },
  { key: 'symptom', label: '症状分析' },
  { key: 'lab', label: '化验检查' },
  { key: 'diagnosis', label: '诊断分析' },
  { key: 'treatment', label: '治疗建议' },
  { key: 'done', label: '完成' },
]

function stepIndex(status) {
  // main_status: 0创建 1症状 2待化验 3诊断 4治疗 5完成 99失败
  if (status === 99) return 5
  return status
}

export default function Workspace() {
  const [history, setHistory] = useState([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [currentId, setCurrentId] = useState(null)
  const [status, setStatus] = useState(null)
  // 轮询定时器用 ref 管理：state 会在闭包里捕获旧值，导致终态/切换记录后轮询不清
  const pollTimerRef = useRef(null)

  // 表单
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [patientGender, setPatientGender] = useState('男')
  const [allergies, setAllergies] = useState('')
  const [symptoms, setSymptoms] = useState('')
  const [useRag, setUseRag] = useState(true)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')

  // 化验回填
  const [labRows, setLabRows] = useState([{ test_name: '', result_value: '', unit: '', reference_range: '', is_abnormal: false }])
  const [submittingLab, setSubmittingLab] = useState(false)

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }

  const refreshHistory = () => {
    setHistoryLoading(true)
    apiListHistory()
      .then((rows) => { setHistory(rows); setHistoryError('') })
      .catch((e) => setHistoryError(e.message))
      .finally(() => setHistoryLoading(false))
  }

  useEffect(() => {
    refreshHistory()
    return stopPolling
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const watchStatus = (id) => {
    stopPolling()
    setCurrentId(id)
    const load = () => apiWorkflowStatus(id).then((s) => {
      setStatus(s)
      if ([5, 99].includes(s.main_status)) {
        stopPolling()
        refreshHistory()
      }
    }).catch(() => {})
    load()
    pollTimerRef.current = setInterval(load, 3000)
  }

  const start = async (e) => {
    e.preventDefault()
    setError('')
    setStarting(true)
    setStatus(null)
    try {
      const resp = await apiStartWorkflow({
        patient_name: patientName,
        patient_age: patientAge ? Number(patientAge) : null,
        patient_gender: patientGender,
        allergy_history: allergies.split(/[,，、\s]+/).filter(Boolean),
        symptom_description: symptoms,
        use_rag: useRag,
      })
      watchStatus(resp.inquiry_record_id)
      refreshHistory()
    } catch (err) {
      setError(err.message)
    } finally {
      setStarting(false)
    }
  }

  const submitLab = async () => {
    setSubmittingLab(true)
    setError('')
    try {
      const rows = labRows.filter((r) => r.test_name.trim())
      if (!rows.length) {
        setError('请至少填写一行化验结果（检查项目不能为空）')
        return
      }
      await apiSubmitLabResults(currentId, rows)
      watchStatus(currentId)
      setLabRows([{ test_name: '', result_value: '', unit: '', reference_range: '', is_abnormal: false }])
    } catch (err) {
      // 兜底：非 Error 抛出物（如对象）字符串化，避免界面显示 [object Object]
      setError(typeof err?.message === 'string' ? err.message : JSON.stringify(err))
    } finally {
      setSubmittingLab(false)
    }
  }

  const idx = status ? stepIndex(status.main_status) : -1
  const waitingLab = status?.main_status === 2

  return (
    <div className="workspace">
      {/* 新建问诊（桌面在左上，手机在最前） */}
      <div className="card workspace-form">
        <h3>🩺 新建 AI 辅助诊疗</h3>
        <form onSubmit={start}>
          <div className="row">
            <label className="field">患者姓名
              <span className="control"><input value={patientName} onChange={(e) => setPatientName(e.target.value)} required /></span>
            </label>
            <label className="field" style={{ maxWidth: 90 }}>年龄
              <span className="control"><input type="number" min="0" max="150" value={patientAge} onChange={(e) => setPatientAge(e.target.value)} /></span>
            </label>
          </div>
          <div className="row">
            <label className="field" style={{ maxWidth: 110 }}>性别
              <span className="control">
                <select value={patientGender} onChange={(e) => setPatientGender(e.target.value)}>
                  <option>男</option><option>女</option>
                </select>
              </span>
            </label>
            <label className="field">过敏史 <span className="hint">逗号/顿号分隔，如：青霉素, 磺胺类</span>
              <span className="control"><input value={allergies} onChange={(e) => setAllergies(e.target.value)} placeholder="无则留空" /></span>
            </label>
          </div>
          <label className="field">症状描述
            <span className="control">
              <textarea rows="4" value={symptoms} onChange={(e) => setSymptoms(e.target.value)}
                placeholder="例：患者男，52岁，近一个月头晕头痛，午后明显，测得诊室血压156/96mmHg，偶有心悸…" required />
            </span>
          </label>
          <label className="field" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" style={{ width: 'auto' }} checked={useRag} onChange={(e) => setUseRag(e.target.checked)} />
            启用知识库证据增强（RAG grounding）
          </label>
          {error && <div className="error-text">{error}</div>}
          <button className="btn" style={{ width: '100%' }} disabled={starting || !patientName || !symptoms}>
            {starting ? '工作流执行中…（首个问题会构建索引，稍慢）' : '▶ 启动诊疗工作流'}
          </button>
        </form>
      </div>

      {/* 历史列表（桌面在左下、内部滚动；手机在最后） */}
      <div className="card workspace-history">
        <h3>
          📋 我的问诊记录
          {history.length > 0 && <span className="badge">{history.length} 条</span>}
        </h3>
        {historyError ? (
          <div className="state-text error">
            历史记录加载失败：{historyError}
            <button type="button" className="btn small secondary retry-link" onClick={refreshHistory}>重试</button>
          </div>
        ) : historyLoading ? (
          <div className="state-text">历史记录加载中…</div>
        ) : (
          <div className="table-wrap history-scroll">
            <table className="data">
              <thead><tr><th>患者</th><th>状态</th><th>RAG</th><th>时间</th></tr></thead>
              <tbody>
                {history.map((r) => (
                  <tr key={r.id} style={{ cursor: 'pointer' }} className={r.id === currentId ? 'active-row' : ''}
                      onClick={() => watchStatus(r.id)}>
                    <td>{r.patient_name}</td>
                    <td><span className={`status-pill s${r.main_status}`}>{r.status_label}</span></td>
                    <td>{r.use_rag ? '✓' : '—'}</td>
                    <td className="muted">{r.created_at}</td>
                  </tr>
                ))}
                {!history.length && <tr><td colSpan="4" className="muted">暂无记录</td></tr>}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 工作流详情 */}
      <div className="workspace-detail">
        {!status && (
          <div className="card state-text" style={{ padding: '60px 0' }}>
            选择左侧历史记录，或新建一次诊疗开始体验
          </div>
        )}

        {status && (
          <>
            <div className="card">
              <h3>
                问诊 #{status.inquiry_record_id} · {status.patient.name}
                <span className={`status-pill s${status.main_status}`}>{status.status_label}</span>
                {status.use_rag && <span className="badge">RAG 增强</span>}
              </h3>

              <div className="stepper">
                {STEPS.map((s, i) => (
                  <div key={s.key} className={`step ${i < idx ? 'done' : ''} ${i === idx ? (waitingLab ? 'waiting active' : 'active') : ''}`}>
                    <span className="dot">{i < idx ? '✓' : i + 1}</span>
                    <div>{waitingLab && i === 2 ? '等待化验中…' : s.label}</div>
                  </div>
                ))}
              </div>

              <div className="muted">主诉：{status.patient.symptom_description}</div>
              {status.patient.allergy_history?.length > 0 && (
                <div className="mt8" style={{ color: 'var(--danger)' }}>
                  ⚠️ 过敏史：{status.patient.allergy_history.join('、')}
                </div>
              )}
            </div>

            {waitingLab && (
              <div className="card">
                <h3>🧪 化验结果回填 <span className="badge">回填后工作流自动恢复</span></h3>
                <div className="muted mb12">建议检查：{status.lab_tests_needed?.join('、') || '—'}</div>
                <div className="lab-grid-head">
                  <span>检查项目</span><span>结果值</span><span>单位</span><span>参考范围</span><span>异常</span><span></span>
                </div>
                {labRows.map((row, i) => (
                  <div className="lab-form-row" key={i}>
                    <input value={row.test_name} placeholder="如：血肌酐"
                      onChange={(e) => setLabRows(labRows.map((r, j) => j === i ? { ...r, test_name: e.target.value } : r))} />
                    <input value={row.result_value} placeholder="如：98"
                      onChange={(e) => setLabRows(labRows.map((r, j) => j === i ? { ...r, result_value: e.target.value } : r))} />
                    <input value={row.unit} placeholder="μmol/L"
                      onChange={(e) => setLabRows(labRows.map((r, j) => j === i ? { ...r, unit: e.target.value } : r))} />
                    <input value={row.reference_range} placeholder="57-97"
                      onChange={(e) => setLabRows(labRows.map((r, j) => j === i ? { ...r, reference_range: e.target.value } : r))} />
                    <input type="checkbox" style={{ width: 'auto' }} aria-label="标记为异常" checked={row.is_abnormal}
                      onChange={(e) => setLabRows(labRows.map((r, j) => j === i ? { ...r, is_abnormal: e.target.checked } : r))} />
                    <button type="button" className="btn small danger" aria-label="删除此行" onClick={() => setLabRows(labRows.filter((_, j) => j !== i))}>×</button>
                  </div>
                ))}
                <div className="mt8">
                  <button type="button" className="btn secondary small"
                    onClick={() => setLabRows([...labRows, { test_name: '', result_value: '', unit: '', reference_range: '', is_abnormal: false }])}>
                    + 添加项目
                  </button>{' '}
                  <button className="btn small" disabled={submittingLab} onClick={submitLab}>
                    {submittingLab ? '提交中…' : '提交化验结果，恢复工作流'}
                  </button>
                </div>
              </div>
            )}

            {status.symptom_analysis && (
              <div className="card">
                <h3>① 症状分析结果</h3>
                <Citations citations={status.symptom_citations} />
                <div className="result-block">
                  <h4>关键症状 / 可能原因</h4>
                  <div className="chip-list">
                    {status.symptom_analysis.key_symptoms?.map((s) => <span key={s} className="chip">{s}</span>)}
                  </div>
                </div>
                {status.symptom_analysis.possible_diseases?.length > 0 && (
                  <div className="result-block">
                    <h4>可能疾病（按概率）</h4>
                    {status.symptom_analysis.possible_diseases.map((d, i) => (
                      <div key={i} className="disease-row">
                        <span style={{ width: 150 }}>{d.disease}</span>
                        <div className="disease-bar-bg"><div className="disease-bar" style={{ width: `${Math.min(d.probability, 100)}%` }} /></div>
                        <span className="prob">{d.probability}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {status.hypothesis_verifications?.length > 0 && (
              <div className="card">
                <h3>②' 鉴别假设并行核验 <span className="badge">Send fan-out · {status.hypothesis_verifications.length} 个独立 Agent</span></h3>
                <div className="hint mb12">每个假设由独立 Agent 对照患者症状、化验结果与知识库分别核验，按支持度排序后汇总给诊断 Agent。</div>
                {status.hypothesis_verifications.map((v, i) => (
                  <div className="result-block" key={i}>
                    <div className="disease-row">
                      <span style={{ width: 170, fontWeight: 600 }}>{i + 1}. {v.disease}</span>
                      <div className="disease-bar-bg"><div className="disease-bar" style={{ width: `${Math.min(v.support_score, 100)}%` }} /></div>
                      <span className="prob">{v.support_score}分</span>
                    </div>
                    <div className="muted mt8">{v.rationale}</div>
                    {v.matched_criteria?.length > 0 && (
                      <div className="mt8" style={{ fontSize: 12 }}>✔ 符合依据：{v.matched_criteria.join('；')}</div>
                    )}
                    {v.missing_information?.length > 0 && (
                      <div className="mt8" style={{ fontSize: 12, color: 'var(--warning)' }}>缺口信息：{v.missing_information.join('；')}</div>
                    )}
                    <Citations citations={v.citations || []} />
                  </div>
                ))}
              </div>
            )}

            {status.diagnosis && (
              <div className="card">
                <h3>② 诊断分析结果 {status.diagnosis.certainty != null && <span className="badge">确定性 {status.diagnosis.certainty}%</span>}</h3>
                <Citations citations={status.diagnosis_citations} />
                <div className="result-block">
                  <h4>主诊断：{status.diagnosis.main_diagnosis}</h4>
                  {status.diagnosis.diagnosis_basis?.length > 0 && (
                    <ul style={{ paddingLeft: 18, lineHeight: 1.9 }}>
                      {status.diagnosis.diagnosis_basis.map((b, i) => <li key={i}>{b}</li>)}
                    </ul>
                  )}
                </div>
                {status.diagnosis.differential_diagnoses?.length > 0 && (
                  <div className="result-block">
                    <h4>鉴别诊断</h4>
                    <div className="chip-list">
                      {status.diagnosis.differential_diagnoses.map((d, i) => <span key={i} className="chip">{d}</span>)}
                    </div>
                  </div>
                )}
              </div>
            )}

            {status.treatment && (
              <div className="card">
                <h3>③ 治疗建议</h3>
                <Citations citations={status.treatment_citations} />
                {status.treatment.medication_advice?.length > 0 && (
                  <div className="result-block">
                    <h4>用药建议</h4>
                    <div className="table-wrap">
                      <table className="data">
                        <thead><tr><th>药物</th><th>用法用量</th><th>疗程</th><th>注意事项</th></tr></thead>
                        <tbody>
                          {status.treatment.medication_advice.map((m, i) => (
                            <tr key={i}>
                              <td style={{ fontWeight: 600 }}>{m.drug_name}</td>
                              <td>{m.dosage}</td>
                              <td>{m.duration}</td>
                              <td className="muted">{m.precautions}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                <div className="row">
                  {status.treatment.lifestyle_advice?.length > 0 && (
                    <div className="result-block">
                      <h4>生活方式建议</h4>
                      <ul style={{ paddingLeft: 18, lineHeight: 1.9 }}>{status.treatment.lifestyle_advice.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    </div>
                  )}
                  {status.treatment.follow_up_advice?.length > 0 && (
                    <div className="result-block">
                      <h4>随访建议</h4>
                      <ul style={{ paddingLeft: 18, lineHeight: 1.9 }}>{status.treatment.follow_up_advice.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    </div>
                  )}
                </div>
                {status.treatment.warning_signals?.length > 0 && (
                  <div className="result-block" style={{ borderLeft: '3px solid var(--danger)' }}>
                    <h4 style={{ color: 'var(--danger)' }}>⚠️ 警示信号</h4>
                    <ul style={{ paddingLeft: 18, lineHeight: 1.9 }}>{status.treatment.warning_signals.map((s, i) => <li key={i}>{s}</li>)}</ul>
                  </div>
                )}
              </div>
            )}

            <div className="card">
              <h3>🕘 流程时间线</h3>
              <div className="timeline">
                {status.events?.map((e, i) => (
                  <div key={i} className={`tl-item ${e.status}`}>
                    <span className="tl-time">{e.time}</span>
                    <b>{e.stage}</b> · {e.status}{e.detail ? ` — ${e.detail}` : ''}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
