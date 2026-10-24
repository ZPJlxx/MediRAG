import { useEffect, useRef, useState } from 'react'
import { apiPatientChat, apiPatientChatHistory, apiPatientSessions } from '../api.js'
import Citations from '../components/Citations.jsx'
import robotAvatar from '../assets/robot.png'
import userAvatar from '../assets/zpj_user.png'

// 意图徽章配色（与后端 INTENT_LABELS 对应）
const INTENT_BADGES = {
  chitchat: { label: '闲聊', cls: 'intent-chitchat' },
  consult: { label: '健康咨询', cls: 'intent-consult' },
  inquiry: { label: '症状问诊', cls: 'intent-inquiry' },
  emergency: { label: '急症警示', cls: 'intent-emergency' },
}

const WELCOME = {
  role: 'assistant',
  content: '您好，我是智慧医疗诊疗助手 👋\n\n您可以问我：\n· 健康知识：如「二甲双胍是什么药」「高血压的诊断标准」\n· 身体不适：如「最近老是头晕怎么回事」，我会帮您分析并建议就诊科室\n\n⚠️ 紧急情况（胸痛、呼吸困难、大出血等）请直接拨打 120！',
  intent: '',
  citations: [],
  welcome: true,
}

const DISCLAIMER_TEXT = '本回复由 AI 助手生成，仅供健康科普参考，不能替代执业医师的诊断和治疗建议。'

// 本浏览器开过的会话台账（localStorage，最近在前，上限 10 个；跨浏览器不共享——免登录下的隐私边界）
const loadSessionLedger = () => {
  try { return JSON.parse(localStorage.getItem('medirag_patient_sessions') || '[]') } catch { return [] }
}
const saveSessionLedger = (list) =>
  localStorage.setItem('medirag_patient_sessions', JSON.stringify(list.slice(0, 10)))
const rememberSession = (sid) => {
  if (!sid) return
  const list = [sid, ...loadSessionLedger().filter((s) => s !== sid)]
  saveSessionLedger(list)
}

export default function PatientChat() {
  const [messages, setMessages] = useState([WELCOME])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [sessionsPanel, setSessionsPanel] = useState(null) // null=关，[]/=面板数据
  const listRef = useRef(null)
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem('medirag_patient_session') || ''
  )

  // 拉取某个会话的消息并渲染（挂载恢复 / 历史面板切换共用）
  const loadSessionMessages = (sid) => {
    if (!sid) {
      setMessages([WELCOME])
      setLoadingHistory(false)
      return
    }
    setLoadingHistory(true)
    apiPatientChatHistory(sid)
      .then((resp) => {
        const history = (resp.messages || []).map((m) => ({
          role: m.role,
          content: m.content,
          intent: m.role === 'user' ? m.intent : '',
          citations: m.citations || [],
          ...(m.role === 'assistant' ? { disclaimer: DISCLAIMER_TEXT } : {}),
        }))
        setMessages(history.length ? [WELCOME, ...history] : [WELCOME])
      })
      .catch(() => setMessages([WELCOME]))
      .finally(() => setLoadingHistory(false))
  }

  // 挂载时恢复当前会话历史
  useEffect(() => {
    if (sessionId) {
      loadSessionMessages(sessionId)
    } else {
      setLoadingHistory(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 新消息时滚动到底部
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, sending])

  // 新开会话：清当前关联（会话进台账，历史面板里仍可切回）
  const newSession = () => {
    localStorage.removeItem('medirag_patient_session')
    setSessionId('')
    setMessages([WELCOME])
    setInput('')
    setSessionsPanel(null)
  }

  // 历史会话面板：拉台账内会话的摘要
  const openSessions = () => {
    if (sessionsPanel) { setSessionsPanel(null); return }
    const ids = loadSessionLedger()
    if (!ids.length) { setSessionsPanel([]); return }
    apiPatientSessions(ids)
      .then((resp) => setSessionsPanel(resp.sessions || []))
      .catch(() => setSessionsPanel([]))
  }

  // 切换回某个历史会话
  const switchSession = (sid) => {
    if (sending || sid === sessionId) { setSessionsPanel(null); return }
    localStorage.setItem('medirag_patient_session', sid)
    setSessionId(sid)
    setSessionsPanel(null)
    loadSessionMessages(sid)
  }

  const send = async (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setSending(true)
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    try {
      const resp = await apiPatientChat(sessionId, text)
      if (resp.session_id && resp.session_id !== sessionId) {
        localStorage.setItem('medirag_patient_session', resp.session_id)
        setSessionId(resp.session_id)
        rememberSession(resp.session_id) // 新会话进台账，历史面板可见
      }
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: resp.reply,
          intent: resp.intent,
          intentLabel: resp.intent_label,
          byRule: resp.by_rule,
          citations: resp.citations || [],
          disclaimer: resp.disclaimer,
          emergency: resp.intent === 'emergency',
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `抱歉，出了点问题：${err?.message || '请稍后再试'}`, error: true },
      ])
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="chat-page">
      <header className="chat-topbar">
        <div className="chat-brand">
          <img className="brand-avatar" src={robotAvatar} alt="" />
          🩺 智慧医疗诊疗助手
        </div>
        <div className="muted">健康科普 · 症状咨询 · 就医引导</div>
        <button type="button" className="btn small secondary" onClick={openSessions}>🕘 历史会话</button>
        {sessionId && messages.length > 1 && (
          <button type="button" className="btn small secondary" onClick={newSession}>🔄 新开会话</button>
        )}
        <a className="btn small secondary" href="#/">医生入口 →</a>
      </header>

      {sessionsPanel && (
        <div className="sessions-panel card">
          <h4>🕘 历史会话 <span className="muted">（本浏览器最近 10 个）</span></h4>
          {sessionsPanel.length === 0 && (
            <div className="muted sessions-empty">还没有聊过的会话</div>
          )}
          {sessionsPanel.map((s) => (
            <div
              key={s.session_id}
              className={`sessions-item ${s.session_id === sessionId ? 'active' : ''}`}
              onClick={() => switchSession(s.session_id)}
            >
              <div className="sessions-title">{s.first_message || '（无提问）'}</div>
              <div className="muted sessions-meta">{s.last_time} · {s.message_count} 条消息</div>
            </div>
          ))}
        </div>
      )}

      <div className="chat-window card">
        <div className="msg-list" ref={listRef}>
          {loadingHistory && (
            <div className="msg-row bot">
              <img className="msg-avatar bot-avatar" src={robotAvatar} alt="诊疗助手" />
              <div className="msg-bubble bot typing">正在恢复聊天记录…</div>
            </div>
          )}
          {messages.map((m, i) => {
            if (m.role === 'user') {
              return (
                <div key={i} className="msg-row user">
                  <div className="msg-bubble user">{m.content}</div>
                  <img className="msg-avatar user-avatar" src={userAvatar} alt="患者" />
                </div>
              )
            }
            const badge = m.intent ? INTENT_BADGES[m.intent] : null
            return (
              <div key={i} className="msg-row bot">
                <img className="msg-avatar bot-avatar" src={robotAvatar} alt="诊疗助手" />
                <div className={`msg-bubble bot ${m.emergency ? 'emergency' : ''} ${m.error ? 'error' : ''}`}>
                  {badge && (
                    <div className={`intent-badge ${badge.cls}`}>
                      {badge.label}{m.byRule ? ' · 规则' : ''}
                    </div>
                  )}
                  <div className="msg-text">{m.content}</div>
                  {m.citations?.length > 0 && (
                    <div className="mt8">
                      <Citations citations={m.citations} />
                    </div>
                  )}
                  {m.disclaimer && <div className="chat-disclaimer">{m.disclaimer}</div>}
                </div>
              </div>
            )
          })}
          {sending && (
            <div className="msg-row bot">
              <img className="msg-avatar bot-avatar" src={robotAvatar} alt="诊疗助手" />
              <div className="msg-bubble bot typing">助手正在思考…</div>
            </div>
          )}
        </div>

        <form className="chat-input" onSubmit={send}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="请输入您的问题，如：最近老是头晕怎么办？"
            maxLength={2000}
            disabled={sending}
          />
          <button className="btn" disabled={sending || !input.trim()}>发送</button>
        </form>
      </div>

      <footer className="chat-footer">
        本助手由 AI 生成回复，仅供健康科普参考，不能替代执业医师的诊断和治疗。
        紧急情况请立即拨打 120 或前往急诊。
      </footer>
    </div>
  )
}
