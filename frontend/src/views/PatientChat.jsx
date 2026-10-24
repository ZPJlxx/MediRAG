import { useEffect, useRef, useState } from 'react'
import { apiPatientChat } from '../api.js'
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

export default function PatientChat() {
  const [messages, setMessages] = useState([WELCOME])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const listRef = useRef(null)
  const sessionId = localStorage.getItem('medirag_patient_session') || ''

  // 新消息时滚动到底部
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, sending])

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
        <a className="btn small secondary" href="#/">医生入口 →</a>
      </header>

      <div className="chat-window card">
        <div className="msg-list" ref={listRef}>
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
