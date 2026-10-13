import { useState } from 'react'
import { apiLogin } from '../api.js'

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const user = await apiLogin(username, password)
      onLogin(user)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>MediRAG</h1>
        <div className="sub">智慧医疗辅助诊疗平台 · RAG 增强的多 Agent 诊疗</div>

        <label className="field">
          用户名
          <span className="control">
            <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="用户名" autoFocus />
          </span>
        </label>
        <label className="field">
          密码
          <span className="control">
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="密码" />
          </span>
        </label>

        {error && <div className="error-text">{error}</div>}

        <button className="btn" style={{ width: '100%' }} disabled={loading || !username || !password}>
          {loading ? '登录中…' : '登 录'}
        </button>

        <div className="demo">
          演示账号：admin / admin123（管理员）· doctor / doctor123（医生）
        </div>
      </form>
    </div>
  )
}
