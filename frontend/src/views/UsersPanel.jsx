import { useEffect, useState } from 'react'
import { apiListUsers, apiCreateUser, apiUpdateUser, apiDeleteUser } from '../api.js'

export default function UsersPanel() {
  const [users, setUsers] = useState([])
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const [form, setForm] = useState({ username: '', password: '', nickname: '', role: 'doctor' })
  const [creating, setCreating] = useState(false)

  const refresh = () => apiListUsers().then(setUsers).catch((e) => setError(e.message))
  useEffect(() => { refresh() }, [])

  const create = async (e) => {
    e.preventDefault()
    setCreating(true); setError(''); setMessage('')
    try {
      await apiCreateUser(form)
      setMessage(`用户 ${form.username} 创建成功`)
      setForm({ username: '', password: '', nickname: '', role: 'doctor' })
      refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  const toggleActive = async (user) => {
    try {
      await apiUpdateUser(user.id, { nickname: user.nickname, role: user.role, is_active: !user.is_active, password: '' })
      refresh()
    } catch (err) { setError(err.message) }
  }

  const remove = async (user) => {
    if (!window.confirm(`确认删除用户 ${user.username}？`)) return
    try {
      await apiDeleteUser(user.id)
      setMessage(`用户 ${user.username} 已删除`)
      refresh()
    } catch (err) { setError(err.message) }
  }

  return (
    <div className="layout-cols-form">
      <div>
        <div className="card">
          <h3>➕ 新增用户</h3>
          <form onSubmit={create}>
            <label className="field">用户名
              <span className="control"><input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required minLength={2} /></span>
            </label>
            <label className="field">初始密码 <span className="hint">至少 6 位，将加密存储（bcrypt）</span>
              <span className="control"><input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required minLength={6} /></span>
            </label>
            <label className="field">姓名/昵称
              <span className="control"><input value={form.nickname} onChange={(e) => setForm({ ...form, nickname: e.target.value })} /></span>
            </label>
            <label className="field">角色
              <span className="control">
                <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                  <option value="doctor">医生</option>
                  <option value="admin">管理员</option>
                </select>
              </span>
            </label>
            <button className="btn" style={{ width: '100%' }} disabled={creating}>{creating ? '创建中…' : '创建用户'}</button>
          </form>
        </div>
      </div>

      <div>
        <div className="card">
          <h3>👤 用户列表 <span className="badge">{users.length} 个账号</span></h3>
          {message && <div className="ok-text">{message}</div>}
          {error && <div className="error-text">{error}</div>}
          <div className="table-wrap">
            <table className="data">
            <thead><tr><th>ID</th><th>用户名</th><th>姓名</th><th>角色</th><th>状态</th><th style={{ width: 150 }}>操作</th></tr></thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td className="muted">{u.id}</td>
                  <td style={{ fontWeight: 600 }}>{u.username}</td>
                  <td>{u.nickname}</td>
                  <td><span className={`status-pill ${u.role === 'admin' ? 's3' : 's1'}`}>{u.role === 'admin' ? '管理员' : '医生'}</span></td>
                  <td>{u.is_active ? '✅ 正常' : '⛔ 禁用'}</td>
                  <td>
                    <button className="btn small secondary" onClick={() => toggleActive(u)}>{u.is_active ? '禁用' : '启用'}</button>{' '}
                    <button className="btn small danger" onClick={() => remove(u)}>删除</button>
                  </td>
                </tr>
              ))}
            </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
