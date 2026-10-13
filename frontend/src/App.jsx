import { useState } from 'react'
import Login from './views/Login.jsx'
import Workspace from './views/Workspace.jsx'
import KnowledgePanel from './views/KnowledgePanel.jsx'
import UsersPanel from './views/UsersPanel.jsx'
import { currentUser, apiLogout } from './api.js'

export default function App() {
  const [user, setUser] = useState(currentUser())
  const [tab, setTab] = useState('workspace')

  if (!user) {
    return <Login onLogin={setUser} />
  }

  const tabs = [
    { key: 'workspace', label: '🩺 诊疗工作台' },
    { key: 'knowledge', label: '📚 医学知识库' },
    ...(user.role === 'admin' ? [{ key: 'users', label: '👤 用户管理' }] : []),
  ]

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          MediRAG<small>智慧医疗辅助诊疗平台 · 证据可溯源</small>
        </div>
        <div className="session">
          <span>{user.nickname || user.username}</span>
          <span className="role-tag">{user.role === 'admin' ? '管理员' : '医生'}</span>
          <button type="button" className="logout-btn" onClick={apiLogout}>退出登录</button>
        </div>
      </div>

      <div className="tabs">
        {tabs.map((t) => (
          <button key={t.key} className={`tab ${tab === t.key ? 'active' : ''}`} onClick={() => setTab(t.key)}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="main">
        {tab === 'workspace' && <Workspace user={user} />}
        {tab === 'knowledge' && <KnowledgePanel />}
        {tab === 'users' && <UsersPanel />}
      </div>

      <div className="disclaimer">
        ⚠️ 本系统仅供演示与辅助参考，不能替代执业医师的诊断与治疗决策；医疗决策请以临床规范和医师判断为准。
      </div>
    </div>
  )
}
