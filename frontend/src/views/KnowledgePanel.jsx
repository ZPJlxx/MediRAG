import { useEffect, useRef, useState } from 'react'
import {
  apiListKnowledge,
  apiUploadKnowledge,
  apiDeleteKnowledge,
  apiSearchKnowledge,
} from '../api.js'

export default function KnowledgePanel() {
  const [files, setFiles] = useState([])
  const [runtime, setRuntime] = useState(null)
  const [ragEnabled, setRagEnabled] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const fileInput = useRef(null)

  // 检索测试
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [results, setResults] = useState(null)

  const refresh = () => {
    apiListKnowledge().then((data) => {
      setFiles(data.files || [])
      setRuntime(data.runtime)
      setRagEnabled(data.rag_enabled)
    }).catch((e) => setError(e.message))
  }

  useEffect(() => { refresh() }, [])

  const upload = async (file) => {
    if (!file) return
    setUploading(true)
    setError(''); setMessage('')
    try {
      await apiUploadKnowledge(file)
      setMessage(`《${file.name}》上传成功，首次检索时会自动构建索引`)
      refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const remove = async (filename) => {
    if (!window.confirm(`确认删除《${filename}》？向量索引将一并清除。`)) return
    try {
      await apiDeleteKnowledge(filename)
      setMessage(`《${filename}》已删除`)
      refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const testSearch = async () => {
    if (!query.trim()) return
    setSearching(true)
    setError(''); setResults(null)
    try {
      const data = await apiSearchKnowledge(query.trim(), 5)
      setResults(data.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="layout-cols">
      <div>
        <div className="card">
          <h3>
            📚 医学知识库
            <span className="badge">{files.length} 份文档</span>
            {ragEnabled ? <span className="badge">诊疗 RAG 已启用</span> : <span className="badge" style={{ background: '#fdf3e0', color: '#b7791f' }}>诊疗 RAG 已关闭</span>}
          </h3>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 14 }}>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.txt,.md,.csv"
              style={{ flex: 1, minWidth: 0 }}
              disabled={uploading}
              onChange={(e) => upload(e.target.files?.[0])}
            />
          </div>
          <div className="hint mb12">支持 pdf / docx / txt / md / csv，单文件 ≤100MB。临床指南、药品说明书、院内规范均可上传；诊断与治疗建议将引用这些资料作为证据。</div>

          {uploading && <div className="ok-text">上传中…</div>}
          {message && <div className="ok-text">{message}</div>}
          {error && <div className="error-text">{error}</div>}

          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>文件名</th><th style={{ width: 110 }}>大小</th><th style={{ width: 140 }}>上传时间</th><th style={{ width: 70 }}>操作</th></tr></thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.filename}>
                    <td>{f.filename}</td>
                    <td className="muted">{(f.size_bytes / 1024).toFixed(1)} KB</td>
                    <td className="muted">{f.created_at}</td>
                    <td><button className="btn small danger" onClick={() => remove(f.filename)}>删除</button></td>
                  </tr>
                ))}
                {!files.length && <tr><td colSpan="4" className="muted">知识库为空，上传文档后 AI 诊疗将获得证据支撑</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="layout-side">
        <div className="card">
          <h3>🔍 检索测试 <span className="badge">三路混合 + RRF + 重排</span></h3>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              value={query}
              placeholder="如：高血压 2 级的诊断标准是什么"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && testSearch()}
            />
            <button className="btn" disabled={searching || !query.trim()} onClick={testSearch}>
              {searching ? '检索中…' : '检索'}
            </button>
          </div>

          {results && (
            <div className="mt12">
              {results.length === 0 && <div className="muted">未检索到相关内容</div>}
              {results.map((c, i) => (
                <div key={i} className="citation-item">
                  <div className="cite-head">
                    <span className="cite-ref">[{i + 1}]</span>
                    <span>《{c.source}》</span>
                    <span className="cite-loc">{c.location}</span>
                  </div>
                  <div className="cite-snippet">"{c.snippet}"</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {runtime && (
          <div className="card">
            <h3>⚙️ 检索引擎状态</h3>
            <div className="muted" style={{ lineHeight: 2 }}>
              <div>Embedding：<b>{runtime.embedding?.provider}::{runtime.embedding?.model}</b> {runtime.embedding?.available ? '✅' : '❌'}</div>
              <div>Rerank：<b>{runtime.rerank?.provider}::{runtime.rerank?.model || '关闭'}</b> {runtime.rerank?.available ? '✅' : (runtime.rerank?.enabled ? '❌' : '—')}</div>
              <div>{runtime.embedding?.status_message}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
