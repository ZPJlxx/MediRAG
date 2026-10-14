const BASE = import.meta.env.VITE_API_BASE_URL || ''

function authHeaders() {
  const token = localStorage.getItem('medirag_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function extractDetail(data, status) {
  if (!data) return `请求失败 (${status})`
  // 统一错误体 {code, message, request_id}：优先展示 message，报障时附上 request_id
  if (typeof data.message === 'string' && data.message) {
    return data.request_id ? `${data.message}（追踪号 ${data.request_id}）` : data.message
  }
  const detail = data.detail
  if (typeof detail === 'string') return detail
  // FastAPI/Pydantic 校验错误是数组：[{loc, msg, type}, ...]
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.join('.') : d.loc
        return `${field || ''}: ${d.msg || d.type || '无效'}`.replace(/^: /, '')
      })
      .join('; ')
  }
  if (detail && typeof detail === 'object') return detail.message || detail.msg || JSON.stringify(detail)
  return `请求失败 (${status})`
}

async function handle(response, { authRedirect = true } = {}) {
  if (response.status === 401 && authRedirect) {
    localStorage.removeItem('medirag_token')
    localStorage.removeItem('medirag_user')
    window.location.reload()
    throw new Error('登录已过期，请重新登录')
  }
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(extractDetail(data, response.status))
  }
  return data
}

export async function apiLogin(username, password) {
  const response = await fetch(`${BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  // 登录本身的 401 是凭据错误，需展示后端文案，不能触发"会话过期"整页刷新
  const data = await handle(response, { authRedirect: false })
  localStorage.setItem('medirag_token', data.token)
  localStorage.setItem('medirag_user', JSON.stringify(data.user))
  return data.user
}

export function apiLogout() {
  localStorage.removeItem('medirag_token')
  localStorage.removeItem('medirag_user')
  window.location.reload()
}

export function currentUser() {
  try {
    return JSON.parse(localStorage.getItem('medirag_user'))
  } catch {
    return null
  }
}

// ---- 用户管理（admin）----
export const apiListUsers = () =>
  fetch(`${BASE}/api/users`, { headers: authHeaders() }).then(handle)

export const apiCreateUser = (body) =>
  fetch(`${BASE}/api/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  }).then(handle)

export const apiUpdateUser = (id, body) =>
  fetch(`${BASE}/api/users/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  }).then(handle)

export const apiDeleteUser = (id) =>
  fetch(`${BASE}/api/users/${id}`, { method: 'DELETE', headers: authHeaders() }).then(handle)

// ---- 知识库 ----
export const apiListKnowledge = () =>
  fetch(`${BASE}/api/knowledge/files`, { headers: authHeaders() }).then(handle)

export async function apiUploadKnowledge(file) {
  const form = new FormData()
  form.append('file', file)
  return handle(await fetch(`${BASE}/api/knowledge/upload`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  }))
}

export const apiDeleteKnowledge = (filename) =>
  fetch(`${BASE}/api/knowledge/files/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  }).then(handle)

export const apiSearchKnowledge = (query, topK = 5) =>
  fetch(`${BASE}/api/knowledge/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ query, top_k: topK }),
  }).then(handle)

// ---- 诊疗工作流 ----
export const apiStartWorkflow = (body) =>
  fetch(`${BASE}/api/workflow/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  }).then(handle)

export const apiWorkflowStatus = (id) =>
  fetch(`${BASE}/api/workflow/${id}/status`, { headers: authHeaders() }).then(handle)

export const apiSubmitLabResults = (id, labResults) =>
  fetch(`${BASE}/api/workflow/${id}/lab-results`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ lab_results: labResults }),
  }).then(handle)

export const apiListHistory = (page = 1, pageSize = 20, patient = '') => {
  const params = new URLSearchParams({ page, page_size: pageSize, patient })
  return fetch(`${BASE}/api/workflow/list/history?${params}`, { headers: authHeaders() }).then(handle)
}
