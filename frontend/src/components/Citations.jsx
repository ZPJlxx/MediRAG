export default function Citations({ citations = [] }) {
  if (!citations.length) {
    return <div className="citations"><div className="citations-title">📎 证据引用</div><div className="citation-empty">本阶段无知识库引用（结论基于模型医学常识或知识库未覆盖）</div></div>
  }
  return (
    <div className="citations">
      <div className="citations-title">📎 证据引用 · {citations.length} 条</div>
      {citations.map((c, i) => (
        <details className="citation-item" key={i}>
          <summary className="cite-head" title={`${c.source} ${c.location}`}>
            <span className="cite-ref">[{i + 1}]</span>
            <span className="cite-source">《{c.source}》</span>
            <span className="cite-loc">{c.location}</span>
            <span className="cite-toggle" aria-hidden="true"></span>
          </summary>
          <div className="cite-snippet">"{c.snippet}"</div>
        </details>
      ))}
    </div>
  )
}
