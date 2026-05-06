import { useState, useRef } from 'react'
import './App.css'

type Valence = 'positive' | 'negative'
type Result = { prediction: string; message: string; confidence: number }

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [valence, setValence] = useState<Valence>('positive')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null
    setFile(selected)
    setResult(null)
    setError(null)
  }

  async function handleSubmit(e: { preventDefault(): void }) {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    setResult(null)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('context', valence)

    try {
      const res = await fetch('http://localhost:8000/analyze-deception', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail ?? `Server error ${res.status}`)
      }

      const data: Result = await res.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <main className="page">
      <header className="header">
        <h1>Deception Detection Demo</h1>
        <p className="subtitle">Upload an MP4 video and select the emotional valence context to analyze for deception.</p>
      </header>

      <form className="card" onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="video-upload" className="label">Video file</label>
          <div className="file-drop">
            <input
              ref={inputRef}
              id="video-upload"
              type="file"
              accept="video/mp4,.mp4"
              onChange={handleFileChange}
              className="file-input"
            />
            <span className="file-label">
              {file ? file.name : 'Choose an MP4 file…'}
            </span>
          </div>
          <span className="hint">Only .mp4 files are accepted.</span>
        </div>

        <div className="field">
          <span className="label">Valence context</span>
          <div className="toggle-group" role="group" aria-label="Valence context">
            <label className={`toggle ${valence === 'positive' ? 'active' : ''}`}>
              <input
                type="radio"
                name="valence"
                value="positive"
                checked={valence === 'positive'}
                onChange={() => setValence('positive')}
              />
              Positive
            </label>
            <label className={`toggle ${valence === 'negative' ? 'active' : ''}`}>
              <input
                type="radio"
                name="valence"
                value="negative"
                checked={valence === 'negative'}
                onChange={() => setValence('negative')}
              />
              Negative
            </label>
          </div>
          <span className="hint">The emotional context in which the statement was made.</span>
        </div>

        <div className="actions">
          <button type="submit" className="btn-primary" disabled={!file || loading}>
            {loading ? 'Analysing…' : 'Analyse video'}
          </button>
          {(file || result) && (
            <button type="button" className="btn-ghost" onClick={handleReset}>
              Reset
            </button>
          )}
        </div>
      </form>

      {error && (
        <div className="result error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
    <div className={`result ${result.prediction === 'Deceptive' ? 'deceptive' : 'truthful'}`}>
    <div className="prediction-label">Result</div>
    <div className="prediction-value">{result.prediction}</div>
    <div className="prediction-label" style={{ marginTop: '8px' }}>Confidence</div>
    <div className="prediction-value" style={{ fontSize: '20px' }}>
      {(result.confidence * 100).toFixed(1)}%
    </div>
  </div>
)}
    </main>
  )
}

export default App
