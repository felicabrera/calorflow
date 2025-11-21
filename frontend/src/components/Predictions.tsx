import { useState } from 'react'

export default function Predictions() {
  const [process, setProcess] = useState<'FCC' | 'CCR'>('FCC')
  const [file, setFile] = useState<File | null>(null)
  const [predictions, setPredictions] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handlePredict = async () => {
    if (!file) {
      setError('Por favor selecciona un archivo CSV')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(
        `http://localhost:8000/api/v1/predict/csv?process=${process}`,
        {
          method: 'POST',
          body: formData,
        }
      )

      if (!response.ok) {
        throw new Error('Error en la predicción')
      }

      const data = await response.json()
      setPredictions(data)
    } catch (err: any) {
      setError(err.message || 'Error realizando predicción')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="predictions">
      <h1>Predicciones</h1>
      <p className="subtitle">Realiza predicciones de PCI y H2 desde datos operacionales</p>

      <div className="card">
        <h2>Configuración</h2>
        
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#333' }}>
            Proceso:
          </label>
          <select
            value={process}
            onChange={(e) => setProcess(e.target.value as 'FCC' | 'CCR')}
            style={{
              padding: '0.5rem',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '1rem',
            }}
          >
            <option value="FCC">FCC - Fluid Catalytic Cracking</option>
            <option value="CCR">CCR - Catalytic Reforming</option>
          </select>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#333' }}>
            Archivo CSV:
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            style={{
              padding: '0.5rem',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '1rem',
            }}
          />
        </div>

        <button
          className="button"
          onClick={handlePredict}
          disabled={loading || !file}
        >
          {loading ? 'Prediciendo...' : 'Realizar Predicción'}
        </button>
      </div>

      {error && (
        <div className="error">
          <p>{error}</p>
        </div>
      )}

      {predictions && (
        <div className="card">
          <h2>Resultados</h2>
          
          <div className="grid">
            <div className="metric-card">
              <h4>Predicciones</h4>
              <div className="value">{predictions.n_predictions}</div>
              <div className="label">Total de muestras</div>
            </div>

            <div className="metric-card">
              <h4>PCI Promedio</h4>
              <div className="value">{predictions.statistics?.pci?.mean?.toFixed(2)}</div>
              <div className="label">kcal/Nm³</div>
            </div>

            <div className="metric-card">
              <h4>H2 Promedio</h4>
              <div className="value">{predictions.statistics?.h2?.mean?.toFixed(2)}</div>
              <div className="label">%</div>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', maxHeight: '400px', overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  <th style={{ padding: '0.75rem', textAlign: 'left', color: '#333' }}>#</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', color: '#333' }}>PCI Predicho</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', color: '#333' }}>H2 Predicho</th>
                </tr>
              </thead>
              <tbody>
                {predictions.predictions?.slice(0, 50).map((pred: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '0.75rem', color: '#333' }}>{idx + 1}</td>
                    <td style={{ padding: '0.75rem', color: '#333' }}>{pred.PCI_pred?.toFixed(2)}</td>
                    <td style={{ padding: '0.75rem', color: '#333' }}>{pred.H2_pred?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
