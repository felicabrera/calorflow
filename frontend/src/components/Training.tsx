import { useState } from 'react'

export default function Training() {
  const [process, setProcess] = useState<'FCC' | 'CCR'>('FCC')
  const [nTrials, setNTrials] = useState(100)
  const [cvFolds, setCvFolds] = useState(5)
  const [loading, setLoading] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleStartTraining = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch('http://localhost:8000/api/v1/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          process,
          config: {
            n_trials: nTrials,
            cv_folds: cvFolds,
            use_autogluon: false,
          },
        }),
      })

      if (!response.ok) {
        throw new Error('Error iniciando entrenamiento')
      }

      const data = await response.json()
      setTaskId(data.task_id)
    } catch (err: any) {
      setError(err.message || 'Error en el entrenamiento')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="training">
      <h1>Entrenamiento de Modelos</h1>
      <p className="subtitle">Configura y entrena modelos de ML para predicción de PCI y H2</p>

      <div className="card">
        <h2>Configuración de Entrenamiento</h2>

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
              width: '100%',
            }}
          >
            <option value="FCC">FCC - Fluid Catalytic Cracking</option>
            <option value="CCR">CCR - Catalytic Reforming</option>
          </select>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#333' }}>
            Trials de Optuna: {nTrials}
          </label>
          <input
            type="range"
            min="10"
            max="300"
            value={nTrials}
            onChange={(e) => setNTrials(Number(e.target.value))}
            style={{ width: '100%' }}
          />
          <small style={{ color: '#666' }}>
            Más trials = mejor optimización pero más tiempo de entrenamiento
          </small>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#333' }}>
            CV Folds: {cvFolds}
          </label>
          <input
            type="range"
            min="2"
            max="10"
            value={cvFolds}
            onChange={(e) => setCvFolds(Number(e.target.value))}
            style={{ width: '100%' }}
          />
          <small style={{ color: '#666' }}>
            Número de particiones para validación cruzada
          </small>
        </div>

        <button
          className="button"
          onClick={handleStartTraining}
          disabled={loading}
        >
          {loading ? 'Iniciando entrenamiento...' : 'Iniciar Entrenamiento'}
        </button>

        {taskId && (
          <div
            style={{
              marginTop: '1rem',
              padding: '1rem',
              background: '#f0f9ff',
              borderRadius: '6px',
              border: '1px solid #0ea5e9',
            }}
          >
            <p style={{ color: '#0369a1' }}>
              ✅ Entrenamiento iniciado
            </p>
            <p style={{ color: '#666', fontSize: '0.875rem', marginTop: '0.5rem' }}>
              Task ID: <code>{taskId}</code>
            </p>
          </div>
        )}

        {error && (
          <div className="error" style={{ marginTop: '1rem' }}>
            <p>{error}</p>
          </div>
        )}
      </div>

      <div className="card">
        <h2>ℹ️ Información sobre el Entrenamiento</h2>
        <div style={{ color: '#333', lineHeight: '1.8' }}>
          <p><strong>Modelos:</strong> XGBoost, LightGBM, CatBoost</p>
          <p><strong>Optimización:</strong> Bayesian Optimization con Optuna</p>
          <p><strong>Ensemble:</strong> Meta-learner con Ridge Regression</p>
          <p><strong>Validación:</strong> K-Fold Cross Validation</p>
          <p><strong>Tiempo estimado:</strong> {Math.round(nTrials / 10)} - {Math.round(nTrials / 5)} minutos</p>
        </div>
      </div>
    </div>
  )
}
