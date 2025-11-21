import { useState, useEffect } from 'react'

interface TrainingStatus {
  status: string
  progress: number
  current_step?: string
  results?: {
    pci_metrics: { rmse: number; mae: number; r2: number }
    h2_metrics: { rmse: number; mae: number; r2: number }
    config: { n_trials: number; cv_folds: number }
  }
  error?: string
}

export default function Training() {
  const [process, setProcess] = useState<'FCC' | 'CCR'>('FCC')
  const [nTrials, setNTrials] = useState(10)
  const [cvFolds, setCvFolds] = useState(5)
  const [loading, setLoading] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)

  // Polling para obtener el estado del entrenamiento
  useEffect(() => {
    if (!taskId) return

    const pollStatus = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/train/status/${taskId}`)
        if (response.ok) {
          const status: TrainingStatus = await response.json()
          setTrainingStatus(status)

          // Detener polling si completó o falló
          if (status.status === 'completed' || status.status === 'failed') {
            setLoading(false)
          }
        }
      } catch (err) {
        console.error('Error polling status:', err)
      }
    }

    // Poll cada 2 segundos
    const interval = setInterval(pollStatus, 2000)
    pollStatus() // Ejecutar inmediatamente

    return () => clearInterval(interval)
  }, [taskId])

  const handleStartTraining = async () => {
    try {
      setLoading(true)
      setError(null)
      setTrainingStatus(null)
      
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
        throw new Error(`Error iniciando entrenamiento: ${response.status}`)
      }

      const data = await response.json()
      setTaskId(data.task_id)
      
    } catch (err: any) {
      console.error('Training error:', err)
      setError(err.message || 'Error en el entrenamiento')
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
          {loading ? 'Entrenando...' : 'Iniciar Entrenamiento'}
        </button>

        {/* Progress Bar */}
        {trainingStatus && trainingStatus.status === 'training' && (
          <div
            style={{
              marginTop: '1.5rem',
              padding: '1.5rem',
              background: '#f0f9ff',
              borderRadius: '8px',
              border: '1px solid #0ea5e9',
            }}
          >
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: 600, color: '#0369a1' }}>Entrenando...</span>
                <span style={{ color: '#0369a1' }}>{trainingStatus.progress}%</span>
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                background: '#dbeafe',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: `${trainingStatus.progress}%`, 
                  height: '100%', 
                  background: '#0ea5e9',
                  transition: 'width 0.3s ease'
                }} />
              </div>
            </div>
            {trainingStatus.current_step && (
              <p style={{ color: '#0369a1', fontSize: '0.875rem', margin: 0 }}>
                📊 {trainingStatus.current_step}
              </p>
            )}
          </div>
        )}

        {/* Results */}
        {trainingStatus && trainingStatus.status === 'completed' && trainingStatus.results && (
          <div
            style={{
              marginTop: '1.5rem',
              padding: '1.5rem',
              background: '#f0fdf4',
              borderRadius: '8px',
              border: '1px solid #22c55e',
            }}
          >
            <h3 style={{ color: '#15803d', marginTop: 0, marginBottom: '1rem' }}>
              ✅ Entrenamiento Completado
            </h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
              {/* PCI Metrics */}
              <div style={{ 
                padding: '1rem', 
                background: 'white', 
                borderRadius: '6px',
                border: '1px solid #e5e7eb'
              }}>
                <h4 style={{ margin: '0 0 0.75rem 0', color: '#374151' }}>📈 Métricas PCI</h4>
                <div style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: '1.8' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>RMSE:</span>
                    <strong style={{ color: '#111827' }}>{trainingStatus.results.pci_metrics.rmse.toFixed(4)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>MAE:</span>
                    <strong style={{ color: '#111827' }}>{trainingStatus.results.pci_metrics.mae.toFixed(4)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>R²:</span>
                    <strong style={{ color: '#111827' }}>{trainingStatus.results.pci_metrics.r2.toFixed(4)}</strong>
                  </div>
                </div>
              </div>

              {/* H2 Metrics */}
              <div style={{ 
                padding: '1rem', 
                background: 'white', 
                borderRadius: '6px',
                border: '1px solid #e5e7eb'
              }}>
                <h4 style={{ margin: '0 0 0.75rem 0', color: '#374151' }}>📊 Métricas H2</h4>
                <div style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: '1.8' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>RMSE:</span>
                    <strong style={{ color: '#111827' }}>{trainingStatus.results.h2_metrics.rmse.toFixed(4)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>MAE:</span>
                    <strong style={{ color: '#111827' }}>{trainingStatus.results.h2_metrics.mae.toFixed(4)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>R²:</span>
                    <strong style={{ color: '#111827' }}>{trainingStatus.results.h2_metrics.r2.toFixed(4)}</strong>
                  </div>
                </div>
              </div>
            </div>

            <div style={{ 
              marginTop: '1rem', 
              padding: '0.75rem', 
              background: '#dbeafe',
              borderRadius: '4px',
              fontSize: '0.875rem',
              color: '#1e40af'
            }}>
              ℹ️ Modelos guardados en: <code>models/{process}/</code>
            </div>
          </div>
        )}

        {/* Error */}
        {trainingStatus && trainingStatus.status === 'failed' && (
          <div
            style={{
              marginTop: '1rem',
              padding: '1rem',
              background: '#fef2f2',
              borderRadius: '6px',
              border: '1px solid #ef4444',
            }}
          >
            <p style={{ color: '#dc2626', margin: 0 }}>
              ❌ Error: {trainingStatus.error || 'Error desconocido'}
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
          <p><strong>Modelos a entrenar:</strong> {nTrials <= 20 ? '2 modelos (XGBoost, LightGBM)' : '4 modelos (XGBoost, LightGBM, CatBoost, RandomForest)'}</p>
          <p><strong>Targets:</strong> PCI y H2 (2 modelos por cada uno)</p>
          <p><strong>Trials por modelo:</strong> {nTrials}</p>
          <p><strong>Total de trials:</strong> {nTrials <= 20 ? nTrials * 2 * 2 : nTrials * 4 * 2} trials ({nTrials <= 20 ? '2' : '4'} modelos × 2 targets × {nTrials} trials)</p>
          <p><strong>Optimización:</strong> Bayesian Optimization con Optuna</p>
          <p><strong>Ensemble:</strong> Meta-learner con Ridge Regression</p>
          <p><strong>Validación:</strong> {cvFolds}-Fold Cross Validation</p>
          <p><strong>⏱️ Tiempo estimado:</strong> {nTrials <= 20 ? `${Math.round(nTrials * 0.5)}-${Math.round(nTrials * 1)}` : `${Math.round(nTrials * 2)}-${Math.round(nTrials * 4)}`} minutos</p>
        </div>
        <div style={{ 
          marginTop: '1rem', 
          padding: '0.75rem', 
          background: '#fef3c7',
          borderRadius: '4px',
          fontSize: '0.875rem',
          color: '#92400e'
        }}>
          💡 <strong>Tip:</strong> Para entrenamientos rápidos (≤20 trials), solo se entrenan XGBoost y LightGBM. Para entrenamientos completos (&gt;20 trials), se entrenan los 4 modelos.
        </div>
      </div>
    </div>
  )
}
