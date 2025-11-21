import { useEffect, useState } from 'react'
import { getVisualizationData, getMetrics } from '../services/api'
import DistributionChart from './charts/DistributionChart'
import TimeSeriesChart from './charts/TimeSeriesChart'
import MetricsCard from './MetricsCard'

interface DashboardData {
  fcc?: any
  ccr?: any
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData>({})
  const [metrics, setMetrics] = useState<any>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)
      
      // Cargar datos de visualización para FCC y CCR
      const [fccData, ccrData, fccMetrics, ccrMetrics] = await Promise.all([
        getVisualizationData('FCC'),
        getVisualizationData('CCR'),
        getMetrics('FCC').catch(() => null),
        getMetrics('CCR').catch(() => null)
      ])

      setData({
        fcc: fccData,
        ccr: ccrData
      })

      setMetrics({
        fcc: fccMetrics,
        ccr: ccrMetrics
      })

      setError(null)
    } catch (err: any) {
      setError(err.message || 'Error cargando datos')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="loading">
        <p>Cargando dashboard...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error">
        <h3>Error</h3>
        <p>{error}</p>
        <button className="button" onClick={loadDashboardData}>Reintentar</button>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <h1>Dashboard - Análisis de Datos</h1>
      <p className="subtitle">Visualización de métricas y distribuciones de los procesos FCC y CCR</p>

      {/* Métricas generales */}
      {metrics.fcc && (
        <section>
          <h2>📊 FCC - Fluid Catalytic Cracking</h2>
          <MetricsCard metrics={metrics.fcc} process="FCC" />
          
          <div className="grid">
            <div className="card">
              <h3>Distribución PCI</h3>
              <DistributionChart 
                data={data.fcc?.data?.target_distributions?.pci?.values || []}
                title="PCI (kcal/Nm³)"
                color="#667eea"
              />
            </div>
            <div className="card">
              <h3>Distribución H2</h3>
              <DistributionChart 
                data={data.fcc?.data?.target_distributions?.h2?.values || []}
                title="H2 (%)"
                color="#764ba2"
              />
            </div>
          </div>

          {data.fcc?.data?.time_series && (
            <div className="card">
              <h3>Series Temporales</h3>
              <TimeSeriesChart 
                data={data.fcc.data.time_series}
                process="FCC"
              />
            </div>
          )}
        </section>
      )}

      {metrics.ccr && (
        <section style={{ marginTop: '3rem' }}>
          <h2>⚗️ CCR - Catalytic Reforming</h2>
          <MetricsCard metrics={metrics.ccr} process="CCR" />
          
          <div className="grid">
            <div className="card">
              <h3>Distribución PCI</h3>
              <DistributionChart 
                data={data.ccr?.data?.target_distributions?.pci?.values || []}
                title="PCI (kcal/Nm³)"
                color="#667eea"
              />
            </div>
            <div className="card">
              <h3>Distribución H2</h3>
              <DistributionChart 
                data={data.ccr?.data?.target_distributions?.h2?.values || []}
                title="H2 (%)"
                color="#764ba2"
              />
            </div>
          </div>

          {data.ccr?.data?.time_series && (
            <div className="card">
              <h3>Series Temporales</h3>
              <TimeSeriesChart 
                data={data.ccr.data.time_series}
                process="CCR"
              />
            </div>
          )}
        </section>
      )}
    </div>
  )
}
