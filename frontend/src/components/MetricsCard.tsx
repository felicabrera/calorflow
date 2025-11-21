interface MetricsCardProps {
  metrics: any
  process: string
}

export default function MetricsCard({ metrics, process }: MetricsCardProps) {
  if (!metrics?.metrics) {
    return (
      <div className="card">
        <p>No hay métricas disponibles para {process}</p>
      </div>
    )
  }

  const { metrics_pci, metrics_h2, competition_score } = metrics.metrics

  return (
    <div className="grid">
      <div className="metric-card">
        <h4>PCI RMSE</h4>
        <div className="value">{metrics_pci?.PCI_rmse?.toFixed(2) || 'N/A'}</div>
        <div className="label">Error cuadrático medio</div>
      </div>

      <div className="metric-card">
        <h4>H2 RMSE</h4>
        <div className="value">{metrics_h2?.H2_rmse?.toFixed(2) || 'N/A'}</div>
        <div className="label">Error cuadrático medio</div>
      </div>

      <div className="metric-card">
        <h4>RMSE Promedio</h4>
        <div className="value">{competition_score?.rmse_prom?.toFixed(2) || 'N/A'}</div>
        <div className="label">Métrica de competición</div>
      </div>

      <div className="metric-card">
        <h4>R² PCI</h4>
        <div className="value">{metrics_pci?.PCI_r2?.toFixed(3) || 'N/A'}</div>
        <div className="label">Coeficiente de determinación</div>
      </div>

      <div className="metric-card">
        <h4>R² H2</h4>
        <div className="value">{metrics_h2?.H2_r2?.toFixed(3) || 'N/A'}</div>
        <div className="label">Coeficiente de determinación</div>
      </div>

      <div className="metric-card">
        <h4>Dentro de ±10%</h4>
        <div className="value">
          {competition_score?.within_10_pci || 0} / {competition_score?.within_10_h2 || 0}
        </div>
        <div className="label">PCI / H2</div>
      </div>
    </div>
  )
}
