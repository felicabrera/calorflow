import Plot from 'react-plotly.js'

interface DistributionChartProps {
  data: number[]
  title: string
  color: string
}

export default function DistributionChart({ data, title, color }: DistributionChartProps) {
  if (!data || data.length === 0) {
    return <p style={{ color: '#666', textAlign: 'center' }}>No hay datos disponibles</p>
  }

  return (
    <Plot
      data={[
        {
          x: data,
          type: 'histogram',
          marker: { color: color },
          nbinsx: 30,
        },
      ]}
      layout={{
        title: title,
        xaxis: { title: title },
        yaxis: { title: 'Frecuencia' },
        height: 300,
        margin: { l: 50, r: 20, t: 40, b: 50 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
