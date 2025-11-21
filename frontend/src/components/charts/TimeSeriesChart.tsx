import Plot from 'react-plotly.js'

interface TimeSeriesChartProps {
  data: {
    dates: string[]
    pci: number[]
    h2: number[]
  }
  process: string
}

export default function TimeSeriesChart({ data, process }: TimeSeriesChartProps) {
  if (!data || !data.dates || data.dates.length === 0) {
    return <p style={{ color: '#666', textAlign: 'center' }}>No hay datos de series temporales</p>
  }

  return (
    <Plot
      data={[
        {
          x: data.dates,
          y: data.pci,
          type: 'scatter',
          mode: 'lines',
          name: 'PCI',
          line: { color: '#667eea', width: 2 },
        },
        {
          x: data.dates,
          y: data.h2,
          type: 'scatter',
          mode: 'lines',
          name: 'H2',
          line: { color: '#764ba2', width: 2 },
          yaxis: 'y2',
        },
      ]}
      layout={{
        title: `${process} - Series Temporales`,
        xaxis: { title: 'Fecha' },
        yaxis: { title: 'PCI (kcal/Nm³)' },
        yaxis2: {
          title: 'H2 (%)',
          overlaying: 'y',
          side: 'right',
        },
        height: 400,
        margin: { l: 60, r: 60, t: 40, b: 80 },
        legend: { x: 0.5, y: 1.1, orientation: 'h' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
