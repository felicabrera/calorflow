/**
 * Servicio API para comunicarse con el backend
 */

const API_BASE_URL = 'http://localhost:8000/api/v1'

export const api = {
  get: async (endpoint: string) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  post: async (endpoint: string, data: any) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },
}

// Endpoints específicos
export const getVisualizationData = async (process: string) => {
  return api.get(`/visualizations/${process}/training`)
}

export const getMetrics = async (process: string) => {
  return api.get(`/metrics/${process}`)
}

export const getModels = async () => {
  return api.get('/models')
}

export const predict = async (data: any) => {
  return api.post('/predict', data)
}

export const trainModel = async (data: any) => {
  return api.post('/train', data)
}
