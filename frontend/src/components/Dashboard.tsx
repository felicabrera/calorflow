import { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Chip,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Divider,
  Stack,
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  ModelTraining,
  Analytics,
  Assessment,
  Description,
} from '@mui/icons-material';
import { api } from '../services/api';
import type { HealthCheckResponse } from '../types/api';

export default function Dashboard() {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadHealthStatus();
    const interval = setInterval(loadHealthStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadHealthStatus = async () => {
    try {
      const data = await api.healthCheck();
      setHealth(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load health status');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md" sx={{ mt: 8 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="h6">Connection Error</Typography>
          <Typography>{error}</Typography>
        </Alert>
        <Button variant="contained" onClick={loadHealthStatus} fullWidth>
          Retry Connection
        </Button>
      </Container>
    );
  }

  const isHealthy = health?.status === 'healthy';

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" fontWeight={700} gutterBottom>
          Calorflow Dashboard
        </Typography>
        <Chip
          icon={isHealthy ? <CheckCircle /> : <Warning />}
          label={isHealthy ? 'System Healthy' : 'System Unhealthy'}
          color={isHealthy ? 'success' : 'warning'}
          sx={{ fontSize: '1rem', px: 2, py: 2.5 }}
        />
      </Box>

      <Stack spacing={3}>
        {/* Info Cards Row */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
          {/* System Info Card */}
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight={600}>
                System Information
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography color="text.secondary">Version</Typography>
                <Typography fontWeight={500}>{health?.version}</Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography color="text.secondary">Status</Typography>
                <Chip
                  label={health?.status}
                  size="small"
                  color={isHealthy ? 'success' : 'warning'}
                />
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography color="text.secondary">Last Check</Typography>
                <Typography fontWeight={500}>
                  {health?.timestamp && new Date(health.timestamp).toLocaleString()}
                </Typography>
              </Box>
            </CardContent>
          </Card>

          {/* Models Availability Card */}
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight={600}>
                Models Availability
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Stack spacing={2}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body1" fontWeight={600}>FCC Model</Typography>
                  <Chip
                    icon={health?.models_available?.FCC ? <CheckCircle /> : <Warning />}
                    label={health?.models_available?.FCC ? 'Ready' : 'Not Trained'}
                    color={health?.models_available?.FCC ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body1" fontWeight={600}>CCR Model</Typography>
                  <Chip
                    icon={health?.models_available?.CCR ? <CheckCircle /> : <Warning />}
                    label={health?.models_available?.CCR ? 'Ready' : 'Not Trained'}
                    color={health?.models_available?.CCR ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Box>

        {/* Quick Actions */}
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom fontWeight={600} sx={{ mb: 3 }}>
            Quick Actions
          </Typography>
          
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(4, 1fr)' }, 
            gap: 2 
          }}>
            <Button
              variant="contained"
              fullWidth
              size="large"
              startIcon={<ModelTraining />}
              sx={{ py: 2 }}
              onClick={() => alert('Training interface coming soon')}
            >
              Train Models
            </Button>
            
            <Button
              variant="outlined"
              fullWidth
              size="large"
              startIcon={<Analytics />}
              sx={{ py: 2 }}
              onClick={() => alert('Prediction interface coming soon')}
            >
              Predictions
            </Button>
            
            <Button
              variant="outlined"
              fullWidth
              size="large"
              startIcon={<Assessment />}
              sx={{ py: 2 }}
              onClick={() => alert('Data quality interface coming soon')}
            >
              Data Quality
            </Button>
            
            <Button
              variant="outlined"
              fullWidth
              size="large"
              startIcon={<Description />}
              sx={{ py: 2 }}
              onClick={() => window.open(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/docs`, '_blank')}
            >
              API Docs
            </Button>
          </Box>
        </Paper>
      </Stack>
    </Container>
  );
}
