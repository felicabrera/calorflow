import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './components/Dashboard'
import Predictions from './components/Predictions'
import Training from './components/Training'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="navbar-brand">
            <h1>🔬 Calorflow</h1>
            <span className="subtitle">ANCAP DataChallenge 2025</span>
          </div>
          <div className="navbar-links">
            <Link to="/">Dashboard</Link>
            <Link to="/predictions">Predicciones</Link>
            <Link to="/training">Entrenamiento</Link>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
