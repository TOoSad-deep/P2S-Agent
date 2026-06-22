import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import ShaderPlayground from './pages/ShaderPlayground'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ShaderPlayground />
  </StrictMode>,
)
