import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import ShaderPreview from './components/ShaderPreview'
import './index.css'

const isShaderPreview = new URLSearchParams(window.location.search).has('shader')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isShaderPreview ? <ShaderPreview /> : <App />}
  </StrictMode>,
)
