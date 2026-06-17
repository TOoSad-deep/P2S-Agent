import { createContext, useContext } from 'react'
import type { PngShaderViewProps } from '../components/PngShaderView'

const PngShaderContext = createContext<PngShaderViewProps | null>(null)

export function PngShaderProvider({
  value,
  children,
}: {
  value: PngShaderViewProps
  children: React.ReactNode
}) {
  return <PngShaderContext.Provider value={value}>{children}</PngShaderContext.Provider>
}

export function usePngShaderContext(): PngShaderViewProps {
  const ctx = useContext(PngShaderContext)
  if (!ctx) {
    throw new Error('usePngShaderContext must be used inside <PngShaderProvider>')
  }
  return ctx
}
