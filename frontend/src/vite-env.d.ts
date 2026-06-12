/// <reference types="vite/client" />

interface Window {
  __shaderReady?: boolean;
  __shaderError?: string | null;
  __setShaderTime?: (timeSeconds: number) => boolean;
}
