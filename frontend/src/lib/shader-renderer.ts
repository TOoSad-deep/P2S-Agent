// shader-renderer.ts
import * as THREE from "three";

/**
 * Per-frame decision for the value written to `u_time`.
 *
 * When `frozenTime` is non-null the shader time is pinned to that value
 * (deterministic control via setTime()/__setShaderTime), so the free-running
 * clock cannot overwrite it. When it is null the running clock value is used.
 *
 * Note: 0 and negative values are valid frozen times — only `null` means
 * "not frozen", so this must NOT use a truthiness check.
 */
export function nextShaderTime(
  frozenTime: number | null,
  clockElapsed: number,
): number {
  return frozenTime !== null ? frozenTime : clockElapsed;
}

const VERTEX_SHADER = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// Shadertoy 兼容的片段着色器包装
function wrapFragmentShader(userShader: string): string {
  return `
precision highp float;

// System uniforms (Three.js naming)
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
varying vec2 vUv;

// Shadertoy aliases (for compatibility with generated shaders)
#define iTime u_time
#define iResolution u_resolution
#define iMouse vec4(u_mouse, 0.0, 0.0)

${userShader}

void main() {
  vec4 fragColor;
  mainImage(fragColor, gl_FragCoord.xy);
  gl_FragColor = fragColor;
}
`;
}

export class ShaderRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private mesh: THREE.Mesh | null = null;
  private clock: THREE.Clock;
  private animationId: number | null = null;
  private mousePos = new THREE.Vector2(0, 0);
  private backdropTexture: THREE.Texture | null = null;
  private userTexture: THREE.Texture | null = null;
  private defaultTexture: THREE.Texture;
  private onFrameCallback: (() => void) | null = null;
  // When non-null, u_time is pinned to this value (deterministic control via
  // setTime()/__setShaderTime) and the rAF loop must not overwrite it with the
  // free-running clock. null = follow the running clock.
  private frozenTime: number | null = null;

  constructor(container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.clock = new THREE.Clock();

    // 默认 1x1 白色纹理（未绑定 channel 时使用，避免采样报错）
    const canvas = document.createElement("canvas");
    canvas.width = 1; canvas.height = 1;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 1, 1);
    this.defaultTexture = new THREE.CanvasTexture(canvas);
  }

  compileShader(fragShaderSource: string): { success: boolean; error: string | null } {
    const fullFrag = wrapFragmentShader(fragShaderSource);
    let compileError: string | null = null;

    // 清除旧的 mesh
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.ShaderMaterial).dispose();
      this.mesh = null;
    }

    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: fullFrag,
      uniforms: {
        u_time: { value: 0.0 },
        u_resolution: { value: new THREE.Vector2(
          this.renderer.domElement.width,
          this.renderer.domElement.height
        )},
        u_mouse: { value: this.mousePos },
        // 纹理通道：iChannel0 = 系统backdrop, iChannel1 = 用户纹理
        iChannel0: { value: this.backdropTexture || this.defaultTexture },
        iChannel1: { value: this.userTexture || this.defaultTexture },
      },
    });

    const geometry = new THREE.PlaneGeometry(2, 2);
    this.mesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.mesh);

    const debug = this.renderer.debug as unknown as {
      checkShaderErrors: boolean;
      onShaderError: (
        | ((gl: WebGLRenderingContext, program: WebGLProgram, vertexShader: WebGLShader, fragmentShader: WebGLShader) => void)
        | null
      );
    };
    const previousCheckShaderErrors = debug.checkShaderErrors;
    const previousOnShaderError = debug.onShaderError;
    debug.checkShaderErrors = true;
    debug.onShaderError = (gl, program, vertexShader, fragmentShader) => {
      const logs = [
        gl.getProgramInfoLog(program),
        gl.getShaderInfoLog(vertexShader),
        gl.getShaderInfoLog(fragmentShader),
      ]
        .map((log) => log?.trim())
        .filter(Boolean);
      compileError = logs.join("\n") || "Shader compile error";
    };

    // 尝试编译
    try {
      this.renderer.render(this.scene, this.camera);
    } catch (err) {
      compileError = err instanceof Error ? err.message : "Shader compile error";
    } finally {
      debug.checkShaderErrors = previousCheckShaderErrors;
      debug.onShaderError = previousOnShaderError;
    }

    if (compileError) {
      this.scene.remove(this.mesh);
      geometry.dispose();
      material.dispose();
      this.mesh = null;
      return { success: false, error: compileError };
    }

    return { success: true, error: null };
  }

  startRendering() {
    this.clock.start();
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      if (this.mesh) {
        const mat = (this.mesh.material as THREE.ShaderMaterial);
        // Respect a frozen time so setTime()/__setShaderTime survives across
        // frames; otherwise track the running clock.
        mat.uniforms.u_time.value = nextShaderTime(
          this.frozenTime,
          this.clock.getElapsedTime(),
        );
        mat.uniforms.u_resolution.value.set(
          this.renderer.domElement.width,
          this.renderer.domElement.height
        );
      }
      this.renderer.render(this.scene, this.camera);
      
      // Call frame callback for FPS tracking
      if (this.onFrameCallback) {
        this.onFrameCallback();
      }
    };
    animate();
  }
  
  setFrameCallback(callback: () => void) {
    this.onFrameCallback = callback;
  }

  stopRendering() {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  updateMouse(x: number, y: number) {
    const canvas = this.renderer.domElement;
    this.mousePos.set(x, canvas.height - y);
  }

  setTime(t: number) {
    // 设置渲染时间（供 Playwright 截图时控制动画帧）。
    // Freeze the time so the rAF loop can't overwrite it with the running
    // clock on the next frame — required for deterministic screenshots.
    this.frozenTime = t;
    if (this.mesh) {
      const mat = (this.mesh.material as THREE.ShaderMaterial);
      mat.uniforms.u_time.value = t;
    }
    this.renderer.render(this.scene, this.camera);
  }

  /** Resume the free-running clock after a setTime() freeze. */
  unfreezeTime() {
    this.frozenTime = null;
  }

  resize(width: number, height: number) {
    this.renderer.setSize(width, height);
    if (this.mesh) {
      const mat = (this.mesh.material as THREE.ShaderMaterial);
      mat.uniforms.u_resolution.value.set(width * devicePixelRatio, height * devicePixelRatio);
    }
  }

  dispose() {
    this.stopRendering();

    // Free GPU resources that the renderer's own dispose() does not own:
    // the current mesh, and every texture this instance allocated. Leaking
    // these (especially the default CanvasTexture) contributes to WebGL
    // context/texture exhaustion over long sessions.
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.ShaderMaterial).dispose();
      this.mesh = null;
    }
    this.defaultTexture.dispose();
    this.backdropTexture?.dispose();
    this.backdropTexture = null;
    this.userTexture?.dispose();
    this.userTexture = null;

    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
