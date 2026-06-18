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

/**
 * Pure decision for what to do when a WebGL context is restored after a
 * `webglcontextlost` event (Bug 2: context-loss recovery).
 *
 * After restoration the GL program/buffers are gone, so the renderer must:
 *  - recompile the last fragment source — but only if one was ever compiled
 *    (a non-empty saved source), otherwise there is nothing to restore;
 *  - resume the rAF render loop — but only if it was actively rendering when
 *    the context was lost, so we don't spin up a loop the caller had paused.
 */
export function restoreActions(
  wasRenderingBeforeLoss: boolean,
  savedFragSource: string | null,
): { shouldRecompile: boolean; shouldResume: boolean } {
  return {
    shouldRecompile: !!savedFragSource,
    shouldResume: wasRenderingBeforeLoss,
  };
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
  // Bug 2: WebGL context-loss recovery state. We keep the last compiled
  // fragment source so a restored context can recompile it, and we remember
  // whether the rAF loop was running so restore can resume it.
  private lastFragSource: string | null = null;
  private contextLost = false;
  private wasRenderingBeforeLoss = false;
  private readonly onContextLost: (event: Event) => void;
  private readonly onContextRestored: () => void;

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

    // Bug 2: handle WebGL context loss/restore. Context loss is common under
    // context exhaustion; without these handlers a lost context silently
    // black-screens with no recovery. preventDefault() on loss tells the
    // browser the context is recoverable so it will fire a restore event.
    this.onContextLost = (event: Event) => {
      event.preventDefault();
      this.contextLost = true;
      // Remember whether we were actively rendering so restore can resume it,
      // then pause the loop (its GL program is now invalid).
      this.wasRenderingBeforeLoss = this.animationId !== null;
      this.stopRendering();
    };
    this.onContextRestored = () => {
      this.contextLost = false;
      const { shouldRecompile, shouldResume } = restoreActions(
        this.wasRenderingBeforeLoss,
        this.lastFragSource,
      );
      // GL resources (programs/buffers/textures) were destroyed with the old
      // context; recompile the last shader to rebuild the mesh + material.
      if (shouldRecompile && this.lastFragSource !== null) {
        this.compileShader(this.lastFragSource);
      }
      if (shouldResume) {
        this.startRendering();
      }
    };
    this.renderer.domElement.addEventListener("webglcontextlost", this.onContextLost, false);
    this.renderer.domElement.addEventListener("webglcontextrestored", this.onContextRestored, false);
  }

  compileShader(fragShaderSource: string): { success: boolean; error: string | null } {
    // Remember the source so a restored WebGL context (Bug 2) can recompile it.
    this.lastFragSource = fragShaderSource;
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
    // Idempotent: never run two concurrent rAF loops (e.g. a restore that
    // resumes while one is already scheduled, or a redundant caller).
    if (this.animationId !== null) {
      return;
    }
    // While the context is lost rendering would throw; resume happens via the
    // webglcontextrestored handler instead.
    if (this.contextLost) {
      this.wasRenderingBeforeLoss = true;
      return;
    }
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

    // Bug 2: detach context-loss listeners before tearing down the canvas so
    // they cannot fire (or leak) after disposal.
    this.renderer.domElement.removeEventListener("webglcontextlost", this.onContextLost, false);
    this.renderer.domElement.removeEventListener("webglcontextrestored", this.onContextRestored, false);

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
