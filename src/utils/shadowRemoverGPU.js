import * as THREE from 'three'

const VERTEX_SHADER = `
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const FRAGMENT_SHADER = `
uniform sampler2D tDiffuse;
uniform float uStrength;
uniform float uThreshold;
uniform float uSoftness;
uniform float uMidtoneProtection;
varying vec2 vUv;

void main() {
  vec4 texel = texture2D(tDiffuse, vUv);
  float luminance = dot(texel.rgb, vec3(0.2126, 0.7152, 0.0722));
  float shadowStart = max(0.0, uThreshold - uSoftness);
  float shadowEnd = min(1.0, uThreshold + uSoftness * 1.35);
  float shadowMask = 1.0 - smoothstep(shadowStart, shadowEnd, luminance);
  shadowMask = pow(shadowMask, 2.2);

  float protectStart = mix(0.18, 0.42, uMidtoneProtection);
  float protectEnd = mix(0.48, 0.82, uMidtoneProtection);
  float highlightGuard = smoothstep(protectStart, protectEnd, luminance);
  float effectiveMask = shadowMask * (1.0 - highlightGuard);

  float targetLuminance = min(1.0, luminance + effectiveMask * uStrength * (0.22 + (1.0 - luminance) * 0.38));
  float safeLuminance = max(luminance, 0.0001);
  float liftRatio = targetLuminance / safeLuminance;
  vec3 lifted = mix(texel.rgb, clamp(texel.rgb * liftRatio, 0.0, 1.0), effectiveMask);

  float grayscale = dot(lifted, vec3(0.299, 0.587, 0.114));
  float neutralize = effectiveMask * uStrength * 0.08;
  lifted = mix(lifted, vec3(grayscale), neutralize);

  gl_FragColor = vec4(clamp(lifted, 0.0, 1.0), texel.a);
}
`

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function createCanvas(width, height) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

function cloneCanvas(sourceCanvas) {
  if (!sourceCanvas) return null
  const cloned = createCanvas(sourceCanvas.width, sourceCanvas.height)
  cloned.getContext('2d').drawImage(sourceCanvas, 0, 0)
  return cloned
}

function normalizeSettings(settings = {}) {
  return {
    strength: clamp((Number(settings.strength) || 0) / 100, 0, 1),
    threshold: clamp((Number(settings.threshold) || 0) / 100, 0, 1),
    softness: clamp((Number(settings.softness) || 0) / 100, 0.01, 1),
    midtoneProtection: clamp((Number(settings.midtoneProtection) || 0) / 100, 0, 1)
  }
}

function applyShadowRemoverCpu(sourceCanvas, settings) {
  const normalized = normalizeSettings(settings)
  if (normalized.strength <= 0) {
    return cloneCanvas(sourceCanvas)
  }

  const outputCanvas = createCanvas(sourceCanvas.width, sourceCanvas.height)
  const context = outputCanvas.getContext('2d')
  context.drawImage(sourceCanvas, 0, 0)

  const imageData = context.getImageData(0, 0, outputCanvas.width, outputCanvas.height)
  const data = imageData.data

  for (let index = 0; index < data.length; index += 4) {
    const alpha = data[index + 3]
    if (alpha === 0) {
      continue
    }

    const red = data[index] / 255
    const green = data[index + 1] / 255
    const blue = data[index + 2] / 255
    const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722

    const shadowStart = Math.max(0, normalized.threshold - normalized.softness)
    const shadowEnd = Math.min(1, normalized.threshold + normalized.softness * 1.35)
    let shadowMask = 1 - smoothstep(shadowStart, shadowEnd, luminance)
    shadowMask = Math.pow(shadowMask, 2.2)

    const protectStart = 0.18 + (0.42 - 0.18) * normalized.midtoneProtection
    const protectEnd = 0.48 + (0.82 - 0.48) * normalized.midtoneProtection
    const highlightGuard = smoothstep(protectStart, protectEnd, luminance)
    const effectiveMask = shadowMask * (1 - highlightGuard)

    const targetLuminance = Math.min(1, luminance + effectiveMask * normalized.strength * (0.22 + (1 - luminance) * 0.38))
    const safeLuminance = Math.max(luminance, 0.0001)
    const liftRatio = targetLuminance / safeLuminance

    let nextRed = red + (clamp(red * liftRatio, 0, 1) - red) * effectiveMask
    let nextGreen = green + (clamp(green * liftRatio, 0, 1) - green) * effectiveMask
    let nextBlue = blue + (clamp(blue * liftRatio, 0, 1) - blue) * effectiveMask

    const grayscale = nextRed * 0.299 + nextGreen * 0.587 + nextBlue * 0.114
    const neutralize = effectiveMask * normalized.strength * 0.08
    nextRed = nextRed + (grayscale - nextRed) * neutralize
    nextGreen = nextGreen + (grayscale - nextGreen) * neutralize
    nextBlue = nextBlue + (grayscale - nextBlue) * neutralize

    data[index] = Math.round(clamp(nextRed, 0, 1) * 255)
    data[index + 1] = Math.round(clamp(nextGreen, 0, 1) * 255)
    data[index + 2] = Math.round(clamp(nextBlue, 0, 1) * 255)
  }

  context.putImageData(imageData, 0, 0)
  return outputCanvas
}

function smoothstep(edge0, edge1, value) {
  if (edge0 === edge1) {
    return value < edge0 ? 0 : 1
  }

  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return t * t * (3 - 2 * t)
}

class ShadowRemoverRenderer {
  constructor() {
    const canvas = document.createElement('canvas')
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      canvas,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true
    })
    this.renderer.setPixelRatio(1)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.NoToneMapping

    this.scene = new THREE.Scene()
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        uStrength: { value: 0 },
        uThreshold: { value: 0.5 },
        uSoftness: { value: 0.2 },
        uMidtoneProtection: { value: 0.5 }
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false
    })

    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material)
    this.scene.add(this.mesh)
    this.width = 0
    this.height = 0
  }

  render(sourceCanvas, settings) {
    if (!sourceCanvas?.width || !sourceCanvas?.height) {
      return null
    }

    if (this.width !== sourceCanvas.width || this.height !== sourceCanvas.height) {
      this.width = sourceCanvas.width
      this.height = sourceCanvas.height
      this.renderer.setSize(this.width, this.height, false)
    }

    const normalized = normalizeSettings(settings)
    if (normalized.strength <= 0) {
      return cloneCanvas(sourceCanvas)
    }

    const texture = new THREE.CanvasTexture(sourceCanvas)
    texture.colorSpace = THREE.NoColorSpace
    texture.minFilter = THREE.LinearFilter
    texture.magFilter = THREE.LinearFilter

    this.material.uniforms.tDiffuse.value = texture
    this.material.uniforms.uStrength.value = normalized.strength
    this.material.uniforms.uThreshold.value = normalized.threshold
    this.material.uniforms.uSoftness.value = normalized.softness
    this.material.uniforms.uMidtoneProtection.value = normalized.midtoneProtection

    this.renderer.render(this.scene, this.camera)

    const outputCanvas = createCanvas(this.width, this.height)
    outputCanvas.getContext('2d').drawImage(this.renderer.domElement, 0, 0, this.width, this.height)

    texture.dispose()
    this.material.uniforms.tDiffuse.value = null
    return outputCanvas
  }

  dispose() {
    this.mesh.geometry.dispose()
    this.material.dispose()
    this.renderer.dispose()
  }
}

let rendererInstance = null

function getRenderer() {
  if (!rendererInstance) {
    rendererInstance = new ShadowRemoverRenderer()
  }

  return rendererInstance
}

export function applyShadowRemoverToCanvas(sourceCanvas, settings) {
  if (!sourceCanvas) {
    return { canvas: null, mode: 'gpu', fallbackReason: '' }
  }

  const normalized = normalizeSettings(settings)
  if (normalized.strength <= 0) {
    return { canvas: cloneCanvas(sourceCanvas), mode: 'bypass', fallbackReason: '' }
  }

  try {
    const renderer = getRenderer()
    const canvas = renderer.render(sourceCanvas, settings)
    if (canvas) {
      return { canvas, mode: 'gpu', fallbackReason: '' }
    }
  } catch (error) {
    const canvas = applyShadowRemoverCpu(sourceCanvas, settings)
    return {
      canvas,
      mode: 'cpu',
      fallbackReason: error instanceof Error ? error.message : 'WebGL initialization failed.'
    }
  }

  return { canvas: applyShadowRemoverCpu(sourceCanvas, settings), mode: 'cpu', fallbackReason: 'GPU rendering returned no output.' }
}

export function disposeShadowRemoverRenderer() {
  rendererInstance?.dispose()
  rendererInstance = null
}