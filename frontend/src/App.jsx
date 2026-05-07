import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const CAPTURE_MS = 1500

// Guide box: centered square, 65% of the shorter dimension
const getBox = (w, h) => {
  const s = Math.min(w, h) * 0.65
  return { x: (w - s) / 2, y: (h - s) / 2, s }
}

function App() {
  const [status,     setStatus]     = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [isScanning, setIsScanning] = useState(false)

  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const rafRef     = useRef(null)
  const streamRef  = useRef(null)
  const timerRef   = useRef(null)
  const sendingRef = useRef(false)

  // --- Draw loop: mirror video + animated guide box ---
  const drawLoop = useCallback(() => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2 || !video.videoWidth) {
      rafRef.current = requestAnimationFrame(drawLoop)
      return
    }
    const W = video.videoWidth, H = video.videoHeight
    canvas.width = W; canvas.height = H
    const ctx = canvas.getContext('2d')

    // Mirror draw
    ctx.save()
    ctx.scale(-1, 1)
    ctx.translate(-W, 0)
    ctx.drawImage(video, 0, 0)
    ctx.restore()

    // Dim outside box
    const { x, y, s } = getBox(W, H)
    ctx.fillStyle = 'rgba(0,0,0,0.45)'
    ctx.fillRect(0, 0, W, y)
    ctx.fillRect(0, y + s, W, H - y - s)
    ctx.fillRect(0, y, x, s)
    ctx.fillRect(x + s, y, W - x - s, s)

    // Pulsing border
    const alpha = 0.55 + 0.45 * Math.sin(Date.now() / 450)
    ctx.strokeStyle = `rgba(192,132,252,${alpha})`
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, s, s)

    // Corner accents
    const cl = 22
    ctx.strokeStyle = '#c084fc'
    ctx.lineWidth = 4
    ctx.lineCap = 'round'
    const drawCorner = (cx, cy, sx, sy) => {
      ctx.beginPath(); ctx.moveTo(cx + sx * cl, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + sy * cl); ctx.stroke()
    }
    drawCorner(x,     y,     1,  1)
    drawCorner(x + s, y,    -1,  1)
    drawCorner(x,     y + s, 1, -1)
    drawCorner(x + s, y + s,-1, -1)

    rafRef.current = requestAnimationFrame(drawLoop)
  }, [])

  // --- Capture guide box region → Flask ---
  const captureAndPredict = useCallback(async () => {
    if (sendingRef.current) return
    const video = videoRef.current
    if (!video || video.readyState < 2 || !video.videoWidth) return

    sendingRef.current = true
    setIsScanning(true)
    try {
      const { x, y, s } = getBox(video.videoWidth, video.videoHeight)
      const crop = document.createElement('canvas')
      crop.width = s; crop.height = s
      // Draw non-mirrored video crop (matches training data orientation)
      crop.getContext('2d').drawImage(video, x, y, s, s, 0, 0, s, s)

      const blob = await new Promise(r => crop.toBlob(r, 'image/jpeg', 0.9))
      if (!blob) return

      const fd = new FormData()
      fd.append('file', blob, 'frame.jpg')
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001'
      const res  = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd })
      const data = await res.json()
      if (data.status === 'success') {
        setPrediction(data.prediction)
        setConfidence(data.confidence)
      }
    } catch (err) {
      console.error('Predict error:', err)
    } finally {
      sendingRef.current = false
      setIsScanning(false)
    }
  }, [])

  // --- Start camera ---
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      })
      streamRef.current = stream
      videoRef.current.srcObject = stream
      await new Promise(r => { videoRef.current.onloadedmetadata = r })
      await videoRef.current.play()
      setStatus('running')
      rafRef.current = requestAnimationFrame(drawLoop)
      timerRef.current = setInterval(captureAndPredict, CAPTURE_MS)
    } catch (err) {
      console.error('Camera error:', err)
      setStatus('error')
    }
  }

  // --- Stop camera ---
  const stopCamera = () => {
    cancelAnimationFrame(rafRef.current)
    clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setStatus('idle')
    setPrediction(null)
    setConfidence(null)
    setIsScanning(false)
  }

  useEffect(() => () => { stopCamera() }, [])

  return (
    <div className="app-container">
      <h1>MSL Translator</h1>

      <p className={`status-badge ${status === 'running' ? 'active' : ''}`}>
        {status === 'idle'    && '👋 Ready — click Start Camera'}
        {status === 'running' && (isScanning ? '🔍 Analysing…' : '🎥 Place your hand in the box')}
        {status === 'error'   && '❌ Camera access failed'}
      </p>

      <div className={`camera-wrapper ${status === 'running' ? 'live' : ''}`}>
        {/* Video kept rendered (not display:none) so browser maintains frame pipeline */}
        <video ref={videoRef} autoPlay playsInline muted
          style={{ position:'absolute', inset:0, width:'100%', height:'100%', opacity:0, pointerEvents:'none' }}
        />
        <canvas ref={canvasRef} className={`camera-canvas ${status !== 'running' ? 'hidden' : ''}`} />
        {status !== 'running' && (
          <div className="camera-placeholder">
            <div className="placeholder-icon">🤟</div>
            <p>Camera feed will appear here</p>
          </div>
        )}
      </div>

      <div className="controls">
        {status !== 'running'
          ? <button className="btn-primary" onClick={startCamera} disabled={status === 'error'}>Start Camera</button>
          : <button className="btn-stop"    onClick={stopCamera}>Stop Camera</button>
        }
      </div>

      {prediction !== null ? (
        <div className={`result-section ${isScanning ? 'dimmed' : ''}`}>
          <div className="letter-display">{prediction}</div>
          {confidence !== null && (
            <p className="confidence">
              Confidence: <strong>{confidence}%</strong>
              {isScanning && <span className="scan-indicator"> · Scanning…</span>}
            </p>
          )}
        </div>
      ) : status === 'running' && (
        <div className="result-section empty">
          <p>Hold your sign still inside the box…</p>
        </div>
      )}
    </div>
  )
}

export default App