import { useState, useEffect, useRef } from 'react'
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import './App.css'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prediction, setPrediction] = useState('')
  const [loading, setLoading] = useState(false)
  const [landmarker, setLandmarker] = useState(null)

  // References to interact with the raw DOM elements
  const imageRef = useRef(null)
  const canvasRef = useRef(null)

  // --- 1. INITIALIZE MEDIAPIPE ON LOAD ---
  useEffect(() => {
    const initializeMediaPipe = async () => {
      // Load the WebAssembly core from Google's CDN
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      )
      // Load the specific Hand Tracking model
      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU" // Use the user's local GPU for tracking
        },
        runningMode: "IMAGE",
        numHands: 1
      })
      setLandmarker(handLandmarker)
    }
    initializeMediaPipe()
  }, [])

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
      setPrediction('')
    }
  }

  // --- 2. THE PROCESSING PIPELINE ---
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!image || !landmarker || !imageRef.current || !canvasRef.current) return

    setLoading(true)

    // A. Scan the image for hands
    const result = landmarker.detect(imageRef.current)

    if (result.handLandmarks.length === 0) {
      setPrediction("Error: No hand detected in the image.")
      setLoading(false)
      return
    }

    // B. Calculate Bounding Box
    const landmarks = result.handLandmarks[0]
    const imgWidth = imageRef.current.naturalWidth
    const imgHeight = imageRef.current.naturalHeight

    // Find the extreme edges of the hand
    let minX = Math.min(...landmarks.map(l => l.x)) * imgWidth
    let maxX = Math.max(...landmarks.map(l => l.x)) * imgWidth
    let minY = Math.min(...landmarks.map(l => l.y)) * imgHeight
    let maxY = Math.max(...landmarks.map(l => l.y)) * imgHeight

    // Add 40 pixels of padding so we don't cut off fingertips
    const padding = 40
    minX = Math.max(0, minX - padding)
    minY = Math.max(0, minY - padding)
    maxX = Math.min(imgWidth, maxX + padding)
    maxY = Math.min(imgHeight, maxY + padding)

    const cropWidth = maxX - minX
    const cropHeight = maxY - minY

    // C. Crop the Image using Canvas
    const canvas = canvasRef.current
    canvas.width = cropWidth
    canvas.height = cropHeight
    const ctx = canvas.getContext('2d')
    
    ctx.drawImage(
      imageRef.current,
      minX, minY, cropWidth, cropHeight, // Source coordinates (the crop)
      0, 0, cropWidth, cropHeight        // Destination coordinates (the canvas)
    )

    // D. Send the Cropped Data to PyTorch
    canvas.toBlob(async (blob) => {
      const formData = new FormData()
      // Package the blob exactly like a normal file upload
      formData.append('file', blob, 'cropped_hand.jpg') 

      try {
        const response = await fetch('http://127.0.0.1:5001/predict', {
          method: 'POST',
          body: formData,
        })
        const data = await response.json()
        
        if (data.status === 'success') {
          setPrediction(data.prediction)
        } else {
          setPrediction('Error: ' + data.error)
        }
      } catch (error) {
        console.error('Error connecting to AI:', error)
        setPrediction('Failed to connect to backend.')
      } finally {
        setLoading(false)
      }
    }, 'image/jpeg')
  }

  return (
    <div className="app-container">
      <h1>MSL Translator</h1>
      <p>{landmarker ? "AI System Ready. Upload an image." : "Loading AI Models..."}</p>
      
      <form onSubmit={handleSubmit} className="upload-form">
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit" disabled={!image || loading || !landmarker}>
          {loading ? 'Processing...' : 'Translate Sign'}
        </button>
      </form>
      
      {/* Hidden canvas used purely for math and cropping */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {preview && (
        <div className="preview-section" style={{ marginTop: '20px' }}>
          {/* We must render the image so MediaPipe can read its pixels */}
          <img 
            ref={imageRef}
            src={preview} 
            alt="Uploaded sign" 
            style={{ maxWidth: '300px', borderRadius: '8px' }} 
            crossOrigin="anonymous"
          />
        </div>
      )}
      
      {prediction && (
        <div className="result-section" style={{ marginTop: '20px' }}>
          <h2>Detected Letter: <span style={{ color: '#4CAF50' }}>{prediction}</span></h2>
        </div>
      )}
    </div>
  )
}

export default App