<template>
  <div class="app-container">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <h1 class="header-title">Real-Time Object Counter</h1>
        <p class="header-subtitle">Using YOLOv8 + FastAPI</p>
      </div>
    </header>

    <!-- Main Content - 3 Panel Layout -->
    <main class="main-content">
      <!-- Left Panel - Controls & Options -->
      <aside class="left-panel">
        <div class="panel-header">
          <h2><i class="fas fa-sliders-h"></i> Controls & Options</h2>
        </div>
        
        <!-- Mode Selection -->
        <div class="control-section">
          <h3>Mode Selection</h3>
          <div class="radio-group">
            <label class="radio-option" :class="{ active: mode === 'webcam' }">
              <input type="radio" v-model="mode" value="webcam" @change="handleModeChange">
              <i class="fas fa-video"></i>
              <span>Live Webcam</span>
            </label>
            <label class="radio-option" :class="{ active: mode === 'image' }">
              <input type="radio" v-model="mode" value="image" @change="handleModeChange">
              <i class="fas fa-image"></i>
              <span>Upload Image</span>
            </label>
            <label class="radio-option" :class="{ active: mode === 'video' }">
              <input type="radio" v-model="mode" value="video" @change="handleModeChange">
              <i class="fas fa-film"></i>
              <span>Upload Video</span>
            </label>
          </div>
        </div>

        <!-- File Upload (for Image/Video modes) -->
        <div class="control-section" v-if="mode !== 'webcam'">
          <h3>Upload File</h3>
          <div class="upload-area" 
               @dragover.prevent="dragOver" 
               @dragleave.prevent="dragLeave" 
               @drop.prevent="dropFile"
               :class="{ 'drag-over': isDragging }">
            <input type="file" 
                   :accept="mode === 'image' ? 'image/*' : 'video/*'" 
                   @change="handleFileUpload"
                   ref="fileInput"
                   style="display: none">
            <div class="upload-content" @click="$refs.fileInput.click()">
              <i :class="mode === 'image' ? 'fas fa-cloud-upload-alt' : 'fas fa-video'"></i>
              <p>Click or drag to upload {{ mode }}</p>
              <span class="upload-hint">{{ selectedFileName || 'No file selected' }}</span>
            </div>
          </div>
        </div>

        <!-- Object Filter -->
        <div class="control-section">
          <h3>Object Filter <span class="optional">(Optional)</span></h3>
          <div class="object-filters">
            <label v-for="obj in availableObjects" :key="obj" class="filter-checkbox">
              <input type="checkbox" v-model="selectedObjects" :value="obj">
              <span class="checkbox-custom"></span>
              <span class="filter-label">{{ obj }}</span>
            </label>
          </div>
        </div>

        <!-- Settings -->
        <div class="control-section">
          <h3>Settings</h3>
          
          <!-- Confidence Threshold -->
          <div class="setting-item">
            <label class="setting-label">
              <span>Confidence Threshold</span>
              <span class="setting-value">{{ confidenceThreshold.toFixed(1) }}</span>
            </label>
            <input type="range" 
                   v-model.number="confidenceThreshold" 
                   min="0.1" 
                   max="1.0" 
                   step="0.1"
                   class="slider">
          </div>

          <!-- Bounding Boxes Toggle -->
          <div class="setting-item">
            <label class="toggle-label">
              <span>Show Bounding Boxes</span>
              <div class="toggle-switch">
                <input type="checkbox" v-model="showBoundingBoxes">
                <span class="toggle-slider"></span>
              </div>
            </label>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="control-section buttons-section">
          <button class="btn btn-primary" 
                  @click="startDetection" 
                  :disabled="isProcessing || (mode !== 'webcam' && !selectedFileName)">
            <i class="fas fa-play"></i> Start Detection
          </button>
          <button class="btn btn-secondary" 
                  @click="stopDetection" 
                  :disabled="!isProcessing">
            <i class="fas fa-stop"></i> Stop Detection
          </button>
        </div>
      </aside>

      <!-- Center Panel - Live Feed / Output -->
      <section class="center-panel">
        <div class="panel-header">
          <h2><i class="fas fa-desktop"></i> Live Feed</h2>
          <div class="live-indicator" v-if="isProcessing">
            <span class="pulse"></span> Live
          </div>
        </div>
        
        <div class="video-container" :class="{ 'has-overlay': showBoundingBoxes && detections.length > 0 }">
          <!-- Webcam Video -->
          <video v-show="mode === 'webcam' && showVideo" 
                 ref="webcamVideo" 
                 autoplay 
                 playsinline
                 muted></video>
          
          <!-- Image Display -->
          <img v-show="mode === 'image' && uploadedImage" 
               :src="uploadedImage" 
               alt="Uploaded Image"
               ref="uploadedImageEl">
          
          <!-- Video Display -->
          <video v-show="mode === 'video' && showVideo" 
                 ref="uploadedVideo" 
                 controls
                 :src="uploadedVideoUrl"></video>

          <!-- Placeholder when no feed -->
          <div v-if="!showVideo && !uploadedImage" class="video-placeholder">
            <i class="fas fa-video-slash"></i>
            <p>Select a mode and start detection</p>
          </div>

          <!-- Bounding Boxes Overlay -->
          <div class="bounding-boxes-overlay" v-if="showBoundingBoxes && detections.length > 0">
            <div v-for="(detection, index) in detections" 
                 :key="index"
                 class="bounding-box"
                 :style="{
                   left: detection.x + 'px',
                   top: detection.y + 'px',
                   width: detection.width + 'px',
                   height: detection.height + 'px',
                   borderColor: detection.color
                 }">
              <span class="box-label" :style="{ backgroundColor: detection.color }">
                {{ detection.class }} ({{ (detection.confidence * 100).toFixed(0) }}%)
              </span>
            </div>
          </div>

          <!-- Total Count Overlay -->
          <div class="count-overlay" v-if="detections.length > 0">
            <span class="count-badge">Total: {{ totalObjectCount }}</span>
          </div>
        </div>
      </section>

      <!-- Right Panel - Object Counts & Stats -->
      <aside class="right-panel">
        <div class="panel-header">
          <h2><i class="fas fa-chart-bar"></i> Object Counts</h2>
        </div>

        <!-- Total Count -->
        <div class="total-count-section">
          <div class="total-count-card">
            <span class="total-label">Total Objects</span>
            <span class="total-value">{{ totalObjectCount }}</span>
          </div>
        </div>

        <!-- Object Counts Table -->
        <div class="counts-table-section">
          <h3>Count by Object</h3>
          <div class="counts-table" v-if="objectCounts.length > 0">
            <div class="table-header">
              <span>Object</span>
              <span>Count</span>
            </div>
            <div v-for="(item, index) in objectCounts" 
                 :key="index" 
                 class="table-row"
                 :style="{ borderLeftColor: item.color }">
              <span class="object-name">
                <i class="fas fa-circle" :style="{ color: item.color }"></i>
                {{ item.name }}
              </span>
              <span class="object-count">{{ item.count }}</span>
            </div>
          </div>
          <div class="empty-state" v-else>
            <i class="fas fa-inbox"></i>
            <p>No objects detected yet</p>
          </div>
        </div>

        <!-- Bar Chart -->
        <div class="chart-section" v-if="objectCounts.length > 0">
          <h3>Visualization</h3>
          <div class="bar-chart">
            <div v-for="(item, index) in objectCounts" 
                 :key="index" 
                 class="bar-item">
              <span class="bar-label">{{ item.name }}</span>
              <div class="bar-container">
                <div class="bar" 
                     :style="{ 
                       width: (item.count / maxCount * 100) + '%',
                       backgroundColor: item.color 
                     }"></div>
              </div>
              <span class="bar-value">{{ item.count }}</span>
            </div>
          </div>
        </div>
      </aside>
    </main>

    <!-- Footer -->
    <footer class="footer">
      <div class="footer-content">
        <p class="footer-credit">
          <i class="fas fa-code"></i> Real-Time Object Counting Demo
        </p>
        <div class="footer-links">
          <a href="https://github.com" target="_blank" title="GitHub">
            <i class="fab fa-github"></i> GitHub
          </a>
          <a href="https://linkedin.com" target="_blank" title="LinkedIn">
            <i class="fab fa-linkedin"></i> LinkedIn
          </a>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

// State
const mode = ref('webcam')
const selectedFileName = ref('')
const selectedFile = ref(null)
const selectedObjects = ref([])
const confidenceThreshold = ref(0.5)
const showBoundingBoxes = ref(true)
const isProcessing = ref(false)
const showVideo = ref(false)
const uploadedImage = ref('')
const uploadedVideoUrl = ref('')

// Detection state
const detections = ref([])
const objectCounts = ref([])

// Refs
const webcamVideo = ref(null)
const uploadedVideo = ref(null)
const fileInput = ref(null)
const uploadedImageEl = ref(null)

// Media stream
let mediaStream = null

// Available objects for filtering
const availableObjects = [
  'Person', 'Car', 'Bicycle', 'Motorbike', 'Bus', 'Truck',
  'Dog', 'Cat', 'Bird', 'Chair', 'Bottle', 'Cup', ' Bowl',
  'Apple', 'Banana', 'Orange', 'Book', 'Phone', 'Laptop'
]

// Color palette for different object classes
const colorPalette = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
  '#F8B500', '#00CED1', '#FF69B4', '#32CD32', '#FF4500'
]

// Drag state
const isDragging = ref(false)

// Computed
const totalObjectCount = computed(() => {
  return detections.value.length
})

const maxCount = computed(() => {
  if (objectCounts.value.length === 0) return 1
  return Math.max(...objectCounts.value.map(c => c.count))
})

// Methods
const handleModeChange = () => {
  stopDetection()
  selectedFileName.value = ''
  selectedFile.value = null
  uploadedImage.value = ''
  uploadedVideoUrl.value = ''
}

const handleFileUpload = (event) => {
  const file = event.target.files[0]
  if (file) {
    processFile(file)
  }
}

const dragOver = (event) => {
  isDragging.value = true
}

const dragLeave = (event) => {
  isDragging.value = false
}

const dropFile = (event) => {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file) {
    processFile(file)
  }
}

const processFile = (file) => {
  selectedFile.value = file
  selectedFileName.value = file.name
  
  if (mode.value === 'image') {
    const reader = new FileReader()
    reader.onload = (e) => {
      uploadedImage.value = e.target.result
    }
    reader.readAsDataURL(file)
  } else if (mode.value === 'video') {
    const url = URL.createObjectURL(file)
    uploadedVideoUrl.value = url
  }
}

const startDetection = async () => {
  isProcessing.value = true
  detections.value = []
  objectCounts.value = []
  
  if (mode.value === 'webcam') {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      if (webcamVideo.value) {
        webcamVideo.value.srcObject = mediaStream
        showVideo.value = true
      }
      // Start detection loop
      startWebcamDetection()
    } catch (error) {
      console.error('Error accessing webcam:', error)
      alert('Could not access webcam. Please check permissions.')
      isProcessing.value = false
    }
  } else if (mode.value === 'video') {
    showVideo.value = true
    startVideoDetection()
  } else if (mode.value === 'image' && uploadedImage.value) {
    // Process single image
    processImage()
  }
}

const stopDetection = () => {
  isProcessing.value = false
  
  // Stop webcam
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop())
    mediaStream = null
  }
  
  showVideo.value = false
  detections.value = []
  objectCounts.value = []
}

// Simulated detection for demo (replace with actual API calls)
const startWebcamDetection = () => {
  if (!isProcessing.value) return
  
  // Simulate detection results
  simulateDetection()
  
  // Continue detection loop
  if (isProcessing.value) {
    setTimeout(startWebcamDetection, 1000)
  }
}

const startVideoDetection = () => {
  if (!uploadedVideo.value) return
  
  uploadedVideo.value.onplay = () => {
    const detectFrame = () => {
      if (!isProcessing.value || uploadedVideo.value.paused || uploadedVideo.value.ended) {
        return
      }
      simulateDetection()
      requestAnimationFrame(detectFrame)
    }
    detectFrame()
  }
}

const processImage = () => {
  // Simulate image detection
  simulateDetection()
}

const simulateDetection = () => {
  // Generate random detections for demo
  const classes = ['Person', 'Car', 'Bottle', 'Chair', 'Dog', 'Cup']
  const numDetections = Math.floor(Math.random() * 5) + 1
  const newDetections = []
  
  for (let i = 0; i < numDetections; i++) {
    const className = classes[Math.floor(Math.random() * classes.length)]
    const colorIndex = classes.indexOf(className) % colorPalette.length
    
    newDetections.push({
      class: className,
      confidence: Math.random() * 0.5 + 0.5,
      x: Math.random() * 400 + 50,
      y: Math.random() * 300 + 50,
      width: Math.random() * 100 + 50,
      height: Math.random() * 100 + 50,
      color: colorPalette[colorIndex]
    })
  }
  
  detections.value = newDetections
  
  // Update object counts
  updateObjectCounts()
}

const updateObjectCounts = () => {
  const counts = {}
  
  detections.value.forEach(detection => {
    if (selectedObjects.value.length === 0 || selectedObjects.value.includes(detection.class)) {
      if (!counts[detection.class]) {
        counts[detection.class] = 0
      }
      counts[detection.class]++
    }
  })
  
  objectCounts.value = Object.entries(counts).map(([name, count], index) => ({
    name,
    count,
    color: colorPalette[index % colorPalette.length]
  }))
}

// Cleanup on unmount
onUnmounted(() => {
  stopDetection()
})
</script>
