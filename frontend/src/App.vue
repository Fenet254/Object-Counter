<template>
  <div class="app-container">
    <!-- Header -->
    <Header />

    <!-- Main Content - 3 Panel Layout -->
    <main class="main-content">
      <!-- Left Panel - Controls & Options -->
      <ControlPanel 
        v-model:mode="mode"
        v-model:selectedFileName="selectedFileName"
        v-model:selectedObjects="selectedObjects"
        v-model:confidenceThreshold="confidenceThreshold"
        v-model:showBoundingBoxes="showBoundingBoxes"
        :isProcessing="isProcessing"
        @startDetection="startDetection"
        @stopDetection="stopDetection"
        @fileUpload="handleFileUpload"
      />

      <!-- Center Panel - Live Feed / Output -->
      <VideoFeed 
        :mode="mode"
        :showVideo="showVideo"
        :uploadedImage="uploadedImage"
        :uploadedVideoUrl="uploadedVideoUrl"
        :showBoundingBoxes="showBoundingBoxes"
        :detections="detections"
        :isProcessing="isProcessing"
      />

      <!-- Right Panel - Object Counts & Stats -->
      <StatsPanel 
        :detections="detections"
        :objectCounts="objectCounts"
      />
    </main>

    <!-- Footer -->
    <Footer />
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
