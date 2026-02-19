<template>
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
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  mode: {
    type: String,
    default: 'webcam'
  },
  showVideo: {
    type: Boolean,
    default: false
  },
  uploadedImage: {
    type: String,
    default: ''
  },
  uploadedVideoUrl: {
    type: String,
    default: ''
  },
  showBoundingBoxes: {
    type: Boolean,
    default: true
  },
  detections: {
    type: Array,
    default: () => []
  },
  isProcessing: {
    type: Boolean,
    default: false
  }
})

const totalObjectCount = computed(() => {
  return props.detections.length
})
</script>

<style scoped>
.center-panel {
  flex: 1;
  background: var(--bg-dark);
  display: flex;
  flex-direction: column;
  min-width: 0;
  position: relative;
  overflow: hidden;
}

.center-panel::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: 
    radial-gradient(circle at 30% 30%, rgba(99, 102, 241, 0.1) 0%, transparent 40%),
    radial-gradient(circle at 70% 70%, rgba(168, 85, 247, 0.08) 0%, transparent 40%);
  pointer-events: none;
}

.panel-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--card-border);
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header h2 {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0;
}

.panel-header h2 i {
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.live-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  font-weight: 700;
  color: var(--success-color);
  text-transform: uppercase;
  letter-spacing: 1px;
}

.pulse {
  width: 10px;
  height: 10px;
  background-color: var(--success-color);
  border-radius: 50%;
  position: relative;
}

.pulse::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100%;
  height: 100%;
  background: var(--success-color);
  border-radius: 50%;
  animation: pulse-ring 1.5s infinite;
}

@keyframes pulse-ring {
  0% {
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
  }
  70% {
    box-shadow: 0 0 0 15px rgba(16, 185, 129, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
  }
}

.video-container {
  flex: 1;
  position: relative;
  background-color: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: var(--radius-lg);
  margin: 16px;
  box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.5);
}

.video-container video,
.video-container img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: var(--radius-md);
}

.video-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
  color: var(--text-muted);
  position: relative;
  z-index: 1;
}

.video-placeholder i {
  font-size: 72px;
  opacity: 0.2;
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

.video-placeholder p {
  font-size: 16px;
  font-weight: 500;
}

.bounding-boxes-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.bounding-box {
  position: absolute;
  border: 3px solid;
  border-radius: 8px;
  box-shadow: 0 0 15px currentColor;
  animation: box-appear 0.3s ease-out;
}

@keyframes box-appear {
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.box-label {
  position: absolute;
  top: -32px;
  left: -3px;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  color: white;
  border-radius: 8px;
  white-space: nowrap;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.count-overlay {
  position: absolute;
  top: 20px;
  right: 20px;
}

.count-badge {
  display: inline-block;
  padding: 14px 28px;
  background: var(--gradient-primary);
  color: white;
  font-size: 20px;
  font-weight: 700;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-glow), 0 4px 20px rgba(99, 102, 241, 0.4);
  animation: badge-appear 0.5s ease-out;
}

@keyframes badge-appear {
  from {
    opacity: 0;
    transform: scale(0.5) translateY(-20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}
</style>
