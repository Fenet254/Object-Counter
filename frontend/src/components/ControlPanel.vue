<template>
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
      <ObjectFilter 
        :availableObjects="availableObjects" 
        v-model="selectedObjects"
      />
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
</template>

<script setup>
import { ref, computed } from 'vue'
import ObjectFilter from './ObjectFilter.vue'

const props = defineProps({
  mode: {
    type: String,
    default: 'webcam'
  },
  selectedFileName: {
    type: String,
    default: ''
  },
  selectedObjects: {
    type: Array,
    default: () => []
  },
  confidenceThreshold: {
    type: Number,
    default: 0.5
  },
  showBoundingBoxes: {
    type: Boolean,
    default: true
  },
  isProcessing: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits([
  'update:mode', 
  'update:selectedFileName', 
  'update:selectedObjects',
  'update:confidenceThreshold', 
  'update:showBoundingBoxes',
  'startDetection',
  'stopDetection',
  'fileUpload'
])

const mode = computed({
  get() { return props.mode },
  set(value) { emit('update:mode', value) }
})

const selectedFileName = computed({
  get() { return props.selectedFileName },
  set(value) { emit('update:selectedFileName', value) }
})

const selectedObjects = computed({
  get() { return props.selectedObjects },
  set(value) { emit('update:selectedObjects', value) }
})

const confidenceThreshold = computed({
  get() { return props.confidenceThreshold },
  set(value) { emit('update:confidenceThreshold', value) }
})

const showBoundingBoxes = computed({
  get() { return props.showBoundingBoxes },
  set(value) { emit('update:showBoundingBoxes', value) }
})

const isDragging = ref(false)
const fileInput = ref(null)

const availableObjects = [
  'Person', 'Car', 'Bicycle', 'Motorbike', 'Bus', 'Truck',
  'Dog', 'Cat', 'Bird', 'Chair', 'Bottle', 'Cup', 'Bowl',
  'Apple', 'Banana', 'Orange', 'Book', 'Phone', 'Laptop'
]

const handleModeChange = () => {
  emit('stopDetection')
  emit('update:selectedFileName', '')
}

const handleFileUpload = (event) => {
  const file = event.target.files[0]
  if (file) {
    emit('fileUpload', file)
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
    emit('fileUpload', file)
  }
}

const startDetection = () => {
  emit('startDetection')
}

const stopDetection = () => {
  emit('stopDetection')
}
</script>

<style scoped>
.left-panel {
  width: var(--left-panel-width);
  min-width: var(--left-panel-width);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  border-right: 1px solid var(--card-border);
  background: var(--bg-glass);
  backdrop-filter: blur(var(--blur-md));
  -webkit-backdrop-filter: blur(var(--blur-md));
}

.panel-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--card-border);
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
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

.control-section {
  padding: 20px 24px;
  border-bottom: 1px solid var(--card-border);
  transition: background var(--transition-normal);
}

.control-section:hover {
  background: var(--card-glow);
}

.control-section h3 {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 16px;
}

.optional {
  font-weight: 400;
  text-transform: none;
  color: var(--text-muted);
  font-size: 11px;
  opacity: 0.7;
}

.radio-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.radio-option {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 18px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.radio-option::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.radio-option:hover {
  border-color: rgba(99, 102, 241, 0.4);
  transform: translateX(4px);
}

.radio-option.active {
  border-color: var(--primary-500);
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
}

.radio-option.active::before {
  opacity: 0.1;
}

.radio-option input {
  display: none;
}

.radio-option i {
  font-size: 20px;
  color: var(--text-muted);
  width: 28px;
  text-align: center;
  position: relative;
  z-index: 1;
  transition: all var(--transition-normal);
}

.radio-option.active i {
  color: var(--primary-500);
  text-shadow: 0 0 20px var(--primary-500);
}

.radio-option span {
  font-size: 14px;
  color: var(--text-primary);
  font-weight: 500;
  position: relative;
  z-index: 1;
}

.upload-area {
  border: 2px dashed rgba(99, 102, 241, 0.3);
  border-radius: var(--radius-lg);
  padding: 32px 24px;
  text-align: center;
  cursor: pointer;
  transition: all var(--transition-normal);
  background: var(--card-bg);
  position: relative;
  overflow: hidden;
}

.upload-area::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.upload-area:hover,
.upload-area.drag-over {
  border-color: var(--primary-500);
  transform: scale(1.02);
  box-shadow: var(--shadow-glow);
}

.upload-area:hover::before,
.upload-area.drag-over::before {
  opacity: 0.1;
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  position: relative;
  z-index: 1;
}

.upload-content i {
  font-size: 42px;
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  transition: transform var(--transition-normal);
}

.upload-area:hover .upload-content i {
  transform: scale(1.1);
}

.upload-content p {
  font-size: 14px;
  color: var(--text-primary);
  margin: 0;
  font-weight: 500;
}

.upload-hint {
  font-size: 12px;
  color: var(--text-muted);
}

.setting-item {
  margin-bottom: 20px;
}

.setting-item:last-child {
  margin-bottom: 0;
}

.setting-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.setting-label span {
  font-size: 13px;
  color: var(--text-primary);
  font-weight: 500;
}

.setting-value {
  font-weight: 700 !important;
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.slider {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--bg-dark);
  outline: none;
  -webkit-appearance: none;
  appearance: none;
  position: relative;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--gradient-primary);
  cursor: pointer;
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
  transition: all var(--transition-fast);
  position: relative;
  z-index: 2;
}

.slider::-webkit-slider-thumb:hover {
  transform: scale(1.2);
  box-shadow: 0 0 30px rgba(99, 102, 241, 0.7);
}

.toggle-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.toggle-switch {
  position: relative;
  width: 52px;
  height: 28px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--bg-dark);
  border: 1px solid var(--card-border);
  border-radius: 28px;
  transition: all var(--transition-normal);
}

.toggle-slider::before {
  position: absolute;
  content: "";
  height: 22px;
  width: 22px;
  left: 2px;
  bottom: 2px;
  background: var(--text-secondary);
  border-radius: 50%;
  transition: all var(--transition-normal);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.toggle-switch input:checked + .toggle-slider {
  background: var(--gradient-primary);
  border-color: transparent;
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
}

.toggle-switch input:checked + .toggle-slider::before {
  transform: translateX(24px);
  background: white;
  box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
}

.buttons-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 16px 24px;
  font-size: 14px;
  font-weight: 600;
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-normal);
  text-transform: uppercase;
  letter-spacing: 1px;
  position: relative;
  overflow: hidden;
}

.btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s ease;
}

.btn:hover::before {
  left: 100%;
}

.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn:disabled::before {
  display: none;
}

.btn-primary {
  background: var(--gradient-primary);
  color: white;
  box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-3px);
  box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
}

.btn-primary:active:not(:disabled) {
  transform: translateY(-1px);
}

.btn-secondary {
  background: var(--card-bg);
  color: var(--text-primary);
  border: 1px solid var(--card-border);
}

.btn-secondary:hover:not(:disabled) {
  background: rgba(99, 102, 241, 0.2);
  border-color: var(--primary-500);
  transform: translateY(-2px);
}
</style>
