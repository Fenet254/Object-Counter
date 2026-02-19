<template>
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
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  detections: {
    type: Array,
    default: () => []
  },
  objectCounts: {
    type: Array,
    default: () => []
  }
})

const totalObjectCount = computed(() => {
  return props.detections.length
})

const maxCount = computed(() => {
  if (props.objectCounts.length === 0) return 1
  return Math.max(...props.objectCounts.map(c => c.count))
})
</script>

<style scoped>
.right-panel {
  width: var(--right-panel-width);
  min-width: var(--right-panel-width);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  border-left: 1px solid var(--card-border);
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

.total-count-section {
  padding: 24px;
  border-bottom: 1px solid var(--card-border);
}

.total-count-card {
  background: var(--gradient-primary);
  border-radius: var(--radius-xl);
  padding: 28px;
  text-align: center;
  box-shadow: var(--shadow-glow), 0 8px 32px rgba(99, 102, 241, 0.3);
  position: relative;
  overflow: hidden;
}

.total-count-card::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 60%);
  animation: shimmer 3s infinite;
}

@keyframes shimmer {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.total-label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.85);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 8px;
  position: relative;
}

.total-value {
  display: block;
  font-size: 56px;
  font-weight: 800;
  color: white;
  line-height: 1;
  position: relative;
  text-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
}

.counts-table-section {
  padding: 20px 24px;
  border-bottom: 1px solid var(--card-border);
}

.counts-table-section h3,
.chart-section h3 {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 16px;
}

.counts-table {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  padding: 8px 16px;
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
}

.table-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  background: var(--card-bg);
  border-radius: var(--radius-md);
  border-left: 4px solid;
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.table-row::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
  transform: translateX(-100%);
  transition: transform 0.5s ease;
}

.table-row:hover {
  transform: translateX(6px);
  box-shadow: var(--shadow-md);
}

.table-row:hover::before {
  transform: translateX(100%);
}

.object-name {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 14px;
  color: var(--text-primary);
  font-weight: 500;
}

.object-name i {
  font-size: 12px;
  filter: drop-shadow(0 0 8px currentColor);
}

.object-count {
  font-size: 20px;
  font-weight: 700;
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 24px;
  color: var(--text-muted);
  text-align: center;
}

.empty-state i {
  font-size: 56px;
  opacity: 0.2;
  margin-bottom: 16px;
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.empty-state p {
  font-size: 14px;
  font-weight: 500;
}

.chart-section {
  padding: 20px 24px;
}

.bar-chart {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.bar-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.bar-label {
  font-size: 12px;
  color: var(--text-primary);
  width: 65px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
}

.bar-container {
  flex: 1;
  height: 24px;
  background: var(--bg-dark);
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}

.bar {
  height: 100%;
  border-radius: 12px;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
  min-width: 4px;
  position: relative;
  overflow: hidden;
}

.bar::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
  transform: translateX(-100%);
  animation: bar-shine 2s infinite;
}

@keyframes bar-shine {
  0% { transform: translateX(-100%); }
  50%, 100% { transform: translateX(100%); }
}

.bar-value {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
  width: 35px;
  text-align: right;
}
</style>
