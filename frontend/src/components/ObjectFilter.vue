<template>
  <div class="object-filters">
    <label v-for="obj in availableObjects" :key="obj" class="filter-checkbox">
      <input type="checkbox" v-model="selectedObjects" :value="obj">
      <span class="checkbox-custom"></span>
      <span class="filter-label">{{ obj }}</span>
    </label>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  availableObjects: {
    type: Array,
    required: true
  },
  modelValue: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['update:modelValue'])

const selectedObjects = computed({
  get() {
    return props.modelValue
  },
  set(value) {
    emit('update:modelValue', value)
  }
})
</script>

<style scoped>
.object-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  max-height: 200px;
  overflow-y: auto;
  padding: 4px;
}

.filter-checkbox {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 20px;
  cursor: pointer;
  transition: all var(--transition-normal);
  font-size: 12px;
}

.filter-checkbox:hover {
  background: rgba(99, 102, 241, 0.2);
  border-color: rgba(99, 102, 241, 0.3);
  transform: translateY(-2px);
}

.filter-checkbox input {
  display: none;
}

.checkbox-custom {
  width: 18px;
  height: 18px;
  border: 2px solid var(--card-border);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition-normal);
  position: relative;
}

.filter-checkbox input:checked + .checkbox-custom {
  background: var(--gradient-primary);
  border-color: transparent;
  box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
}

.filter-checkbox input:checked + .checkbox-custom::after {
  content: '';
  width: 6px;
  height: 6px;
  background: white;
  border-radius: 50%;
}

.filter-label {
  font-size: 12px;
  color: var(--text-primary);
  font-weight: 500;
}
</style>
