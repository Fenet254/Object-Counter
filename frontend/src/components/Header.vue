<template>
  <header class="header">
    <div class="header-content">
      <h1 class="header-title">Real-Time Object Counter</h1>
      <p class="header-subtitle">Using YOLOv8 + FastAPI</p>
    </div>
  </header>
</template>

<script setup>
// Header component - Simple presentational component
</script>

<style scoped>
.header {
  height: var(--header-height);
  background: var(--gradient-header);
  backdrop-filter: blur(var(--blur-md));
  -webkit-backdrop-filter: blur(var(--blur-md));
  border-bottom: 1px solid var(--card-border);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 32px;
  flex-shrink: 0;
  position: relative;
  z-index: 10;
}

.header::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
}

.header-content {
  text-align: center;
}

.header-title {
  font-size: 26px;
  font-weight: 700;
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  letter-spacing: -0.5px;
  text-shadow: 0 0 40px rgba(99, 102, 241, 0.3);
}

.header-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 4px 0 0;
  font-weight: 400;
  letter-spacing: 0.5px;
}
</style>
