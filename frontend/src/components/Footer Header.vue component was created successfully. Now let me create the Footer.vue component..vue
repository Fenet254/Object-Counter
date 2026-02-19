<template>
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
</template>

<script setup>
// Footer component - Simple presentational component
</script>

<style scoped>
.footer {
  height: var(--footer-height);
  background: var(--gradient-header);
  backdrop-filter: blur(var(--blur-sm));
  -webkit-backdrop-filter: blur(var(--blur-sm));
  border-top: 1px solid var(--card-border);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 32px;
  flex-shrink: 0;
  position: relative;
}

.footer::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
}

.footer-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  max-width: 1400px;
}

.footer-credit {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.footer-credit i {
  background: var(--text-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.footer-links {
  display: flex;
  gap: 24px;
}

.footer-links a {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all var(--transition-fast);
}

.footer-links a:hover {
  color: var(--primary-500);
  transform: translateY(-2px);
}

.footer-links i {
  font-size: 18px;
}
</style>
