async function loadCSV(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) {
      console.error(`Failed to load CSV: ${path} (status ${res.status})`);
      return null;
    }
    const txt = await res.text();
    const lines = txt.trim().split('\n');
    const header = lines[0].split(',');
    return lines.slice(1).map(line => {
      const cols = line.split(',');
      const row = {};
      header.forEach((h, i) => row[h] = cols[i]);
      return row;
    });
  } catch (err) {
    console.error(`Error fetching CSV ${path}:`, err);
    return null;
  }
}

async function initCharts() {
  const ablation = await loadCSV('assets/data/ablation_study_results.csv') || [];
  const benchmark = await loadCSV('assets/data/multi_scheduler_benchmark.csv') || [];

  if (ablation.length) {
    const labels = ablation.map(r => r.removed_feature);
    const values = ablation.map(r => Number(r.r2_drop));
    new Chart(document.getElementById('ablationChart'), {
      type: 'bar',
      data: { labels, datasets: [{ label: 'R² drop', data: values, backgroundColor: '#38bdf8' }] },
      options: { plugins: { legend: { labels: { color: '#e2e8f0' } } }, scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } } }
    });
  }

  if (benchmark.length) {
    const labels = benchmark.map(r => r.scheduler);
    const values = benchmark.map(r => Number(r.mean_wait));
    new Chart(document.getElementById('schedulerChart'), {
      type: 'line',
      data: { labels, datasets: [{ label: 'Mean wait', data: values, borderColor: '#34d399', tension: 0.25 }] },
      options: { plugins: { legend: { labels: { color: '#e2e8f0' } } }, scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } } }
    });
  }
}

document.addEventListener('DOMContentLoaded', initCharts);
