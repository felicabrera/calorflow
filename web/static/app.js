const preprocessBtn = document.getElementById('preprocessBtn');
const trainBtn = document.getElementById('trainBtn');
const trainStatus = document.getElementById('trainStatus');
const processSelect = document.getElementById('processSelect');
const nTrialsInput = document.getElementById('nTrialsInput');
const modelsList = document.getElementById('modelsList');
const refreshModelsBtn = document.getElementById('refreshModelsBtn');
const deleteAllModelsBtn = document.getElementById('deleteAllModelsBtn');
const cancelBtn = document.getElementById('cancelBtn');
const predictBtn = document.getElementById('predictBtn');
const sampleInput = document.getElementById('sampleInput');
const predictOut = document.getElementById('predictOut');
const logOutput = document.getElementById('logOutput');

let metricChart = null;
let metricsHistory = {pci: [], h2: []};
let metricsByProcess = {FCC: null, CCR: null};
let cachedModels = [];

// Helper for API calls
async function fetchJson(url, method='GET', body=null){
  const opts = {method, headers: {'Content-Type': 'application/json'}};
  if(body!=null) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  const data = await res.json().catch(()=>null);
  if(!res.ok){
    const err = new Error(res.statusText || 'Request failed');
    err.response = data;
    throw err;
  }
  return data;
}

function log(msg){
  const now = new Date().toLocaleTimeString();
  logOutput.innerText = `${now} - ${msg}\n` + logOutput.innerText;
}

// --- Event Handlers ---

preprocessBtn.onclick = async () => {
  preprocessBtn.disabled = true;
  preprocessBtn.innerText = 'Running...';
  try{
    const res = await fetchJson('/api/preprocess', 'POST');
    log('Preprocess done: ' + (res.files ? res.files.length + ' files' : 'OK'));
  }catch(e){
    log('Preprocess error: ' + e);
  }
  preprocessBtn.disabled = false;
  preprocessBtn.innerText = 'Run Preprocess';
}

trainBtn.onclick = async () => {
  const process = processSelect.value;
  const nTrials = parseInt(nTrialsInput.value || '20');
  
  // Reset UI and metrics
  metricsHistory = {pci: [], h2: []};
  if(metricChart) {
    metricChart.data.labels = [];
    metricChart.data.datasets[0].data = [];
    metricChart.data.datasets[1].data = [];
    metricChart.update();
  }
  document.getElementById('metricEmptyMsg').style.display = 'block';
  
  trainBtn.disabled = true;
  cancelBtn.style.display = 'inline-block';
  cancelBtn.disabled = false;

  try {
    const body = {process, n_trials: nTrials, use_optuna: false};
    await fetchJson('/api/train', 'POST', body);
    
    if(process === 'BOTH'){
      pollStatus('FCC');
      pollStatus('CCR');
    } else {
      pollStatus(process);
    }
  } catch(e) {
    log('Train start error: ' + e);
    trainBtn.disabled = false;
    cancelBtn.style.display = 'none';
  }
}

cancelBtn.onclick = async () =>{
  const process = processSelect.value;
  try{
    const res = await fetchJson('/api/train/cancel', 'POST', {process});
    log('Cancel requested: ' + (res.message || 'OK'));
  }catch(e){
    log('Cancel error: ' + e);
  }
}

refreshModelsBtn.onclick = async () =>{
  try {
    const res = await fetchJson('/api/models');
    renderModels(res.models);
  } catch(e) {
    log('Fetch models error: ' + e);
  }
}

deleteAllModelsBtn.onclick = async () =>{
  if(!confirm('Delete all models? This action cannot be undone.')) return;
  try{
    const res = await fetchJson('/api/models?delete_all=true', 'DELETE');
    log('Deleted ' + res.deleted + ' models');
    refreshModelsBtn.click();
  }catch(e){ log('Delete error: ' + e); }
}

predictBtn.onclick = async () =>{
  const process = processSelect.value;
  if(process === 'BOTH'){
    alert('Select a single process (FCC or CCR) to predict.');
    return;
  }
  predictBtn.disabled = true;
  predictOut.innerText = 'Predicting...';
  try{
    const records = JSON.parse(sampleInput.value);
    const res = await fetchJson('/api/predict', 'POST', {process, records});
    predictOut.innerText = JSON.stringify(res.predictions, null, 2);
  }catch(e){
    let msg = e.toString();
    if(e.response && e.response.detail) msg = JSON.stringify(e.response.detail);
    predictOut.innerText = 'Error: ' + msg;
  }
  predictBtn.disabled = false;
}

// --- UI Logic ---

function renderModels(models){
  cachedModels = models || [];
  modelsList.innerHTML = '';
  if(cachedModels.length === 0) {
    modelsList.innerHTML = '<li class="list-group-item text-muted text-center small">No models found</li>';
  } else {
    cachedModels.forEach(m =>{
      const li = document.createElement('li');
      li.className = 'list-group-item d-flex justify-content-between align-items-center small';
      li.innerHTML = `
        <div class="text-truncate" title="${m.file}">
          <span class="fw-bold">${m.file.split('_')[0].toUpperCase()}</span> 
          ${m.file.substring(4)}
        </div>
        <button class='btn btn-sm btn-outline-danger delete-model py-0 px-2' data-path='${m.path}'>&times;</button>
      `;
      modelsList.appendChild(li);
    });
  }

  // Attach delete handlers
  document.querySelectorAll('.delete-model').forEach(btn =>{
    btn.onclick = async (e) =>{
      const path = btn.dataset.path;
      if(!confirm('Delete this model?')) return;
      await fetchJson(`/api/models?path=${encodeURIComponent(path)}`, 'DELETE');
      refreshModelsBtn.click();
    }
  });

  updateModelCounts();
  updatePredictState();
}

function updateModelCounts() {
  const fccCount = cachedModels.filter(m => m.file.toLowerCase().startsWith('fcc_')).length;
  const ccrCount = cachedModels.filter(m => m.file.toLowerCase().startsWith('ccr_')).length;
  document.getElementById('fccModelCount').innerText = fccCount;
  document.getElementById('ccrModelCount').innerText = ccrCount;
}

function updatePredictState() {
  const selected = processSelect.value;
  let hasModels = false;
  
  if(selected !== 'BOTH') {
    const prefix = selected.toLowerCase() + '_';
    hasModels = cachedModels.some(m => m.file.toLowerCase().startsWith(prefix));
  }
  
  predictBtn.disabled = !hasModels;
  if(selected === 'BOTH') predictBtn.disabled = true;
}

processSelect.onchange = () => {
  updatePredictState();
  // Check if running to show cancel button
  checkRunningStatus();
};

async function checkRunningStatus() {
  const process = processSelect.value;
  if(process === 'BOTH') {
    // simple check
    cancelBtn.style.display = 'none'; 
    return;
  }
  try {
    const status = await fetchJson(`/api/train/status?process=${process}`);
    if(status.running) {
      cancelBtn.style.display = 'inline-block';
      cancelBtn.disabled = false;
      trainBtn.disabled = true;
    } else {
      cancelBtn.style.display = 'none';
      trainBtn.disabled = false;
    }
  } catch(e) {}
}

// --- Polling & WebSocket ---

async function pollStatus(process){
  const isFCC = process === 'FCC';
  const statusEl = document.getElementById(isFCC ? 'trainStatusFCC' : 'trainStatusCCR');
  const progressBar = document.getElementById(isFCC ? 'trainProgressBarFCC' : 'trainProgressBarCCR');
  
  statusEl.innerText = 'Starting...';
  progressBar.style.width = '0%';
  
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/training/${process}`);
  
  ws.onmessage = (ev) => {
    try{
      const msg = JSON.parse(ev.data);
      const message = msg.message || msg.event || 'Update';
      
      if(msg.progress !== undefined){
        const pct = Math.round(msg.progress * 100);
        progressBar.style.width = `${pct}%`;
        statusEl.innerText = `${message} (${pct}%)`;
      } else {
        statusEl.innerText = message;
      }

      if(msg.metrics){
        updateMetrics(msg.metrics, process);
      }
      
      if(msg.event === 'model_saved'){
        refreshModelsBtn.click();
      }
      
      if(msg.event === 'log' || msg.event === 'log_message') {
        log(`[${process}] ${msg.message}`);
      }

    } catch(e){ console.error(e); }
  }

  // Fallback polling loop to detect completion if WS fails or for robust finish
  while(true){
    await new Promise(r => setTimeout(r, 2000));
    try {
      const status = await fetchJson(`/api/train/status?process=${process}`);
      const pct = Math.round(status.progress * 100);
      progressBar.style.width = `${pct}%`;
      
      if(status.metrics) updateMetrics(status.metrics, process);

      if(!status.running) {
        statusEl.innerText = 'Completed';
        progressBar.style.width = '100%';
        ws.close();
        
        // If we are in BOTH mode, we need to check if the other one is also done
        if(processSelect.value === 'BOTH') {
           const other = process === 'FCC' ? 'CCR' : 'FCC';
           const otherStatus = await fetchJson(`/api/train/status?process=${other}`);
           if(!otherStatus.running) {
             trainBtn.disabled = false;
             cancelBtn.style.display = 'none';
           }
        } else {
           trainBtn.disabled = false;
           cancelBtn.style.display = 'none';
        }
        refreshModelsBtn.click();
        break;
      }
    } catch(e) { break; }
  }
}

function updateMetrics(metrics, process) {
  // Update chart data
  if(metrics.pci && metrics.pci.val_rmse) metricsHistory.pci.push(metrics.pci.val_rmse);
  if(metrics.h2 && metrics.h2.val_rmse) metricsHistory.h2.push(metrics.h2.val_rmse);
  
  updateChart(metricsHistory);
  document.getElementById('metricEmptyMsg').style.display = 'none';

  // Update Real-time cards
  metricsByProcess[process] = metrics;
  
  if(metrics.pci) {
    document.getElementById('pciRmse').innerText = (metrics.pci.val_rmse || 0).toFixed(2);
    document.getElementById('pciWithin').innerText = (metrics.pci.pct_within_10 || 0).toFixed(1);
  }
  if(metrics.h2) {
    document.getElementById('h2Rmse').innerText = (metrics.h2.val_rmse || 0).toFixed(2);
    document.getElementById('h2Within').innerText = (metrics.h2.pct_within_10 || 0).toFixed(1);
  }

  // Aggregate
  if(metricsByProcess.FCC && metricsByProcess.CCR) {
    const avgPci = ((metricsByProcess.FCC.pci?.val_rmse || 0) + (metricsByProcess.CCR.pci?.val_rmse || 0)) / 2;
    const avgH2 = ((metricsByProcess.FCC.h2?.val_rmse || 0) + (metricsByProcess.CCR.h2?.val_rmse || 0)) / 2;
    document.getElementById('avgPciRmse').innerText = avgPci.toFixed(2);
    document.getElementById('avgH2Rmse').innerText = avgH2.toFixed(2);
  }
}

function updateChart(metrics){
  if(!metricChart){
    const ctx = document.getElementById('metricChart').getContext('2d');
    metricChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: metrics.pci.map((_,i)=>i+1), 
        datasets: [
          {label: 'PCI RMSE', data: metrics.pci, borderColor: '#0d6efd', tension: 0.3, pointRadius: 2},
          {label: 'H2 RMSE', data: metrics.h2, borderColor: '#dc3545', tension: 0.3, pointRadius: 2},
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: { legend: { position: 'bottom' } }
      }
    });
  } else {
    metricChart.data.labels = metrics.pci.map((_,i)=>i+1);
    metricChart.data.datasets[0].data = metrics.pci;
    metricChart.data.datasets[1].data = metrics.h2;
    metricChart.update();
  }
}

// Init
(function(){
  refreshModelsBtn.click();
  checkRunningStatus();
})();
