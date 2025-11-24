const preprocessBtn = document.getElementById('preprocessBtn');
const trainBtn = document.getElementById('trainBtn');
const trainStatus = document.getElementById('trainStatus');
const trainProgress = document.getElementById('trainProgress');
const trainProgressBar = document.getElementById('trainProgressBar');
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

preprocessBtn.onclick = async () => {
  preprocessBtn.disabled = true;
  preprocessBtn.innerText = 'Running Preprocess…';
  try{
    const res = await fetchJson('/api/preprocess', 'POST');
    log('Preprocess done: ' + JSON.stringify(res.files || res));
  }catch(e){
    log('Preprocess error: ' + e);
  }
  preprocessBtn.disabled = false;
  preprocessBtn.innerText = 'Run Preprocess';
}

trainBtn.onclick = async () => {
  const process = processSelect.value;
  const nTrials = parseInt(nTrialsInput.value || '20');
  trainBtn.disabled = true;
  const body = {process, n_trials: nTrials, use_optuna: false};
  await fetchJson('/api/train', 'POST', body);
  // if process == BOTH, schedule FIFO polling for both, else single
    if(process === 'BOTH'){
    pollStatus('FCC');
    pollStatus('CCR');
    // enable cancel while sequential training runs
      cancelBtn.disabled = false;
      cancelBtn.style.display = 'inline-block';
  } else {
    pollStatus(process);
      cancelBtn.disabled = false;
      cancelBtn.style.display = 'inline-block';
  }
}

refreshModelsBtn.onclick = async () =>{
  const res = await fetchJson('/api/models');
  renderModels(res.models);
}

deleteAllModelsBtn.onclick = async () =>{
  if(!confirm('Delete all models? This action cannot be undone.')) return;
  try{
    const res = await fetchJson('/api/models?delete_all=true', 'DELETE');
    log('Delete all models result: ' + JSON.stringify(res));
    refreshModelsBtn.click();
  }catch(e){ log('Delete all models error: ' + e); }
}

cancelBtn.onclick = async () =>{
  const process = processSelect.value;
  try{
    const res = await fetchJson('/api/train/cancel', 'POST', {process});
    log('Cancel requested: ' + JSON.stringify(res));
  }catch(e){
    if(e.response){ log('Cancel error: ' + JSON.stringify(e.response)); }
    else { log('Cancel error: ' + e.toString()); }
  }
}

predictBtn.onclick = async () =>{
  const process = processSelect.value;
  if(process === 'BOTH'){
    alert('Select a single process to predict (FCC or CCR)');
    return;
  }
  try{
    const records = JSON.parse(sampleInput.value);
    const res = await fetchJson('/api/predict', 'POST', {process, records});
    predictOut.innerText = JSON.stringify(res.predictions, null, 2);
  }catch(e){
    // Show the server error message if available
    if(e.response){
      predictOut.innerText = 'Error: ' + JSON.stringify(e.response);
    } else {
      predictOut.innerText = 'Error: ' + e.toString();
    }
  }
}

function log(msg){
  const now = new Date().toLocaleTimeString();
  logOutput.innerText = `${now} - ${msg}\n` + logOutput.innerText;
}

function renderModels(models){
  cachedModels = models || [];
  modelsList.innerHTML = '';
  models.forEach(m =>{
    const li = document.createElement('li');
    li.className = 'list-group-item';
    li.innerHTML = `<div class="d-flex justify-content-between align-items-center"><span>${m.file} — modified ${m.modified}</span><div><button class='btn btn-sm btn-outline-danger ms-2 delete-model' data-path='${m.path}'>Delete</button></div></div>`;
    modelsList.appendChild(li);
  });
  // Attach delete handlers
  document.querySelectorAll('.delete-model').forEach(btn =>{
    btn.onclick = async (e) =>{
      const path = btn.dataset.path;
      if(!confirm('Delete model ' + path + '?')) return;
      const res = await fetchJson(`/api/models?path=${encodeURIComponent(path)}`, 'DELETE');
      log('Delete model result: ' + JSON.stringify(res));
      refreshModelsBtn.click();
    }
  });
  // Disable predict if no models available
  const anyModels = cachedModels && cachedModels.length > 0;
  // Enable predict if there are models for the selected process
  const selected = processSelect.value;
  let hasModelsForSelected = false;
  if(selected === 'BOTH'){
    // If BOTH selected, we disable predict by design
    hasModelsForSelected = false;
  } else {
    const prefix = selected.toLowerCase() + '_';
    if(anyModels){
      hasModelsForSelected = cachedModels.some(m => m.file.toLowerCase().startsWith(prefix));
    }
  }
  predictBtn.disabled = !hasModelsForSelected;
  // Update counters for FCC and CCR in UI
  const fccCount = cachedModels.filter(m => m.file.toLowerCase().startsWith('fcc_')).length;
  const ccrCount = cachedModels.filter(m => m.file.toLowerCase().startsWith('ccr_')).length;
  const fccModelCount = document.getElementById('fccModelCount');
  const ccrModelCount = document.getElementById('ccrModelCount');
  if(fccModelCount) fccModelCount.innerText = fccCount.toString();
  if(ccrModelCount) ccrModelCount.innerText = ccrCount.toString();
}

async function pollStatus(process){
  // Poll until training stops
  const isFCC = process === 'FCC';
  const statusEl = document.getElementById(isFCC ? 'trainStatusFCC' : 'trainStatusCCR');
  const progressContainer = document.getElementById(isFCC ? 'trainProgressFCC' : 'trainProgressCCR');
  const progressBar = document.getElementById(isFCC ? 'trainProgressBarFCC' : 'trainProgressBarCCR');
  progressContainer.style.display = 'block';
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/training/${process}`);
  ws.onmessage = (ev) => {
    try{
      const msg = JSON.parse(ev.data);
      const message = msg.message || msg.event || JSON.stringify(msg);
      log(`WS ${process}: ${message}`);
      if(msg.progress !== undefined){
        const pct = Math.round(msg.progress * 100);
        progressBar.style.width = `${pct}%`;
        progressBar.innerText = `${pct}%`;
        statusEl.innerText = `${process}: ${message} — ${pct}%`;
      }
      if(msg.metrics){
        metricsHistory.pci.push(msg.metrics.pci?.val_rmse || 0);
        metricsHistory.h2.push(msg.metrics.h2?.val_rmse || 0);
        updateChart(metricsHistory);
        // Realtime metrics display
        const proc = process.toUpperCase();
        if(msg.metrics.pci){
          const pciRmseEl = document.getElementById('pciRmse');
          const pciWithinEl = document.getElementById('pciWithin');
          if(pciRmseEl) pciRmseEl.innerText = (msg.metrics.pci.val_rmse || 0).toFixed(2);
          if(pciWithinEl) pciWithinEl.innerText = (msg.metrics.pci.pct_within_10 || 0).toFixed(2);
        }
        if(msg.metrics.h2){
          const h2RmseEl = document.getElementById('h2Rmse');
          const h2WithinEl = document.getElementById('h2Within');
          if(h2RmseEl) h2RmseEl.innerText = (msg.metrics.h2.val_rmse || 0).toFixed(2);
          if(h2WithinEl) h2WithinEl.innerText = (msg.metrics.h2.pct_within_10 || 0).toFixed(2);
        }
        metricsByProcess[proc] = msg.metrics;
        if(metricsByProcess.FCC && metricsByProcess.CCR){
          const avgPci = ((metricsByProcess.FCC.pci.val_rmse || 0) + (metricsByProcess.CCR.pci.val_rmse || 0)) / 2;
          const avgH2 = ((metricsByProcess.FCC.h2.val_rmse || 0) + (metricsByProcess.CCR.h2.val_rmse || 0)) / 2;
          const avgPciEl = document.getElementById('avgPciRmse');
          const avgH2El = document.getElementById('avgH2Rmse');
          if(avgPciEl) avgPciEl.innerText = avgPci.toFixed(2);
          if(avgH2El) avgH2El.innerText = avgH2.toFixed(2);
        }
      }
      if(msg.event === 'model_saved'){
        log(`Model saved: ${msg.path} (${msg.target}/${msg.model_type})`);
        refreshModelsBtn.click();
      }
    }
    catch(e){ log(`WS ${process}: ${ev.data}`); }
  }
  while(true){
    const status = await fetchJson(`/api/train/status?process=${process}`);
    const pct = Math.round(status.progress * 100);
    progressBar.style.width = pct + '%';
    progressBar.innerText = pct + '%';
    statusEl.innerText = `${process}: Running: ${status.running} — Progress: ${pct}%`;
    if(status.metrics){
      // Update chart
      metricsHistory.pci.push(status.metrics.pci?.val_rmse || 0);
      metricsHistory.h2.push(status.metrics.h2?.val_rmse || 0);
      updateChart(metricsHistory);
      document.getElementById('metricEmptyMsg').style.display = 'none';
      document.getElementById('logsEmptyMsg').style.display = 'none';
    }
    // Update cancel button visibility
    if(status.running){
      cancelBtn.disabled = false;
      cancelBtn.style.display = 'inline-block';
    } else {
      cancelBtn.disabled = true;
      cancelBtn.style.display = 'none';
    }
    if(process === 'BOTH'){
      // When training BOTH we check both FCC and CCR status to ensure sequence finished
      const sFCC = await fetchJson('/api/train/status?process=FCC');
      const sCCR = await fetchJson('/api/train/status?process=CCR');
      if(!sFCC.running && !sCCR.running){
        log(`${process} training completed`);
        trainBtn.disabled = false;
        refreshModelsBtn.click();
        break;
      }
      // Otherwise continue polling
      await new Promise(r => setTimeout(r, 2000));
      continue;
    }
    if(!status.running) { 
      log(`${process} training completed`);
      trainBtn.disabled = false;
      refreshModelsBtn.click();
      break;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
}

  // No longer using fetchLogs - web socket is used for real-time logs.

function updateChart(metrics){
  if(!metricChart){
    const ctx = document.getElementById('metricChart').getContext('2d');
    metricChart = new Chart(ctx, {
      type: 'line',
      data: {labels: metrics.pci.map((_,i)=>i+1), datasets: [
        {label: 'PCI RMSE', data: metrics.pci, borderColor: 'steelblue', tension: 0.2},
        {label: 'H2 RMSE', data: metrics.h2, borderColor: 'coral', tension: 0.2},
      ]},
      options: {responsive:true}
    });
  } else {
    metricChart.data.labels = metrics.pci.map((_,i)=>i+1);
    metricChart.data.datasets[0].data = metrics.pci;
    metricChart.data.datasets[1].data = metrics.h2;
    metricChart.update();
  }
}

(function(){
  // Initial load of models
  refreshModelsBtn.click();
  // disable and hide cancel initially
  cancelBtn.disabled = true;
  cancelBtn.style.display = 'none';
  // ensure predict disabled until models are loaded
  predictBtn.disabled = true;
  processSelect.onchange = () => {
    const anyModels = cachedModels && cachedModels.length > 0;
    predictBtn.disabled = !anyModels || processSelect.value === 'BOTH';
  }
  processSelect.onblur = async () => {
    // Query training status for selected process to set Cancel button accordingly
    try{
      const status = await fetchJson(`/api/train/status?process=${processSelect.value}`);
      cancelBtn.disabled = !status.running;
    }catch(e){
      cancelBtn.disabled = true;
    }
  }
})();
