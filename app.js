// app.js
// Main application glue: UI, wiring data-loader and gru model, plotting.
// This is an ES module entry point referenced by index.html
import { DataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

const fileInput = document.getElementById('file-input');
const prepareBtn = document.getElementById('prepare-btn');
const trainBtn = document.getElementById('train-btn');
const predictBtn = document.getElementById('predict-btn');
const saveBtn = document.getElementById('save-btn');
const loadBtn = document.getElementById('load-btn');
const trainSplitInput = document.getElementById('train-split');
const epochsInput = document.getElementById('epochs');
const batchSizeInput = document.getElementById('batch-size');

const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const samplesMeta = document.getElementById('samples-meta');
const symbolsMeta = document.getElementById('symbols-meta');
const warningDiv = document.getElementById('warning');
const logEl = document.getElementById('log');

const accuracyBarCanvas = document.getElementById('accuracy-bar');
const stockSelect = document.getElementById('stock-select');
const stockTimelineCanvas = document.getElementById('stock-timeline');
const stockStats = document.getElementById('stock-stats');
const timelinesContainer = document.getElementById('timelines-container');

let dataLoader = new DataLoader({ sequenceLength: 12, forecastHorizon: 3 });
dataLoader.setLogger((...args) => appLog(...args));

let model = new GRUModel({ inputShape: [12, 20], gruUnits: 64, denseUnits: 30, learningRate: 0.001 });

let charts = { accuracyBar: null, stockTimeline: null, perStockSmall: [] };
let preparedMeta = null;
let tensors = null;
let evalResults = null;

// simple logger
function appLog(...args) {
  const s = args.map(a => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');
  console.log(...args);
  const now = new Date().toISOString().slice(11,23);
  logEl.textContent += `[${now}] ${s}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

// UI helpers
function setProgress(pct, text='') {
  progressBar.style.width = `${Math.round(pct)}%`;
  progressText.textContent = text || `${Math.round(pct)}%`;
}

// prepare data
prepareBtn.addEventListener('click', async () => {
  try {
    if (!fileInput.files.length) { alert('Please choose a CSV file first'); return; }
    setProgress(0, 'Parsing CSV...');
    clearState();
    const file = fileInput.files[0];
    await dataLoader.loadFromFile(file);
    setProgress(20, 'Preparing dataset...');
    const trainSplitPercent = Number(trainSplitInput.value || 80);
    preparedMeta = dataLoader.prepareDataset({ trainSplitPercent });
    tensors = dataLoader.getTensors();
    samplesMeta.textContent = `${preparedMeta.samples} (train ${preparedMeta.trainSamples} / test ${preparedMeta.testSamples})`;
    symbolsMeta.textContent = preparedMeta.symbols.join(', ');
    warningDiv.textContent = preparedMeta.symbols.length !== 10 ? `Found ${preparedMeta.symbols.length} symbols (expected 10)` : '';
    setProgress(100,'Dataset ready');
    trainBtn.disabled = false;
    predictBtn.disabled = true;
    saveBtn.disabled = true;
    // populate stock select
    stockSelect.innerHTML = '';
    for (const s of preparedMeta.symbols) {
      const opt = document.createElement('option'); opt.value = s; opt.textContent = s; stockSelect.appendChild(opt);
    }
    renderEmptyAccuracy();
  } catch (err) {
    appLog("Error preparing data:", err.message || err);
    alert("Error preparing data: " + (err.message || err));
    setProgress(0, 'Error');
  }
});

// build and train model
trainBtn.addEventListener('click', async () => {
  try {
    if (!tensors) { alert('Prepare dataset first'); return; }
    trainBtn.disabled = true;
    setProgress(0, 'Building model...');
    model.buildModel();
    setProgress(5, 'Starting training...');
    const epochs = Number(epochsInput.value || 30);
    const batchSize = Number(batchSizeInput.value || 32);

    const onEpochEnd = (epoch, logs) => {
      const pct = Math.min(90, 5 + ((epoch+1) / epochs) * 85);
      setProgress(pct, `Epoch ${epoch+1}/${epochs} — loss:${(logs.loss||0).toFixed(4)} acc:${(logs.binaryAccuracy||0).toFixed(4)}`);
    };

    await model.fit(tensors.X_train, tensors.y_train, { epochs, batchSize, onEpochEnd });
    setProgress(95, 'Training finished');
    predictBtn.disabled = false;
    saveBtn.disabled = false;
    appLog('Training completed');
  } catch (err) {
    appLog('Training error:', err.message || err);
    alert('Training error: ' + (err.message || err));
    trainBtn.disabled = false;
    setProgress(0,'Error');
  }
});

// evaluate/predict
predictBtn.addEventListener('click', async () => {
  try {
    if (!tensors) { alert('Prepare dataset first'); return; }
    if (!model.model) { alert('Model not built/trained'); return; }

    setProgress(0, 'Predicting on test set...');
    const res = await model.evaluateTestSet(tensors.X_test, tensors.y_test, preparedMeta.symbols, 3);
    evalResults = res;
    setProgress(60, 'Computing visuals...');
    // compute averaged accuracy per stock (already provided)
    const perStockAcc = res.perStockAcc.map(v => Number((v*100).toFixed(2)));
    const zipped = preparedMeta.symbols.map((s,i) => ({ symbol: s, acc: perStockAcc[i], idx: i }));
    const sorted = zipped.slice().sort((a,b) => b.acc - a.acc);
    renderAccuracyBar(sorted);
    renderPerStockTimelines(res.perStockTimeline, preparedMeta.symbols, preparedMeta.sampleDatesTest);
    renderStockTimelineForSelected(preparedMeta.symbols, res.perStockTimeline, res.perStockAcc, preparedMeta.sampleDatesTest);
    setProgress(100, 'Evaluation complete');
    appLog(`Overall accuracy: ${(res.overallAcc*100).toFixed(2)}%`);
  } catch (err) {
    appLog('Prediction error:', err.message || err);
    alert('Prediction error: ' + (err.message || err));
    setProgress(0,'Error');
  }
});

// save / load weights
saveBtn.addEventListener('click', async () => {
  try {
    setProgress(0,'Saving weights...');
    await model.saveToLocalStorage('gru-multi-stock-model');
    setProgress(100,'Saved to localstorage');
    appLog('Model weights saved to localstorage.');
  } catch (err) {
    appLog('Save error:', err.message || err);
    alert('Save error: ' + (err.message || err));
    setProgress(0,'Error');
  }
});

loadBtn.addEventListener('click', async () => {
  try {
    setProgress(0,'Loading model from localstorage (if present)...');
    await model.loadFromLocalStorage('gru-multi-stock-model');
    setProgress(100,'Loaded (if present)');
    appLog('Model loaded from localstorage (if exists).');
    predictBtn.disabled = false;
    saveBtn.disabled = false;
  } catch (err) {
    appLog('Load error (may be no saved model):', err.message || err);
    setProgress(0,'No saved model found / load failed');
    alert('Load failed (no saved model or incompatible): ' + (err.message || err));
  }
});

// when selecting a stock in dropdown, update timeline panel
stockSelect.addEventListener('change', () => {
  if (!evalResults) return;
  renderStockTimelineForSelected(preparedMeta.symbols, evalResults.perStockTimeline, evalResults.perStockAcc, preparedMeta.sampleDatesTest);
});

// render accuracy bar chart
function renderAccuracyBar(sorted) {
  // sorted: [{symbol, acc, idx}] best->worst
  const labels = sorted.map(s => s.symbol);
  const data = sorted.map(s => s.acc);
  if (charts.accuracyBar) charts.accuracyBar.destroy();
  charts.accuracyBar = new Chart(accuracyBarCanvas, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Accuracy (%)', data }] },
    options: {
      indexAxis: 'y',
      scales: { x: { min:0, max:100 } },
      plugins: { legend: { display:false } }
    }
  });
}

// empty placeholder
function renderEmptyAccuracy() {
  if (charts.accuracyBar) charts.accuracyBar.destroy();
  charts.accuracyBar = new Chart(accuracyBarCanvas, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Accuracy (%)', data: [] }] },
    options: { indexAxis: 'y', plugins: { legend: { display:false } } }
  });
}

// Render per-stock timelines (compact rows)
function renderPerStockTimelines(perStockTimeline, symbols, sampleDates) {
  // perStockTimeline: [S arrays of length testSamples] values 0/1
  timelinesContainer.innerHTML = '';
  // dispose old small charts
  charts.perStockSmall.forEach(c => { try { c.destroy(); } catch (e){} });
  charts.perStockSmall = [];

  const S = symbols.length;
  for (let s = 0; s < S; s++) {
    const row = document.createElement('div');
    row.className = 'timeline-row';
    const label = document.createElement('div');
    label.style.minWidth = '72px';
    label.textContent = symbols[s];
    const canvas = document.createElement('canvas');
    canvas.height = 40;
    // small width based on number of points
    canvas.style.width = '100%';
    canvas.style.flex = '1 1 auto';
    row.appendChild(label);
    row.appendChild(canvas);
    timelinesContainer.appendChild(row);

    const data = perStockTimeline[s].map(v => v ? 1 : 0);
    // colors: map 1->green, 0->red. Chart.js allows segment coloring per bar via backgroundColor array.
    const bg = data.map(v => v ? 'rgba(0,160,80,0.9)' : 'rgba(200,40,40,0.9)');
    const chart = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: sampleDates.map(d=>d),
        datasets: [{ data, backgroundColor: bg, barPercentage: 1.0, categoryPercentage: 1.0 }]
      },
      options: {
        animation: false,
        plugins: { legend: { display:false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false, min:0, max:1 } }
      }
    });
    charts.perStockSmall.push(chart);
  }
}

// Render selected stock timeline in bigger chart and show stats
function renderStockTimelineForSelected(symbols, perStockTimeline, perStockAcc, sampleDates) {
  const selected = stockSelect.value || symbols[0];
  const idx = symbols.indexOf(selected);
  if (idx < 0) return;
  const arr = perStockTimeline[idx];
  const data = arr.map(v => v ? 1 : 0);
  const bg = data.map(v => v ? 'rgba(0,160,80,0.9)' : 'rgba(200,40,40,0.9)');

  if (charts.stockTimeline) charts.stockTimeline.destroy();
  charts.stockTimeline = new Chart(stockTimelineCanvas.getContext('2d'), {
    type: 'bar',
    data: { labels: sampleDates.map(d=>d), datasets: [{ label: `${selected} correctness`, data, backgroundColor: bg }] },
    options: {
      animation:false,
      plugins: { leg
