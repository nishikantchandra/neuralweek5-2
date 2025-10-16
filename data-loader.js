// data-loader.js
// ES module that parses CSV, pivots data, normalizes and prepares sliding-window tensors.
// Exports: DataLoader class
export class DataLoader {
  constructor({ sequenceLength = 12, forecastHorizon = 3 } = {}) {
    this.sequenceLength = sequenceLength;
    this.forecastHorizon = forecastHorizon; // 3
    this.raw = null;
    this.symbols = [];
    this.dates = []; // sorted
    this.perSymbolSeries = {}; // symbol -> array of {date, open, close}
    this.normalizers = {}; // symbol -> {openMin, openMax, closeMin, closeMax}
    this.X_train = null; this.y_train = null; this.X_test = null; this.y_test = null;
    this.trainIndices = null; // metadata
    this.testIndices = null;
  }

  log(...args) { if (this.logger) this.logger(...args); }
  setLogger(fn){ this.logger = fn; }

  // Accept File object from input
  async loadFromFile(file) {
    if (!file) throw new Error("No file provided");
    const text = await file.text();
    return this.loadFromCSVText(text);
  }

  // Accept raw CSV text
  async loadFromCSVText(csvText) {
    this.log("Parsing CSV...");
    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
    if (parsed.errors && parsed.errors.length) {
      this.log("CSV parse errors:", parsed.errors.slice(0,5));
    }
    const rows = parsed.data.map(r => ({
      Date: (r.Date || r.date || r.datetime || r.Timestamp || r.timestamp || "").trim(),
      Symbol: (r.Symbol || r.symbol || r.Ticker || r.ticker || "").trim(),
      Open: parseFloat(r.Open || r.open || r.O || r.o),
      Close: parseFloat(r.Close || r.close || r.C || r.c)
    })).filter(r => r.Date && r.Symbol && !Number.isNaN(r.Open) && !Number.isNaN(r.Close));

    if (!rows.length) throw new Error("No valid rows in CSV");

    // Normalize date strings -> ISO (try common formats)
    rows.forEach(r => {
      // try Date.parse
      const d = new Date(r.Date);
      if (isNaN(d)) {
        // try splitting common formats like dd/mm/yyyy or dd-mm-yyyy
        const mm = r.Date.replace(/\./g,'-').replace(/\//g,'-').trim();
        const d2 = new Date(mm);
        if (!isNaN(d2)) r.Date = d2.toISOString().slice(0,10);
        else r.Date = r.Date; // leave as-is
      } else {
        r.Date = d.toISOString().slice(0,10);
      }
    });

    // collect symbols and dates
    const symbolSet = new Set();
    const dateSet = new Set();
    rows.forEach(r => { symbolSet.add(r.Symbol); dateSet.add(r.Date); });

    this.symbols = Array.from(symbolSet).sort();
    this.dates = Array.from(dateSet).sort((a,b) => new Date(a)-new Date(b));
    if (this.symbols.length !== 10) {
      this.log(`Warning: found ${this.symbols.length} distinct symbols (expected 10).`);
    }

    // pivot: build perSymbolSeries with complete dates if possible
    this.perSymbolSeries = {};
    for (const sym of this.symbols) this.perSymbolSeries[sym] = [];

    // index rows by sym+date for fast lookup
    const lookup = new Map();
    rows.forEach(r => lookup.set(`${r.Symbol}||${r.Date}`, { open: r.Open, close: r.Close }));

    // For each date and symbol, if missing skip (we will allow only samples with full windows)
    for (const sym of this.symbols) {
      for (const date of this.dates) {
        const v = lookup.get(`${sym}||${date}`);
        if (v) this.perSymbolSeries[sym].push({ date, open: v.open, close: v.close });
        else this.perSymbolSeries[sym].push({ date, open: NaN, close: NaN });
      }
    }

    // Clean rows with NaNs by marking them - we'll only create samples with no NaNs
    // compute normalizers per symbol on available values
    this.normalizers = {};
    for (const sym of this.symbols) {
      const arr = this.perSymbolSeries[sym];
      const opens = arr.map(a=>a.open).filter(v=>!Number.isNaN(v));
      const closes = arr.map(a=>a.close).filter(v=>!Number.isNaN(v));
      const openMin = Math.min(...opens), openMax = Math.max(...opens);
      const closeMin = Math.min(...closes), closeMax = Math.max(...closes);
      // protect against zero range
      this.normalizers[sym] = {
        openMin: openMin, openMax: openMax === openMin ? openMin + 1e-6 : openMax,
        closeMin: closeMin, closeMax: closeMax === closeMin ? closeMin + 1e-6 : closeMax
      };
    }

    this.raw = { rows, parsedCount: rows.length };
    this.log(`Parsed ${rows.length} rows. Symbols: ${this.symbols.length}. Dates: ${this.dates.length}.`);
    return { symbols: this.symbols, dates: this.dates };
  }

  // Create sliding-window dataset given trainSplitPercent (chronological)
  prepareDataset({ trainSplitPercent = 80 } = {}) {
    if (!this.raw) throw new Error("No data loaded. Call loadFromFile or loadFromCSVText first.");
    const S = this.symbols.length;
    const D = this.dates.length;

    // For each possible index idx that corresponds to day D_i as the "anchor D"
    // anchor index i refers to date D_i; we need previous sequenceLength days: i - sequenceLength + 1 .. i
    // but user asked: Input: for each date D provide last 12 days' features (so anchor is D, using last 12 days up to D)
    // and outputs compare Close(t+offset) > Close(D) for offset 1..3 -> need up to i + forecastHorizon
    const seq = this.sequenceLength;
    const h = this.forecastHorizon;

    const inputSamples = [];
    const outputSamples = [];
    const sampleDates = []; // anchor date for each sample (D)

    for (let i = 0; i < D; i++) {
      const seqStart = i - seq + 1;
      const futureEnd = i + h;
      if (seqStart < 0) continue;
      if (futureEnd >= D) continue;

      // check if any NaN in required data
      let bad = false;
      // also we need close(D) per symbol
      for (let s = 0; s < S; s++) {
        for (let t = seqStart; t <= i; t++) {
          const v = this.perSymbolSeries[this.symbols[s]][t];
          if (Number.isNaN(v.open) || Number.isNaN(v.close)) { bad = true; break; }
        }
        // check future close existence for offsets 1..h
        for (let offset = 1; offset <= h; offset++) {
          const v2 = this.perSymbolSeries[this.symbols[s]][i + offset];
          if (!v2 || Number.isNaN(v2.close)) { bad = true; break; }
        }
        if (bad) break;
      }
      if (bad) continue;

      // Build input: shape [seq, S*2]
      const sampleInput = [];
      for (let t = seqStart; t <= i; t++) {
        const row = [];
        for (let s = 0; s < S; s++) {
          const v = this.perSymbolSeries[this.symbols[s]][t];
          const norm = this.normalizers[this.symbols[s]];
          // min-max normalize per-feature per stock
          const on = (v.open - norm.openMin) / (norm.openMax - norm.openMin);
          const cn = (v.close - norm.closeMin) / (norm.closeMax - norm.closeMin);
          row.push(on, cn);
        }
        sampleInput.push(row);
      }

      // Build output: for each stock, for offsets 1..h, label = Close(t+offset) > Close(D) ? 1 : 0
      const sampleOutput = [];
      for (let s = 0; s < S; s++) {
        const closeD = this.perSymbolSeries[this.symbols[s]][i].close;
        for (let offset = 1; offset <= h; offset++) {
          const futureClose = this.perSymbolSeries[this.symbols[s]][i + offset].close;
          sampleOutput.push(futureClose > closeD ? 1 : 0);
        }
      }

      inputSamples.push(sampleInput);
      outputSamples.push(sampleOutput);
      sampleDates.push(this.dates[i]); // anchor date
    }

    if (!inputSamples.length) throw new Error("No valid sliding-window samples could be constructed (missing data or short series).");

    // chronological split
    const total = inputSamples.length;
    const trainCount = Math.round((trainSplitPercent/100) * total);
    const X = tf.tensor3d(inputSamples); // shape [samples, seq, S*2]
    const y = tf.tensor2d(outputSamples); // shape [samples, S*h]

    // split
    const X_train = X.slice([0,0,0],[trainCount, seq, S*2]);
    const X_test = X.slice([trainCount,0,0],[total - trainCount, seq, S*2]);
    const y_train = y.slice([0,0],[trainCount, S*h]);
    const y_test = y.slice([trainCount,0],[total - trainCount, S*h]);

    // store meta
    this.X_train = X_train;
    this.y_train = y_train;
    this.X_test = X_test;
    this.y_test = y_test;
    this.trainIndices = { start:0, count: trainCount };
    this.testIndices = { start:trainCount, count: total - trainCount };
    this.sampleDates = sampleDates; // anchor dates per sample (global)
    this.sampleDatesTrain = sampleDates.slice(0, trainCount);
    this.sampleDatesTest = sampleDates.slice(trainCount);

    // expose shapes
    const meta = {
      samples: total,
      trainSamples: trainCount,
      testSamples: total - trainCount,
      sequenceLength: seq,
      featuresPerStep: S*2,
      outputDim: S*h,
      symbols: this.symbols.slice()
    };

    this.log(`Prepared dataset. Total samples: ${total}. Train: ${trainCount}. Test: ${total - trainCount}.`);

    return meta;
  }

  // Provide getters that return raw tensors (already tf.Tensor) - consumer must not dispose them; but class will have dispose()
  getTensors() {
    if (!this.X_train) throw new Error("Dataset not prepared yet. Call prepareDataset()");
    return {
      X_train: this.X_train,
      y_train: this.y_train,
      X_test: this.X_test,
      y_test: this.y_test,
      symbols: this.symbols,
      sampleDatesTest: this.sampleDatesTest
    };
  }

  // Dispose stored tensors to avoid memory leak
  dispose() {
    if (this.X_train) { this.X_train.dispose(); this.X_train = null; }
    if (this.y_train) { this.y_train.dispose(); this.y_train = null; }
    if (this.X_test) { this.X_test.dispose(); this.X_test = null; }
    if (this.y_test) { this.y_test.dispose(); this.y_test = null; }
    this.raw = null;
  }
}
