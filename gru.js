// gru.js
// ES module: defines the GRU model, training, predict/eval utilities.
// Exports: GRUModel class
export class GRUModel {
  constructor({ inputShape = [12, 20], gruUnits = 64, denseUnits = 30, learningRate = 0.001 } = {}) {
    this.inputShape = inputShape; // [seqLen, features]
    this.gruUnits = gruUnits;
    this.denseUnits = denseUnits; // 10 stocks * 3 days = 30
    this.learningRate = learningRate;
    this.model = null;
  }

  buildModel({ bidirectional = false, returnSequences = false } = {}) {
    // dispose existing model if present
    if (this.model) {
      try { this.model.dispose(); } catch (e) {}
      this.model = null;
    }

    const tf = window.tf;
    // Input layer
    const layers = [];
    // Use functional API for clarity
    const input = tf.input({ shape: this.inputShape });

    let x = input;
    // First GRU
    const gru1 = tf.layers.gru({
      units: this.gruUnits,
      returnSequences: true,
      activation: 'tanh',
      recurrentActivation: 'sigmoid'
    });
    x = gru1.apply(x);
    // Second GRU (can be returnSequences false)
    const gru2 = tf.layers.gru({
      units: Math.max(16, Math.floor(this.gruUnits/2)),
      returnSequences: returnSequences,
      activation: 'tanh',
      recurrentActivation: 'sigmoid'
    });
    x = gru2.apply(x);

    // If second returned sequences, we may need to flatten or take last timestep
    if (Array.isArray(x)) x = x[0];

    // Dense output layer with sigmoid for binary outputs
    const out = tf.layers.dense({ units: this.denseUnits, activation: 'sigmoid' }).apply(x);

    this.model = tf.model({ inputs: input, outputs: out });
    const optimizer = tf.train.adam(this.learningRate);
    this.model.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: [tf.metrics.binaryAccuracy]
    });

    return this.model;
  }

  // Fit with callback to update UI
  async fit(X_train, y_train, { epochs = 30, batchSize = 32, onEpochEnd = null } = {}) {
    if (!this.model) this.buildModel();
    const tf = window.tf;
    const callbacks = {
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) onEpochEnd(epoch, logs);
        await tf.nextFrame();
      }
    };
    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle: false, // time-series => don't shuffle
      callbacks
    });
    return history;
  }

  // Predict on X (tf.Tensor)
  predict(X) {
    if (!this.model) throw new Error("Model not built.");
    return tf.tidy(() => {
      const preds = this.model.predict(X);
      // preds shape [samples, denseUnits]
      return preds.clone();
    });
  }

  // Compute per-stock accuracy averaged over 3 horizons
  // y_true and y_pred are tensors [samples, S*h] where S* h = denseUnits
  // Returns { perStockAcc: number[], overallAcc }
  async computePerStockAccuracy(y_true, y_pred, S, h) {
    const tf = window.tf;
    return tf.tidy(() => {
      // Ensure binary threshold 0.5 for preds
      const predsBin = y_pred.greater(tf.scalar(0.5)).toInt();
      const truths = y_true.toInt();
      // reshape to [samples, S, h]
      const samples = predsBin.shape[0];
      const preds3 = predsBin.reshape([samples, S, h]);
      const truths3 = truths.reshape([samples, S, h]);
      // correctness per sample per stock per horizon
      const correct = preds3.equal(truths3).toFloat(); // 1/0
      // For each stock, average across samples and horizons
      // mean over axis [0,2] => result shape [S]
      const perStockAcc = correct.mean([0,2]); // tensor length S
      const overallAcc = correct.mean(); // scalar
      return {
        perStockAcc: perStockAcc.arraySync(),
        overallAcc: overallAcc.arraySync()
      };
    });
  }

  // Evaluate and compute per-sample correctness array (for timeline plotting)
  // returns an object { perStockAcc, overallAcc, perStockTimeline: [S arrays of 0/1 per test sample], perSamplePreds, perSampleTruths }
  async evaluateTestSet(X_test, y_test, symbols, horizon=3) {
    const tf = window.tf;
    const S = symbols.length;
    const preds = await this.predict(X_test); // tensor
    const samples = preds.shape[0];
    // compute binary predictions
    const predsBin = preds.greater(tf.scalar(0.5)).toInt();
    const truths = y_test.toInt();
    const preds3 = predsBin.reshape([samples, S, horizon]);
    const truths3 = truths.reshape([samples, S, horizon]);

    // compute correctness per stock per sample: average over horizon -> mark correct if all horizons correct? user asked averaged over 3 output days.
    // We'll compute correctness per horizon and also averaged correctness per stock per sample (mean of horizons)
    const correctPerHorizon = preds3.equal(truths3).toInt(); // shape [samples, S, horizon]
    const correctPerSamplePerStock = correctPerHorizon.mean(2).toFloat(); // mean across horizon => 0..1
    // For timeline, we can treat correctness >0.5 as correct, else wrong (i.e., majority of horizons correct)
    const timelineBinary = correctPerSamplePerStock.greater(tf.scalar(0.5)).toInt();

    // convert to JS arrays
    const perStockTimeline = [];
    const cp = timelineBinary.transpose([1,0]); // [S, samples]
    const cpArr = cp.arraySync();
    for (let s = 0; s < S; s++) {
      perStockTimeline.push(cpArr[s]); // array length = samples of 0/1
    }

    // per-stock accuracies averaged over samples and horizons
    const { perStockAcc, overallAcc } = await this.computePerStockAccuracy(y_test, preds, S, horizon);

    // return also raw preds/truths as arrays for optional confusion matrices
    const predArr = preds3.arraySync(); // [samples, S, h]
    const truthArr = truths3.arraySync();

    // dispose local tensors
    preds.dispose(); predsBin.dispose(); preds3.dispose(); truths.dispose(); truths3.dispose();
    correctPerHorizon.dispose(); correctPerSamplePerStock.dispose(); timelineBinary.dispose(); cp.dispose();

    return {
      perStockAcc,
      overallAcc,
      perStockTimeline,
      predArr,
      truthArr
    };
  }

  // Save model weights to localstorage (key)
  async saveToLocalStorage(key = 'gru-multi-stock-model') {
    if (!this.model) throw new Error("No model to save");
    return await this.model.save(`localstorage://${key}`);
  }

  // Load model from localstorage key
  async loadFromLocalStorage(key = 'gru-multi-stock-model') {
    const tf = window.tf;
    const loaded = await tf.loadLayersModel(`localstorage://${key}`);
    // replace model
    if (this.model) try { this.model.dispose(); } catch(e){}
    this.model = loaded;
    this.model.compile({ optimizer: tf.train.adam(this.learningRate), loss: 'binaryCrossentropy', metrics: [tf.metrics.binaryAccuracy] });
    return this.model;
  }

  dispose() {
    if (this.model) {
      try { this.model.dispose(); } catch(e){}
      this.model = null;
    }
  }
}
