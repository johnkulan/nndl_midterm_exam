// Browser-only EDA and ML for Kaggle weather.csv
// - Schema: target = weather, features = precipitation, temp_max, temp_min, wind
// - All processing runs client-side using PapaParse + Chart.js + TensorFlow.js

/* ============================
   Schema & state
   ============================ */
const schema = {
  target: "weather",
  features: ["precipitation", "temp_max", "temp_min", "wind"],
  dateColumn: "date"
};
let rawData = [];      // parsed rows
let charts = {};       // Chart.js instances
let lastSummary = null; // for exports
let weatherModel = null; // GRU model instance
let trainingData = null; // processed training data
let testData = null;    // processed test data
let predictions = null; // model predictions
let labelEncoder = {};  // weather label encoding
let labelDecoder = {};  // reverse mapping

/* ============================
   Utilities / stats
   ============================ */
const $ = id => document.getElementById(id);
function isMissing(v){ return v === null || v === undefined || v === "" || (typeof v === "number" && isNaN(v)); }
function mean(arr){ const a = arr.filter(x=>!isMissing(x)).map(Number); return a.length ? a.reduce((s,v)=>s+v,0)/a.length : null; }
function median(arr){ const a = arr.filter(x=>!isMissing(x)).map(Number).sort((p,q)=>p-q); if(!a.length) return null; const m=Math.floor(a.length/2); return a.length%2? a[m] : (a[m-1]+a[m])/2; }
function std(arr){ const a = arr.filter(x=>!isMissing(x)).map(Number); if(a.length<2) return 0; const m = mean(a); const v = a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1); return Math.sqrt(v); }
function round(n, d=3){ return (n === null || n === undefined || isNaN(n)) ? "" : Number(n.toFixed(d)); }

/* Pearson correlation */
function pearson(a,b){
  const x=[], y=[];
  for(let i=0;i<a.length;i++){
    if(!isMissing(a[i]) && !isMissing(b[i])){ x.push(Number(a[i])); y.push(Number(b[i])); }
  }
  if(x.length<2) return 0;
  const mx = mean(x), my = mean(y);
  let num=0, sx=0, sy=0;
  for(let i=0;i<x.length;i++){ num += (x[i]-mx)*(y[i]-my); sx += (x[i]-mx)**2; sy += (y[i]-my)**2; }
  const denom = Math.sqrt(sx*sy);
  return denom === 0 ? 0 : num/denom;
}

/* ============================
   DOM: preview & overview
   ============================ */
function showPreview(rows, limit = 10){
  const el = $("preview");
  if(!rows || rows.length === 0){ el.innerHTML = "<div class='muted small'>No preview available.</div>"; return; }
  const cols = Object.keys(rows[0]);
  let html = "<table><thead><tr>" + cols.map(c=>`<th>${c}</th>`).join("") + "</tr></thead><tbody>";
  for(const r of rows.slice(0,limit)){
    html += "<tr>" + cols.map(c=>`<td>${r[c] ?? ""}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";
  el.innerHTML = html;
}

/* ============================
   CSV loading (file input)
   ============================ */
function handleFileLoad(file){
  if(!file){ alert("No file chosen. Please select weather.csv."); return; }
  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    quotes: true,
    dynamicTyping: true,
    transformHeader: h => h.trim(),
    complete: results => {
      rawData = results.data;
      $("overview").innerText = `Loaded ${rawData.length} rows. Columns: ${Object.keys(rawData[0] || {}).join(", ")}`;
      showPreview(rawData, 15);
      clearAllCharts();
      disableExportLinks();
      resetModelState();
    },
    error: err => {
      console.error("Parse error", err);
      alert("Error parsing file: " + (err.message || err));
    }
  });
}

function resetModelState() {
  weatherModel = null;
  trainingData = null;
  testData = null;
  predictions = null;
  labelEncoder = {};
  labelDecoder = {};
  $("trainBtn").disabled = false;
  $("evaluateBtn").disabled = true;
}

/* ============================
   Missing values
   ============================ */
function computeMissingPercent(data){
  if(!data || data.length === 0) return {};
  const cols = Object.keys(data[0]);
  const res = {};
  for(const c of cols){
    let miss = 0;
    for(const r of data) if(isMissing(r[c])) miss++;
    res[c] = +(100 * miss / data.length).toFixed(2);
  }
  return res;
}
function renderMissingChart(obj){
  const ctx = $("missingChart").getContext("2d");
  if(charts.missing) charts.missing.destroy();
  const labels = Object.keys(obj);
  const data = labels.map(l=>obj[l]);
  charts.missing = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "% missing", data }] },
    options: { responsive: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: t => t.parsed.y + "%" } } }, scales: { y: { ticks: { callback: v => v + "%" } } } }
  });
}

/* ============================
   Stats summary
   ============================ */
function computeStats(data){
  if(!data || data.length === 0) return null;
  const stats = { numeric: {}, categorical: {}, groups: {} };
  // numeric
  for(const f of schema.features){
    const vals = data.map(r=>r[f]);
    stats.numeric[f] = {
      mean: round(mean(vals), 4),
      median: round(median(vals), 4),
      std: round(std(vals), 4),
      missing: vals.filter(isMissing).length
    };
  }
  // categorical counts
  const tgt = schema.target;
  stats.categorical[tgt] = {};
  for(const r of data){
    const v = r[tgt] === undefined || r[tgt] === null ? "" : String(r[tgt]);
    stats.categorical[tgt][v] = (stats.categorical[tgt][v] || 0) + 1;
  }
  // group by target
  const groups = {};
  for(const r of data){
    const key = r[tgt] === undefined || r[tgt] === null ? "" : String(r[tgt]);
    groups[key] = groups[key] || { count: 0 };
    groups[key].count++;
    for(const f of schema.features){
      groups[key][f + "_vals"] = groups[key][f + "_vals"] || [];
      if(!isMissing(r[f])) groups[key][f + "_vals"].push(Number(r[f]));
    }
  }
  for(const k of Object.keys(groups)){
    for(const f of schema.features){
      const arr = groups[k][f + "_vals"];
      groups[k][f + "_mean"] = arr.length ? round(mean(arr),4) : "";
      groups[k][f + "_median"] = arr.length ? round(median(arr),4) : "";
      groups[k][f + "_std"] = arr.length ? round(std(arr),4) : "";
      delete groups[k][f + "_vals"];
    }
  }
  stats.groups = groups;
  return stats;
}
function renderStatsTable(stats){
  const el = $("stats");
  if(!stats){ el.innerHTML = "<div class='muted small'>No stats yet.</div>"; return; }

  // numeric summary
  let html = "<strong>Numeric summary</strong>";
  html += "<div class='preview' style='max-height:220px;overflow:auto;margin-top:8px'><table><thead><tr><th>feature</th><th>mean</th><th>median</th><th>std</th><th>missing</th></tr></thead><tbody>";
  for(const f of Object.keys(stats.numeric)){
    const s = stats.numeric[f];
    html += `<tr><td>${f}</td><td>${s.mean}</td><td>${s.median}</td><td>${s.std}</td><td>${s.missing}</td></tr>`;
  }
  html += "</tbody></table></div>";

  // categorical counts
  html += "<div style='margin-top:10px'><strong>Categorical counts</strong><div class='preview' style='max-height:140px;overflow:auto;margin-top:8px'><table>";
  const cat = stats.categorical[schema.target];
  const keys = Object.keys(cat).sort((a,b)=>cat[b]-cat[a]);
  for(const k of keys) html += `<tr><td>${k}</td><td>${cat[k]}</td></tr>`;
  html += "</table></div></div>";

  // group by
  html += "<div style='margin-top:10px'><strong>Group (by " + schema.target + ")</strong><div class='preview' style='max-height:200px;overflow:auto;margin-top:8px'><table><thead><tr><th>group</th><th>count</th>";
  for(const f of schema.features) html += `<th>${f}_mean</th><th>${f}_median</th><th>${f}_std</th>`;
  html += "</tr></thead><tbody>";
  for(const g of Object.keys(stats.groups)){
    const gr = stats.groups[g];
    html += `<tr><td>${g}</td><td>${gr.count}</td>`;
    for(const f of schema.features) html += `<td>${gr[f + "_mean"]}</td><td>${gr[f + "_median"]}</td><td>${gr[f + "_std"]}</td>`;
    html += "</tr>";
  }
  html += "</tbody></table></div></div>";

  el.innerHTML = html;
}

/* ============================
   Visualizations
   ============================ */

function clearAllCharts(){
  for(const k in charts) try{ charts[k].destroy(); }catch(e){}
  charts = {};
}

/* Histogram builder */
function renderHistogram(canvasId, values, label, bins = 20){
  const ctx = document.getElementById(canvasId).getContext("2d");
  if(charts[canvasId]) charts[canvasId].destroy();
  const nums = values.filter(v=>!isMissing(v)).map(Number);
  if(nums.length === 0){
    charts[canvasId] = new Chart(ctx,{type:'bar',data:{labels:[],datasets:[]},options:{plugins:{legend:{display:false}}}});
    return;
  }
  const min = Math.min(...nums), max = Math.max(...nums);
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  for(const v of nums){
    const idx = Math.min(bins - 1, Math.floor((v - min) / (step || 1)));
    counts[idx]++;
  }
  const labels = Array.from({length:bins},(_,i)=>`${round(min + i*step,2)}-${round(min + (i+1)*step,2)}`);
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label, data: counts }] },
    options: { responsive:true, plugins:{legend:{display:false}}, scales:{ x:{ ticks:{ maxRotation:0 } } } }
  });
}

/* Doughnut with percentage labels on slices (custom plugin) */
function renderDoughnutWithPercents(canvasId, countsObj){
  const ctx = document.getElementById(canvasId).getContext("2d");
  if(charts[canvasId]) charts[canvasId].destroy();
  const labels = Object.keys(countsObj);
  const values = labels.map(l=>countsObj[l]);
  const total = values.reduce((a,b)=>a+b,0);

  // plugin: draw percent text on each slice
  const percentPlugin = {
    id: 'slicePercent',
    afterDraw(chart){
      const {ctx, chartArea: {width, height}} = chart;
      chart.data.datasets.forEach((dataset, datasetIndex) => {
        const meta = chart.getDatasetMeta(datasetIndex);
        meta.data.forEach((arc, i) => {
          const value = dataset.data[i];
          if(!value) return;
          const percent = (value / total) * 100;
          const pos = arc.getCenterPoint ? arc.getCenterPoint() : arc.tooltipPosition();
          ctx.save();
          ctx.fillStyle = '#fff';
          ctx.font = '12px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(percent.toFixed(1) + '%', pos.x, pos.y);
          ctx.restore();
        });
      });
    }
  };

  charts[canvasId] = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values }] },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => {
              const v = ctx.parsed;
              const pct = total ? (v / total * 100).toFixed(1) : 0;
              return `${ctx.label}: ${pct}% (${v})`;
            }
          }
        },
        legend: { position: 'right' }
      }
    },
    plugins: [percentPlugin]
  });
}

/* ============================
   Export helpers
   ============================ */
function enableExportLinks(summary){
  lastSummary = summary;
  try{
    const csvRows = summary.csvRows || [];
    const csv = Papa.unparse(csvRows);
    $("downloadCSV").href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const json = JSON.stringify(summary.jsonDesc || {}, null, 2);
    $("downloadJSON").href = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
  }catch(e){
    console.error("Export error", e);
    alert("Error preparing exports: " + e.message);
  }
}
function disableExportLinks(){
  $("downloadCSV").href = "#";
  $("downloadJSON").href = "#";
  lastSummary = null;
}

/* ============================
   ROC AUC Utilities
   ============================ */

// Calculate ROC curve points for a single class
function calculateROCCurve(scores, trueLabels, positiveClass) {
  const thresholds = Array.from({ length: 101 }, (_, i) => i / 100).reverse();
  const rocPoints = [];
  
  for (const threshold of thresholds) {
    let truePositives = 0;
    let falsePositives = 0;
    let trueNegatives = 0;
    let falseNegatives = 0;
    
    for (let i = 0; i < scores.length; i++) {
      const predictedPositive = scores[i] >= threshold;
      const actuallyPositive = trueLabels[i] === positiveClass;
      
      if (predictedPositive && actuallyPositive) truePositives++;
      else if (predictedPositive && !actuallyPositive) falsePositives++;
      else if (!predictedPositive && actuallyPositive) falseNegatives++;
      else trueNegatives++;
    }
    
    const tpr = truePositives / (truePositives + falseNegatives) || 0;
    const fpr = falsePositives / (falsePositives + trueNegatives) || 0;
    
    rocPoints.push({
      threshold,
      tpr,
      fpr,
      truePositives,
      falsePositives,
      trueNegatives,
      falseNegatives
    });
  }
  
  return rocPoints;
}

// Calculate AUC using trapezoidal rule
function calculateAUC(rocPoints) {
  let auc = 0;
  for (let i = 1; i < rocPoints.length; i++) {
    const prev = rocPoints[i - 1];
    const curr = rocPoints[i];
    auc += (curr.fpr - prev.fpr) * (curr.tpr + prev.tpr) / 2;
  }
  return Math.abs(auc);
}

// Calculate multiclass ROC AUC using One-vs-Rest approach
function calculateMulticlassROCAUC(predictions, trueLabels, numClasses) {
  const classAUCs = [];
  const rocCurves = [];
  
  for (let classIndex = 0; classIndex < numClasses; classIndex++) {
    const scores = predictions.map(pred => pred[classIndex]);
    const binaryTrueLabels = trueLabels.map(label => label === classIndex ? 1 : 0);
    
    const rocPoints = calculateROCCurve(scores, binaryTrueLabels, 1);
    const auc = calculateAUC(rocPoints);
    
    classAUCs.push({
      class: classIndex,
      className: labelDecoder[classIndex],
      auc: auc
    });
    
    rocCurves.push({
      classIndex,
      className: labelDecoder[classIndex],
      rocPoints,
      auc
    });
  }
  
  // Calculate macro-average AUC
  const macroAUC = classAUCs.reduce((sum, cls) => sum + cls.auc, 0) / numClasses;
  
  return {
    classAUCs,
    rocCurves,
    macroAUC
  };
}

/* ============================
   GRU Model Implementation
   ============================ */

class WeatherGRUModel {
    constructor(numClasses, numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.model = null;
        this.isTrained = false;
        this.buildModel();
    }

    buildModel() {
        // Clear any existing model
        if (this.model) {
            this.model.dispose();
        }

        this.model = tf.sequential({
            layers: [
                // Input shape: [null, 12, numFeatures] - [batch, timesteps, features]
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: [12, this.numFeatures]
                }),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.gru({
                    units: 32,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.3 }),
                
                tf.layers.dense({
                    units: 16,
                    activation: 'relu'
                }),
                
                // Multi-output classification layer
                tf.layers.dense({
                    units: this.numClasses,
                    activation: 'softmax'
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log('GRU model built successfully');
    }

    async train(X, y, options = {}) {
        const {
            epochs = 50,
            batchSize = 32,
            validationSplit = 0.2,
            callbacks = {}
        } = options;

        try {
            const history = await this.model.fit(X, y, {
                epochs,
                batchSize,
                validationSplit,
                shuffle: true,
                callbacks: callbacks
            });

            this.isTrained = true;
            return history;
        } catch (error) {
            console.error('Training error:', error);
            throw new Error(`Model training failed: ${error.message}`);
        }
    }

    predict(X) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before making predictions');
        }
        return this.model.predict(X);
    }

    async evaluate(X, y) {
        if (!this.isTrained) {
            throw new Error('Model must be trained before evaluation');
        }

        const results = this.model.evaluate(X, y);
        const loss = await results[0].data();
        const accuracy = await results[1].data();

        // Clean up tensors
        results[0].dispose();
        results[1].dispose();

        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

/* ============================
   ML Model Integration
   ============================ */

async function prepareTrainingData() {
  if (!rawData || rawData.length === 0) {
    throw new Error("No data loaded. Please load CSV first.");
  }

  // Filter out rows with missing values in features or target
  const cleanData = rawData.filter(row => {
    const hasMissingFeatures = schema.features.some(f => isMissing(row[f]));
    const hasMissingTarget = isMissing(row[schema.target]);
    return !hasMissingFeatures && !hasMissingTarget;
  });

  if (cleanData.length < 100) {
    throw new Error(`Insufficient clean data (${cleanData.length} rows). Need at least 100 rows.`);
  }

  // Encode weather labels
  const uniqueLabels = [...new Set(cleanData.map(row => String(row[schema.target])))];
  uniqueLabels.forEach((label, index) => {
    labelEncoder[label] = index;
    labelDecoder[index] = label;
  });

  const numClasses = uniqueLabels.length;

  // Normalize features
  const featureData = {};
  schema.features.forEach(feature => {
    const values = cleanData.map(row => Number(row[feature]));
    const meanVal = mean(values);
    const stdVal = std(values);
    featureData[feature] = { values, mean: meanVal, std: stdVal };
  });

  // Create sequences for time series (using lookback of 12 days)
  const lookback = 12;
  const sequences = [];
  const targets = [];

  for (let i = lookback; i < cleanData.length; i++) {
    const sequence = [];
    for (let j = i - lookback; j < i; j++) {
      const featureVector = schema.features.map(feature => {
        const normalized = (featureData[feature].values[j] - featureData[feature].mean) / featureData[feature].std;
        return isNaN(normalized) ? 0 : normalized;
      });
      sequence.push(featureVector);
    }
    
    sequences.push(sequence);
    targets.push(labelEncoder[String(cleanData[i][schema.target])]);
  }

  // Split into train/test (80/20)
  const splitIndex = Math.floor(sequences.length * 0.8);
  const trainSequences = sequences.slice(0, splitIndex);
  const testSequences = sequences.slice(splitIndex);
  const trainTargets = targets.slice(0, splitIndex);
  const testTargets = targets.slice(splitIndex);

  // Convert to tensors
  trainingData = {
    X: tf.tensor3d(trainSequences),
    y: tf.oneHot(tf.tensor1d(trainTargets, 'int32'), numClasses)
  };

  testData = {
    X: tf.tensor3d(testSequences),
    y: tf.oneHot(tf.tensor1d(testTargets, 'int32'), numClasses),
    originalTargets: testTargets
  };

  return {
    numSequences: sequences.length,
    numTrain: trainSequences.length,
    numTest: testSequences.length,
    numClasses: numClasses,
    featureStats: featureData
  };
}

async function trainModel() {
  try {
    $("trainBtn").disabled = true;
    $("overview").innerText = "Preparing training data...";

    const dataInfo = await prepareTrainingData();
    
    $("overview").innerText = `Training GRU model: ${dataInfo.numTrain} samples, ${dataInfo.numClasses} classes`;
    
    // Create model directly (no import needed)
    weatherModel = new WeatherGRUModel(dataInfo.numClasses, schema.features.length);
    
    // Train model with progress updates
    await weatherModel.train(trainingData.X, trainingData.y, {
      epochs: 30, // Reduced for faster training
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const loss = logs.loss ? logs.loss.toFixed(4) : 'N/A';
          const acc = logs.acc ? logs.acc.toFixed(4) : 'N/A';
          const valLoss = logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A';
          const valAcc = logs.val_acc ? logs.val_acc.toFixed(4) : 'N/A';
          $("overview").innerText = 
            `Epoch ${epoch + 1}/30 - loss: ${loss}, acc: ${acc}, val_loss: ${valLoss}, val_acc: ${valAcc}`;
        }
      }
    });

    $("overview").innerText = `Training complete! Model ready for evaluation.`;
    $("evaluateBtn").disabled = false;

    // Clean up tensors
    trainingData.X.dispose();
    trainingData.y.dispose();

  } catch (error) {
    console.error("Training error:", error);
    alert("Training failed: " + error.message);
    $("trainBtn").disabled = false;
  }
}

async function evaluateModel() {
  if (!weatherModel || !testData) {
    alert("No trained model or test data available. Please train the model first.");
    return;
  }

  try {
    $("evaluateBtn").disabled = true;
    $("overview").innerText = "Running evaluation...";

    // Make predictions
    const predictionsTensor = weatherModel.predict(testData.X);
    const predictedClasses = await predictionsTensor.argMax(1).array();
    const predictionProbabilities = await predictionsTensor.array();
    const trueClasses = testData.originalTargets;

    // Calculate accuracy per class
    const classAccuracy = {};
    const classCounts = {};
    
    trueClasses.forEach((trueClass, index) => {
      const predictedClass = predictedClasses[index];
      const className = labelDecoder[trueClass];
      
      if (!classAccuracy[className]) {
        classAccuracy[className] = { correct: 0, total: 0 };
        classCounts[className] = 0;
      }
      
      classCounts[className]++;
      if (trueClass === predictedClass) {
        classAccuracy[className].correct++;
      }
      classAccuracy[className].total++;
    });

    // Convert to percentage
    const accuracyData = {};
    Object.keys(classAccuracy).forEach(className => {
      accuracyData[className] = (classAccuracy[className].correct / classAccuracy[className].total) * 100;
    });

    // Calculate ROC AUC
    const rocResults = calculateMulticlassROCAUC(predictionProbabilities, trueClasses, Object.keys(labelDecoder).length);

    // Render all charts
    renderAccuracyChart(accuracyData, classCounts);
    renderPredictionTimeline(trueClasses, predictedClasses, labelDecoder);
    renderROCCurve(rocResults);

    // Calculate overall accuracy
    const correctPredictions = trueClasses.filter((trueClass, index) => trueClass === predictedClasses[index]).length;
    const overallAccuracy = (correctPredictions / trueClasses.length) * 100;

    $("overview").innerText = `Evaluation complete! Overall accuracy: ${overallAccuracy.toFixed(2)}%, Macro AUC: ${rocResults.macroAUC.toFixed(4)}`;

    // Clean up
    predictionsTensor.dispose();
    testData.X.dispose();
    testData.y.dispose();

  } catch (error) {
    console.error("Evaluation error:", error);
    alert("Evaluation failed: " + error.message);
    $("evaluateBtn").disabled = false;
  }
}

function renderAccuracyChart(accuracyData, countsData) {
  const ctx = document.getElementById('accuracyChart').getContext('2d');
  if (charts.accuracy) charts.accuracy.destroy();

  const labels = Object.keys(accuracyData);
  const accuracies = labels.map(label => accuracyData[label]);
  const counts = labels.map(label => countsData[label]);

  // Sort by accuracy (descending)
  const sortedIndices = accuracies.map((_, i) => i).sort((a, b) => accuracies[b] - accuracies[a]);
  const sortedLabels = sortedIndices.map(i => labels[i]);
  const sortedAccuracies = sortedIndices.map(i => accuracies[i]);
  const sortedCounts = sortedIndices.map(i => counts[i]);

  charts.accuracy = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sortedLabels,
      datasets: [{
        label: 'Accuracy (%)',
        data: sortedAccuracies,
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            afterLabel: (context) => {
              const index = sortedLabels.indexOf(context.label);
              return `Samples: ${sortedCounts[index]}`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: 'Accuracy (%)'
          }
        }
      }
    }
  });
}

function renderPredictionTimeline(trueClasses, predictedClasses, decoder) {
  const ctx = document.getElementById('timelineChart').getContext('2d');
  if (charts.timeline) charts.timeline.destroy();

  // Show first 100 predictions for clarity
  const displayLimit = Math.min(100, trueClasses.length);
  const indices = Array.from({length: displayLimit}, (_, i) => i);
  const displayTrue = trueClasses.slice(0, displayLimit).map(c => decoder[c]);
  const displayPred = predictedClasses.slice(0, displayLimit).map(c => decoder[c]);

  const correct = indices.map(i => displayTrue[i] === displayPred[i] ? 1 : 0);

  charts.timeline = new Chart(ctx, {
    type: 'line',
    data: {
      labels: indices,
      datasets: [
        {
          label: 'Correct Predictions',
          data: correct,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointBackgroundColor: (context) => 
            correct[context.dataIndex] === 1 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: {
            callback: (value) => value === 1 ? 'Correct' : (value === 0 ? 'Wrong' : '')
          }
        },
        x: {
          title: {
            display: true,
            text: 'Test Sample Index'
          }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (context) => {
              const index = context.dataIndex;
              return [
                `True: ${displayTrue[index]}`,
                `Predicted: ${displayPred[index]}`,
                `Status: ${correct[index] ? 'Correct' : 'Wrong'}`
              ];
            }
          }
        }
      }
    }
  });
}

function renderROCCurve(rocResults) {
  const ctx = document.getElementById('rocChart').getContext('2d');
  if (charts.roc) charts.roc.destroy();

  // Prepare datasets for each class
  const datasets = rocResults.rocCurves.map((rocCurve, index) => {
    const colors = [
      'rgba(255, 99, 132, 0.8)',
      'rgba(54, 162, 235, 0.8)',
      'rgba(255, 206, 86, 0.8)',
      'rgba(75, 192, 192, 0.8)',
      'rgba(153, 102, 255, 0.8)',
      'rgba(255, 159, 64, 0.8)'
    ];
    
    return {
      label: `${rocCurve.className} (AUC = ${rocCurve.auc.toFixed(4)})`,
      data: rocCurve.rocPoints.map(point => ({ x: point.fpr, y: point.tpr })),
      borderColor: colors[index % colors.length],
      backgroundColor: colors[index % colors.length].replace('0.8', '0.1'),
      fill: false,
      tension: 0.4,
      pointRadius: 0
    };
  });

  // Add diagonal reference line
  datasets.push({
    label: 'Random Classifier (AUC = 0.5)',
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: 'rgba(128, 128, 128, 0.5)',
    borderDash: [5, 5],
    fill: false,
    pointRadius: 0
  });

  charts.roc = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: datasets
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `ROC Curves (Macro AUC = ${rocResults.macroAUC.toFixed(4)})`
        },
        tooltip: {
          callbacks: {
            title: (items) => `FPR: ${items[0].parsed.x.toFixed(3)}, TPR: ${items[0].parsed.y.toFixed(3)}`,
            label: (context) => {
              const datasetLabel = context.dataset.label;
              // For the diagonal line, don't show additional info
              if (datasetLabel.includes('Random Classifier')) {
                return datasetLabel;
              }
              return datasetLabel;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: {
            display: true,
            text: 'False Positive Rate'
          },
          min: 0,
          max: 1
        },
        y: {
          type: 'linear',
          title: {
            display: true,
            text: 'True Positive Rate'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
}

/* ============================
   Main EDA orchestration
   ============================ */
function runEDA(){
  if(!rawData || rawData.length === 0){ alert("No data loaded. Click Load and choose weather.csv."); return; }

  // overview already shown; recompute missing
  const missing = computeMissingPercent(rawData);
  renderMissingChart(missing);

  // stats
  const stats = computeStats(rawData);
  renderStatsTable(stats);

  // histograms for numeric features
  for(const f of schema.features){
    const vals = rawData.map(r => r[f]);
    const canvasId = f === "precipitation" ? "hist_precip" : (f === "temp_max" ? "hist_tempmax" : (f === "temp_min" ? "hist_tempmin" : "hist_wind"));
    // set title text (ensures top label shows column name)
    const titleEl = $(`title_${f}`) || null;
    if(titleEl) titleEl.innerText = f;
    renderHistogram(canvasId, vals, f, 24);
  }

  // doughnut with percentages (weather distribution)
  const dist = stats.categorical[schema.target] || {};
  renderDoughnutWithPercents("weather_doughnut", dist);

  $("overview").innerText = `EDA complete — ${rawData.length} rows analyzed.`;

  // Prepare summary for export: csvRows (simple summary) + jsonDesc
  const csvRows = [];
  for(const f of Object.keys(stats.numeric)){
    const s = stats.numeric[f];
    csvRows.push({ metric: 'mean', feature: f, value: s.mean });
    csvRows.push({ metric: 'median', feature: f, value: s.median });
    csvRows.push({ metric: 'std', feature: f, value: s.std });
    csvRows.push({ metric: 'missing', feature: f, value: s.missing });
  }
  // add categorical counts
  for(const [k,v] of Object.entries(stats.categorical[schema.target] || {})){
    csvRows.push({ metric: 'count', feature: `${schema.target}:${k}`, value: v });
  }
  const jsonDesc = { overview: { rows: rawData.length, columns: Object.keys(rawData[0] || {}) }, missing, stats };
  enableExportLinks({ csvRows, jsonDesc });
}

/* ============================
   Events wiring
   ============================ */
document.addEventListener("DOMContentLoaded", () => {
  $("loadBtn").addEventListener("click", () => {
    const file = $("fileInput").files[0];
    if(!file){ alert("Please select a CSV file before clicking Load."); return; }
    handleFileLoad(file);
  });

  $("runBtn").addEventListener("click", () => {
    // quick schema validation
    const sample = rawData[0] || {};
    const missingCols = [];
    [schema.target, ...schema.features].forEach(c => { if(!(c in sample)) missingCols.push(c); });
    if(missingCols.length){
      const proceed = confirm("The loaded CSV is missing expected columns: " + missingCols.join(", ") + ". Continue anyway?");
      if(!proceed) return;
    }
    runEDA();
  });

  $("trainBtn").addEventListener("click", () => {
    trainModel();
  });

  $("evaluateBtn").addEventListener("click", () => {
    evaluateModel();
  });

  disableExportLinks();
  $("evaluateBtn").disabled = true;
});