// Global State
let globalData = [];
let selectedPoints = [];
let cachedEmbeddings = {}; // Cache for t-SNE results
let currentPositions = null; // Store current t-SNE positions
let currentModality = null; // Store current modality

// Store positions for both plots
let textPositions = null;
let imagePositions = null;

const setStatus = (msg) => {
    document.getElementById('status').innerText = msg;
};

/**
 * Generate a cache key based on data and parameters
 */
function getCacheKey(modality, dataLength, perplexity, maxIter) {
    return `tsne_${modality}_${dataLength}_${perplexity}_${maxIter}`;
}

/**
 * Save t-SNE results to localStorage
 */
function saveTSNEToCache(cacheKey, embedding) {
    try {
        const cacheData = {
            embedding: embedding,
            timestamp: Date.now(),
            version: '1.0'
        };
        localStorage.setItem(cacheKey, JSON.stringify(cacheData));
        console.log(`Cached t-SNE results with key: ${cacheKey}`);
    } catch (e) {
        console.warn('Failed to cache t-SNE results (localStorage full?):', e);
    }
}

/**
 * Load t-SNE results from localStorage
 */
function loadTSNEFromCache(cacheKey) {
    try {
        const cached = localStorage.getItem(cacheKey);
        if (!cached) return null;
        
        const cacheData = JSON.parse(cached);
        
        // Validate cache structure
        if (!cacheData || !cacheData.embedding) {
            console.warn('Invalid cache structure, removing...');
            localStorage.removeItem(cacheKey);
            return null;
        }
        
        // Check if cache is still valid (less than 7 days old)
        const cacheAge = Date.now() - cacheData.timestamp;
        const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days
        
        if (cacheAge > maxAge) {
            console.log('Cache expired, removing...');
            localStorage.removeItem(cacheKey);
            return null;
        }
        
        // Validate embedding is an array
        if (!Array.isArray(cacheData.embedding)) {
            console.warn('Cache contains non-array embedding, removing...');
            localStorage.removeItem(cacheKey);
            return null;
        }
        
        console.log(`Loaded t-SNE results from cache (age: ${Math.round(cacheAge / 1000 / 60)} minutes)`);
        return cacheData.embedding;
    } catch (e) {
        console.warn('Failed to load from cache:', e);
        return null;
    }
}

/**
 * Clear all t-SNE caches
 */
function clearAllCaches() {
    const keys = Object.keys(localStorage);
    let cleared = 0;
    for (let key of keys) {
        if (key.startsWith('tsne_')) {
            localStorage.removeItem(key);
            cleared++;
        }
    }
    console.log(`Cleared ${cleared} cached t-SNE results`);
    alert(`Cleared ${cleared} cached visualizations`);
}

// Pyodide initialization
let pyodide = null;

async function initPyodide() {
    if (pyodide) return pyodide;
    
    console.log('Loading Pyodide...');
    setStatus('Initializing Python environment (Pyodide)...');
    
    pyodide = await loadPyodide();
    
    // Load required packages
    setStatus('Installing Python packages (scikit-learn, numpy)...');
    await pyodide.loadPackage(['numpy', 'scikit-learn']);
    
    console.log('Pyodide ready with scikit-learn');
    return pyodide;
}

/**
 * Run t-SNE using Python's scikit-learn via Pyodide
 */
async function runTSNEwithPython(embeddings, dims = 3, perplexity = 30, maxIter = 500) {
    // Initialize Pyodide if needed
    await initPyodide();
    
    const n = embeddings.length;
    const d = embeddings[0].length;
    
    console.log(`Running Python t-SNE on ${n} samples with ${d} features...`);
    setStatus(`Phase 4/4 (60%): Running Python t-SNE with scikit-learn...`);
    
    // Convert embeddings to a flat array for Python
    const flatData = embeddings.flat();
    
    // Pass data to Python
    pyodide.globals.set('data_flat', flatData);
    pyodide.globals.set('n_samples', n);
    pyodide.globals.set('n_features', d);
    pyodide.globals.set('n_components', dims);
    pyodide.globals.set('perplexity_val', perplexity);
    pyodide.globals.set('n_iter_val', maxIter);
    
    // Run t-SNE in Python
    const result = await pyodide.runPythonAsync(`
import numpy as np
from sklearn.manifold import TSNE

# Reshape data
X = np.array(data_flat).reshape(n_samples, n_features)

# Run t-SNE
print(f"Running t-SNE: n_samples={n_samples}, n_features={n_features}, n_components={n_components}")
tsne = TSNE(
    n_components=n_components,
    perplexity=perplexity_val,
    n_iter=n_iter_val,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

embedding = tsne.fit_transform(X)
print(f"t-SNE complete. Output shape: {embedding.shape}")

# Convert to list for JavaScript
embedding.tolist()
    `);
    
    // Convert Pyodide proxy to JavaScript array if needed
    let jsResult;
    if (result && result.toJs) {
        jsResult = result.toJs();
    } else {
        jsResult = result;
    }
    
    console.log('Python t-SNE complete, result type:', typeof jsResult, 'isArray:', Array.isArray(jsResult));
    return jsResult;
}

const EMBEDDINGS_ZIP_URL = "https://raw.githubusercontent.com/epiverse/tcgadata/main/TITAN/text_image_embeddings.tsv.zip";

function parseEmbedding(embStr) {
    return embStr
        .replace(/[\[\]"]/g, '')
        .split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n));
}

async function loadData() {
    console.log("Fetching embeddings ZIP file...");
    setStatus('Phase 1/4 (0%): Downloading data...');
    try {
        const response = await fetch(EMBEDDINGS_ZIP_URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch embeddings file. HTTP Status: ${response.status}`);
        }

        const dataBuffer = await response.arrayBuffer();
        setStatus('Unzipping and parsing data...');

        if (typeof JSZip === 'undefined') {
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';
                script.onload = () => resolve();
                script.onerror = () => reject(new Error('Dynamic JSZip load failed'));
                document.head.appendChild(script);
                setTimeout(() => reject(new Error('Timed out loading JSZip')), 8000);
            }).catch(err => {
                console.warn('Dynamic JSZip load failed:', err);
            });
        }

        if (typeof JSZip === 'undefined') {
             throw new Error("JSZip library is required but not loaded in HTML.");
        }

        const zip = await JSZip.loadAsync(dataBuffer);
        const fileName = 'text_image_embeddings.tsv';
        const file = zip.file(fileName);
        if (!file) {
            throw new Error(`File '${fileName}' not found in the ZIP archive.`);
        }

        setStatus('Phase 2/4 (25%): Extracting and parsing...');
        const uint8array = await file.async('uint8array');

        const decoder = new TextDecoder('utf-8');
        const CHUNK_SIZE = 128 * 1024;
        let pos = 0;
        let carry = '';
        let headers = null;
        const data = [];

        while (pos < uint8array.length) {
            const end = Math.min(pos + CHUNK_SIZE, uint8array.length);
            const chunk = uint8array.subarray(pos, end);
            pos = end;

            let text = decoder.decode(chunk, { stream: true });
            text = carry + text;

            const lines = text.split('\n');
            carry = lines.pop();

            for (let li = 0; li < lines.length; li++) {
                const rawLine = lines[li].replace(/\r$/, '');
                if (!rawLine) continue;

                if (!headers) {
                    headers = rawLine.split('\t');
                    continue;
                }

                const values = rawLine.split('\t');
                const row = {};
                headers.forEach((header, index) => {
                    row[header.trim()] = values[index] ? values[index].trim() : '';
                });

                data.push({
                    i: parseInt(row.i),
                    id: row.id,
                    patient_id: row.patient_id,
                    cancer_type: row.cancer_type,
                    text_embedding: parseEmbedding(row.embeddings),
                    image_embedding: parseEmbedding(row.image_embedding)
                });
            }

            if (pos % (256 * 1024) === 0) {
                await new Promise(resolve => requestAnimationFrame(resolve));
            }
        }

        if (carry) {
            const lastLine = carry.replace(/\r$/, '');
            if (lastLine && headers) {
                const values = lastLine.split('\t');
                const row = {};
                headers.forEach((header, index) => {
                    row[header.trim()] = values[index] ? values[index].trim() : '';
                });
                data.push({
                    i: parseInt(row.i),
                    id: row.id,
                    patient_id: row.patient_id,
                    cancer_type: row.cancer_type,
                    text_embedding: parseEmbedding(row.embeddings),
                    image_embedding: parseEmbedding(row.image_embedding)
                });
            }
        }

        const validData = data.filter(d => d.text_embedding.length > 0 && d.image_embedding.length > 0);
        console.log(`Loaded ${validData.length} valid samples`);

        setStatus('Phase 2/4 (50%): Data loaded');
        return validData;

    } catch (error) {
        console.error("Error during data loading:", error);
        setStatus(`Failed to load data: ${error.message}`);
        alert(`Failed to load data: ${error.message}`);
        return [];
    }
}

/**
 * GPU-accelerated PCA using TensorFlow.js with SVD
 * Fixed: Proper tensor disposal to prevent memory leaks
 */
async function performPCATensorFlow(data, targetDim = 50) {
    if (data.length === 0) return data;
    
    const n = data.length;
    const d = data[0].length;
    
    if (d <= targetDim) {
        console.log(`Data dimension ${d} is already <= target ${targetDim}, skipping PCA`);
        return data;
    }
    
    console.log(`Running GPU-accelerated PCA: ${d}D → ${targetDim}D`);
    
    // Use simplified variance-based PCA for high dimensions to save memory
    if (d > 300) {
        console.log('Using memory-efficient variance-based PCA');
        
        // 1. Center the data (CPU to save GPU memory)
        const mean = new Array(d).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                mean[j] += data[i][j];
            }
        }
        for (let j = 0; j < d; j++) mean[j] /= n;
        
        const centered = data.map(row => row.map((val, j) => val - mean[j]));
        
        // 2. Calculate variance for each dimension (CPU)
        const variances = new Array(d).fill(0);
        for (let j = 0; j < d; j++) {
            for (let i = 0; i < n; i++) {
                variances[j] += centered[i][j] * centered[i][j];
            }
            variances[j] /= n;
        }
        
        // 3. Get indices of top variance dimensions
        const indices = variances
            .map((v, i) => ({ variance: v, index: i }))
            .sort((a, b) => b.variance - a.variance)
            .slice(0, targetDim)
            .map(item => item.index);
        
        // 4. Project onto top variance dimensions
        const reduced = centered.map(row => indices.map(idx => row[idx]));
        
        console.log(`Memory-efficient PCA complete: ${d}D → ${reduced[0].length}D`);
        return reduced;
    }
    
    // For smaller dimensions, use GPU SVD
    return await tf.tidy(() => {
        // Convert to tensor
        const dataTensor = tf.tensor2d(data);
        
        // Center the data
        const mean = dataTensor.mean(0);
        const centered = dataTensor.sub(mean);
        
        const k = Math.min(targetDim, Math.min(n, d) - 1);
        
        // Use SVD on smaller matrix
        let matrixForSVD = centered;
        let transposeNeeded = false;
        
        if (n > d) {
            matrixForSVD = centered.transpose();
            transposeNeeded = true;
        }
        
        // Compute SVD (GPU-accelerated)
        const { u, v } = tf.linalg.svd(matrixForSVD);
        
        // Get top k principal components
        let components;
        if (transposeNeeded) {
            components = u.slice([0, 0], [d, k]);
        } else {
            components = v.slice([0, 0], [d, k]);
        }
        
        // Project data
        const reduced = centered.matMul(components);
        const result = reduced.arraySync();
        
        console.log(`GPU PCA complete: ${d}D → ${result[0].length}D`);
        return result;
    });
}

/**
 * Compute distance matrix using CPU (more stable than GPU for large datasets)
 */
function computeDistanceMatrixCPU(embeddings) {
    const n = embeddings.length;
    const d = embeddings[0].length;
    const distances = Array(n).fill(0).map(() => Array(n).fill(0));
    
    console.log(`Computing ${n}x${n} distance matrix on CPU...`);
    
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let distSq = 0;
            for (let k = 0; k < d; k++) {
                const diff = embeddings[i][k] - embeddings[j][k];
                distSq += diff * diff;
            }
            const dist = Math.sqrt(distSq);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    
    return distances;
}

/**
 * Compute P matrix (affinities) in high-dimensional space with GPU acceleration
 */
function computePMatrixGPU(distances, perplexity = 30) {
    const n = distances.length;
    const P = Array(n).fill(0).map(() => Array(n).fill(0));
    const targetEntropy = Math.log(perplexity);
    
    for (let i = 0; i < n; i++) {
        // Binary search for sigma (beta = 1/(2*sigma^2))
        let betaMin = -Infinity;
        let betaMax = Infinity;
        let beta = 1.0;
        
        for (let iter = 0; iter < 50; iter++) {
            let sumP = 0;
            let entropy = 0;
            
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const pij = Math.exp(-distances[i][j] * distances[i][j] * beta);
                P[i][j] = pij;
                sumP += pij;
            }
            
            if (sumP === 0) sumP = 1e-10;
            
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                P[i][j] /= sumP;
                if (P[i][j] > 1e-10) {
                    entropy -= P[i][j] * Math.log(P[i][j]);
                }
            }
            
            const diff = entropy - targetEntropy;
            if (Math.abs(diff) < 1e-5) break;
            
            if (diff > 0) {
                betaMin = beta;
                beta = (betaMax === Infinity) ? beta * 2 : (beta + betaMax) / 2;
            } else {
                betaMax = beta;
                beta = (betaMin === -Infinity) ? beta / 2 : (beta + betaMin) / 2;
            }
        }
    }
    
    // Symmetrize
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const pij = (P[i][j] + P[j][i]) / (2 * n);
            P[i][j] = Math.max(pij, 1e-12);
            P[j][i] = P[i][j];
        }
    }
    
    return P;
}

/**
 * CPU-based gradient computation (stable, no WebGL issues)
 */
function computeGradientCPU(Y, P) {
    const n = Y.length;
    const dims = Y[0].length;
    
    // Compute Q matrix
    const Q = Array(n).fill(0).map(() => Array(n).fill(0));
    let sumQ = 0;
    
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            let distSq = 0;
            for (let d = 0; d < dims; d++) {
                const diff = Y[i][d] - Y[j][d];
                distSq += diff * diff;
            }
            const qij = 1 / (1 + distSq);
            Q[i][j] = qij;
            Q[j][i] = qij;
            sumQ += 2 * qij;
        }
    }
    
    // Normalize Q
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            Q[i][j] = Math.max(Q[i][j] / sumQ, 1e-12);
        }
    }
    
    // Compute gradient
    const gradient = Array(n).fill(0).map(() => Array(dims).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const mult = (P[i][j] - Q[i][j]) * Q[i][j] * sumQ;
            for (let d = 0; d < dims; d++) {
                gradient[i][d] += 4 * mult * (Y[i][d] - Y[j][d]);
            }
        }
    }
    
    return gradient;
}

/**
 * Fast t-SNE with memory management (uses CPU for stability)
 */
async function runTSNEwithGPU(embeddings, dims = 3, perplexity = 30, maxIter = 500) {
    const n = embeddings.length;
    
    // Log initial GPU memory
    if (tf.memory) {
        console.log('GPU memory before t-SNE:', tf.memory());
    }
    
    console.log('Computing distance matrix on CPU (more stable)...');
    setStatus('Phase 4/4 (60%): Computing distances...');
    const distances = computeDistanceMatrixCPU(embeddings);
    
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    console.log('Computing P matrix...');
    setStatus('Phase 4/4 (65%): Computing affinities...');
    const P = computePMatrixGPU(distances, perplexity);
    
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    // Initialize Y randomly
    const Y = Array(n).fill(0).map(() => 
        Array(dims).fill(0).map(() => (Math.random() - 0.5) * 0.0001)
    );
    
    // Momentum and learning rate
    let momentum = 0.5;
    const eta = 200;
    let gains = Array(n).fill(0).map(() => Array(dims).fill(1));
    let iY = Array(n).fill(0).map(() => Array(dims).fill(0));
    
    console.log('Running t-SNE iterations...');
    
    for (let iter = 0; iter < maxIter; iter++) {
        // Compute gradient (CPU-based for stability)
        const gradient = computeGradientCPU(Y, P);
        
        // Update with momentum
        for (let i = 0; i < n; i++) {
            for (let d = 0; d < dims; d++) {
                // Adaptive learning rate
                if ((gradient[i][d] > 0) !== (iY[i][d] > 0)) {
                    gains[i][d] += 0.2;
                } else {
                    gains[i][d] = Math.max(gains[i][d] * 0.8, 0.01);
                }
                
                iY[i][d] = momentum * iY[i][d] - eta * gains[i][d] * gradient[i][d];
                Y[i][d] += iY[i][d];
            }
        }
        
        // Center Y
        const mean = Array(dims).fill(0);
        for (let i = 0; i < n; i++) {
            for (let d = 0; d < dims; d++) {
                mean[d] += Y[i][d];
            }
        }
        for (let d = 0; d < dims; d++) mean[d] /= n;
        for (let i = 0; i < n; i++) {
            for (let d = 0; d < dims; d++) {
                Y[i][d] -= mean[d];
            }
        }
        
        // Switch to higher momentum
        if (iter === 250) {
            momentum = 0.8;
        }
        
        // Progress updates
        if (iter % 10 === 0) {
            const progress = Math.floor((iter / maxIter) * 30) + 70;
            setStatus(`Phase 4/4 (${progress}%): t-SNE iteration ${iter}/${maxIter}...`);
            
            // Yield to browser and clean up GPU memory periodically
            if (iter % 50 === 0) {
                await new Promise(resolve => requestAnimationFrame(resolve));
                
                // Force garbage collection of unused tensors
                if (tf.memory && iter % 100 === 0) {
                    const memInfo = tf.memory();
                    console.log(`Iteration ${iter} - GPU memory: ${memInfo.numBytes / 1024 / 1024} MB, Tensors: ${memInfo.numTensors}`);
                }
            }
        }
    }
    
    // Final memory check
    if (tf.memory) {
        console.log('GPU memory after t-SNE:', tf.memory());
    }
    
    console.log('t-SNE complete');
    return Y;
}

/**
 * Fast k-NN using original embeddings (not t-SNE output)
 */
function euclideanDistance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        const diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

function getKNN(index, allEmbeddings, k = 10) {
    const target = allEmbeddings[index];
    const distances = [];
    
    for (let i = 0; i < allEmbeddings.length; i++) {
        if (i === index) continue;
        distances.push({
            index: i,
            dist: euclideanDistance(target, allEmbeddings[i])
        });
    }
    
    if (distances.length <= k) {
        return distances.map(d => d.index);
    }
    
    distances.sort((a, b) => a.dist - b.dist);
    return distances.slice(0, k).map(d => d.index);
}

async function runVisualization() {
    const modality = document.getElementById('embeddingType').value;
    if (globalData.length === 0) {
        globalData = await loadData();
    }

    // t-SNE parameters
    const perplexity = Math.min(30, Math.max(5, Math.floor(globalData.length / 3)));
    const maxIter = 500;
    
    // Generate cache key
    const cacheKey = getCacheKey(modality, globalData.length, perplexity, maxIter);
    
    // Check if we have cached results
    const cachedResult = loadTSNEFromCache(cacheKey);
    
    if (cachedResult) {
        const statusEl = document.getElementById('status');
        statusEl.className = 'cached';
        setStatus('⚡ Loaded from cache! (instant - no computation needed)');
        renderPlot(cachedResult, modality);
        setStatus("✅ Ready! (Loaded from cache) Click two points for SNN analysis. Use 'Clear Cache' to recompute.");
        return;
    }

    // No cache - compute from scratch
    const statusEl = document.getElementById('status');
    statusEl.className = '';
    setStatus('Phase 3/4 (55%): Preparing Python environment...');
    
    // Initialize Pyodide first
    await initPyodide();
    
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    let embeddings = globalData.map(d => d[modality]);
    
    // PCA preprocessing
    setStatus(`Phase 3/4 (60%): PCA dimensionality reduction to 50D...`);
    embeddings = await performPCATensorFlow(embeddings, 50);
    
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    // Python-based t-SNE with scikit-learn
    const embedding = await runTSNEwithPython(embeddings, 3, perplexity, maxIter);
    
    // Cache the results
    saveTSNEToCache(cacheKey, embedding);
    
    setStatus('Rendering visualization...');
    renderPlot(embedding, modality);
    statusEl.className = 'cached';
    setStatus("✅ Ready! Click two points for SNN. Results cached - next load will be instant! 💾");
}

function renderPlot(pos, modality) {
    // Ensure pos is a proper JavaScript array
    if (!pos || !Array.isArray(pos)) {
        console.error('Invalid position data:', pos);
        setStatus('Error: Invalid t-SNE result format');
        return;
    }
    
    // Save for reset functionality
    currentPositions = pos;
    currentModality = modality;
    
    const x = pos.map(p => p[0]);
    const y = pos.map(p => p[1]);
    const z = pos.map(p => p[2]);
    
    const uniqueTypes = [...new Set(globalData.map(d => d.cancer_type))];
    const colorMap = {};
    uniqueTypes.forEach((type, i) => {
        colorMap[type] = i;
    });
    
    const colors = globalData.map(d => colorMap[d.cancer_type]);
    const labels = globalData.map(d => 
        `<b>${d.id}</b><br>Patient: ${d.patient_id}<br>Type: <b>${d.cancer_type}</b>`
    );

    const trace = {
        x: x, y: y, z: z,
        mode: 'markers',
        type: 'scatter3d',
        text: labels,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
            size: 4,
            color: colors,
            colorscale: 'Portland',
            opacity: 0.75,
            line: {
                color: 'rgba(255, 255, 255, 0.3)',
                width: 1
            }
        },
        name: 'Patients'
    };

    const layout = {
        margin: { l: 0, r: 0, b: 0, t: 50 },
        title: {
            text: `${modality === 'text_embedding' ? '📝 Text' : '🔬 Image'} Embeddings (GPU-Accelerated t-SNE)`,
            font: { size: 18, color: '#2c3e50' }
        },
        scene: { 
            xaxis: { 
                title: 't-SNE 1', 
                showgrid: true, 
                gridcolor: '#e8e8e8',
                backgroundcolor: '#fafafa'
            }, 
            yaxis: { 
                title: 't-SNE 2', 
                showgrid: true, 
                gridcolor: '#e8e8e8',
                backgroundcolor: '#fafafa'
            }, 
            zaxis: { 
                title: 't-SNE 3', 
                showgrid: true, 
                gridcolor: '#e8e8e8',
                backgroundcolor: '#fafafa'
            },
            bgcolor: '#ffffff',
            camera: {
                eye: { x: 1.4, y: 1.4, z: 1.2 }
            }
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1.02,
            y: 1,
            xanchor: 'left',
            yanchor: 'top'
        },
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot('plot-container', [trace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan3d', 'select3d', 'lasso3d']
    });

    // Store positions globally for click handler
    window.currentPositions = pos;
    
    const plotDiv = document.getElementById('plot-container');
    
    // Remove any existing click handlers to prevent duplicates
    plotDiv.removeAllListeners('plotly_click');
    
    plotDiv.on('plotly_click', function(data) {
        const pn = data.points[0].pointNumber;
        
        // Only process clicks on the base trace (original points)
        if (data.points[0].curveNumber === 0) {
            handleSelection(pn, window.currentPositions);
        }
    });
}

function handleSelection(index, positions) {
    // If we already have 2 points, clear the old highlights and start fresh
    if (selectedPoints.length === 2) {
        selectedPoints = [];
        
        // Remove all highlight traces by redrawing just the base trace
        try {
            const plotDiv = document.getElementById('plot-container');
            const baseTrace = plotDiv.data[0]; // Keep the original scatter plot
            
            // Temporarily disable click events
            plotDiv.removeAllListeners('plotly_click');
            
            // Redraw with only the base trace
            Plotly.react('plot-container', [baseTrace], plotDiv.layout).then(() => {
                // Re-attach click handler after update completes
                plotDiv.on('plotly_click', function(data) {
                    const pn = data.points[0].pointNumber;
                    if (data.points[0].curveNumber === 0) {
                        handleSelection(pn, window.currentPositions);
                    }
                });
            });
        } catch (e) {
            console.warn('Error clearing traces:', e);
        }
        
        document.getElementById('selectionInfo').innerHTML = `
            <strong>💡 Interactive SNN Analysis:</strong> Click any two points in the 3D plot to find Shared Nearest Neighbors.
            <div>
                <span class="feature-tag">🎯 Interactive 3D t-SNE</span>
                <span class="feature-tag">🔍 Customizable k-NN</span>
                <span class="feature-tag">📊 Cancer Type Clustering</span>
                <span class="feature-tag">⚡ 5-10x Faster</span>
                <span class="feature-tag">💾 Cached Results</span>
            </div>
        `;
    }
    
    selectedPoints.push(index);

    if (selectedPoints.length === 2) {
        const modality = document.getElementById('embeddingType').value;
        const k = parseInt(document.getElementById('kValue').value) || 20;
        const embeddings = globalData.map(d => d[modality]);
        
        // Get k nearest neighbors for each selected point
        const knn1 = getKNN(selectedPoints[0], embeddings, k);
        const knn2 = getKNN(selectedPoints[1], embeddings, k);
        
        // Find shared neighbors (intersection)
        const shared = knn1.filter(idx => knn2.includes(idx));
        
        // Find unique neighbors for each point
        const unique1 = knn1.filter(idx => !knn2.includes(idx));
        const unique2 = knn2.filter(idx => !knn1.includes(idx));
        
        highlightNeighbors(shared, unique1, unique2, selectedPoints, positions);
        
        const pt1 = globalData[selectedPoints[0]];
        const pt2 = globalData[selectedPoints[1]];
        
        document.getElementById('selectionInfo').innerHTML = `
            <div style="background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%); padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #667eea;">
                <strong>🎯 k-NN Analysis Results (k=${k}):</strong><br><br>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                    <div>
                        <strong>📍 Point 1:</strong> ${pt1.id}<br>
                        <span style="color: #666;">${pt1.cancer_type}</span><br>
                        <span style="color: #3498db;">🔵 Unique neighbors: ${unique1.length}</span>
                    </div>
                    <div>
                        <strong>📍 Point 2:</strong> ${pt2.id}<br>
                        <span style="color: #666;">${pt2.cancer_type}</span><br>
                        <span style="color: #9b59b6;">🟣 Unique neighbors: ${unique2.length}</span>
                    </div>
                </div>
                
                <div style="margin-top: 15px; padding: 10px; background: rgba(255,68,68,0.1); border-radius: 6px;">
                    <strong style="color: #e74c3c;">🔴 Shared Neighbors: ${shared.length}</strong> 
                    <span style="color: #666;">(${Math.round(shared.length / k * 100)}% overlap)</span>
                </div>
                
                <div style="margin-top: 10px; font-size: 0.85em; color: #666;">
                    <strong>Legend:</strong> 
                    🔵 Blue = Point 1 only | 
                    🟣 Purple = Point 2 only | 
                    🔴 Red = Shared | 
                    ⬛ Black = Selected
                    <br>
                    <em>Click any point to start a new comparison</em>
                </div>
            </div>
        `;
    } else if (selectedPoints.length === 1) {
        const pt1 = globalData[selectedPoints[0]];
        document.getElementById('selectionInfo').innerHTML = `
            <div style="background: #f0f0f0; padding: 10px; border-radius: 6px; margin-top: 10px;">
                <strong>First point selected:</strong> ${pt1.id} (${pt1.cancer_type})<br>
                <span style="color: #666;">Click another point to find shared neighbors</span>
            </div>
        `;
    }
}

function highlightNeighbors(sharedIdx, unique1Idx, unique2Idx, selectedIdx, pos, containerId = 'plot-container') {
    const plot = getThreeJSPlot(containerId);
    if (!plot) return;
    
    // Highlight neighbors with Three.js
    plot.highlightPoints(sharedIdx, unique1Idx, unique2Idx, selectedIdx);
}

// Highlight only the selected points in the other plot (without neighbors)
function highlightSelectedPointsOnly(selectedIdx, pos, containerId) {
    const plot = getThreeJSPlot(containerId);
    if (!plot) return;
    
    // Highlight only selected points
    plot.highlightPoints([], [], [], selectedIdx);
}

// Dual visualization function for side-by-side plots
async function runDualVisualization() {
    if (globalData.length === 0) {
        globalData = await loadData();
    }

    // Initialize Pyodide once for both visualizations
    await initPyodide();

    const perplexity = Math.min(30, Math.max(5, Math.floor(globalData.length / 3)));
    const maxIter = 500;
    
    // Generate both visualizations
    await Promise.all([
        runSingleVisualization('text_embedding', 'plot-container-text', perplexity, maxIter),
        runSingleVisualization('image_embedding', 'plot-container-image', perplexity, maxIter)
    ]);
    
    const statusEl = document.getElementById('status');
    statusEl.className = 'cached';
    setStatus("✅ Ready! Both visualizations complete. Click two points in either plot for SNN analysis.");
}

async function runSingleVisualization(modality, containerId, perplexity, maxIter) {
    const cacheKey = getCacheKey(modality, globalData.length, perplexity, maxIter);
    let cachedResult = loadTSNEFromCache(cacheKey);
    
    // Validate cached result
    if (cachedResult && (!Array.isArray(cachedResult) || cachedResult.length === 0)) {
        console.warn('Invalid cache data detected, clearing...');
        localStorage.removeItem(cacheKey);
        cachedResult = null;
    }
    
    let embedding;
    if (cachedResult) {
        console.log(`Loaded ${modality} from cache`);
        embedding = cachedResult;
    } else {
        console.log(`Computing ${modality}...`);
        setStatus(`Computing ${modality === 'text_embedding' ? 'Text' : 'Image'} embeddings...`);
        
        let embeddings = globalData.map(d => d[modality]);
        embeddings = await performPCATensorFlow(embeddings, 50);
        embedding = await runTSNEwithPython(embeddings, 3, perplexity, maxIter);
        
        // Validate before caching
        if (embedding && Array.isArray(embedding) && embedding.length > 0) {
            saveTSNEToCache(cacheKey, embedding);
        } else {
            console.error('Invalid embedding result from Python:', embedding);
            setStatus('Error: Invalid t-SNE computation result');
            return;
        }
    }
    
    renderPlotToContainer(embedding, modality, containerId);
}

function renderPlotToContainer(pos, modality, containerId) {
    // Ensure pos is a proper JavaScript array
    if (!pos || !Array.isArray(pos)) {
        console.error('Invalid position data:', pos);
        setStatus('Error: Invalid t-SNE result format');
        return;
    }
    
    // Store positions globally
    if (modality === 'text_embedding') {
        textPositions = pos;
    } else {
        imagePositions = pos;
    }
    
    // Initialize or get Three.js plot
    let plot = getThreeJSPlot(containerId);
    if (!plot) {
        plot = initThreeJSPlot(containerId);
    }
    
    // Render points
    plot.renderPoints(pos, globalData);
    
    // Set click handler
    plot.onPointClick = (pointIndex) => {
        window.currentPositions = pos;
        window.activeModality = modality;
        handleSelectionForContainer(pointIndex, pos, modality, containerId);
    };
}

function handleSelectionForContainer(index, positions, modality, containerId) {
    if (selectedPoints.length === 2) {
        selectedPoints = [];
        
        // Clear both plots
        try {
            ['plot-container-text', 'plot-container-image'].forEach(cid => {
                const plot = getThreeJSPlot(cid);
                if (plot) {
                    plot.clearHighlights();
                }
            });
        } catch (e) {
            console.warn('Error clearing highlights:', e);
        }
        
        document.getElementById('selectionInfo').innerHTML = `
            <strong>💡 Interactive SNN Analysis:</strong> Click any two points in the 3D plots to find Shared Nearest Neighbors.
        `;
    }
    
    selectedPoints.push(index);

    if (selectedPoints.length === 2) {
        const k = parseInt(document.getElementById('kValue').value) || 20;
        const embeddings = globalData.map(d => d[modality]);
        
        const knn1 = getKNN(selectedPoints[0], embeddings, k);
        const knn2 = getKNN(selectedPoints[1], embeddings, k);
        
        const shared = knn1.filter(idx => knn2.includes(idx));
        const unique1 = knn1.filter(idx => !knn2.includes(idx));
        const unique2 = knn2.filter(idx => !knn1.includes(idx));
        
        // Highlight in the clicked plot with full analysis
        highlightNeighbors(shared, unique1, unique2, selectedPoints, positions, containerId);
        
        // Highlight just the selected points in the other plot
        const otherContainerId = containerId === 'plot-container-text' ? 'plot-container-image' : 'plot-container-text';
        const otherPositions = containerId === 'plot-container-text' ? imagePositions : textPositions;
        
        if (otherPositions) {
            highlightSelectedPointsOnly(selectedPoints, otherPositions, otherContainerId);
        }
        
        const pt1 = globalData[selectedPoints[0]];
        const pt2 = globalData[selectedPoints[1]];
        const modalityName = modality === 'text_embedding' ? 'Text' : 'Image';
        
        document.getElementById('selectionInfo').innerHTML = `
            <div style="background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%); padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #667eea;">
                <strong style="font-size: 1.1em; color: #2c3e50;">📊 SNN Analysis (${modalityName} Embeddings, k=${k})</strong>
                <div style="margin-top: 10px;">
                    <div style="margin: 5px 0;"><strong>Point 1:</strong> ${pt1.id} (${pt1.cancer_type})</div>
                    <div style="margin: 5px 0;"><strong>Point 2:</strong> ${pt2.id} (${pt2.cancer_type})</div>
                    <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 6px;">
                        <strong style="color: #ffd700;">⭐ Shared:</strong> ${shared.length}/${k} (${((shared.length/k)*100).toFixed(1)}%)
                    </div>
                    <div style="display: flex; gap: 10px; margin-top: 10px;">
                        <div style="flex: 1; padding: 8px; background: rgba(0, 191, 255, 0.1); border-radius: 6px;">
                            <strong style="color: #00bfff;">🔵 Point 1 Unique:</strong> ${unique1.length}
                        </div>
                        <div style="flex: 1; padding: 8px; background: rgba(255, 0, 255, 0.1); border-radius: 6px;">
                            <strong style="color: #ff00ff;">🟣 Point 2 Unique:</strong> ${unique2.length}
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.85em; color: #666;">
                        <strong>Legend:</strong> 
                        🟢 Green = Selected Points | 
                        🔵 Cyan = Point 1 Only | 
                        🟣 Magenta = Point 2 Only | 
                        ⭐ Gold = Shared Neighbors
                        <br>
                        <em>Click any point to start a new comparison</em>
                    </div>
                </div>
            </div>
        `;
    }
}

// Expose functions globally
window.runDualVisualization = runDualVisualization;

// Expose cache clearing function globally
window.clearTSNECache = clearAllCaches;