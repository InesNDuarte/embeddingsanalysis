// Global State
let globalData = [];
let selectedPoints = [];
let cachedEmbeddings = {}; // Cache for t-SNE results

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
        
        // Check if cache is still valid (less than 7 days old)
        const cacheAge = Date.now() - cacheData.timestamp;
        const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days
        
        if (cacheAge > maxAge) {
            console.log('Cache expired, removing...');
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
 * GPU-accelerated distance matrix calculation for t-SNE
 */
function computeDistanceMatrixGPU(embeddings) {
    return tf.tidy(() => {
        const X = tf.tensor2d(embeddings);
        const n = embeddings.length;
        
        // Compute pairwise squared distances efficiently on GPU
        // ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
        const sqNorms = X.square().sum(1);
        const sqNormsCol = sqNorms.expandDims(1);
        const sqNormsRow = sqNorms.expandDims(0);
        const dotProduct = X.matMul(X.transpose());
        
        const distSq = sqNormsCol.add(sqNormsRow).sub(dotProduct.mul(2));
        
        // Clamp to avoid numerical issues
        const dist = distSq.clipByValue(0, Infinity).sqrt();
        
        return dist.arraySync();
    });
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
 * Memory-efficient t-SNE gradient computation
 * Uses smaller GPU operations to prevent context loss
 */
function computeGradientGPU(Y, P) {
    const n = Y.length;
    const dims = Y[0].length;
    
    // For large datasets, use CPU to prevent GPU memory issues
    if (n > 2000) {
        return computeGradientCPU(Y, P);
    }
    
    return tf.tidy(() => {
        const Ytensor = tf.tensor2d(Y);
        const Ptensor = tf.tensor2d(P);
        
        // Compute pairwise squared distances
        const sqNorms = Ytensor.square().sum(1);
        const sqNormsCol = sqNorms.expandDims(1);
        const sqNormsRow = sqNorms.expandDims(0);
        const dotProduct = Ytensor.matMul(Ytensor.transpose());
        
        const distSq = sqNormsCol.add(sqNormsRow).sub(dotProduct.mul(2));
        
        // Compute Q matrix
        const Q_unnorm = distSq.add(1).pow(-1);
        
        // Zero diagonal
        const eye = tf.eye(n);
        const Q_nodiag = Q_unnorm.mul(tf.sub(1, eye));
        
        // Normalize Q
        const sumQ = Q_nodiag.sum().arraySync();
        const Q = Q_nodiag.div(Math.max(sumQ, 1e-12));
        
        // Compute (P - Q)
        const PQ = Ptensor.sub(Q);
        const mult = PQ.mul(Q_nodiag);
        
        // Compute gradient - extract to CPU early to save GPU memory
        const multArray = mult.arraySync();
        const YArray = Y;
        
        const gradient = Array(n).fill(0).map(() => Array(dims).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                for (let d = 0; d < dims; d++) {
                    gradient[i][d] += 4 * multArray[i][j] * (YArray[i][d] - YArray[j][d]);
                }
            }
        }
        
        return gradient;
    });
}

/**
 * CPU-based gradient computation (fallback for large datasets)
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
 * Fast t-SNE with GPU acceleration and memory management
 */
async function runTSNEwithGPU(embeddings, dims = 3, perplexity = 30, maxIter = 500) {
    const n = embeddings.length;
    
    // Log initial GPU memory
    if (tf.memory) {
        console.log('GPU memory before t-SNE:', tf.memory());
    }
    
    console.log('Computing distance matrix on GPU...');
    setStatus('Phase 4/4 (60%): Computing distances on GPU...');
    const distances = computeDistanceMatrixGPU(embeddings);
    
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
        // Compute gradient (uses GPU for small datasets, CPU for large)
        const gradient = computeGradientGPU(Y, P);
        
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
    setStatus('Phase 3/4 (55%): Initializing GPU acceleration...');
    
    try {
        await tf.ready();
        const backend = tf.getBackend();
        console.log(`TensorFlow.js backend: ${backend}`);
        setStatus(`Phase 3/4 (58%): Using ${backend.toUpperCase()} backend...`);
    } catch (e) {
        console.warn('TensorFlow.js not ready:', e);
    }
    
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    let embeddings = globalData.map(d => d[modality]);
    
    // GPU-accelerated PCA
    setStatus(`Phase 3/4 (60%): GPU-accelerated PCA...`);
    embeddings = await performPCATensorFlow(embeddings, 50);
    
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    // GPU-accelerated t-SNE
    const embedding = await runTSNEwithGPU(embeddings, 3, perplexity, maxIter);
    
    // Cache the results
    saveTSNEToCache(cacheKey, embedding);
    
    setStatus('Rendering visualization...');
    renderPlot(embedding, modality);
    statusEl.className = 'cached';
    setStatus("✅ Ready! Click two points for SNN. Results cached - next load will be instant! 💾");
}

function renderPlot(pos, modality) {
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
        showlegend: false,
        paper_bgcolor: '#ffffff'
    };

    Plotly.newPlot('plot-container', [trace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan3d', 'select3d', 'lasso3d']
    });

    document.getElementById('plot-container').on('plotly_click', function(data) {
        const pn = data.points[0].pointNumber;
        handleSelection(pn, pos);
    });
}

function handleSelection(index, positions) {
    selectedPoints.push(index);
    if (selectedPoints.length > 2) selectedPoints.shift();

    if (selectedPoints.length === 2) {
        const modality = document.getElementById('embeddingType').value;
        const embeddings = globalData.map(d => d[modality]);
        
        const knn1 = getKNN(selectedPoints[0], embeddings, 20);
        const knn2 = getKNN(selectedPoints[1], embeddings, 20);
        
        const shared = knn1.filter(idx => knn2.includes(idx));
        
        highlightShared(shared, selectedPoints, positions);
        
        const pt1 = globalData[selectedPoints[0]];
        const pt2 = globalData[selectedPoints[1]];
        
        document.getElementById('selectionInfo').innerHTML = `
            <div style="background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%); padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #667eea;">
                <strong>🎯 Shared Nearest Neighbors:</strong> <span style="color: #667eea; font-size: 1.2em; font-weight: bold;">${shared.length}</span> found<br>
                <strong>📍 Point 1:</strong> ${pt1.id} <span style="color: #666;">(${pt1.cancer_type})</span><br>
                <strong>📍 Point 2:</strong> ${pt2.id} <span style="color: #666;">(${pt2.cancer_type})</span>
            </div>
        `;
    }
}

function highlightShared(sharedIdx, selectedIdx, pos) {
    const sharedTrace = {
        x: sharedIdx.map(i => pos[i][0]),
        y: sharedIdx.map(i => pos[i][1]),
        z: sharedIdx.map(i => pos[i][2]),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Shared Neighbors',
        text: sharedIdx.map(i => `<b>Shared</b><br>${globalData[i].id}<br>${globalData[i].cancer_type}`),
        hovertemplate: '%{text}<extra></extra>',
        marker: { 
            size: 7, 
            color: '#ff4444',
            symbol: 'circle',
            line: {
                color: '#cc0000',
                width: 2
            },
            opacity: 0.9
        }
    };

    const selectedTrace = {
        x: selectedIdx.map(i => pos[i][0]),
        y: selectedIdx.map(i => pos[i][1]),
        z: selectedIdx.map(i => pos[i][2]),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Selected Points',
        text: selectedIdx.map(i => `<b>Selected</b><br>${globalData[i].id}<br>${globalData[i].cancer_type}`),
        hovertemplate: '%{text}<extra></extra>',
        marker: { 
            size: 12, 
            color: '#2c3e50',
            symbol: 'diamond',
            line: {
                color: '#000000',
                width: 3
            }
        }
    };

    Plotly.addTraces('plot-container', [sharedTrace, selectedTrace]);
}

document.getElementById('processBtn').addEventListener('click', runVisualization);

// Expose cache clearing function globally
window.clearTSNECache = clearAllCaches;