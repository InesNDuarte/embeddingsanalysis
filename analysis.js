// Global State
let globalData = [];
let selectedPoints = [];

const setStatus = (msg) => {
    document.getElementById('status').innerText = msg;
};

// // 1. DATA LOADING AND SETUP
// Use the raw.githubusercontent.com URL to fetch the ZIP directly (avoids HTML redirects)
const EMBEDDINGS_ZIP_URL = "https://raw.githubusercontent.com/epiverse/tcgadata/main/TITAN/text_image_embeddings.tsv.zip";

// Helper function to convert the string representation of an embedding into a float array
function parseEmbedding(embStr) {
    // Handles array strings like "[0.1, 0.2, ...]"
    return embStr
        .replace(/[\[\]"]/g, '')
        .split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n));
}

/**
 * Fetches, unzips, and parses the TSV data from the GitHub URL.
 * @returns {Promise<Array<Object>>} A promise that resolves to the parsed array of data objects.
 */
async function loadData() {
    console.log("Fetching embeddings ZIP file...");
    setStatus('Phase 1/5 (0%): Downloading data...');
    try {
        // 1. Fetch the ZIP file
        const response = await fetch(EMBEDDINGS_ZIP_URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch embeddings file. HTTP Status: ${response.status}`);
        }

        const dataBuffer = await response.arrayBuffer();
    console.log("Successfully fetched ZIP file. Unzipping...");
    setStatus('Unzipping and parsing data...');

        // 2. Unzip the file
        // If JSZip isn't loaded yet, try to dynamically load a CDN copy before failing.
        if (typeof JSZip === 'undefined') {
            console.warn('JSZip not present. Attempting to load JSZip dynamically from CDN...');
            // Try jsDelivr CDN first
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';
                script.onload = () => resolve();
                script.onerror = () => reject(new Error('Dynamic JSZip load failed'));
                document.head.appendChild(script);
                // Timeout after 8s
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

        // Get the file as a Uint8Array and stream-parse it in chunks to avoid
        // allocating a giant string or a large array of lines at once.
        console.log("Successfully fetched ZIP file. Parsing TSV entry as stream...");
        setStatus('Phase 2/5 (20%): Extracting ZIP...');
        const uint8array = await file.async('uint8array');

        // Stream parser that decodes chunks and processes completed lines.
        const decoder = new TextDecoder('utf-8');
        const CHUNK_SIZE = 64 * 1024; // 64KB
        let pos = 0;
        let carry = '';
        let headers = null;
        const data = [];
        let lastPercent = -1;

        setStatus('Phase 3/5 (40%): Parsing data - 0%');
        while (pos < uint8array.length) {
            const end = Math.min(pos + CHUNK_SIZE, uint8array.length);
            const chunk = uint8array.subarray(pos, end);
            pos = end;

            // Decode the current chunk and prepend any carry-over from the previous chunk
            let text = decoder.decode(chunk, { stream: true });
            text = carry + text;

            // Split into lines, keep the last partial line as carry
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

            // Update percent progress, but only when it changes to avoid UI thrash
            const percent = Math.floor((pos / uint8array.length) * 100);
            if (percent !== lastPercent) {
                lastPercent = percent;
                setStatus(`Phase 3/5 (40%): Parsing data - ${percent}%`);
                // Yield to the browser so it can repaint the UI and show the updated percent.
                // Using requestAnimationFrame is lighter than setTimeout(0) and schedules
                // the continuation after the next paint.
                await new Promise(resolve => requestAnimationFrame(resolve));
            }
        }

        // Process any remaining carry as the last line
        if (carry) {
            const lastLine = carry.replace(/\r$/, '');
            if (lastLine) {
                if (!headers) {
                    headers = lastLine.split('\t');
                } else {
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
        }

        console.log(`Data parsing complete. Loaded ${data.length} samples.`);

        // Sanity-check: filter valid samples (both text and image embeddings present)
        const validData = data.filter(d => d.text_embedding.length > 0 && d.image_embedding.length > 0);
        const validCount = validData.length;
        // Print a short summary and the first valid sample for runtime inspection
        console.log('Sanity check after loadData():', {
            totalRows: data.length,
            validSamples: validCount,
            firstValidSample: validData.length ? validData[0] : null
        });

        setStatus('Phase 3/5 (60%): Data loaded');
        return validData;

    } catch (error) {
        console.error("Error during data loading:", error);
        setStatus(`Failed to load data: ${error.message}`);
        alert(`Failed to load data: ${error.message}`);
        return []; // Return an empty array on failure
    }
}

// Global variable to hold the loaded data (will be populated async)
let data = [];

/**
 * Simple PCA implementation for dimensionality reduction
 * @param {Array<Array>} data - Input data (n samples x d features)
 * @param {number} targetDim - Target dimensionality (default 50)
 * @returns {Array<Array>} Reduced data (n samples x targetDim)
 */
function performPCA(data, targetDim = 50) {
    if (data.length === 0) return data;
    
    const n = data.length;
    const d = data[0].length;
    
    // If data is already lower dimensional than target, return as is
    if (d <= targetDim) {
        console.log(`Data dimension ${d} is already <= target ${targetDim}, skipping PCA`);
        return data;
    }
    
    // 1. Compute mean
    const mean = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            mean[j] += data[i][j];
        }
    }
    for (let j = 0; j < d; j++) mean[j] /= n;
    
    // 2. Center data
    const centered = data.map(row => row.map((val, j) => val - mean[j]));
    
    // 3. Compute covariance matrix (simplified: use correlation instead for stability)
    const cov = Array(d).fill(0).map(() => Array(d).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            for (let k = j; k < d; k++) {
                cov[j][k] += centered[i][j] * centered[i][k];
                if (j !== k) cov[k][j] = cov[j][k];
            }
        }
    }
    for (let j = 0; j < d; j++) {
        for (let k = 0; k < d; k++) {
            cov[j][k] /= n;
        }
    }
    
    // 4. Power iteration method to find top eigenvectors
    const eigenvectors = [];
    let matrix = cov.map(row => [...row]);
    
    for (let eig = 0; eig < Math.min(targetDim, d); eig++) {
        // Power iteration
        let v = new Array(d).fill(0);
        v[eig % d] = 1; // Initialize
        
        for (let iter = 0; iter < 20; iter++) {
            // Multiply matrix by vector
            const mv = new Array(d).fill(0);
            for (let i = 0; i < d; i++) {
                for (let j = 0; j < d; j++) {
                    mv[i] += matrix[i][j] * v[j];
                }
            }
            
            // Normalize
            let norm = Math.sqrt(mv.reduce((sum, x) => sum + x * x, 0));
            if (norm < 1e-10) break;
            v = mv.map(x => x / norm);
        }
        
        eigenvectors.push(v);
        
        // Deflate matrix
        for (let i = 0; i < d; i++) {
            for (let j = 0; j < d; j++) {
                matrix[i][j] -= v[i] * v[j];
            }
        }
    }
    
    // 5. Project data onto eigenvectors
    const reduced = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < eigenvectors.length; j++) {
            let proj = 0;
            for (let k = 0; k < d; k++) {
                proj += centered[i][k] * eigenvectors[j][k];
            }
            row.push(proj);
        }
        reduced.push(row);
    }
    
    console.log(`PCA: reduced ${d} dimensions to ${reduced[0].length}`);
    return reduced;
}

/**
 * Calculates Euclidean distance between two vectors 
 */
function euclideanDistance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        const diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

/**
 * Finds K nearest neighbors for a given index
 */
function getKNN(index, allEmbeddings, k = 10) {
    const target = allEmbeddings[index];
    const distances = [];
    
    // Calculate distances, skipping the point itself
    for (let i = 0; i < allEmbeddings.length; i++) {
        if (i === index) continue;
        distances.push({
            index: i,
            dist: euclideanDistance(target, allEmbeddings[i])
        });
    }
    
    // Sort and return top k
    if (distances.length <= k) {
        return distances.map(d => d.index);
    }
    
    distances.sort((a, b) => a.dist - b.dist);
    return distances.slice(0, k).map(d => d.index);
}

/**
 * Runs t-SNE with PCA preprocessing and renders the Plotly 3D scatter
 */
async function runVisualization() {
    const modality = document.getElementById('embeddingType').value;
    if (globalData.length === 0) {
        globalData = await loadData();
    }

    setStatus(`Phase 4/5 (65%): Reducing dimensionality with PCA to 50D...`);
    
    let embeddings = globalData.map(d => d[modality]);
    
    // Apply PCA reduction to 50 dimensions
    embeddings = performPCA(embeddings, 50);
    
    // Yield to browser after PCA
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    setStatus(`Phase 4/5 (70%): Initializing t-SNE for ${modality}...`);
    
    // Initialize t-SNE with optimized parameters for faster convergence
    const tsne = new tsnejs.tSNE({
        dim: 3,
        perplexity: Math.min(20, Math.max(5, globalData.length / 30)), // Adaptive perplexity
        epsilon: 30, // Increased learning rate for faster convergence
        metric: 'euclidean'
    });

    // Yield to browser before initialization
    await new Promise(resolve => requestAnimationFrame(resolve));
    
    try {
        tsne.initDataRaw(embeddings);
    } catch (err) {
        console.error("Error during t-SNE initialization:", err);
        setStatus(`Error: Failed to initialize t-SNE: ${err.message}`);
        return;
    }

    const maxIterations = 200; // Reduced iterations for faster completion
    setStatus(`Phase 5/5 (80%): Running t-SNE iterations - 0%`);
    
    // Run t-SNE iterations with periodic browser yields
    // Process 10 iterations per batch to keep UI responsive
    for (let k = 0; k < maxIterations; k++) {
        tsne.step();
        
        // Update progress every 20 iterations
        if (k % 20 === 0) {
            const progress = Math.floor((k / maxIterations) * 20) + 80; // 80-100%
            setStatus(`Phase 5/5 (${progress}%): Running t-SNE iteration ${k}/${maxIterations}...`);
        }
        
        // Yield to browser every 10 iterations for responsiveness
        if (k % 10 === 0 && k > 0) {
            await new Promise(resolve => requestAnimationFrame(resolve));
        }
    }

    setStatus('Phase 5/5 (95%): Rendering visualization...');
    const pos = tsne.getSolution(); // Returns [x, y, z] array

    renderPlot(pos, modality);
    setStatus("Ready. Click two points to find Shared Nearest Neighbors.");
}

function renderPlot(pos, modality) {
    const x = pos.map(p => p[0]);
    const y = pos.map(p => p[1]);
    const z = pos.map(p => p[2]);
    const labels = globalData.map(d => `ID: ${d.id}<br>Type: ${d.cancer_type}`);

    const trace = {
        x: x, y: y, z: z,
        mode: 'markers',
        type: 'scatter3d',
        text: labels,
        marker: {
            size: 5,
            color: globalData.map((_, i) => i),
            colorscale: 'Viridis',
            opacity: 0.8
        }
    };

    const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        scene: { xaxis: { title: 't-SNE 1' }, yaxis: { title: 't-SNE 2' }, zaxis: { title: 't-SNE 3' } },
        hovermode: 'closest'
    };

    Plotly.newPlot('plot-container', [trace], layout);

    // Click Event for SNN
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
        
        // Find intersection (Shared Neighbors)
        const shared = knn1.filter(idx => knn2.includes(idx));
        
        highlightShared(shared, selectedPoints, positions);
        
        document.getElementById('selectionInfo').innerHTML = `
            <strong>Shared Neighbors:</strong> ${shared.length} found between 
            Point ${selectedPoints[0]} and Point ${selectedPoints[1]}.
        `;
    }
}

function highlightShared(sharedIdx, selectedIdx, pos) {
    // Add a new trace to the plot for the shared points
    const sharedTrace = {
        x: sharedIdx.map(i => pos[i][0]),
        y: sharedIdx.map(i => pos[i][1]),
        z: sharedIdx.map(i => pos[i][2]),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Shared Neighbors',
        marker: { size: 8, color: 'red', symbol: 'circle' }
    };

    const selectedTrace = {
        x: selectedIdx.map(i => pos[i][0]),
        y: selectedIdx.map(i => pos[i][1]),
        z: selectedIdx.map(i => pos[i][2]),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Selected Points',
        marker: { size: 10, color: 'black', symbol: 'diamond' }
    };

    // Update plot (keeping the original points)
    Plotly.addTraces('plot-container', [sharedTrace, selectedTrace]);
}

document.getElementById('processBtn').addEventListener('click', runVisualization);