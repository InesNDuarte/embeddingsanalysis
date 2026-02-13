// Three.js rendering system for 3D scatter plots
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class ThreeJSPlot {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointsData = [];
        this.colorMap = {};
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.pointMeshes = [];
        this.highlightMeshes = [];
        this.tooltip = null;
        this.hoveredPoint = null;
        this.onPointClick = null;
        
        this.init();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        // Ensure container can position legend overlay
        this.container.style.position = this.container.style.position || 'relative';
        
        // Create camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(1.4, 1.4, 1.2);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Add orbit controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Add grid
        const gridHelper = new THREE.GridHelper(10, 10, 0xe8e8e8, 0xe8e8e8);
        this.scene.add(gridHelper);
        
        // Add axes helper
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);
        
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Create tooltip
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tooltip';
        this.container.appendChild(this.tooltip);
        
        // Event listeners
        this.renderer.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.renderer.domElement.addEventListener('click', this.onClick.bind(this));
        window.addEventListener('resize', this.onWindowResize.bind(this));
        
        // Start animation loop
        this.animate();
    }
    
    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check intersections with base points only
        const basePoints = this.pointMeshes.filter(m => m.userData.isBase);
        const intersects = this.raycaster.intersectObjects(basePoints);
        
        if (intersects.length > 0) {
            const pointIndex = intersects[0].object.userData.index;
            this.hoveredPoint = pointIndex;
            
            // Show tooltip
            const data = this.pointsData[pointIndex];
            this.tooltip.innerHTML = `<b>${data.id}</b><br>Patient: ${data.patient_id}<br>Type: <b>${data.cancer_type}</b>`;
            this.tooltip.style.display = 'block';
            this.tooltip.style.left = event.clientX - rect.left + 10 + 'px';
            this.tooltip.style.top = event.clientY - rect.top + 10 + 'px';
        } else {
            this.hoveredPoint = null;
            this.tooltip.style.display = 'none';
        }
    }
    
    onClick(event) {
        if (this.hoveredPoint !== null && this.onPointClick) {
            this.onPointClick(this.hoveredPoint);
        }
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    // Generate color palette for cancer types (lighter pastel colors)
    generateColorPalette(uniqueTypes) {
        const colors = [
            0xe6194b, 0x3cb44b, 0xffe119, 0x4363d8, 0xf58231, 
            0x911eb4, 0x46f0f0, 0xf032e6, 0xbcf60c, 0xfabebe, 
            0x008080, 0xe6beff, 0x9a6324, 0xfffac8, 0x800000
        ];
        
        const colorMap = {};
        uniqueTypes.forEach((type, i) => {
            colorMap[type] = colors[i % colors.length];
        });
        return colorMap;
    }
    
    renderPoints(positions, data) {
        // Clear existing points
        this.pointMeshes.forEach(mesh => this.scene.remove(mesh));
        this.pointMeshes = [];
        this.highlightMeshes.forEach(mesh => this.scene.remove(mesh));
        this.highlightMeshes = [];
        
        this.pointsData = data;
        
        // Get unique cancer types and generate colors
        const uniqueTypes = [...new Set(data.map(d => d.cancer_type))];
        this.colorMap = this.generateColorPalette(uniqueTypes);

        // Create or update legend overlay for this plot
        this._updateLegend(uniqueTypes, this.colorMap);
        
        // Normalize positions for better visualization
        const bounds = this.calculateBounds(positions);
        const scale = 5 / Math.max(bounds.xRange, bounds.yRange, bounds.zRange);
        
        // Create point geometry
        const geometry = new THREE.SphereGeometry(0.02, 8, 8);
        
        positions.forEach((pos, i) => {
            const x = (pos[0] - bounds.xCenter) * scale;
            const y = (pos[1] - bounds.yCenter) * scale;
            const z = (pos[2] - bounds.zCenter) * scale;
            
            const color = this.colorMap[data[i].cancer_type];
            const material = new THREE.MeshPhongMaterial({
                color: color,
                transparent: true,
                opacity: 1.0
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, y, z);
            mesh.userData = {
                index: i,
                isBase: true,
                originalColor: color
            };
            
            this.scene.add(mesh);
            this.pointMeshes.push(mesh);
        });
        
        // Center camera on data
        this.centerCamera(bounds, scale);
    }

    _updateLegend(uniqueTypes, colorMap) {
        try {
            let legend = this.container.querySelector('.threejs-legend');
            if (!legend) {
                legend = document.createElement('div');
                legend.className = 'threejs-legend';
                legend.style.position = 'absolute';
                legend.style.top = '10px';
                legend.style.right = '10px';
                legend.style.background = 'rgba(255, 255, 255, 0.9)';
                legend.style.border = '1px solid rgba(0,0,0,0.08)';
                legend.style.padding = '8px';
                legend.style.borderRadius = '6px';
                legend.style.maxHeight = '60%';
                legend.style.overflow = 'auto';
                legend.style.fontSize = '12px';
                legend.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
                this.container.appendChild(legend);
            } else {
                legend.innerHTML = '';
            }

            uniqueTypes.forEach(type => {
                const colorNum = colorMap[type];
                const hex = '#' + (colorNum.toString(16).padStart(6, '0'));
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.gap = '8px';
                row.style.marginBottom = '6px';

                const sw = document.createElement('span');
                sw.style.display = 'inline-block';
                sw.style.width = '14px';
                sw.style.height = '14px';
                sw.style.borderRadius = '3px';
                sw.style.background = hex;
                sw.style.border = '1px solid rgba(0,0,0,0.08)';

                const label = document.createElement('span');
                label.textContent = type;
                label.style.color = '#333';

                row.appendChild(sw);
                row.appendChild(label);
                legend.appendChild(row);
            });
        } catch (err) {
            console.warn('Failed to update ThreeJS legend:', err);
        }
    }
    
    calculateBounds(positions) {
        const xs = positions.map(p => p[0]);
        const ys = positions.map(p => p[1]);
        const zs = positions.map(p => p[2]);
        
        const xMin = Math.min(...xs);
        const xMax = Math.max(...xs);
        const yMin = Math.min(...ys);
        const yMax = Math.max(...ys);
        const zMin = Math.min(...zs);
        const zMax = Math.max(...zs);
        
        return {
            xCenter: (xMin + xMax) / 2,
            yCenter: (yMin + yMax) / 2,
            zCenter: (zMin + zMax) / 2,
            xRange: xMax - xMin,
            yRange: yMax - yMin,
            zRange: zMax - zMin
        };
    }
    
    centerCamera(bounds, scale) {
        const distance = 8;
        this.camera.position.set(distance, distance, distance);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }
    
    highlightPoints(sharedIdx, unique1Idx, unique2Idx, selectedIdx) {
        // Clear existing highlights
        this.highlightMeshes.forEach(mesh => this.scene.remove(mesh));
        this.highlightMeshes = [];
        
        // Dim all base points first when highlighting
        this.pointMeshes.forEach(mesh => {
            if (mesh.userData.isBase) {
                mesh.material.opacity = 0.15;
            }
        });
        
        // Helper to create highlight mesh
        const createHighlight = (indices, color, size, shape = 'sphere', glowIntensity = 1.0) => {
            indices.forEach(idx => {
                const baseMesh = this.pointMeshes[idx];
                if (!baseMesh) return;
                
                let geometry;
                if (shape === 'diamond') {
                    geometry = new THREE.OctahedronGeometry(size, 0);
                } else {
                    geometry = new THREE.SphereGeometry(size, 16, 16);
                }
                
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: glowIntensity * 0.3,
                    transparent: false,
                    opacity: 1.0
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.copy(baseMesh.position);
                mesh.userData = { isHighlight: true };
                
                this.scene.add(mesh);
                this.highlightMeshes.push(mesh);
            });
        };
        
        // Create highlights with different colors and sizes
        // Point 1 neighbors only (user-requested pastel cyan: #9fe7ff)
        if (unique1Idx && unique1Idx.length > 0) {
            createHighlight(unique1Idx, 0x9fe7ff, 0.03, 'sphere', 1.2);
        }
        
        // Point 2 neighbors only (user-requested pastel magenta: #ff9fff)
        if (unique2Idx && unique2Idx.length > 0) {
            createHighlight(unique2Idx, 0xff9fff, 0.03, 'sphere', 1.2);
        }
        
        // Shared neighbors (bright yellow/gold)
        if (sharedIdx && sharedIdx.length > 0) {
            createHighlight(sharedIdx, 0xffd700, 0.05, 'diamond', 1.5);
        }
        
        // Selected points (bright green)
        if (selectedIdx && selectedIdx.length > 0) {
            createHighlight(selectedIdx, 0x00ff00, 0.05, 'diamond', 2.0);
        }
    }
    
    clearHighlights() {
        this.highlightMeshes.forEach(mesh => this.scene.remove(mesh));
        this.highlightMeshes = [];
        
        // Restore base point opacity
        this.pointMeshes.forEach(mesh => {
            if (mesh.userData.isBase) {
                mesh.material.opacity = 1.0;
            }
        });
    }
    
    clear() {
        this.pointMeshes.forEach(mesh => this.scene.remove(mesh));
        this.pointMeshes = [];
        this.clearHighlights();
    }
    
    dispose() {
        this.renderer.domElement.removeEventListener('mousemove', this.onMouseMove);
        this.renderer.domElement.removeEventListener('click', this.onClick);
        window.removeEventListener('resize', this.onWindowResize);
        
        this.pointMeshes.forEach(mesh => {
            mesh.geometry.dispose();
            mesh.material.dispose();
        });
        
        this.highlightMeshes.forEach(mesh => {
            mesh.geometry.dispose();
            mesh.material.dispose();
        });
        
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
        if (this.tooltip) {
            this.container.removeChild(this.tooltip);
        }
    }
}

// Global plot instances
window.threejsPlots = {};

// Initialize a plot
function initThreeJSPlot(containerId) {
    if (window.threejsPlots[containerId]) {
        window.threejsPlots[containerId].dispose();
    }
    window.threejsPlots[containerId] = new ThreeJSPlot(containerId);
    return window.threejsPlots[containerId];
}

// Get existing plot instance
function getThreeJSPlot(containerId) {
    return window.threejsPlots[containerId];
}

// Export to window for use in analysis.js
window.initThreeJSPlot = initThreeJSPlot;
window.getThreeJSPlot = getThreeJSPlot;
window.ThreeJSPlot = ThreeJSPlot;
