// API Base URL
const API_URL = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws/stream`;

// Global state
let selectedFile = null;
let currentTranscription = null;
let streamWebSocket = null;
let transcriptionHistory = JSON.parse(localStorage.getItem('transcriptionHistory') || '[]');

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    initStream();
    initChat();
    checkSystemStatus();
    loadHistory();
});

// Navigation
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tab = item.dataset.tab;
            
            // Update active states
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Show tab
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(`${tab}-tab`).classList.add('active');
        });
    });
}

// System Status
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        document.getElementById('gpu-status').textContent = data.gpu_available 
            ? `✓ ${data.gpu_name?.split(' ').slice(-2).join(' ') || 'Available'}` 
            : '✗ Not Found';
        
        document.getElementById('whisper-status').textContent = data.models_loaded 
            ? `✓ ${data.whisper_model}` 
            : '✗ Not Loaded';
        
        document.getElementById('llm-status').textContent = data.llm_loaded 
            ? `✓ ${data.llm_model || 'Ready'}` 
            : '✗ Not Loaded';
    } catch (error) {
        console.error('Failed to check system status:', error);
    }
}

// Upload Tab
function initUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const transcribeBtn = document.getElementById('transcribe-btn');
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Transcribe button
    transcribeBtn.addEventListener('click', transcribeFile);
    
    // Result tabs
    document.querySelectorAll('.result-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            document.querySelectorAll('.result-text').forEach(t => t.classList.remove('active'));
            document.getElementById(`result-${tab.dataset.result}`).classList.add('active');
        });
    });
    
    // Copy button
    document.getElementById('copy-btn').addEventListener('click', () => {
        const activeResult = document.querySelector('.result-text.active');
        navigator.clipboard.writeText(activeResult.textContent);
        showNotification('Copied to clipboard!');
    });
}

function handleFileSelect(file) {
    selectedFile = file;
    
    const uploadText = document.querySelector('.upload-text');
    uploadText.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    
    document.getElementById('transcribe-btn').disabled = false;
    
    // Hide previous results
    document.getElementById('results-section').classList.add('hidden');
}

async function transcribeFile() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // Get translation options
    const translations = Array.from(document.querySelectorAll('input[name="translate"]:checked'))
        .map(cb => cb.value)
        .join(',');
    
    if (translations) {
        formData.append('translate_to', translations);
    }
    
    formData.append('word_timestamps', document.getElementById('word-timestamps').checked);
    
    // Show progress
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';
    
    try {
        // Show detailed progress stages
        const stageMessages = [
            { text: '✅ File uploaded', delay: 500 },
            { text: '⏳ Extracting audio...', delay: 1500 },
            { text: '✅ Audio extracted', delay: 2000 },
            { text: '⏳ Running Voice Activity Detection...', delay: 2500 },
            { text: '✅ VAD complete (removing silence)', delay: 3000 },
            { text: '⏳ Detecting language...', delay: 3500 },
            { text: '✅ Language detected', delay: 4000 },
            { text: '⏳ Transcribing with Whisper Turbo...', delay: 4500 },
        ];
        
        let currentStageIndex = 0;
        let startTime = Date.now();
        
        const updateStages = () => {
            const elapsed = Date.now() - startTime;
            
            // Update stage text
            while (currentStageIndex < stageMessages.length && elapsed >= stageMessages[currentStageIndex].delay) {
                progressText.innerHTML = stageMessages.slice(0, currentStageIndex + 1)
                    .map(s => s.text)
                    .join('<br>');
                currentStageIndex++;
            }
            
            // Update progress bar
            if (currentStageIndex < stageMessages.length) {
                const progress = (currentStageIndex / stageMessages.length) * 100;
                progressFill.style.width = `${progress}%`;
            } else {
                // Transcribing - show elapsed time
                const transcribeElapsed = Math.floor((elapsed - 4500) / 1000);
                progressText.innerHTML = stageMessages.map(s => s.text).join('<br>') + 
                    `<br><span style="color: #10b981">⏱️ Transcribing... ${transcribeElapsed}s</span>`;
                
                // Slowly progress from 50% to 95%
                const progress = Math.min(95, 50 + (transcribeElapsed * 0.5));
                progressFill.style.width = `${progress}%`;
            }
        };
        
        const progressInterval = setInterval(updateStages, 100);
        window.transcribeInterval = progressInterval;
        
        const response = await fetch(`${API_URL}/transcribe`, {
            method: 'POST',
            body: formData
        });
        
        if (window.transcribeInterval) {
            clearInterval(window.transcribeInterval);
        }
        progressFill.style.width = '100%';
        
        // Show completion message
        progressText.innerHTML = stageMessages.map(s => s.text).join('<br>') + 
            '<br><span style="color: #10b981; font-weight: bold">✅ Transcription Complete!</span>';
        
        if (!response.ok) {
            throw new Error('Transcription failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
        // Save to history
        saveToHistory(result);
        
        setTimeout(() => {
            progressSection.classList.add('hidden');
        }, 1000);
        
    } catch (error) {
        console.error('Error:', error);
        progressText.textContent = 'Error: ' + error.message;
        showNotification('Transcription failed', 'error');
    }
}

function displayResults(result) {
    currentTranscription = result;
    
    // Show results section
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');
    
    // Update info
    document.getElementById('detected-lang').textContent = result.detected_language.toUpperCase();
    document.getElementById('duration').textContent = formatDuration(result.duration);
    document.getElementById('proc-time').textContent = `${result.processing_time.toFixed(2)}s`;
    
    // Update transcription
    document.getElementById('result-transcription').textContent = result.transcription;
    
    // Update translations
    if (result.translations) {
        if (result.translations.english) {
            document.getElementById('result-english').textContent = result.translations.english;
        }
        if (result.translations.russian) {
            document.getElementById('result-russian').textContent = result.translations.russian;
        }
    }
}

// Stream Tab
function initStream() {
    const startBtn = document.getElementById('stream-start-btn');
    const stopBtn = document.getElementById('stream-stop-btn');
    
    startBtn.addEventListener('click', startStreaming);
    stopBtn.addEventListener('click', stopStreaming);
}

async function startStreaming() {
    const streamUrl = document.getElementById('stream-url').value.trim();
    if (!streamUrl) {
        showNotification('Please enter a stream URL', 'error');
        return;
    }
    
    const quality = document.getElementById('stream-quality').value;
    const chunkDuration = parseInt(document.getElementById('chunk-duration').value);
    
    // Update UI
    document.getElementById('stream-start-btn').classList.add('hidden');
    document.getElementById('stream-stop-btn').classList.remove('hidden');
    document.getElementById('stream-status').textContent = 'Connecting...';
    document.getElementById('stream-text').textContent = '';
    
    // Connect WebSocket
    streamWebSocket = new WebSocket(WS_URL);
    
    streamWebSocket.onopen = () => {
        console.log('WebSocket connected');
        streamWebSocket.send(JSON.stringify({
            action: 'start',
            stream_url: streamUrl,
            quality: quality,
            chunk_duration: chunkDuration
        }));
    };
    
    streamWebSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'status') {
            document.getElementById('stream-status').textContent = data.message;
        } else if (data.type === 'transcription') {
            const streamText = document.getElementById('stream-text');
            const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
            streamText.innerHTML += `<div><strong>[${timestamp}] ${data.language.toUpperCase()}:</strong> ${data.text}</div>`;
            streamText.scrollTop = streamText.scrollHeight;
        } else if (data.type === 'error') {
            showNotification('Stream error: ' + data.message, 'error');
        }
    };
    
    streamWebSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showNotification('Connection error', 'error');
        stopStreaming();
    };
    
    streamWebSocket.onclose = () => {
        console.log('WebSocket closed');
        stopStreaming();
    };
}

function stopStreaming() {
    if (streamWebSocket) {
        streamWebSocket.send(JSON.stringify({ action: 'stop' }));
        streamWebSocket.close();
        streamWebSocket = null;
    }
    
    document.getElementById('stream-start-btn').classList.remove('hidden');
    document.getElementById('stream-stop-btn').classList.add('hidden');
    document.getElementById('stream-status').textContent = 'Idle';
}

// Chat Tab
function initChat() {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    
    sendBtn.addEventListener('click', sendChatMessage);
    
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

async function sendChatMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    // Add user message
    addChatMessage('user', message);
    chatInput.value = '';
    
    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message assistant typing-indicator';
    typingDiv.innerHTML = `
        <div class="chat-avatar">AI</div>
        <div class="chat-bubble">Thinking...</div>
    `;
    document.getElementById('chat-messages').appendChild(typingDiv);
    
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: 'web-session',
                context: currentTranscription?.transcription || null
            })
        });
        
        // Remove typing indicator
        typingDiv.remove();
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        // Handle different response formats
        let responseText = data.response || data.message || data.text || 'No response';
        
        // Clean up response if it contains extra formatting
        responseText = responseText.trim();
        
        addChatMessage('assistant', responseText);
        
    } catch (error) {
        // Remove typing indicator
        typingDiv.remove();
        console.error('Chat error:', error);
        addChatMessage('assistant', 'Sorry, I encountered an error. Please check if the LLM model is loaded.');
    }
}

function addChatMessage(role, text) {
    const messagesDiv = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    messageDiv.innerHTML = `
        <div class="chat-avatar">${role === 'user' ? 'U' : 'AI'}</div>
        <div class="chat-bubble">${escapeHtml(text)}</div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// History
function loadHistory() {
    const historyList = document.getElementById('history-list');
    
    if (transcriptionHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                </svg>
                <p>No transcriptions yet</p>
            </div>
        `;
        return;
    }
    
    // Only show transcription results (filename, text preview, duration, timestamp)
    historyList.innerHTML = transcriptionHistory.map((item, index) => `
        <div class="history-item" onclick="loadFromHistory(${index})">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #10b981;">${item.filename || 'Transcription'}</h4>
                <span style="font-size: 0.75rem; color: #6b7280;">${item.detected_language?.toUpperCase() || ''} • ${formatDuration(item.duration || 0)}</span>
            </div>
            <p style="margin: 8px 0; color: #d1d5db; font-size: 0.875rem; line-height: 1.4;">${(item.transcription || '').substring(0, 200)}${(item.transcription || '').length > 200 ? '...' : ''}</p>
            <small style="color: #6b7280;">${new Date(item.timestamp).toLocaleString()}</small>
        </div>
    `).join('');
}

function saveToHistory(result) {
    transcriptionHistory.unshift({
        ...result,
        timestamp: Date.now()
    });
    
    // Keep only last 50
    if (transcriptionHistory.length > 50) {
        transcriptionHistory = transcriptionHistory.slice(0, 50);
    }
    
    localStorage.setItem('transcriptionHistory', JSON.stringify(transcriptionHistory));
    loadHistory();
}

function loadFromHistory(index) {
    const item = transcriptionHistory[index];
    currentTranscription = item;
    
    // Pin to chat and switch to chat tab
    pinToChat(item);
}

function pinToChat(item) {
    currentTranscription = item;
    
    // Update pinned context UI
    const pinnedContext = document.getElementById('pinned-context');
    document.getElementById('pinned-filename').textContent = item.filename || 'Transcription';
    
    const preview = (item.transcription || '').substring(0, 100);
    document.getElementById('pinned-preview').textContent = preview + (preview.length >= 100 ? '...' : '');
    
    pinnedContext.classList.remove('hidden');
    
    // Switch to chat tab
    document.querySelector('.nav-item[data-tab="chat"]').click();
    
    // Update placeholder
    document.getElementById('chat-input').placeholder = `Ask about "${item.filename || 'this transcription'}"...`;
    document.getElementById('chat-input').focus();
    
    // Add system message
    addChatMessage('assistant', `📌 Context loaded: "${item.filename || 'Transcription'}". You can now ask me questions about this content!`);
}

function removePinnedContext() {
    currentTranscription = null;
    document.getElementById('pinned-context').classList.add('hidden');
    document.getElementById('chat-input').placeholder = 'Ask me anything...';
    addChatMessage('assistant', '📌 Context removed. I\'ll now answer general questions.');
}

function showHistoryModal(item) {
    const modal = document.getElementById('history-modal');
    
    // Set modal content
    document.getElementById('modal-title').textContent = item.filename || 'Transcription';
    document.getElementById('modal-lang').textContent = (item.detected_language || 'Unknown').toUpperCase();
    document.getElementById('modal-duration').textContent = formatDuration(item.duration || 0);
    document.getElementById('modal-date').textContent = new Date(item.timestamp).toLocaleString();
    
    // Set text content
    document.getElementById('modal-original').textContent = item.transcription || 'No transcription';
    document.getElementById('modal-english').textContent = item.translations?.english || 'No English translation';
    document.getElementById('modal-russian').textContent = item.translations?.russian || 'No Russian translation';
    
    // Reset tabs
    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.modal-text').forEach(t => t.classList.remove('active'));
    document.querySelector('.modal-tab[data-tab="original"]').classList.add('active');
    document.getElementById('modal-original').classList.add('active');
    
    // Setup tab switching
    document.querySelectorAll('.modal-tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.modal-text').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`modal-${tab.dataset.tab}`).classList.add('active');
        };
    });
    
    // Show modal
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeHistoryModal() {
    document.getElementById('history-modal').classList.add('hidden');
    document.body.style.overflow = '';
}

function copyModalText() {
    const activeText = document.querySelector('.modal-text.active');
    navigator.clipboard.writeText(activeText.textContent);
    alert('Copied to clipboard!');
}

function useInChat() {
    closeHistoryModal();
    // Switch to chat tab
    document.querySelector('.nav-item[data-tab="chat"]').click();
    // Pre-fill context hint
    document.getElementById('chat-input').placeholder = 'Ask about: ' + (currentTranscription?.filename || 'this transcription');
}

// Utilities
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type = 'success') {
    // Simple notification (you can enhance this)
    alert(message);
}
