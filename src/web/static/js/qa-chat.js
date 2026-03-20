/**
 * IRIS QA Chat Module
 * Handles interactive Q&A with the literature database.
 */

(function() {
    'use strict';

    // State management
    const state = {
        sessionIds: {},
        currentMode: 'global',
        currentPaperId: null,
        isMinimized: false
    };

    // Initialize when DOM is ready
    function init() {
        console.log('[QA Chat] Initializing...');
        initGlobalChat();
        initEmbeddedChat();
        bindGlobalEvents();
        console.log('[QA Chat] Initialization complete');
    }

    // Initialize global floating chat
    function initGlobalChat() {
        const container = document.createElement('div');
        container.id = 'qa-chat-global';
        container.innerHTML = `
            <div class="qa-chat-container" id="qa-chat-container" data-mode="global">
                <!-- Chat Header -->
                <div class="qa-chat-header">
                    <div class="qa-chat-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                        </svg>
                        <span>Ask AI - Global Search</span>
                    </div>
                    <div class="qa-chat-controls">
                        <button class="qa-btn-minimize" title="Minimize">−</button>
                        <button class="qa-btn-close" title="Close">×</button>
                    </div>
                </div>

                <!-- Chat Messages -->
                <div class="qa-chat-messages" id="qa-chat-messages">
                    <div class="qa-message qa-message-system">
                        <div class="qa-message-content">
                            Ask questions about the entire literature database. I'll search across all papers to find relevant information.
                        </div>
                    </div>
                </div>

                <!-- Chat Input -->
                <div class="qa-chat-input-container">
                    <textarea
                        id="qa-chat-input-global"
                        class="qa-chat-input"
                        placeholder="Type your question here..."
                        rows="1"
                    ></textarea>
                    <div class="qa-chat-actions">
                        <span class="qa-chat-status" id="qa-chat-status"></span>
                        <button class="qa-btn-send" id="qa-btn-send-global" disabled>
                            <span class="qa-send-text">Send</span>
                            <span class="qa-send-loader"></span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Floating Toggle Button -->
            <button class="qa-chat-toggle" id="qa-chat-toggle" title="Ask AI">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
                <span>Ask AI</span>
            </button>
        `;
        document.body.appendChild(container);
        bindChatEvents('global');
    }

    // Initialize embedded chat (for detail pages)
    function initEmbeddedChat() {
        console.log('[QA Chat] Looking for embedded chat container...');
        const embeddedContainer = document.querySelector('.qa-chat-embedded');
        console.log('[QA Chat] Embedded container found:', !!embeddedContainer);

        if (!embeddedContainer) {
            console.log('[QA Chat] No embedded container found, skipping embedded chat init');
            return;
        }

        const mode = embeddedContainer.dataset.mode || 'specific';
        const paperId = embeddedContainer.dataset.paperId;

        console.log('[QA Chat] Initializing embedded chat with mode:', mode, 'paperId:', paperId);

        const chatContainer = document.createElement('div');
        chatContainer.className = 'qa-chat-container qa-chat-embedded';
        chatContainer.dataset.mode = mode;
        if (paperId) chatContainer.dataset.paperId = paperId;

        chatContainer.innerHTML = `
            <div class="qa-chat-header">
                <div class="qa-chat-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                    <span>Ask Questions About This Paper</span>
                </div>
            </div>

            <div class="qa-chat-messages" id="qa-chat-messages-${mode}">
                <div class="qa-message qa-message-system">
                    <div class="qa-message-content">
                        Ask questions about this paper. I'll search through its content to find answers.
                    </div>
                </div>
            </div>

            <div class="qa-chat-input-container">
                <textarea
                    id="qa-chat-input-${mode}"
                    class="qa-chat-input"
                    placeholder="Ask a question about this paper..."
                    rows="1"
                ></textarea>
                <div class="qa-chat-actions">
                    <span class="qa-chat-status" id="qa-chat-status-${mode}"></span>
                    <button class="qa-btn-send" id="qa-btn-send-${mode}" disabled>
                        <span class="qa-send-text">Send</span>
                        <span class="qa-send-loader"></span>
                    </button>
                </div>
            </div>

            ${mode === 'specific' ? `
            <div class="qa-mode-toggle">
                <button class="qa-mode-btn active" data-mode="specific">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                    <span>This Paper</span>
                </button>
                <button class="qa-mode-btn" data-mode="global">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    <span>All Papers</span>
                </button>
            </div>
            ` : ''}
        `;

        // Append the chat container to the embedded container
        embeddedContainer.innerHTML = '';
        embeddedContainer.appendChild(chatContainer);

        bindChatEvents(mode, paperId);

        // Bind mode toggle events
        if (mode === 'specific') {
            const modeBtns = chatContainer.querySelectorAll('.qa-mode-btn');
            modeBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    modeBtns.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    state.currentMode = btn.dataset.mode;
                });
            });
        }

        console.log('[QA Chat] Embedded chat initialized successfully');
    }

    // Bind chat events for a specific chat instance
    function bindChatEvents(chatId, paperId = null) {
        let container;
        if (chatId === 'global') {
            container = document.querySelector('#qa-chat-global .qa-chat-container');
        } else {
            // For embedded chats, find the container inside .qa-chat-embedded
            const embeddedWrapper = document.querySelector('.qa-chat-embedded');
            if (embeddedWrapper) {
                container = embeddedWrapper.querySelector('.qa-chat-container');
            }
        }

        if (!container) return;

        const input = document.getElementById(`qa-chat-input-${chatId === 'global' ? 'global' : chatId}`);
        const sendBtn = document.getElementById(`qa-btn-send-${chatId === 'global' ? 'global' : chatId}`);
        const messagesDiv = document.getElementById(`qa-chat-messages-${chatId === 'global' ? '' : chatId}`) || container.querySelector('.qa-chat-messages');
        const statusSpan = document.getElementById(`qa-chat-status-${chatId === 'global' ? 'global' : chatId}`) || container.querySelector('.qa-chat-status');

        // Auto-resize textarea
        if (input) {
            input.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                sendBtn.disabled = this.value.trim() === '';
            });

            // Send on Enter (Shift+Enter for new line)
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (this.value.trim()) {
                        sendMessage(chatId, paperId);
                    }
                }
            });
        }

        // Send button click
        if (sendBtn) {
            sendBtn.addEventListener('click', () => sendMessage(chatId, paperId));
        }

        // Toggle button (for global chat)
        if (chatId === 'global') {
            const toggle = document.getElementById('qa-chat-toggle');
            const minimizeBtn = container.querySelector('.qa-btn-minimize');
            const closeBtn = container.querySelector('.qa-btn-close');

            if (toggle) {
                toggle.addEventListener('click', () => {
                    container.classList.add('qa-chat-visible');
                    toggle.style.display = 'none';
                });
            }

            if (minimizeBtn) {
                minimizeBtn.addEventListener('click', () => {
                    container.classList.remove('qa-chat-visible');
                    if (toggle) toggle.style.display = 'flex';
                });
            }

            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    container.classList.remove('qa-chat-visible');
                    if (toggle) toggle.style.display = 'flex';
                    // Clear messages
                    if (messagesDiv) {
                        messagesDiv.innerHTML = `
                            <div class="qa-message qa-message-system">
                                <div class="qa-message-content">
                                    Ask questions about the entire literature database. I'll search across all papers to find relevant information.
                                </div>
                            </div>
                        `;
                    }
                    // Reset session
                    delete state.sessionIds['global'];
                });
            }
        }

        // Store references
        state[chatId] = { input, sendBtn, messagesDiv, statusSpan, container };
    }

    // Send a message
    async function sendMessage(chatId, paperId) {
        const chat = state[chatId];
        if (!chat || !chat.input) return;

        const question = chat.input.value.trim();
        if (!question) return;

        // Get current mode (for embedded chats)
        let mode = chatId;
        let currentPaperId = paperId;

        if (chat.container && chat.container.classList.contains('qa-chat-embedded')) {
            const activeModeBtn = chat.container.querySelector('.qa-mode-btn.active');
            if (activeModeBtn) {
                mode = activeModeBtn.dataset.mode;
            }
        }

        // Add user message
        addMessage(chatId, 'user', question);

        // Clear input
        chat.input.value = '';
        chat.input.style.height = 'auto';
        chat.sendBtn.disabled = true;

        // Show loading
        if (chat.statusSpan) {
            chat.statusSpan.textContent = 'Thinking...';
        }
        chat.container.classList.add('qa-chat-loading');

        try {
            // Get or create session
            if (!state.sessionIds[chatId]) {
                const response = await fetch('/api/qa/conversation', {
                    method: 'POST'
                });
                const data = await response.json();
                state.sessionIds[chatId] = data.session_id;
            }

            // Send query
            const response = await fetch(`/api/qa/conversation/${state.sessionIds[chatId]}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    mode: mode,
                    paper_id: mode === 'specific' ? currentPaperId : null,
                    top_k: 5
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();

            // Add assistant response
            addMessage(chatId, 'assistant', data.answer);

        } catch (error) {
            console.error('QA request failed:', error);
            addMessage(chatId, 'system', 'Sorry, something went wrong. Please try again.');
        } finally {
            // Hide loading
            if (chat.statusSpan) {
                chat.statusSpan.textContent = '';
            }
            chat.container.classList.remove('qa-chat-loading');
            chat.input.focus();
        }
    }

    // Add a message to the chat
    function addMessage(chatId, role, content) {
        const chat = state[chatId];
        if (!chat || !chat.messagesDiv) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `qa-message qa-message-${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'qa-message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        chat.messagesDiv.appendChild(messageDiv);

        // Scroll to bottom
        chat.messagesDiv.scrollTop = chat.messagesDiv.scrollHeight;

        // Trigger MathJax rendering
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([contentDiv]).catch(err => console.error('MathJax error:', err));
        }
    }

    // Bind global events
    function bindGlobalEvents() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && state['global']) {
                // Page became visible, refocus input if chat is open
                const chat = state['global'];
                if (chat && chat.container.classList.contains('qa-chat-visible')) {
                    chat.input.focus();
                }
            }
        });
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
