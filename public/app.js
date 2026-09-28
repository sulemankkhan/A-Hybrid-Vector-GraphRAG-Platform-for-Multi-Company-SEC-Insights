document.addEventListener('DOMContentLoaded', () => {
    const loginModal = document.getElementById('login-modal');
    const dashboard = document.getElementById('dashboard');
    const loginForm = document.getElementById('login-form');
    const loginError = document.getElementById('login-error');
    const logoutBtn = document.getElementById('logout-btn');
    
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const loader = document.getElementById('answer-loader');
    
    const answerContent = document.getElementById('answer-content');
    const vectorContent = document.getElementById('vector-content');
    const graphContent = document.getElementById('graph-content');

    // Authentication State
    let authToken = localStorage.getItem('nexus_token');

    if (authToken) {
        showDashboard();
    }

    function showDashboard() {
        loginModal.classList.add('hidden');
        dashboard.classList.remove('hidden');
    }

    function logout() {
        localStorage.removeItem('nexus_token');
        authToken = null;
        dashboard.classList.add('hidden');
        loginModal.classList.remove('hidden');
    }

    logoutBtn.addEventListener('click', logout);

    // Login Handler
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        
        loginError.textContent = '';
        const submitButton = loginForm.querySelector('button');
        submitButton.disabled = true;
        submitButton.innerHTML = 'Authenticating...';

        try {
            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);

            const response = await fetch('/api/main/token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                credentials: 'include',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Invalid credentials');
            }

            const data = await response.json();
            authToken = data.access_token;
            localStorage.setItem('nexus_token', authToken);
            
            showDashboard();
        } catch (error) {
            loginError.textContent = 'Authentication failed. Please check credentials.';
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Authenticate <span class="arrow">→</span>';
        }
    });

    // Query Handler
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // UI Loading State
        submitBtn.disabled = true;
        loader.classList.remove('hidden');
        answerContent.innerHTML = '<p class="placeholder-text">Synthesizing multi-modal financial data...</p>';
        vectorContent.innerHTML = '<p class="placeholder-text">Retrieving semantic vectors from Pinecone...</p>';
        graphContent.innerHTML = '<p class="placeholder-text">Traversing Knowledge Graph...</p>';

        try {
            const response = await fetch('/api/main/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${authToken}`
                },
                credentials: 'include',
                body: JSON.stringify({ query: query })
            });

            if (response.status === 401) {
                logout();
                throw new Error('Session expired. Please log in again.');
            }

            if (!response.ok) {
                const err = await response.text();
                throw new Error(err);
            }

            const data = await response.json();
            
            // Render Answer using marked.js
            answerContent.innerHTML = marked.parse(data.answer);

            // Render Vector Context
            if (data.vector_context && data.vector_context.length > 0) {
                vectorContent.innerHTML = data.vector_context.map((v, i) => 
                    `<div class="context-card">
                        <strong>Rank ${i+1} (${v.source})</strong><br><br>
                        ${v.text}
                    </div>`
                ).join('');
            } else {
                vectorContent.innerHTML = '<p class="placeholder-text">No relevant vectors found.</p>';
            }

            // Render Graph Context
            if (data.graph_context && data.graph_context.length > 0) {
                graphContent.innerHTML = data.graph_context.map(g => 
                    `<div class="context-card">${g}</div>`
                ).join('');
            } else {
                graphContent.innerHTML = '<p class="placeholder-text">No factual graph connections found.</p>';
            }

        } catch (error) {
            answerContent.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}</p>`;
            vectorContent.innerHTML = '';
            graphContent.innerHTML = '';
        } finally {
            submitBtn.disabled = false;
            loader.classList.add('hidden');
        }
    });
});
