// Scheduler Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const statusFilter = document.getElementById('status-filter');
    const typeFilter = document.getElementById('type-filter');
    const refreshBtn = document.getElementById('refresh-btn');
    const runNowBtn = document.getElementById('run-now-btn');
    const taskTbody = document.getElementById('task-tbody');
    const taskModal = document.getElementById('task-modal');
    const modalClose = document.querySelector('.modal-close');

    // Auto-refresh interval (30 seconds)
    let autoRefreshInterval;

    // Fetch and display tasks
    async function fetchTasks() {
        const status = statusFilter.value;
        const taskType = typeFilter.value;

        const params = new URLSearchParams();
        if (status) params.append('status', status);
        if (taskType) params.append('task_type', taskType);

        try {
            const response = await fetch(`/scheduler/tasks?${params}`);
            const data = await response.json();
            updateTaskTable(data.tasks || []);
        } catch (error) {
            console.error('Failed to fetch tasks:', error);
        }
    }

    // Update task table
    function updateTaskTable(tasks) {
        taskTbody.innerHTML = '';

        if (tasks.length === 0) {
            taskTbody.innerHTML = `
                <tr>
                    <td colspan="8" class="no-tasks">No tasks found</td>
                </tr>
            `;
            return;
        }

        tasks.forEach(task => {
            const row = document.createElement('tr');
            row.setAttribute('data-task-id', task.task_id);
            row.setAttribute('data-status', task.status);

            const errorHtml = task.error_message
                ? `<span class="error-icon" title="${escapeHtml(task.error_message)}">⚠️</span>`
                : '';

            row.innerHTML = `
                <td class="task-id" title="${task.task_id}">${task.task_id.substring(0, 8)}...</td>
                <td>${formatTaskType(task.task_type)}</td>
                <td>
                    <span class="task-status status-${task.status}">
                        ${task.status.charAt(0).toUpperCase() + task.status.slice(1)}
                    </span>
                </td>
                <td>${formatDateTime(task.scheduled_time) || '-'}</td>
                <td>${formatDateTime(task.started_time) || '-'}</td>
                <td>${formatDateTime(task.completed_time) || '-'}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${task.progress * 100}%"></div>
                        <span class="progress-text">${Math.round(task.progress * 100)}%</span>
                    </div>
                </td>
                <td>${errorHtml}</td>
            `;

            // Add click handler for task details
            row.addEventListener('click', () => showTaskDetails(task));

            taskTbody.appendChild(row);
        });
    }

    // Format task type for display
    function formatTaskType(type) {
        return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // Format datetime for display
    function formatDateTime(dateStr) {
        if (!dateStr) return null;
        const date = new Date(dateStr);
        return date.toLocaleString();
    }

    // Escape HTML to prevent XSS
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Show task details modal
    async function showTaskDetails(task) {
        const modalBody = document.getElementById('task-modal-body');

        // Fetch full task details
        try {
            const response = await fetch(`/scheduler/tasks/${task.task_id}`);
            const fullTask = await response.json();

            modalBody.innerHTML = `
                <table class="detail-table">
                    <tr><th>Task ID:</th><td><code>${fullTask.task_id}</code></td></tr>
                    <tr><th>Type:</th><td>${formatTaskType(fullTask.task_type)}</td></tr>
                    <tr><th>Status:</th><td><span class="task-status status-${fullTask.status}">${fullTask.status}</span></td></tr>
                    <tr><th>Trigger:</th><td>${fullTask.trigger_type}</td></tr>
                    <tr><th>Scheduled:</th><td>${formatDateTime(fullTask.scheduled_time) || '-'}</td></tr>
                    <tr><th>Started:</th><td>${formatDateTime(fullTask.started_time) || '-'}</td></tr>
                    <tr><th>Completed:</th><td>${formatDateTime(fullTask.completed_time) || '-'}</td></tr>
                    <tr><th>Progress:</th><td>${Math.round(fullTask.progress * 100)}%</td></tr>
                    ${fullTask.error_message ? `<tr><th>Error:</th><td class="error-message">${escapeHtml(fullTask.error_message)}</td></tr>` : ''}
                    ${fullTask.metadata ? `<tr><th>Metadata:</th><td><pre>${JSON.stringify(fullTask.metadata, null, 2)}</pre></td></tr>` : ''}
                </table>
            `;

            taskModal.classList.add('active');
        } catch (error) {
            console.error('Failed to fetch task details:', error);
        }
    }

    // Run now button
    runNowBtn.addEventListener('click', async function() {
        runNowBtn.disabled = true;
        runNowBtn.textContent = 'Running...';

        try {
            const response = await fetch('/scheduler/tasks/run-now', {
                method: 'POST'
            });
            const data = await response.json();

            if (data.success) {
                // Refresh tasks after a short delay
                setTimeout(fetchTasks, 1000);
            } else {
                alert('Failed to run task: ' + data.message);
            }
        } catch (error) {
            console.error('Failed to run task:', error);
            alert('Failed to run task');
        } finally {
            runNowBtn.disabled = false;
            runNowBtn.textContent = 'Run Now';
        }
    });

    // Refresh button
    refreshBtn.addEventListener('click', fetchTasks);

    // Filter change handlers
    statusFilter.addEventListener('change', fetchTasks);
    typeFilter.addEventListener('change', fetchTasks);

    // Modal close handlers
    modalClose.addEventListener('click', () => {
        taskModal.classList.remove('active');
    });

    taskModal.addEventListener('click', (e) => {
        if (e.target === taskModal) {
            taskModal.classList.remove('active');
        }
    });

    // Auto-refresh every 30 seconds
    function startAutoRefresh() {
        autoRefreshInterval = setInterval(fetchTasks, 30000);
    }

    function stopAutoRefresh() {
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
        }
    }

    // Start auto-refresh
    startAutoRefresh();

    // Initial fetch
    fetchTasks();

    // Cleanup on page unload
    window.addEventListener('beforeunload', stopAutoRefresh);
});
