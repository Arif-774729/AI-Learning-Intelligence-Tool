document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    const analyzeBtn = document.getElementById('analyze-btn');
    const dashboard = document.getElementById('dashboard');

    let riskChartInstance = null;
    let difficultyChartInstance = null;

    // File Selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
            analyzeBtn.disabled = false;
        }
    });

    // Analyze Button Click
    analyzeBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        analyzeBtn.textContent = "Analyzing...";
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            // 1. Get Predictions
            const predictResponse = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const predictData = await predictResponse.json();

            // 2. Get Difficulty Insights
            const difficultyResponse = await fetch('/difficulty');
            const difficultyData = await difficultyResponse.json();

            // 3. Update UI
            updateDashboard(predictData.predictions, difficultyData.difficulty_analysis);
            dashboard.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            analyzeBtn.textContent = "Analyze Data";
            analyzeBtn.disabled = false;
        }
    });

    function updateDashboard(predictions, difficultyStats) {
        // --- Summary Stats ---
        const totalStudents = predictions.length;
        const highRisk = predictions.filter(p => p.risk_level === 'High').length;
        const avgProb = predictions.reduce((acc, curr) => acc + curr.completion_probability, 0) / totalStudents;

        document.getElementById('total-students').textContent = totalStudents;
        document.getElementById('high-risk-count').textContent = highRisk;
        document.getElementById('avg-prob').textContent = (avgProb * 100).toFixed(1) + '%';

        // --- Risk Chart ---
        const riskCounts = {
            'High': predictions.filter(p => p.risk_level === 'High').length,
            'Medium': predictions.filter(p => p.risk_level === 'Medium').length,
            'Low': predictions.filter(p => p.risk_level === 'Low').length
        };

        const ctxRisk = document.getElementById('riskChart').getContext('2d');
        if (riskChartInstance) riskChartInstance.destroy();
        
        riskChartInstance = new Chart(ctxRisk, {
            type: 'doughnut',
            data: {
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [{
                    data: [riskCounts.High, riskCounts.Medium, riskCounts.Low],
                    backgroundColor: ['#ef4444', '#eab308', '#22c55e'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#94a3b8' } }
                }
            }
        });

        // --- Difficulty Chart ---
        // Aggregate difficulty by chapter (across all courses for simplicity in this view)
        // Or just show top 10 hardest chapters
        const sortedDifficulty = difficultyStats.sort((a, b) => b.difficulty_score - a.difficulty_score).slice(0, 10);
        
        const ctxDiff = document.getElementById('difficultyChart').getContext('2d');
        if (difficultyChartInstance) difficultyChartInstance.destroy();

        difficultyChartInstance = new Chart(ctxDiff, {
            type: 'bar',
            data: {
                labels: sortedDifficulty.map(d => `${d.course_id} - Ch${d.chapter_id}`),
                datasets: [{
                    label: 'Difficulty Score',
                    data: sortedDifficulty.map(d => d.difficulty_score),
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        // --- High Risk Table ---
        const tableBody = document.querySelector('#results-table tbody');
        tableBody.innerHTML = '';
        
        const highRiskStudents = predictions.filter(p => p.risk_level === 'High').slice(0, 10); // Show top 10
        
        highRiskStudents.forEach(student => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${student.student_id}</td>
                <td>${(student.completion_probability * 100).toFixed(1)}%</td>
                <td class="danger">High</td>
                <td>${student.predicted_completion ? 'Complete' : 'Dropout'}</td>
            `;
            tableBody.appendChild(row);
        });
    }
});
