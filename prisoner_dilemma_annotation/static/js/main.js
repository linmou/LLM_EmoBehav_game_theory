let currentScenario = null;

async function loadNextScenario() {
    const response = await fetch('/api/get-next-scenario');
    const data = await response.json();
    
    if (data.completed) {
        document.querySelector('.container').innerHTML = '<h2>All scenarios have been annotated!</h2>';
        return;
    }
    
    currentScenario = data;
    
    document.getElementById('progress').textContent = 
        `Progress: ${data.completed} / ${data.total}`;
    
    document.getElementById('scenario-title').textContent = data.scenario.scenario;
    document.getElementById('scenario-description').textContent = data.scenario.description;
    
    // Reset form
    document.querySelectorAll('input[name="validity"]').forEach(radio => radio.checked = false);
    document.getElementById('comments').value = '';
}

async function submitAnnotation() {
    const validity = document.querySelector('input[name="validity"]:checked');
    if (!validity) {
        alert('Please select whether the scenario is valid or not.');
        return;
    }
    
    const annotation = {
        scenario_id: currentScenario.scenario_id,
        is_valid: validity.value === 'true',
        comments: document.getElementById('comments').value
    };
    
    await fetch('/api/submit-annotation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(annotation)
    });
    
    loadNextScenario();
}

// Load first scenario when page loads
document.addEventListener('DOMContentLoaded', loadNextScenario); 