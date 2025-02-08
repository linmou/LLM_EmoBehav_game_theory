from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, abort
from database import get_db
import json
from functools import wraps
import io
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Load scenarios from JSON file
with open('group_chat/prisionDelimma_data_samples.json', 'r') as f:
    scenarios = json.load(f)

# Add admin configuration
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@example.com')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    logger.debug(f"Login attempt for email: {email}")
    
    db = get_db()
    try:
        user = db.table('users').select('*').eq('email', email).execute()
        logger.debug(f"Found user: {user.data}")
        
        if not user.data:
            logger.debug("Creating new user")
            user = db.table('users').insert({'email': email}).execute()
            user_id = user.data[0]['id']
        else:
            user_id = user.data[0]['id']
        
        session['user_id'] = user_id
        return redirect(url_for('annotate'))
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return "Database error", 500

@app.route('/annotate')
@login_required
def annotate():
    return render_template('annotate.html')

@app.route('/api/get-next-scenario')
@login_required
def get_next_scenario():
    user_id = session['user_id']
    db = get_db()
    
    try:
        # Get already annotated scenarios
        annotated = db.table('annotations')\
            .select('scenario_id')\
            .eq('user_id', user_id)\
            .execute()
        
        logger.debug(f"Found annotated scenarios: {annotated.data}")
        annotated_ids = set(row['scenario_id'] for row in annotated.data)  # Using set for faster lookups
        
        # Find first unannotated scenario
        available_scenarios = []
        for i, scenario in enumerate(scenarios):
            if i not in annotated_ids:
                available_scenarios.append({
                    'scenario_id': i,
                    'scenario': scenario
                })
        
        if available_scenarios:
            next_scenario = available_scenarios[0]  # Get the first available scenario
            logger.debug(f"Returning scenario {next_scenario['scenario_id']}")
            return jsonify({
                'scenario_id': next_scenario['scenario_id'],
                'scenario': next_scenario['scenario'],
                'total': len(scenarios),
                'completed': len(annotated_ids),
                'remaining': len(available_scenarios)
            })
        
        logger.debug("All scenarios have been annotated")
        return jsonify({
            'completed': True,
            'total': len(scenarios),
            'completed_count': len(annotated_ids)
        })
        
    except Exception as e:
        logger.error(f"Error in get_next_scenario: {str(e)}")
        return jsonify({'error': 'Failed to get next scenario'}), 500

@app.route('/api/submit-annotation', methods=['POST'])
@login_required
def submit_annotation():
    data = request.json
    user_id = session['user_id']
    db = get_db()
    
    try:
        # Validate required fields
        if 'scenario_id' not in data or 'is_valid' not in data:
            logger.error("Missing required fields in annotation submission")
            return jsonify({'error': 'Missing required fields'}), 400

        logger.debug(f"Submitting annotation: user_id={user_id}, data={data}")
        
        # Check if annotation already exists
        existing = db.table('annotations')\
            .select('id')\
            .eq('user_id', user_id)\
            .eq('scenario_id', data['scenario_id'])\
            .execute()
        
        current_time = datetime.utcnow().isoformat()
        
        if existing.data:
            logger.debug(f"Updating existing annotation id={existing.data[0]['id']}")
            result = db.table('annotations')\
                .update({
                    'is_valid': data['is_valid'],
                    'comments': data.get('comments', ''),
                    'updated_at': current_time
                })\
                .eq('id', existing.data[0]['id'])\
                .execute()
        else:
            logger.debug("Creating new annotation")
            result = db.table('annotations')\
                .insert({
                    'user_id': user_id,
                    'scenario_id': data['scenario_id'],
                    'is_valid': data['is_valid'],
                    'comments': data.get('comments', ''),
                    'created_at': current_time,
                    'updated_at': current_time
                })\
                .execute()
        
        logger.debug(f"Database operation result: {result}")
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error in submit_annotation: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to submit annotation',
            'details': str(e)
        }), 500

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        db = get_db()
        user = db.table('users').select('*').eq('id', session['user_id']).execute()
        
        if not user.data or user.data[0]['email'] != ADMIN_EMAIL:
            abort(403)  # Forbidden
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@admin_required
def admin():
    db = get_db()
    
    # Get statistics
    total_scenarios = len(scenarios)
    
    annotators = db.table('users').select('count', count='exact').execute()
    total_annotators = annotators.count
    
    annotations = db.table('annotations').select('count', count='exact').execute()
    total_annotations = annotations.count
    
    return render_template('admin.html',
                         total_scenarios=total_scenarios,
                         total_annotators=total_annotators,
                         total_annotations=total_annotations)

@app.route('/api/download-results')
@admin_required
def download_results():
    db = get_db()
    
    # Get all annotations with user info
    query = '''
    annotations(
        id,
        scenario_id,
        is_valid,
        comments,
        created_at,
        users(email)
    )
    '''
    
    annotations = db.table('annotations')\
        .select(query)\
        .execute()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Email',
        'Scenario ID',
        'Scenario Name',
        'Is Valid',
        'Comments',
        'Created At'
    ])
    
    for row in annotations.data:
        scenario_name = scenarios[row['scenario_id']]['scenario'] if row['scenario_id'] < len(scenarios) else 'Unknown'
        writer.writerow([
            row['users']['email'],
            row['scenario_id'],
            scenario_name,
            row['is_valid'],
            row['comments'],
            row['created_at']
        ])
    
    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'annotations_{timestamp}.csv'
    )

# After loading scenarios
logger.debug(f"Loaded {len(scenarios)} scenarios from JSON file")

if __name__ == '__main__':
    app.run(debug=True) 