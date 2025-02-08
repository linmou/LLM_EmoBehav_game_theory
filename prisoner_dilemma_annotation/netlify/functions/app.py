from flask import Flask, jsonify
from flask.helpers import send_from_directory
from netlify_lambda_wsgi import make_wsgi_handler
import os

# Import your existing Flask app
from ....app import app

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Handler for Netlify Functions
handler = make_wsgi_handler(app) 