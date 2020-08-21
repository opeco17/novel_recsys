import os
import sqlite3
import requests

from flask import request, jsonify
from db_connect import create_connection

from run import app


DB_PATH = app.config['DB_PATH']
DETAILS_TABLE_NAME = app.config['DETAILS_TABLE_NAME']
DB_EXTENSION_PATH = app.config['DB_EXTENSION_PATH']


@app.route('/')
def index():
    return jsonify({
        "message": "Here is index!"
    })


@app.route('/insert_predict_point', methods=['POST'])
def insert_predict_point():
    conn, cur = create_connection(DB_PATH, DB_EXTENSION_PATH)
    conn.commit()
    conn.close()
    return jsonify({
        "message": "Here is prediction!"
    })