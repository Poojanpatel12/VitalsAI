"""
Database initialization script for VitalsAI
Run this to create/update database tables.
"""
from app import app, db

with app.app_context():
    db.create_all()
    print("✓ Database tables created successfully")
    print("  - users table")
    print("  - predictions table")
