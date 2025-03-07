# Setup Guide for Stock Analysis Dashboard

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Development Setup](#development-setup)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Operating System: Windows 10+, macOS 10.15+, or Linux
- RAM: Minimum 4GB (8GB recommended)
- Storage: 1GB free space
- Internet connection

### Required Software
1. **Python**
   - Version 3.8 or higher
   - Download from [python.org](https://www.python.org/downloads/)

2. **Git**
   - Latest version
   - Download from [git-scm.com](https://git-scm.com/downloads)

3. **Code Editor** (recommended)
   - VSCode, PyCharm, or similar
   - Download VSCode from [code.visualstudio.com](https://code.visualstudio.com/)

## Installation Steps

### 1. Clone the Repository
```bash
# Create a directory for your projects
mkdir projects
cd projects

# Clone the repository
git clone https://github.com/yourusername/stock-analysis-dashboard.git
cd stock-analysis-dashboard
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the root directory:
```bash
# .env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
DEBUG_MODE=False
```

### 2. Application Configuration
Copy and modify the configuration file:
```bash
cp config.example.py config.py
```

Edit `config.py` with your settings:
```python
# config.py
CONFIG = {
    'default_ticker': 'AAPL',
    'default_period': '1y',
    'cache_duration': 3600,
    'max_forecast_days': 30
}
```

## Running the Application

### 1. Development Mode
```bash
# Basic run
streamlit run His.py

# Run with specific port
streamlit run His.py --server.port 8501

# Run in debug mode
streamlit run His.py --debug
```

### 2. Production Mode
```bash
# Set production environment
export STREAMLIT_ENV=production

# Run with production settings
streamlit run His.py --server.address 0.0.0.0 --server.port 8501
```

### 3. Accessing the Application
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

## Development Setup

### 1. IDE Configuration
#### VSCode Settings
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### 2. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 3. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up git hooks
pre-commit install
```

### 4. Testing Environment
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Issues
```bash
# If pip install fails, try:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

#### 2. Port Already in Use
```bash
# Find process using port
# Windows
netstat -ano | findstr :8501
# Linux/macOS
lsof -i :8501

# Kill process
# Windows
taskkill /PID <PID> /F
# Linux/macOS
kill -9 <PID>
```

#### 3. Virtual Environment Issues
```bash
# If venv activation fails, recreate it:
rm -rf venv
python -m venv venv
```

### Verification Steps

#### 1. Check Installation
```bash
# Verify Python version
python --version

# Verify pip packages
pip list

# Check streamlit installation
streamlit --version
```

#### 2. Test Data Access
```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
print(ticker.info)
```

## Additional Resources

### Documentation Links
- [Streamlit Documentation](https://docs.streamlit.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Plotly Documentation](https://plotly.com/python/)

### Community Support
- GitHub Issues: [Project Issues Page](https://github.com/yourusername/stock-analysis-dashboard/issues)
- Discussion Forum: [Project Discussions](https://github.com/yourusername/stock-analysis-dashboard/discussions)

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Clear cache
streamlit cache clear
```

## Security Considerations

### 1. API Keys and Secrets
- Store sensitive data in `.env` file
- Never commit `.env` to version control
- Use environment variables for production

### 2. Data Protection
```python
# Example of secure configuration
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
```

### 3. Access Control
- Implement authentication if needed
- Use HTTPS in production
- Regular security updates

## Maintenance

### Regular Updates
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Generate new requirements
pip freeze > requirements.txt
```

### Backup Procedures
```bash
# Backup configuration
cp config.py config.backup.py

# Backup data (if applicable)
cp -r data/ data_backup/
```

Remember to:
- Keep dependencies updated
- Monitor application logs
- Perform regular backups
- Test updates in development before deploying to production
- Document any configuration changes
- Maintain security patches

This setup guide provides comprehensive instructions for installing, configuring, and maintaining the Stock Analysis Dashboard. For additional support, please refer to the main documentation or create an issue on GitHub.
