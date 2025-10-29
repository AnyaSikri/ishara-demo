# Deployment Guide: Drug Script Analysis Web App

## 🚀 Quick Start - Deploy to Streamlit Cloud (Free!)

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

**Time:** 5 minutes  
**Cost:** Free  
**URL:** https://streamlit.io/cloud

#### Steps:
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Set main file: `streamlit_app.py`
7. Click "Deploy"

**That's it!** Your app is live at: `https://your-app-name.streamlit.app`

---

## 🖥️ Option 2: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

App will open at: `http://localhost:8501`

---

## 📦 Option 3: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deploy to any cloud:
- **AWS:** Use Elastic Beanstalk or ECS
- **Google Cloud:** Use Cloud Run
- **Azure:** Use Container Instances
- **DigitalOcean:** Use App Platform

---

## 🔐 Option 4: Production Deployment (Advanced)

### With Authentication & Security

```python
# Add to streamlit_app.py
import streamlit_authenticator as stauth

# Add authentication
names = ['username']
usernames = ['user1']
passwords = ['password123']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    'cookie_name', 'signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'sidebar')
```

---

## 🎨 Alternative Platforms

### Option A: Gradio
```bash
pip install gradio
```

Similar to Streamlit but focuses on ML models.

### Option B: Plotly Dash
More customizable but requires more frontend knowledge.

### Option C: FastAPI + React
Full control but much more complex (2-4 weeks development).

---

## 📊 Comparison

| Platform | Setup Time | Cost | Flexibility | Recommended For |
|----------|-----------|------|-------------|-----------------|
| **Streamlit** | 5 min | Free | Medium | ✅ **Start here!** |
| **Gradio** | 5 min | Free | Low | Quick demos |
| **Plotly Dash** | 30 min | Free/Paid | High | Custom dashboards |
| **FastAPI+React** | 2-4 weeks | Varies | Very High | Full web apps |

---

## 🎯 Recommendation

**Start with Streamlit Cloud** - It's free, takes 5 minutes, and you can always migrate later if needed.

