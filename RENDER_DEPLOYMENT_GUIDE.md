# NASA Air Quality App - Render Deployment Guide

## 🚀 Deploy to Render (Free Tier)

### Prerequisites
1. GitHub account
2. Render account (sign up at https://render.com)
3. Your code pushed to a GitHub repository

### Step 1: Prepare Your Repository

Make sure these files are in your repository:
- `nasa_server.py` (main application)
- `requirements-deploy.txt` (lightweight dependencies)
- `render.yaml` (Render configuration)
- `Procfile` (deployment command)
- `static/` folder (frontend files)
- All other necessary Python files

### Step 2: Create Render Service

1. **Connect GitHub Repository**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select your repository

2. **Configure Service Settings**
   - **Name**: `nasa-air-quality-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `python nasa_server.py`

3. **Set Environment Variables**
   Go to "Environment" tab and add:
   ```
   PORT=10000
   TELEGRAM_BOT_TOKEN=8348450996:AAGI1sl6XjqXyHv_qdXPBDbbS4SvSv7FwL4
   DEFAULT_CHAT_ID=1538644238
   APP_ENV=production
   PYTHON_VERSION=3.11.0
   ```

### Step 3: Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies from `requirements-deploy.txt`
   - Start your application with `python nasa_server.py`

### Step 4: Access Your App

Once deployed, you'll get a URL like:
`https://nasa-air-quality-app.onrender.com`

Your app will have these endpoints:
- `/` - Main air quality interface
- `/health` - Health check endpoint
- `/aqi/auto` - Auto-detect air quality
- `/ml/predict/location/{location}` - ML predictions
- All other existing API endpoints

### Step 5: Configure Telegram (Optional)

If you want to use your own Telegram bot:
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Update the `TELEGRAM_BOT_TOKEN` environment variable in Render
4. Update `DEFAULT_CHAT_ID` with your chat ID

### Free Tier Limitations

Render's free tier includes:
- ✅ 512MB RAM
- ✅ 0.1 CPU cores
- ✅ Apps sleep after 15 minutes of inactivity
- ✅ 750 build hours per month
- ⚠️  Cold starts (15-30 seconds to wake up)

### Troubleshooting

1. **Build Failures**: Check the build logs in Render dashboard
2. **App Not Starting**: Verify the start command and environment variables
3. **Memory Issues**: The lightweight requirements should fit in 512MB
4. **Health Check Fails**: Ensure `/health` endpoint returns HTTP 200

### Features Available in Deployment

✅ Air Quality Monitoring by location/coordinates
✅ Auto-detection using IP geolocation
✅ ML-Enhanced Predictions with dominant pollutant detection
✅ Feature Analysis and Accuracy Reports
✅ Modern minimalistic UI with high contrast text
✅ Telegram Alert System
✅ NASA Space Apps Integration
✅ Real-time AQI calculations

### Local Testing Before Deployment

Test locally with production-like settings:
```bash
# Set environment variables
export PORT=8080
export APP_ENV=production

# Install deployment dependencies
pip install -r requirements-deploy.txt

# Run server
python nasa_server.py
```

Access at: http://localhost:8080

### Support

If you encounter issues:
1. Check Render service logs
2. Verify all environment variables are set
3. Ensure GitHub repository is public or properly connected
4. Check that all required files are committed to the repository