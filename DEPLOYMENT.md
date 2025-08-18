# Mediscope AI - Deployment Guide

## Overview

This guide provides instructions for deploying Mediscope AI to various platforms and troubleshooting common deployment issues.

## Pre-deployment Checklist

### 1. Environment Variables

Make sure to set the following environment variables in your deployment platform:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
HF_HUB_DISABLE_TELEMETRY=1
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=-1
TF_CPP_MIN_LOG_LEVEL=2
```

### 2. Model Files

Ensure the following files are included in your deployment:

- `models/pneumonia_model.h5` - The trained CNN model
- `faiss_index/` - The vector database files

### 3. Python Version

The application requires Python 3.12.0. A `runtime.txt` file is included for platforms that support it.

## Deployment Platforms

### Streamlit Cloud

1. Connect your GitHub repository
2. Set the environment variables in the Streamlit Cloud dashboard
3. Deploy with the following settings:
   - Python version: 3.12
   - Main file: `streamlit_app.py`

### Heroku

1. Create a `Procfile` with: `web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
2. Set environment variables in Heroku dashboard
3. Deploy using Heroku CLI or GitHub integration

### Railway

1. Connect your GitHub repository
2. Set environment variables in Railway dashboard
3. Railway will automatically detect the Python version

## Common Issues and Solutions

### 1. HTTP 429 Errors (Rate Limiting)

**Problem**: HuggingFace API rate limiting during model download
**Solution**: The application now includes:

- Multiple fallback models
- Retry logic with exponential backoff
- Environment variables to reduce telemetry

### 2. CUDA/GPU Issues

**Problem**: TensorFlow trying to use GPU in deployment environment
**Solution**: The application now:

- Forces CPU-only usage
- Disables CUDA devices
- Uses tensorflow-cpu package

### 3. Model Loading Failures

**Problem**: CNN model not loading properly
**Solution**: Enhanced error handling with:

- Model file existence checks
- Test predictions after loading
- Graceful fallbacks

### 4. Memory Issues

**Problem**: Out of memory errors during deployment
**Solution**:

- Use CPU-only TensorFlow
- Reduced model sizes
- Memory-efficient configurations

## Troubleshooting

### Check Logs

Look for these indicators in deployment logs:

- ✅ Success messages for model loading
- ⚠️ Warning messages for fallbacks
- ❌ Error messages for failures

### Common Error Messages

1. **"Model not properly loaded"**

   - Check if `models/pneumonia_model.h5` exists
   - Verify file permissions
   - Check TensorFlow version compatibility

2. **"HTTP Error 429"**

   - Normal during initial deployment
   - Application will retry automatically
   - Should resolve after a few minutes

3. **"CUDA error"**
   - Expected in CPU-only environments
   - Application will fall back to CPU
   - Not a critical error

## Performance Optimization

### For Production Deployment

1. **Pre-load models**: Models are loaded once at startup
2. **Caching**: Streamlit caching is enabled for better performance
3. **Memory management**: CPU-only configuration reduces memory usage
4. **Error handling**: Graceful fallbacks prevent crashes

### Monitoring

- Monitor memory usage
- Check response times
- Watch for error rates
- Monitor HuggingFace API usage

## Support

If you encounter issues not covered in this guide:

1. Check the application logs
2. Verify environment variables are set correctly
3. Ensure all required files are included in deployment
4. Test with a smaller model first

## Environment Variables Reference

| Variable                   | Purpose                       | Required |
| -------------------------- | ----------------------------- | -------- |
| `GEMINI_API_KEY`           | Google Gemini API for chatbot | Yes      |
| `HF_HUB_DISABLE_TELEMETRY` | Disable HuggingFace telemetry | No       |
| `TOKENIZERS_PARALLELISM`   | Disable tokenizer parallelism | No       |
| `CUDA_VISIBLE_DEVICES`     | Force CPU-only mode           | No       |
| `TF_CPP_MIN_LOG_LEVEL`     | Reduce TensorFlow logging     | No       |
