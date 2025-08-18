# Deployment Troubleshooting Guide

## TensorFlow Library Loading Issues

### Problem

```
tensorflow.python.framework.errors_impl.NotFoundError: This app has encountered an error.
File "/mount/src/mediscope-ai/src/cnn_predictor.py", line 1, in <module>
    import tensorflow as tf
```

### Solution

The application has been updated to handle TensorFlow loading issues gracefully:

1. **Environment Configuration**: The app now configures TensorFlow environment variables before import
2. **Graceful Fallback**: If TensorFlow fails to load, the app runs in limited mode
3. **Error Handling**: All TensorFlow operations are wrapped in try-catch blocks

### Key Changes Made

#### 1. Updated `requirements.txt`

- Pinned TensorFlow version to `2.15.0` for better compatibility
- Updated related packages for deployment stability
- Added deployment-specific dependencies

#### 2. Enhanced `src/cnn_predictor.py`

- Added environment configuration before TensorFlow import
- Implemented graceful fallback when TensorFlow is unavailable
- Added `is_model_available()` method to check functionality

#### 3. Updated `streamlit_app.py`

- Added TensorFlow availability checks
- Implemented limited mode when TensorFlow is not available
- Added user-friendly error messages

#### 4. Enhanced `deployment_config.py`

- Added more robust environment configuration
- Implemented TensorFlow availability checking
- Added Python path configuration

## Testing Your Deployment

### Run the Test Script

```bash
python test_tensorflow_loading.py
```

This will test:

- TensorFlow import capability
- CNN predictor initialization
- Streamlit app imports
- Deployment configuration

### Expected Results

#### ✅ Full Functionality

If all tests pass, the app will have:

- Pneumonia detection with AI model
- Medical assistant chatbot
- Full feature set

#### ⚠️ Limited Mode

If TensorFlow fails but other components work:

- Medical assistant chatbot (still functional)
- Pneumonia detection (disabled)
- Clear user notifications about limitations

## Deployment Best Practices

### 1. Environment Variables

Ensure these are set in your deployment environment:

```bash
CUDA_VISIBLE_DEVICES=-1
TF_CPP_MIN_LOG_LEVEL=2
TF_ENABLE_DEPRECATION_WARNINGS=0
```

### 2. Package Versions

Use the exact versions specified in `requirements.txt`:

- `tensorflow-cpu==2.15.0`
- `tf-keras==2.15.0`
- `h5py==3.10.0`

### 3. Model Files

Ensure the model file is available:

- `models/pneumonia_model.h5`
- Or create a fallback model using `create_fallback_model.py`

### 4. Memory Considerations

- TensorFlow CPU-only mode uses less memory
- Consider using lighter models for deployment
- Monitor memory usage during deployment

## Common Issues and Solutions

### Issue: TensorFlow Import Fails

**Solution**: The app will automatically run in limited mode. Users can still use the chatbot.

### Issue: Model Loading Fails

**Solution**: Check if the model file exists and is accessible. Create a fallback model if needed.

### Issue: Memory Issues

**Solution**: Use CPU-only TensorFlow and consider reducing model complexity.

### Issue: Package Conflicts

**Solution**: Use the exact versions in `requirements.txt` and ensure clean environment.

## Monitoring and Logs

### Check Application Status

The app provides clear status indicators:

- ✅ Full functionality available
- ⚠️ Limited mode (chatbot only)
- ❌ Critical errors

### Log Messages

Look for these key messages:

- `✅ TensorFlow imported successfully`
- `⚠️ TensorFlow is not available. Model loading will be skipped.`
- `💡 The app will run in limited mode - only chatbot functionality will be available.`

## Support

If you continue to experience issues:

1. Run the test script to identify specific problems
2. Check the deployment logs for detailed error messages
3. Verify all dependencies are correctly installed
4. Ensure the deployment environment supports the required packages
