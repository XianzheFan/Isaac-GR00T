#!/bin/bash
# Setup Vulkan rendering for SAPIEN/SimplerEnv on headless GPU containers

# 1. Start Xvfb virtual display (skip if already running)
if [ -f /tmp/.X99-lock ]; then
    echo "Xvfb :99 already running, skipping."
else
    Xvfb :99 -screen 0 1024x768x24 &
    echo "Xvfb :99 started."
fi

# 2. Create NVIDIA Vulkan ICD config
cat > /tmp/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.0"
    }
}
EOF
echo "NVIDIA Vulkan ICD created at /tmp/nvidia_icd.json"

# 3. Set environment variables
export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
export DISPLAY=:99
export SAPIEN_DISABLE_RAY_TRACING=1

echo "Done. VK_ICD_FILENAMES=$VK_ICD_FILENAMES, DISPLAY=$DISPLAY"
