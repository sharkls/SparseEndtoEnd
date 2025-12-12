#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DEPLOY_DIR="${SCRIPT_DIR}/deploy"

echo "Start building plugins..."

# Build dfa_plugin
if [ -d "${DEPLOY_DIR}/dfa_plugin" ]; then
    echo "========================================"
    echo "Building dfa_plugin..."
    echo "========================================"
    cd "${DEPLOY_DIR}/dfa_plugin"
    if [ -f "build.sh" ]; then
        bash build.sh
    else
        echo "Error: build.sh not found in dfa_plugin"
        exit 1
    fi
else
    echo "Error: dfa_plugin directory not found"
    exit 1
fi

# Build ln_plugin
if [ -d "${DEPLOY_DIR}/ln_plugin" ]; then
    echo "========================================"
    echo "Building ln_plugin..."
    echo "========================================"
    cd "${DEPLOY_DIR}/ln_plugin"
    if [ -f "build.sh" ]; then
        bash build.sh
    else
        echo "Error: build.sh not found in ln_plugin"
        exit 1
    fi
else
    echo "Error: ln_plugin directory not found"
    exit 1
fi

# Build sparsebox_plugin
if [ -d "${DEPLOY_DIR}/sparsebox_plugin" ]; then
    echo "========================================"
    echo "Building sparsebox_plugin..."
    echo "========================================"
    cd "${DEPLOY_DIR}/sparsebox_plugin"
    if [ -f "build.sh" ]; then
        bash build.sh
    else
        echo "Error: build.sh not found in sparsebox_plugin"
        exit 1
    fi
else
    echo "Error: sparsebox_plugin directory not found"
    exit 1
fi

echo "========================================"
echo "All plugins built successfully."
echo "========================================"
