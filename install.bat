@echo off
echo ================================
echo PyTorch CUDA 12.8 Setup Script
echo ================================

echo.
echo Uninstalling old versions...
pip uninstall torch torchaudio -y

echo.
echo Installing requirements (no cache)...
pip install --no-cache-dir -r requirements.txt

echo.
echo Verifying installation...
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo Done!
pause