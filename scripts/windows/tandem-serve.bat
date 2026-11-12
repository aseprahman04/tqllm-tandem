@echo off
rem Tandem (multi-GPU fork of tqLLM): serve one model split across two GPUs.
rem Usage: tandem-serve.bat [model.gguf]
rem Runs the tqserve.exe beside this script (the release zip), else build\tqserve.exe of the repository. Model paths are
rem relative to the current folder. Override any of these with environment variables before calling.
setlocal
set "TQSERVE=%~dp0tqserve.exe"
if not exist "%TQSERVE%" set "TQSERVE=%~dp0..\..\build\tqserve.exe"
if "%TANDEM_MODEL%"=="" set "TANDEM_MODEL=models\Qwen3.8-27B-TQ_L.gguf"
if not "%~1"=="" set "TANDEM_MODEL=%~1"
if "%TANDEM_DRAFTER%"=="" set "TANDEM_DRAFTER=models\drafters\Qwen3.8-27B-DFlash2-TQ3.gguf"
if "%TANDEM_VOCAB%"=="" set "TANDEM_VOCAB=models\mtp-vocab\atx_65536.txt"
rem first layer of the second range: layers 0..K-1 on CUDA device 1, the rest + head + drafter on device 0
if "%TQLLM_SPLIT%"=="" set "TQLLM_SPLIT=44"
if "%TANDEM_CONTEXT%"=="" set "TANDEM_CONTEXT=524288"
if "%TANDEM_PORT%"=="" set "TANDEM_PORT=8086"
if "%CUDA_VISIBLE_DEVICES%"=="" set "CUDA_VISIBLE_DEVICES=0,1"
set "ROPE="
rem past the trained 262,144 tokens: YaRN by the factor the context needs
if %TANDEM_CONTEXT% GTR 262144 set "ROPE=--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144"
if %TANDEM_CONTEXT% GTR 524288 set "ROPE=--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144"
"%TQSERVE%" --model "%TANDEM_MODEL%" -md "%TANDEM_DRAFTER%" --draft-vocab "%TANDEM_VOCAB%" ^
  --host 127.0.0.1 --port %TANDEM_PORT% --context %TANDEM_CONTEXT% --kv tq3 --kv-host 0 --prompt-cache on %ROPE%
