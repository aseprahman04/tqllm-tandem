@echo off
rem Tensor parallel (TQLLM_TP=1), TQ_XS, no drafter, 8K context. %1 = model file (default TQ_XS).
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set CUDA_VISIBLE_DEVICES=0,1
set TQLLM_TP=1
set "M=F:\models\qwen3.8-27b-tq\Qwen3.8-27B-TQ_XS.gguf"
if not "%~1"=="" set "M=%~1"
"%~dp0build\tqserve.exe" --model "%M%" --host 127.0.0.1 --port 11435 --context 8192 --kv tq3 --kv-host 0 > "%~dp0tq-tp.log" 2>&1
