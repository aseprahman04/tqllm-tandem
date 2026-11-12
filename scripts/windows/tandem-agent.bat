@echo off
rem Tandem: start the split server and a coding agent on it, through tqLLM's launcher (tools\launch\tqllm.py).
rem Usage: tandem-agent.bat claude^|codex^|opencode^|pi^|serve
rem   claude: Claude Code over the Anthropic Messages API; codex: Codex over the Responses API; serve: the server only.
rem The agent must be installed and on PATH. Same variables as tandem-serve.bat; Python 3 must be on PATH.
setlocal
if "%~1"=="" (
  echo usage: tandem-agent.bat claude^|codex^|opencode^|pi^|serve
  exit /b 1
)
set "TQSERVE=%~dp0tqserve.exe"
if not exist "%TQSERVE%" set "TQSERVE=%~dp0..\..\build\tqserve.exe"
set "LAUNCH=%~dp0launch\tqllm.py"
if not exist "%LAUNCH%" set "LAUNCH=%~dp0..\..\tools\launch\tqllm.py"
if "%TANDEM_MODEL%"=="" set "TANDEM_MODEL=models\Qwen3.8-27B-TQ_L.gguf"
if "%TANDEM_DRAFTER%"=="" set "TANDEM_DRAFTER=models\drafters\Qwen3.8-27B-DFlash2-TQ3.gguf"
if "%TANDEM_VOCAB%"=="" set "TANDEM_VOCAB=models\mtp-vocab\atx_65536.txt"
if "%TQLLM_SPLIT%"=="" set "TQLLM_SPLIT=44"
if "%TANDEM_CONTEXT%"=="" set "TANDEM_CONTEXT=524288"
if "%TANDEM_PORT%"=="" set "TANDEM_PORT=8086"
if "%CUDA_VISIBLE_DEVICES%"=="" set "CUDA_VISIBLE_DEVICES=0,1"
set "ROPE="
if %TANDEM_CONTEXT% GTR 262144 set "ROPE=--server-arg=--rope-scaling --server-arg=yarn --server-arg=--rope-scale --server-arg=2 --server-arg=--yarn-orig-ctx --server-arg=262144"
if %TANDEM_CONTEXT% GTR 524288 set "ROPE=--server-arg=--rope-scaling --server-arg=yarn --server-arg=--rope-scale --server-arg=4 --server-arg=--yarn-orig-ctx --server-arg=262144"
set "CMD=launch %~1"
if /i "%~1"=="serve" set "CMD=serve"
python "%LAUNCH%" %CMD% --native "%TQSERVE%" --model "%TANDEM_MODEL%" --drafter "%TANDEM_DRAFTER%" ^
  --draft-vocab "%TANDEM_VOCAB%" --context %TANDEM_CONTEXT% --port %TANDEM_PORT% --log "%TEMP%\tandem-launch.log" ^
  --server-arg=--kv --server-arg=tq3 --server-arg=--kv-host --server-arg=0 %ROPE%
