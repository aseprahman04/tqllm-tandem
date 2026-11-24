@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "F:\tools\tqllm-inkling"
if not exist build-inkling mkdir build-inkling
cl /nologo /std:c++20 /EHsc /O2 /I src/formats/weight /I src/core /I third_party/ggml_mmq/ggml /D_CRT_SECURE_NO_WARNINGS ^
  /Fo:build-inkling\ /Fe:build-inkling\nvfp4_gguf40.exe ^
  tests\nvfp4_gguf40.cpp src\formats\weight\dequant.cpp 2>&1
if %errorlevel% neq 0 (echo COMPILE-FAIL & exit /b 1)
echo COMPILE-OK
