@echo off
rem Syntax-check the inkling2 engine: inkling2.cpp (MSVC host) + inkling2.cu (nvcc device).
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "F:\tools\tqllm-inkling"
if not exist build-inkling mkdir build-inkling

echo === compiling inkling2.cpp (cl) ===
cl /nologo /std:c++20 /EHsc /c /Zc:preprocessor /D_CRT_SECURE_NO_WARNINGS /D_WINDOWS /D_USE_MATH_DEFINES /I include /I src/core /I src/engine /I src/formats/weight /I src/formats/kv /I src/kv /I src/experts /I src/spec /I src/models/inkling2 /I third_party /I "%CUDA%\include" /Fo:build-inkling\ik2_inkling2.obj src\models\inkling2\inkling2.cpp 2>&1
if %errorlevel% neq 0 (echo CL-FAIL & exit /b 1)
echo CL-OK

echo === compiling inkling2.cu (nvcc) ===
"%CUDA%\bin\nvcc.exe" -std=c++20 -Xcompiler /EHsc,/Zc:preprocessor -D_WINDOWS -D_USE_MATH_DEFINES -I include -I src/core -I src/engine -I src/formats/weight -I src/formats/kv -I src/kv -I src/experts -I src/spec -I src/models/inkling2 -I third_party -I "%CUDA%\include" -c src\models\inkling2\inkling2.cu -o build-inkling\ik2_inkling2_cu.obj 2>&1
if %errorlevel% neq 0 (echo NVCC-FAIL & exit /b 1)
echo NVCC-OK
echo ALL-OK
