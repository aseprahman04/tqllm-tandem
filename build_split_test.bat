@echo off
rem Compile + run the CPU split test (no GPU needed).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "F:\tools\tqllm-inkling"
if not exist build-inkling mkdir build-inkling
cl /nologo /std:c++20 /EHsc /O2 /D_CRT_SECURE_NO_WARNINGS /D_USE_MATH_DEFINES /Fe:build-inkling\inkling2_split_test.exe tests\inkling2_split_test.cpp 2>&1
if %errorlevel% neq 0 (echo CL-FAIL & exit /b 1)
echo === running ===
build-inkling\inkling2_split_test.exe
