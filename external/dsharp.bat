@echo off
REM DSHARP WSL wrapper — calls DSHARP in WSL Ubuntu
REM Usage: dsharp.bat <cnf_file_path> [options]
REM The CNF file path will be auto-converted from Windows to WSL format

set "input=%1"
REM Convert Windows path to WSL path
set "wsl_input=%input:\=/%"
set "wsl_input=%wsl_input:C:=/mnt/c%"
set "wsl_input=%wsl_input:D:=/mnt/d%"
set "wsl_input=%wsl_input:E:=/mnt/e%"

REM Shift to get the remaining arguments
shift
set "rest_args="
:parse
if "%~1"=="" goto done
set "rest_args=%rest_args% %~1"
shift
goto parse
:done

wsl ~/solvers/dsharp/dsharp %wsl_input% %rest_args%
