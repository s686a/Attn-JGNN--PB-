@echo off
REM PBCount WSL wrapper — calls PBCount in WSL Ubuntu

set "input=%1"
shift
set "rest_args="
:parse
if "%~1"=="" goto done
set "rest_args=%rest_args% %~1"
shift
goto parse
:done

set "wsl_input=%input:\=/%"
set "wsl_input=%wsl_input:C:=/mnt/c%"
set "wsl_input=%wsl_input:D:=/mnt/d%"
set "wsl_input=%wsl_input:E:=/mnt/e%"

wsl ~/solvers/pbcount/pbcount --cf %wsl_input% %rest_args%
