@echo off
REM PBMC WSL wrapper — calls PBMC in WSL Ubuntu

set "input=%1"
set "wsl_input=%input:\=/%"
set "wsl_input=%wsl_input:C:=/mnt/c%"
set "wsl_input=%wsl_input:D:=/mnt/d%"
set "wsl_input=%wsl_input:E:=/mnt/e%"

wsl ~/solvers/pbmc/bin/pbmc %wsl_input%
