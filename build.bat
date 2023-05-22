@echo off

set src="kedos/src/kedos.c"

gcc %src% -o "build/kedos.exe"

pause