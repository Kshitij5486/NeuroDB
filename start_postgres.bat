@echo off
"C:\Program Files\PostgreSQL\16\bin\pg_ctl.exe" start -D "C:\Program Files\PostgreSQL\16\data" -l "C:\Program Files\PostgreSQL\16\data\log\manual.log"
echo PostgreSQL started.
pause