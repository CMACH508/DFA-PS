cls
cd DJIA
rem cmd /c  R.bat
start "DJIA" /max R-SSM.bat
cd ..

cd HSI
rem cmd /c  R.bat
start "HSI" /max R-SSM.bat
cd ..

cd CSI
cmd /c R-SSM.bat
cd ..


cd US
start "US" /max R-SSM.bat
cd ..

cd HK
cmd /c  R-SSM.bat
cd ..

cd CN-A
cmd /c R-SSM.bat