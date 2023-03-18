set "PATH=C:/windows;C:/windows/system32;C:/windows/syswow64;%~dp0../../../../libtorch/lib;C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64"

cd DJIA
cmd /c R.bat

cmd /c R-test.bat

PAUSE