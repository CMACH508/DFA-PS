clear
cd DJIA
export CUDA_VISIBLE_DEVICES=0
./R-SSM.sh &
cd ..

cd HSI
export CUDA_VISIBLE_DEVICES=1
./R-SSM.sh &
cd ..

cd CSI
export CUDA_VISIBLE_DEVICES=2
./R-SSM.sh
cd ..

cd US
export CUDA_VISIBLE_DEVICES=3
./R-SSM.sh &
cd ..

cd HK
export CUDA_VISIBLE_DEVICES=0
./R-SSM.sh &
cd ..

cd CN-A
export CUDA_VISIBLE_DEVICES=1
./R-SSM.sh
