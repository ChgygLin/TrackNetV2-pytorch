all:
	g++ -o tracknet tracknet.cpp -lncnn -I../../ncnn-20240820-ubuntu-2204-shared/include/ncnn -L../../ncnn-20240820-ubuntu-2204-shared/lib/ `pkg-config --cflags --libs opencv4`

#export LD_LIBRARY_PATH=../../ncnn-20240820-ubuntu-2204-shared/lib/
#./tracknet ../smash_test.mp4