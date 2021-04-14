
output.o: timing.o rrgen.o myfft.o
	g++ -o output.o timing.o rrgen.o myfft.o

timing.o: timing.cpp rrgenV3.h
	g++ -c timing.cpp  -lm -o timing.o

rrgen.o: rrgenV3.c rrgenV3.h 
	gcc -c rrgenV3.c -lm -o rrgen.o

fft.o: myfft.cpp myfft.h
	g++ -c myfft.cpp =lm fft.o
clean:
	rm *.o output

run:
	./output.o

# output.o: main.o rrgen.o myfft.o lombmethods.o
# 	g++ -o output.o main.o rrgen.o myfft.o lombmethods.o

# main.o: main.cpp rrgenV3.h
# 	g++ -c main.cpp  -lm -o main.o

# rrgen.o: rrgenV3.c rrgenV3.h 
# 	gcc -c rrgenV3.c -lm -o rrgen.o

# myfft.o: myfft.cpp myfft.h 
# 	g++ -c myfft.cpp -o myfft.o

# lombmethods.o: lombmethods.cpp lombmethods.h 
# 	g++ -c lombmethods.cpp -o lombmethods.o

# clean:
# 	rm *.o output
# run:
# 	./output.o