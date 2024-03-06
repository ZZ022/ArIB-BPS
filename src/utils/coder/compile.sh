g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` python_interface.cpp -o mixcoder.so
