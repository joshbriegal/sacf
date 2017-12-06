# Generalised Autocorrelation Function 
(credit Larz Kreutzer, c++ implementation by Josh Briegal jtb34@cam.ac.uk)

## Installation
Using CMAKE (https://cmake.org) is the easiest way to build the code. Code should be built into 'build' directory Example for UNIX based system:

1) Navigate to GACF/build
2) run "cmake .."
3) run "make"

Running "./GACF" will run the function main() from main.cpp

## File Structure
<pre>
.
├── CMakeLists.txt
├── build
├── include
│   ├── Correlator.h
│   └── DataStructure.h
└── src
    ├── Correlator.cpp
    ├── Correlator.h
    ├── DataStructure.cpp
    └── main.cpp
 </pre>
