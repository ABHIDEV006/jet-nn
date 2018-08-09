g++ $1.cc `fastjet-config --cxxflags --libs` `pythia8-config --cppflags --libs` -o bin/$1
