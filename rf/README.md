RandomForest - binary classifier

* random-forest in C++ using OpenMP
  * https://github.com/bjoern-andres/random-forest
  * This header file contains a fast C++ implementation of Random Forests as described in: Leo Breiman. Random Forests. Machine Learning 45(1):5-32, 2001.
  * This software was developed by: Bjoern Andres, Steffen Kirchhoff, Evgeny Levinkov
  * Code in this directory is modified from the original version.
  * To build
      mkdir ../chiuw2016/rf/random-forest/build
      cd ../chiuw2016/rf/random-forest/build
      cmake ..
      make
  * To run the code
      ../chiuw2016/rf/random-forest/run_cpp.sh

* chpl port of the bjoern-andres implementation
  * code compiles under 0.12, more thank likely does not run correctly
  * domain use is the compile issue under 0.13 - 'warning here be dragons'

