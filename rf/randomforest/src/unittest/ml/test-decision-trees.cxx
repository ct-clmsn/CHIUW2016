#include <stdexcept>
#include <string>
#include <vector>
#include <random>

#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

#include "andres/marray.hxx"
#include "andres/ml/decision-trees.hxx"

template<typename T>
static inline void loadArr(string& f, andres::Marray<T>& data, int nos) {
  string eol;
  ifstream fs;
  fs.open(f);
getline(fs, eol);
getline(fs, eol);
getline(fs, eol);
  int i, j;
  double k;
  int c = 0;
  while(c < nos && !fs.eof()) {
    fs >> i; fs >> j; fs >> k;
    data(i,j) = k;
    c++;
  }
 
  fs.close();
} 

template<typename T>
static inline void loadVec(string& f, andres::Marray<T>& data, int nos) {
  string eol;
  ifstream fs;
  fs.open(f);
getline(fs, eol);
getline(fs, eol);
getline(fs, eol);
  int i = 0;
  int c = 0;
  double k;
  while(c < nos && !fs.eof()) {
    fs >> k;
    data(i) = k;
    i++;
    c++;
  }

  fs.close();
} 


inline void test(const bool& x) { 
    if(!x) throw std::logic_error("test failed."); 
}

int main(int argc, char** argv) {

/*
double FEATURES [2][100] = { { 0.131538, 0.218959, 0.934693, 0.0345721, 0.00769819, 0.686773, 0.526929, 0.701191, 0.0474645, 0.75641, 0.98255, 0.0726859, 0.436411, 0.274907, 0.897656, 0.504523, 0.493977, 0.0737491, 0.913817, 0.050084, 0.125365, 0.629543, 0.888572, 0.513274, 0.841511, 0.467917, 0.571655, 0.49848, 0.890737, 0.212752, 0.274588, 0.70982, 0.31754, 0.681346, 0.147533, 0.955409, 0.408767, 0.488515, 0.199757, 0.651254, 0.476432, 0.901673, 0.410313, 0.162199, 0.135109, 0.4523, 0.215248, 0.86086, 0.817561, 0.632739, 0.702207, 0.289316, 0.414028, 0.729748, 0.706535, 0.524987, 0.488943, 0.916634, 0.139195, 0.446023, 0.439726, 0.211519, 0.61635, 0.727335, 0.680562, 0.828708, 0.629572, 0.388823, 0.269215, 0.783865, 0.0113162, 0.819726, 0.17688, 0.257169, 0.634717, 0.75294, 0.598217, 0.117437, 0.587989, 0.161688, 0.556836, 0.528548, 0.430848, 0.446866, 0.0782632, 0.676237, 0.944753, 0.460434, 0.60564, 0.343818, 0.0149506, 0.917848, 0.843975, 0.0220837, 0.436638, 0.695679, 0.00425392, 0.903301, 0.00634169, 0.544023}, {0.45865, 0.678865, 0.519416, 0.5297, 0.0668422, 0.930436, 0.653919, 0.762198, 0.328234, 0.365339, 0.753356, 0.884707, 0.477732, 0.166507, 0.0605643, 0.319033, 0.0907329, 0.384142, 0.464446, 0.770205, 0.688455, 0.725412, 0.306322, 0.845982, 0.415395, 0.178328, 0.0330538, 0.748293, 0.84204, 0.130427, 0.414293, 0.239911, 0.652059, 0.387725, 0.845576, 0.148152, 0.564899, 0.961095, 0.629269, 0.803073, 0.20325, 0.142021, 0.885648, 0.365339, 0.455307, 0.931674, 0.908922, 0.505956, 0.462245, 0.824697, 0.954415, 0.514435, 0.876566, 0.715642, 0.0190924, 0.0651939, 0.682049, 0.890019, 0.989362, 0.514659, 0.80665, 0.153604, 0.000878999, 0.417724, 0.83642, 0.0817376, 0.213547, 0.947545, 0.284035, 0.282156, 0.983236, 0.398144, 0.157731, 0.101637, 0.79477, 0.63343, 0.318778, 0.526123, 0.702989, 0.860226, 0.316007, 0.588119, 0.370226, 0.387831, 0.253922, 0.728608, 0.940163, 0.661355, 0.150394, 0.522808, 0.642979, 0.970087, 0.455752, 0.70695, 0.75171, 0.753863, 0.0799169, 0.746679, 0.518478, 0.215826 } };

//static int LABELS [100] = { 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0 };
*/

    typedef double Feature;
    typedef unsigned char Label;

    const size_t numberOfSamples = 50000; // 100;
    const size_t numberOfFeatures = 784; //2;
    const size_t shape[] = {numberOfSamples, numberOfFeatures};

    string datf(argv[1]), datl(argv[2]);
    andres::Marray<Feature> features(shape, shape+2);
    andres::Marray<Label> labels(shape, shape+1);

    loadArr(datf, features, numberOfSamples);
    loadVec(datl, labels, numberOfSamples);

    /*
    // define random feature matrix
    std::default_random_engine RandomNumberGenerator;
    typedef double Feature;
    std::uniform_real_distribution<double> randomDistribution(0.0, 1.0);
    const size_t shape[] = {numberOfSamples, numberOfFeatures};
    andres::Marray<Feature> features(shape, shape + 2);
    for(size_t sample = 0; sample < numberOfSamples; ++sample)
    for(size_t feature = 0; feature < numberOfFeatures; ++feature) {
        //features(sample, feature) = randomDistribution(RandomNumberGenerator);
//std::cout << sample << ' ' << feature << std::endl;
        //features(sample, feature) = FEATURES[sample][feature];
        features(sample, feature) = FEATURES[feature][sample];
    }

    // define labels
    typedef unsigned char Label;
    andres::Marray<Label> labels(shape, shape + 1);
    for(size_t sample = 0; sample < numberOfSamples; ++sample) {
        if((features(sample, 0) <= 0.5 && features(sample, 1) <= 0.5)
        || (features(sample, 0) > 0.5 && features(sample, 1) > 0.5)) {
            labels(sample) = 0;
        }
        else {
            labels(sample) = 1;
        }
        //labels(sample) = (char)LABELS[sample];
    }
*/

    // learn decision forest
    typedef double Probability;
    andres::ml::DecisionForest<Feature, Label, Probability> decisionForest;
    const size_t numberOfDecisionTrees = 10;
//std::cout << "features" << features.asString() << std::endl;
//std::cout << "labels" << labels.asString() << std::endl;

std::clock_t start;
double duration;
start = std::clock();

    decisionForest.learn(features, labels, numberOfDecisionTrees);

duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
std::cout<<"time in seconds: "<< duration <<'\n';

/*
    // predict probabilities for every label and every training sample
    andres::Marray<Probability> probabilities(shape, shape + 2);
    decisionForest.predict(features, probabilities);
    // TODO: test formally

    std::stringstream sstream;
    decisionForest.serialize(sstream);

    andres::ml::DecisionForest<Feature, Label, Probability> decisionForest_2;
    decisionForest_2.deserialize(sstream);

    andres::Marray<Probability> probabilities_2(shape, shape + 2);
    decisionForest.predict(features, probabilities_2);

    size_t cnt = 0;
    for (size_t i = 0; i < numberOfSamples; ++i) {
        std::cout << i << '\t' << probabilities(i) << '|' << probabilities_2(i) << std::endl;
        if (fabs(probabilities(i) - probabilities_2(i)) < std::numeric_limits<double>::epsilon())
            ++cnt;
    }

    if (cnt != numberOfSamples)
        throw std::runtime_error("two predictions coincide");
*/
    return 0;
}
