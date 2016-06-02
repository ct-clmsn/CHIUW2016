use RandomForest;
use MatrixMarket;
use Time;

config const featurefile : string;
config const labelfile : string;

proc main() {
   
   const features = mmread(featurefile);
   const labels = mmread(labelfile):int;

   var decisionForest = new DecisionForest(real, int, real);
   const numberOfDecisionTrees = 10;
//writeln("training started");

var t:Timer;
t.start();
   decisionForest.learn(features, labels, numberOfDecisionTrees);
t.stop();
writeln(t.elapsed());

//writeln("training ended");
   // predict probabilities for every label and every training sample
   //var probs : [0..nos, 0..nof] real;
   //decisionForest.predict(features, probs);

   //writeln(probs);
}

