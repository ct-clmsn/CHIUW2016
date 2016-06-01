// vectorized summation for a sequence of arrays - results in an array
// in numpy, this is equivalent to summing a list of arrays 
//
use Math;
use BlockDist;
use SysBasic;
use Sys;
use Random;

class Model {

   var Dparams = {1..1};
   var params : [Dparams] real;
   var mb_sz:int; // minibath size

   proc getParameters() { assert(false, "getParameters not implemented!"); params(..) = max(int(64)); return params; }

   proc getNumParameters() { assert(false, "getNumParameters not implemented!"); return max(int(64)); }

   proc getLearningRate() { return 1.0; }

}

class LogisticRegressionModel : Model {

   var alpha:real = 1.0;

   proc LogisticRegressionModel(Dx:domain, mbsz:int=1, alph:real=1.0) {
      Dparams = {Dx.low(2)..Dx.high(2)};
      mb_sz = mbsz;
      alpha = alph;
      fillRandom(params);
   }

   inline proc dot(x:[?Dx]real, y:[?Dy] real) {
      assert(Dx.rank == 1 && Dx.rank == Dy.rank /*&& Dx.high == Dy.high*/, "domains not a match");
      return + reduce (x * y);
   }

   inline proc sigmoid(z) { return 1.0 / (1.0 + exp(-z)); }

   proc getParameters() { return params; }   
   proc getNumParameters() { return params.domain.high; }

   proc predict(x) {
      return sigmoid(this.dot(params, x));
   }

   inline proc predict(p, x) {
      return sigmoid(this.dot(p, x));
   }


   proc update(ref w:[?Dw] real, x:[?Dx] real, y:[?Dy] real, alpha:real) {
      var tmpxdom = {1..Dx.high};
      var tmpx : [tmpxdom] real;
      var p = predict(w, tmpx);
      x = tmpx;
      w -= alpha * (p - y) * x; 
   }
    
   proc train(X:[?Dx] real, y:[?Dy] real, alpha:real, ref theta:[?Dw] real) {
      var (n, m) = (X.domain.high(1), X.domain.high(2));

      for i in 1..n { 
         update(theta, X(i,..), y, alpha);
      }

      var torettheta => theta;
      return torettheta;
   }

   proc getLearningRate() { return alpha; }

}

proc train(mb_sz:int, lmbda:real, X:[?Dx] real, y:[?Dy] real) {
   const yy = y; //if Dy.high(1) == 1 then y(1,..) else y(..,1);
   const numLabels = (max reduce yy):int;

   var all_theta : [1..numLabels, 1..Dx.high(2)] real;

   var m = new LogisticRegressionModel(X.domain, mb_sz, lmbda);

   for i in 1..numLabels {
      const Y = (yy==i:real):real;
      const thetatmp = m.train(X, Y, lmbda, all_theta(i,..));
      all_theta(i,..) =  thetatmp;
   }

   delete m;

   return all_theta;
}

proc predict(const X:[?Dx] real, const thetas:[?Dthetas] real) {
   assert(Dx.rank == 1 && Dthetas.rank == 2);

   inline proc sigmoid(z) { return 1.0 / (1.0 + exp(-z)); }

   inline proc dot(x:[?Dx]real, y:[?Dy] real) {
      assert(Dx.rank == 1 && Dx.rank == Dy.rank /*&& Dx.high == Dy.high*/, "domains not a match");
      return + reduce (x * y);
   }

   const estimates : [1..Dthetas.high(1)] real = [ i in 1..Dthetas.high(1) ] sigmoid(dot(X, thetas(i,..)));
   const (maxVal, maxValNum) = maxloc reduce zip(estimates, estimates.domain);

   return maxValNum;
}

config const trainingdata : string = "mnist_data_training.mtx";
config const traininglbls : string = "mnist_lbl_training.mtx";
config const tstdata : string = "mnist_sm_data.mtx";
config const tstlbls : string = "mnist_sm_lbl.mtx";

proc main() {

    use MatrixMarket;
    use Time;

    const X = mmread(trainingdata); //, mmap_sz=(1024*75));
    const y = mmread(traininglbls); //, mmap_sz=(1024*75));
    const mb_sz : int = X.domain.high(1);

var t:Timer;
t.start();
    const weights = train(mb_sz, 0.3, X, y); //X(idx,1..), y(idx,1..));
t.stop();
writeln(t.elapsed());

//    const tsty = mmread(tstlbls); //"mnist_sm_lbl.mtx");
//    const tstx = mmread(tstdata); //"mnist_sm_data.mtx");
/*
    for i in 1..tstx.domain.high(1) {
       const esty = predict(tstx(i,..), weights);
       writeln(("predict", (if tsty.domain.high(1) == 1 then tsty(1,i) else tsty(i,1), esty)));
    }
*/

}

