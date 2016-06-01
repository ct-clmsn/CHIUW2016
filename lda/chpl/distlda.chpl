module distlda {

use Random;
use Math;
use MatrixMarket;
use IO;
use Time;

extern proc drand48():real;

inline proc gibbs(ref z:[?Dz] real, ref wp:[?Dwp] real, ref dp:[?Ddp] real, ref ztot:[?Dztot] real, ref w:[?Dw] real, ref d:[?Dd] real, mn:real=0.0, mx:real=1.0) {
   const N = Dz.high;
   const W = Dwp.high(1);
   const T = Dwp.high(2);
   const D = Ddp.high(1);
   const alpha = 0.1;
   const beta = 0.01;
   var maxprob:real;

   var probs : [1..T] real;
   const wbeta = (W:real) * beta;
   w -= 1.0;
   d -= 1.0;
   //z -= 1.0;
   
   const Nrng = 1..N;
   //const randvals = [ j in Nrng ] drand48();

   for i in Nrng {
      const wi = floor(w(i)):int+1;
      const di = floor(d(i)):int+1;

      var t = floor(z(i)):int;
      ztot(t)-=1.0;
      wp(wi, t)-=1.0;
      dp(di, t)-=1.0;

      probs = (wp(wi, 1..) + beta) * (dp(di, 1..) + alpha) / (ztot + wbeta);

      const totprob = (+ reduce probs);

      var currprob = probs(1);
     const scale = drand48(); 
      maxprob = totprob * scale;
      t = 1; 
      while maxprob > currprob {
         currprob += probs(t);
         t+= if maxprob > currprob then 1 else 0;
      }
      z(i) = t;
      ztot(t)+=1;
      wp(wi, t) += 1.0;
      dp(di, t) += 1.0;
   }

   w += 1.0;
   d += 1.0;

   return (z, wp, dp);
}

proc loadW(W_fn) {
  var toret = mmread(W_fn);
  return toret(1,1):int;
}

proc loadD(D_fn) {
  var toret = mmread(D_fn);
  return toret(1,1):int;
}

proc loadd(D_fn) {
  var toret = mmread(D_fn);
  var ret = toret(1..,1);
  return ret;
}

proc loadw(w_fn) {
  var toret = mmread(w_fn);
  var ret = toret(1..,1);
  return ret:int;
}

proc loadN(N_fn) {
  var toret = mmread(N_fn);
  return toret(1,1):int;
}

proc loadwords(ref wordset, words_fn) {
  var idx = 0;

  var fd = open(words_fn, iomode.r);

  for l in fd.lines() {
    wordset+=idx;
    wordset(idx) = l;
    idx+=1;
  } 

  fd.close();
}

/*
   DistLDA/docword.kos.train.mat.N.mtx -> int
   DistLDA/docword.kos.train.mat.word -> plain text list
   DistLDA/docword.kos.train.mat.w.mtx -> matrix
   DistLDA/docword.kos.train.mat.W.mtx -> int
   DistLDA/docword.kos.train.mat.d.mtx -> matrix
   DistLDA/docword.kos.train.mat.D.mtx -> int
*/

record Data {

   var W:int;

   var wDom = {1..1};
   var w : [wDom] int;

   var D:int;

   var dDom = {1..1};
   var d : [dDom] real;

   var wdom : domain(int);
   var words : [wdom] string;

   var T, N:int;

   proc Data(W_fn:string, w_fn:string, D_fn:string, d_fn:string, N_fn:string, words_fn:string, T:int) {
      this.W = loadW(W_fn);
      const wvals = loadw(w_fn);
      this.wDom = wvals.domain;
      this.w = wvals;
      this.D = loadD(D_fn);
      const dvals = loadd(d_fn);
      this.dDom = dvals.domain;
      this.d = dvals;
      this.N = loadN(N_fn);
      loadwords(this.words, words_fn);
      this.T = T;
   }

}

// reference top voted anwer: http://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
proc getRandom(ref x:[]real, const mn:real, const mx:real) {
   fillRandom(x);
   x = [ i in x.domain ] floor(x(i) * (mx - mn + 1.0)) + mn;
}

proc distlda(data:Data, ITER:int, beta:real, alpha:real, const P=1) {
   const W = data.W;
   const w = data.w;
   const D = data.D;
   const d = data.d;
   const N = data.N;
   const T = data.T;

   const zDom = {1..data.N};
   var zz : [zDom] real;
   getRandom(zz, 1.0, T:real);
   var z : [zDom] int = zz:int;

   var wp : [1..W, 1..T] real = 0.0;
   var dp : [1..D, 1..T] real = 0.0;

   const dataDom = {1..data.N};
   for n in dataDom {
      wp(w(n):int, z(n)) += 1.0; //wp(w(n):int, z(n)) + 1.0;
      dp(d(n):int, z(n)) += 1.0; //dp(d(n):int, z(n)) + 1.0;
   }

   var ztot = [i in wp.domain.low(1)..wp.domain.high(1)] (+ reduce wp(i,..));
   var ztotchk = [ i in dp.domain.low(1)..dp.domain.high(1)] (+ reduce dp(i,..));

   var Dp = D/P;
   var Nstart = 1;
   
   var pwdzDom = {1..Dp};
   var pw, pd, pz : [1..P] [pwdzDom] real;
   var pdp : [1..P] [1..D, 1..T] real;

   for p in 1..P {
      const Dstart = ((p-1)*Dp+1):int;
      const Dend = (p*Dp):int;
      const Nend = (+ reduce dp(Dstart..Dend, 1..) ):int;
      pwdzDom = {1..((Nend-Nstart)+1)};

      pw(p) = w(Nstart..Nend);
      pd(p) = d(Nstart..Nend) - Dstart; //(Dstart-1);
      pz(p) = z(Nstart..Nend);
      pdp(p) = dp(Dstart..Dend, ..);

      Nstart = Nend+1;
      pd(p)+=1.0;
   }

var t:Timer;
t.start();

   // iterate through distributed sampling
   //
   for itr in 1..ITER {
      var wp0 = wp; // have to do a 'put' operation here
      var ztot0 = [ i in wp0.domain.low(2)..wp0.domain.high(2) ] (+ reduce wp0(..,i));
      var dwp : [wp.domain] real;

      for p in 1..P {
         (pz(p), wp, pdp(p)) = gibbs(pz(p), wp0, pdp(p), ztot0, pw(p), pd(p)); //, 1.0, T:real);
         dwp = dwp + (wp - wp0);
      } // end
  
      wp = wp0 + dwp;

   } // end iter loop

t.stop();
writeln(t.elapsed());


}

} // end module end

