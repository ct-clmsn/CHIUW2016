use distlda;

config var Wf:string;
config var wf:string;
config var Df:string;
config var df:string;
config var Nf:string;
config var words_fn:string;
config var T:int;
config var P:int;

proc main() {

  var data = new Data(Wf, wf, Df, df, Nf, words_fn, T);
  const ITER = 500;
  const beta  = 0.01;
  const alpha = 0.1;
  distlda(data, ITER, beta, alpha, P);

}

