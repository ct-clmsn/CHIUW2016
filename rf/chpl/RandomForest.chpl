module RandomForest {

use Sort;
use Random;
use Math;
use List;
use Timer;

inline proc randomInt(const nDom) {
   var pool:[nDom] real;
   sync {
      fillRandom(pool);
   }
   const (val, idx) = maxloc reduce zip(pool, pool.domain);
   return idx;
}

inline iter randomIntPair(const nDom) {
   var pool:[nDom] real;
   fillRandom(pool);

   for i in nDom {
      const (val, idx) = maxloc reduce zip(pool, nDom);
      yield (i, idx);
      pool(idx) = 0.0;
   }
}

proc fisherYatesShuffle(const shuffleDom) { 
   //for i from n − 1 downto 1 do
   //  j ← random integer such that 0 ≤ j ≤ i
   //  exchange a[j] and a[i]
   var shuffleBuf:[shuffleDom] int;

   for (i,j) in randomIntPair(shuffleDom) {
      shuffleBuf(j) = i;
      shuffleBuf(i) = j;
   }

   return shuffleBuf;
}

proc sample(n, r) {
   //Generate r randomly chosen, sorted integers from [0,n)
   var rgen = new RandomStream();
   var pop = n;
   const rdom = {0..r};
   for samp in rdom by -1 {
        var cumprob = 1.0;
        const x = rgen.getNext();
        while x < cumprob {
            cumprob -= cumprob * samp:real / pop:real;
            pop -= 1;
        }
        //yield n-pop-1;
        break;
   }
   return n-pop;
}

record DecisionNode {
   type FEATURE;
   type LABEL;

   var featureIndex:int;
   var threshold:FEATURE;
   var cnDom = {0..1};
   var childNodeIndices:[cnDom] int; // 0 <, 1 >=
   var Label:LABEL;
   var isleaf:bool;
   
   proc DecisionNode(type FEATURE, type LABEL) {
      isleaf = false;
      childNodeIndices(0) = 0;
      childNodeIndices(1) = 0;
   }

   proc isLeaf() :bool {
     return isleaf;  
   }

   proc childNodeIndex(j:int):int {
      return childNodeIndices(j);
   }

   proc assocToArr(d:[?T] int) {
      var toret : [0..d.size] int;
      for i in d.domain {
         toret(i) = d[i];
      }
      return toret;
   }

   proc learn(features:[]FEATURE, labels:[]LABEL, sampleIndices:[?T]int, startIdx:int, endIdx:int) {
      {
         var isLabelUnique = true;
         var firstLabel = labels(sampleIndices(startIdx));
         for j in startIdx..endIdx-1 {
            if labels(sampleIndices(j)) != firstLabel {
               isLabelUnique = false;
               break;
            }
         }

         if isLabelUnique {
            isleaf = true;
            Label = labels(sampleIndices(startIdx));
            return -1;
         }    
      }

      const numFeatures = features.domain.high(2)+1;
      const numFeaturesToAssess = ceil((numFeatures:real ** 0.5)):int;
      var featureIndices : [0..numFeaturesToAssess-1] int;
      var buffer : [0..numFeatures-1] int;
      sampleSubsetWithoutReplacement(numFeatures, 
           numFeaturesToAssess,
           featureIndices,
           buffer
      );

      var noldom : domain(int);
      type bag = [noldom] int;
      var numberOfLabels : [0..1] bag;

      var optsumgini = INFINITY;
      var optfeatureidx :int;
      var optthresholdidx:int;
      var optthreshold:FEATURE;

      for fij in 0..numFeaturesToAssess-1 {
         const fi = if featureIndices(fij) == 0 then featureIndices(fij)+1 else featureIndices(fij);

         QuickSort(sampleIndices(startIdx..endIdx));

         var numberOfElements:[0..1] int;
         numberOfElements(0) = 0; numberOfElements(1) = (endIdx-startIdx);

         for k in startIdx..endIdx-1 {
            const lbl = labels(sampleIndices(k));
            if !numberOfLabels(1).domain.member(lbl) {
               numberOfLabels(0).domain += lbl;
               numberOfLabels(1).domain += lbl;
            }
            numberOfLabels(1)(lbl)+=1;
         }
        
         var thresholdIdx = startIdx;
         while true {
            const thresholdOld = thresholdIdx;
            while thresholdIdx+1 < endIdx && features(sampleIndices(thresholdIdx), fi) == features(sampleIndices(thresholdIdx+1), fi) {
               const lbl = labels(sampleIndices(thresholdIdx));
               numberOfElements(0)+=1;
               numberOfElements(1)-=1;
               if !numberOfLabels(0).domain.member(lbl) {
                  numberOfLabels(0).domain += lbl;
               }
               numberOfLabels(0)(lbl)+=1;

               if !numberOfLabels(1).domain.member(lbl) {
                  numberOfLabels(1).domain += lbl;
               }
               numberOfLabels(1)(lbl)-=1;
               thresholdIdx+=1;
            }

            {
               const labl = labels(sampleIndices(thresholdIdx));
               numberOfElements(0)+=1;
               numberOfElements(1)-=1;
               if !numberOfLabels(0).domain.member(labl) {
                  numberOfLabels(0).domain += labl;
               }
               numberOfLabels(0)(labl)+=1;

               if !numberOfLabels(1).domain.member(labl) {
                  numberOfLabels(1).domain += labl;
               }
               numberOfLabels(1)(labl)-=1;
            }

            thresholdIdx+=1;
            if thresholdIdx == endIdx { 
               break;
            }

            var numbersOfDistinctPairs : [0..1] int;
            for s in 0..1 {
               /*const kcpy = numberOfLabels(s).domain;
               var mcpy = numberOfLabels(s).domain;
               for k in kcpy {
                  mcpy-=k;
                  for m in mcpy {
                     numbersOfDistinctPairs(s) += (numberOfLabels(s)(k) * numberOfLabels(s)(m));
                  }
               }*/
               const arr = assocToArr(numberOfLabels(s));
               for k in 0..arr.size-1 {
                  for m in k+1..arr.size-1 {
                     numbersOfDistinctPairs(s) += (arr(k) * arr(m));
                  }
               }
            }

            var ginicoefs : [0..1] real;
            for s in 0..1 {
               ginicoefs(s) = if numberOfElements(s) < 2 then 0.0 else (numbersOfDistinctPairs(s):real / (numberOfElements(s) * (numberOfElements(s) - 1):real));
            }

            const ginisum = ginicoefs(0) + ginicoefs(1);
            if ginisum < optsumgini {
               optsumgini = ginisum;
               optfeatureidx = fi;
               optthreshold = features(sampleIndices(thresholdIdx), fi);
               optthresholdidx = thresholdIdx;
            }
         }

         for s in 0..1 {
            for k in numberOfLabels[s].domain {
               numberOfLabels[s][k] = 0;
            }
         }
      }
 
      threshold = optthreshold;
      featureIndex = optfeatureidx;

      QuickSort(sampleIndices(startIdx..endIdx));

      return optthresholdidx;
   }

   proc sampleSubsetWithoutReplacement(sz:int, subsetsz:int, indices:[]int, candidates:[]int) {
      for j in 0..sz-1 {
         candidates(j) = j;
      }
      sync {
         for j in 0..subsetsz-1 {
            const dom = {0..(sz-1) - j};
            const idx = randomInt(dom);
            indices(j) = candidates(idx);
            candidates(idx) = candidates(sz-j-1);
         } 
      }
   }
}

record TreeConstructionQueueEntry {
   var nodeIdx:int;
   var sampleIdxStart:int;
   var sampleIdxEnd:int;
   var thresholdIdx:int;

   proc TreeConstructionQueueEntry(const ni = 1, const sis = 1, const sie = 1, const ti = 1) {
      nodeIdx = ni;
      sampleIdxStart = sis;
      sampleIdxEnd = sie;
      thresholdIdx = ti;
   }
}

record DecisionTree {
   type FEATURE;
   type LABEL;

   var nodesDom = {0..0};
   var decisionNodes:[nodesDom] DecisionNode(FEATURE, LABEL);

   proc DecisionTree(type FEATURE, type LABEL, const n_nodes=2) {
      nodesDom = {0..n_nodes-1};
   }

   proc queueAt(q:list(?T), i:int) {
      for (j, v) in zip(0..q.length-1, q) {
         if j == i { return v; } 
      }
      var t : T;
      return t;
   }

   proc learn(features:[]FEATURE, lbls:[]LABEL, sampleIndices:[] int) {
      var queue : list(TreeConstructionQueueEntry);
      {
         var root = new DecisionNode(FEATURE, LABEL);
         const thresholdIdx = root.learn(features, lbls, sampleIndices, sampleIndices.domain.low, sampleIndices.domain.high);
         decisionNodes.insert(decisionNodes.domain.high, root);
         if !decisionNodes(0).isLeaf() {
            var root_tcq = new TreeConstructionQueueEntry(0, 1, sampleIndices.domain.high, thresholdIdx);
            queue.append(root_tcq);
         }
      }

      while queue.length > 0 {
         const qval = queueAt(queue, 0);

         const ni = qval.nodeIdx;
         const sib = qval.sampleIdxStart;
         const sie = qval.sampleIdxEnd;
         const ti = qval.thresholdIdx;

         queue.remove(qval);

         var nodeIdxNew:int = decisionNodes.size;
         var thresholdIdxNew:int;

         var ndn = new DecisionNode(FEATURE, LABEL);
         thresholdIdxNew = ndn.learn(features, lbls, sampleIndices, sib, ti);
         decisionNodes.push_back(ndn);

         decisionNodes(ni).childNodeIndices(0) = nodeIdxNew;
         if !decisionNodes(nodeIdxNew).isLeaf() { 
            var tcq = new TreeConstructionQueueEntry(nodeIdxNew, sib, ti, thresholdIdxNew);
            queue.append(tcq);
         } 

         nodeIdxNew = decisionNodes.size;
         ndn = new DecisionNode(FEATURE, LABEL);
         thresholdIdxNew = ndn.learn(features, lbls, sampleIndices, ti, sie);
         decisionNodes.push_back(ndn);

         decisionNodes(ni).childNodeIndices(1) = nodeIdxNew;
         if !decisionNodes(nodeIdxNew).isLeaf() { 
            var tcq = new TreeConstructionQueueEntry(nodeIdxNew, ti, sie, thresholdIdxNew);
            queue.append(tcq);
         }
      }
   }

   proc predict(const ref features:[]FEATURE, ref lbls:[] LABEL) {
      const nos = features.domain.high(1);
      const nof = features.domain.high(2);
      if lbls.domain.high != nos {
         var newlbldom = {0..nos};
         reshape(lbls, newlbldom);
      }

      for j in 0..nos {
         var ni = 0;
         while true {
            const decisionNode = decisionNodes(ni);
            if decisionNode.isLeaf() {
               lbls(j) = decisionNode.Label;
               break;
            }
            else {
               const fi = decisionNode.featureIndex;
               const threshold = decisionNode.threshold;
               ni = decisionNode.childNodeIndex(if features(j, fi) < threshold then 0 else 1);
            }
         }
      }
   }
}

record DecisionForest {
   type FEATURE;
   type LABEL;
   type PROBABILITY;
   
   var dtdom = {0..0};
   var decisionTrees : [dtdom] DecisionTree(FEATURE, LABEL); 

   proc DecisionForest(type FEATURE, type LABEL, type PROBABILITY) {
   }

   proc clear() {
      decisionTrees.clear();
   }

   proc size() {
      return dtdom.size;
   }

   proc learn(features:[]FEATURE, lbls:[] LABEL, const sz=255) {
      const numberOfSamples = features.domain.high(1);
      const numberOfDecisionTrees = sz;
      clear();
      dtdom = {0..numberOfDecisionTrees-1}; 
      const samplesDom = {1..numberOfSamples};
      forall treeIdx in dtdom {
         var sampleIndices : [samplesDom] int;
         sampleBootstrap(numberOfSamples, sampleIndices); // uses atomic in this method
         decisionTrees(treeIdx).learn(features, lbls, sampleIndices);
      }
   }

   proc predict(features:[]FEATURE, lblprobs:[]PROBABILITY) {
      assert(lblprobs.rank == 2);
      const numberOfSamples = features.domain.high(1);
      const numberOfFeatures = features.domain.high(2);
      const nsamplesdom = {0..numberOfSamples};

      forall treeIdx in decisionTrees.domain {
         var lbls:[nsamplesdom] LABEL;
         const dtree = decisionTrees(treeIdx);
         dtree.predict(features, lbls);
         for sampleIndex in nsamplesdom {
            const lbl = lbls[sampleIndex];
            sync {
               lblprobs(sampleIndex, lbl)+=1;
            }
        }
      }

      forall j in 0..lblprobs.domain.high(1) {
         lblprobs(j,..) /= decisionTrees.domain.size:PROBABILITY;
      }
   }

   proc sampleBootstrap(sz:int, idx:[]int) {
      idx.domain = {idx.domain.low..idx.domain.low+sz};
      // critical section
      sync {
         for j in idx.domain {
            idx[j] = randomInt(idx.domain);
         }
      }
   }
}

/*
proc main() {
   const nos = 100-1;
   const nof = 2-1;

   var features : [0..nos, 0..nof] real;
   for i in 0..features.domain.high(1) {
      fillRandom(features(i,..));
   }

   var labels : [0..nos] int;
   for sample in 0..nos {
      if((features(sample, 0) <= 0.5 && features(sample, 1) <= 0.5)
        || (features(sample, 0) > 0.5 && features(sample, 1) > 0.5)) {
         labels(sample) = 0;
      }
      else {
         labels(sample) = 1;
      }
   }

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

   writeln(probs);
   
}
*/

} // end module
