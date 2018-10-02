frozen=NetTake[trained,{1,7}];
in=frozen/@Keys[testSet1];
testin=frozen/@Keys[testSet2];
out=IdentityMatrix[10][[#+1]]&/@Values@testSet1;

g={LinearLayer[128,"Input"->800],LogisticSigmoid}//NetChain[#]&//
NetInitialize[#,Method->{
"Random",
"Weights"->NormalDistribution[0,.00001],
"Biases"->NormalDistribution[0,.0001] 
}]&;

ginv=PseudoInverse[g/@in];
exNet=(g/@#).ginv.out & ;

exNet@testin;
Values@testSet2;
Transpose@{%,First@First@Position[#,Max@#]-1&/@%%};
SameQ@@@%//Tally//AssociationThread@@Transpose@#&
%@True/(%@False+%@True)//N
