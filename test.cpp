#include "mlp.cpp"
#include <iostream>

using namespace std;

void valueTest(){
  Value* a = new Value(2);
  Value* b = new Value(3);
  Value* d = new Value(4);
  Value* c = *a + *b;
  Value* e = *c + *d;
  Value* f = *c * *e ;
  Value* g = *f / *a;
  Value* h = *g ^ 2;
  
  h->backward();

  cout<<"a: "<<a->toString()<<endl;
  cout<<"b: "<<b->toString()<<endl;
  cout<<"c: "<<c->toString()<<endl;
  cout<<"d: "<<d->toString()<<endl;
  cout<<"e: "<<e->toString()<<endl;
  cout<<"f: "<<f->toString()<<endl;
  cout<<"g: "<<g->toString()<<endl;
  cout<<"h: "<<h->toString()<<endl;

  return;
}

void neuron_test(){
  Neuron n = Neuron(3);
  vector<Value*> in;
  in.push_back(new Value(1));
  in.push_back(new Value(1));
  in.push_back(new Value(2));
  double exp = 10;
  for(int i = 0; i<20; i++){
    Value *res = n(in);
    Value *mse = res->mse(exp);
    mse->backward();
    printf("Value at epoch %d:\n%s\nPrediction: %f\n\n", i+1, n.paramsToString().c_str(), res->getData());
    n.backprop(0.05);
    n.zerograd();
  }
  Value *res = n(in);
  printf("Final values:\n%s\nPrediction: %f\n", n.paramsToString().c_str(), res->getData());
  return;
}

void layer_test(){
  Layer l = Layer(3, 2);
  vector<Value*> in;
  in.push_back(new Value(0));
  in.push_back(new Value(1));
  in.push_back(new Value(2));
  for(int i = 0; i<500; i++){
    vector<Value*> out = l(in);
    cout<<"Epoch: "<<i+1<<endl<<"Prediction: "<<out[0]->getData()<<" "<<out[1]->getData()<<endl;
    Value* mse2 = out[1]->mse(1);
    Value* mse1 = out[0]->mse(1);
    mse2->backward();
    mse1->backward();
    l.backprop(0.1);
    l.zerograd();
  }
  vector<Value*> out = l(in);
  cout<<"Final values: "<<endl;
  cout<<"Prediction: "<<out[0]->getData()<<" "<<out[1]->getData()<<endl;
  cout<<"Layer params: "<<l.paramsToString()<<endl;
  return;
}

void mlp_test(){
  MLP mlp = MLP({3, 2, 2, 1});
  vector<Value*> in;
  in.push_back(new Value(20));
  in.push_back(new Value(-12));
  in.push_back(new Value(3.9));

  cout<<"Initial params: "<<mlp.paramsToString()<<endl;
  for(int i = 0; i<2000; i++){
    vector<Value*> out = mlp(in);
    if((i+1)%25 == 0){
      cout<<"Epoch: "<<i+1<<endl<<"Prediction: "<<out[0]->getData()<<endl;
    }
    Value* mse = out[0]->mse(25);
    mse->backward();
    mlp.backprop(0.01);
    mlp.zerograd();
  }
  vector<Value*> out = mlp(in);
  cout<<"Final values: "<<endl;
  cout<<"Prediction: "<<out[0]->getData()<<endl;
  cout<<"MLP params: "<<mlp.paramsToString()<<endl;
  return;
}

int main(){
  //valueTest();
  //neuron_test();
  //layer_test();
  //mlp_test();
  return 0;
}