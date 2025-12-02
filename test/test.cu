#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main(){

  cudaDeviceProp device_properties;

  cudaGetDeviceProperties(&device_properties, 0);
  cout<<"Device name: "<<device_properties.name<<endl;
  cout<<"Device compute capability: "<<device_properties.major<<"."<<device_properties.minor<<endl;
  cout<<"Device total memory: "<<device_properties.totalGlobalMem<<endl;
  cout<<"Device shared memory per block: "<<device_properties.sharedMemPerBlock<<endl;
  cout<<"Device total shared memory per multiprocessor: "<<device_properties.sharedMemPerMultiprocessor<<endl;
  cout<<"Device registers per block: "<<device_properties.regsPerBlock<<endl;
  cout<<"Device max threads per warp: "<<device_properties.warpSize<<endl;
  cout<<"Device max threads per block: "<<device_properties.maxThreadsPerBlock<<endl;
  cout<<"Device max threads per multiprocessor: "<<device_properties.maxThreadsDim[0]<<"x"<<device_properties.maxThreadsDim[1]<<"x"<<device_properties.maxThreadsDim[2]<<endl;
  cout<<"Device max grid size: "<<device_properties.maxGridSize[0]<<"x"<<device_properties.maxGridSize[1]<<"x"<<device_properties.maxGridSize[2]<<endl;
  cout<<"Device clock rate: "<<device_properties.clockRate<<endl;
  cout<<"Device texture alignment: "<<device_properties.textureAlignment<<endl;
  cout<<"Device total constant memory: "<<device_properties.totalConstMem<<endl;
  return 0;
}