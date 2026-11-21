extern "C" __global__ void __launch_bounds__(128) setmax_kernel(float* a) {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
  a[threadIdx.x] += 1.0f;
}
extern "C" __global__ void __launch_bounds__(128, 16) hog_kernel(const float* a, float* o) {
  float v[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) v[i] = a[threadIdx.x * 128 + i];
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int i = 0; i < 128; ++i) v[i] = v[i] * v[(i + 1) % 128] + v[(i + 7) % 128];
  }
#pragma unroll
  for (int i = 0; i < 128; ++i) o[threadIdx.x * 128 + i] = v[i];
}
