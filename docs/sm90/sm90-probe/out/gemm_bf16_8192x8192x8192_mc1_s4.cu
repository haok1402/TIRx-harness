
#ifdef __CUDACC_RTC__
  #include <cuda/std/cstdint>
  using cuda::std::uint8_t;
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  using cuda::std::int8_t;
  using cuda::std::int16_t;
  using cuda::std::int32_t;
  using cuda::std::int64_t;

  #include <cuda/std/type_traits>
  namespace std {
    using cuda::std::is_same;
    using cuda::std::is_same_v;
    using cuda::std::is_integral;
    using cuda::std::is_signed;
    using cuda::std::is_unsigned;
    using cuda::std::is_floating_point;
    using cuda::std::enable_if;
    using cuda::std::conditional;
  }

  // NVRTC uses asm/volatile instead of __asm__/__volatile__ (gcc extension).
  #ifndef __asm__
  #define __asm__ asm
  #endif
  #ifndef __volatile__
  #define __volatile__ volatile
  #endif
#else
  #include <cstdint>
  #include <type_traits>
  #include <cuda.h>
#endif

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
#include <cuda_bf16.h>
__device__ nv_bfloat16 max(nv_bfloat16 a, nv_bfloat16 b)
{
  return __hgt(a, b) ? a : b;
}
__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)
{
  return __hlt(a, b) ? a : b;
}
#endif // __CUDA_ARCH__ >= 800
// Pack two bfloat16 values.
static inline __device__ __host__ unsigned
__pack_nv_bfloat162(const nv_bfloat16 x, const nv_bfloat16 y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some bfp16 math functions are not supported in cuda_bfp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ nv_bfloat16 HALF_MATH_NAME(nv_bfloat16 x, nv_bfloat16 y) {   \
  float tmp_x = __bfloat162float(x);                                      \
  float tmp_y = __bfloat162float(y);                                      \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2bfloat16(result);                                        \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ nv_bfloat16 HALF_MATH_NAME(nv_bfloat16 x) {          \
  float tmp_x = __bfloat162float(x);                                     \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2bfloat16(result);                                       \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

template <typename T, typename TVec2>
struct __align__(8) half4_bfloat164 {
  T x, y, z, w;
  __host__ __device__ half4_bfloat164() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
  __host__ __device__ half4_bfloat164(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

};

using nv_bfloat164 = half4_bfloat164<nv_bfloat16, nv_bfloat162>;
__host__ __device__ nv_bfloat164 make_nv_bfloat164(nv_bfloat16 x, nv_bfloat16 y, nv_bfloat16 z, nv_bfloat16 w) {
    return nv_bfloat164(x, y, z, w);
}
__host__ __device__ nv_bfloat162 make_nv_bfloat162(nv_bfloat16 x, nv_bfloat16 y) {
    return nv_bfloat162(x, y);
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
#endif

#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union GmmaDescriptor
{
  HOST_DEVICE constexpr
  GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor const& t) noexcept : desc_(t.desc_) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor && t) noexcept : desc_(t.desc_) {}

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor && t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};
__forceinline__ __device__ void tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(uint32_t& __d, float __a, float __b) {
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(__d) : "f"(__a), "f"(__b));
}
__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_arrive_nocount_arrive_shared__cluster_b64(uint32_t __addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :  : "r"(__addr) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_wait_group_wait_group_sync_aligned_1() {
  asm volatile("wgmma.wait_group.sync.aligned 1;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_bf16_ss_mma_async_sync_aligned_m64n256k16_f32_bf16_bf16_1_1_0_0(float& __d0, float& __d1, float& __d2, float& __d3, float& __d4, float& __d5, float& __d6, float& __d7, float& __d8, float& __d9, float& __d10, float& __d11, float& __d12, float& __d13, float& __d14, float& __d15, float& __d16, float& __d17, float& __d18, float& __d19, float& __d20, float& __d21, float& __d22, float& __d23, float& __d24, float& __d25, float& __d26, float& __d27, float& __d28, float& __d29, float& __d30, float& __d31, float& __d32, float& __d33, float& __d34, float& __d35, float& __d36, float& __d37, float& __d38, float& __d39, float& __d40, float& __d41, float& __d42, float& __d43, float& __d44, float& __d45, float& __d46, float& __d47, float& __d48, float& __d49, float& __d50, float& __d51, float& __d52, float& __d53, float& __d54, float& __d55, float& __d56, float& __d57, float& __d58, float& __d59, float& __d60, float& __d61, float& __d62, float& __d63, float& __d64, float& __d65, float& __d66, float& __d67, float& __d68, float& __d69, float& __d70, float& __d71, float& __d72, float& __d73, float& __d74, float& __d75, float& __d76, float& __d77, float& __d78, float& __d79, float& __d80, float& __d81, float& __d82, float& __d83, float& __d84, float& __d85, float& __d86, float& __d87, float& __d88, float& __d89, float& __d90, float& __d91, float& __d92, float& __d93, float& __d94, float& __d95, float& __d96, float& __d97, float& __d98, float& __d99, float& __d100, float& __d101, float& __d102, float& __d103, float& __d104, float& __d105, float& __d106, float& __d107, float& __d108, float& __d109, float& __d110, float& __d111, float& __d112, float& __d113, float& __d114, float& __d115, float& __d116, float& __d117, float& __d118, float& __d119, float& __d120, float& __d121, float& __d122, float& __d123, float& __d124, float& __d125, float& __d126, float& __d127, uint64_t __a_desc, uint64_t __b_desc, uint32_t __scale_d) {
  asm volatile("{ .reg .pred ps0; setp.ne.b32 ps0, %130, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, ps0, 1, 1, 0, 0; }" : "+f"(__d0), "+f"(__d1), "+f"(__d2), "+f"(__d3), "+f"(__d4), "+f"(__d5), "+f"(__d6), "+f"(__d7), "+f"(__d8), "+f"(__d9), "+f"(__d10), "+f"(__d11), "+f"(__d12), "+f"(__d13), "+f"(__d14), "+f"(__d15), "+f"(__d16), "+f"(__d17), "+f"(__d18), "+f"(__d19), "+f"(__d20), "+f"(__d21), "+f"(__d22), "+f"(__d23), "+f"(__d24), "+f"(__d25), "+f"(__d26), "+f"(__d27), "+f"(__d28), "+f"(__d29), "+f"(__d30), "+f"(__d31), "+f"(__d32), "+f"(__d33), "+f"(__d34), "+f"(__d35), "+f"(__d36), "+f"(__d37), "+f"(__d38), "+f"(__d39), "+f"(__d40), "+f"(__d41), "+f"(__d42), "+f"(__d43), "+f"(__d44), "+f"(__d45), "+f"(__d46), "+f"(__d47), "+f"(__d48), "+f"(__d49), "+f"(__d50), "+f"(__d51), "+f"(__d52), "+f"(__d53), "+f"(__d54), "+f"(__d55), "+f"(__d56), "+f"(__d57), "+f"(__d58), "+f"(__d59), "+f"(__d60), "+f"(__d61), "+f"(__d62), "+f"(__d63), "+f"(__d64), "+f"(__d65), "+f"(__d66), "+f"(__d67), "+f"(__d68), "+f"(__d69), "+f"(__d70), "+f"(__d71), "+f"(__d72), "+f"(__d73), "+f"(__d74), "+f"(__d75), "+f"(__d76), "+f"(__d77), "+f"(__d78), "+f"(__d79), "+f"(__d80), "+f"(__d81), "+f"(__d82), "+f"(__d83), "+f"(__d84), "+f"(__d85), "+f"(__d86), "+f"(__d87), "+f"(__d88), "+f"(__d89), "+f"(__d90), "+f"(__d91), "+f"(__d92), "+f"(__d93), "+f"(__d94), "+f"(__d95), "+f"(__d96), "+f"(__d97), "+f"(__d98), "+f"(__d99), "+f"(__d100), "+f"(__d101), "+f"(__d102), "+f"(__d103), "+f"(__d104), "+f"(__d105), "+f"(__d106), "+f"(__d107), "+f"(__d108), "+f"(__d109), "+f"(__d110), "+f"(__d111), "+f"(__d112), "+f"(__d113), "+f"(__d114), "+f"(__d115), "+f"(__d116), "+f"(__d117), "+f"(__d118), "+f"(__d119), "+f"(__d120), "+f"(__d121), "+f"(__d122), "+f"(__d123), "+f"(__d124), "+f"(__d125), "+f"(__d126), "+f"(__d127) : "l"(__a_desc), "l"(__b_desc), "r"(__scale_d));
}

__forceinline__ __device__ int32_t tvm_builtin_ptx_fetch_register_cluster_ctaid_x() {
    uint32_t x;
    asm volatile("mov.u32 %0, %cluster_ctaid.x;" : "=r"(x));
    return (int32_t)x;
}
__forceinline__ __device__ void tvm_builtin_ptx_prefetch_tensormap(const void* __addr) {
  asm volatile("prefetch.tensormap [%0];" :  : "l"(__addr) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_mapa_u32_shared__cluster_u32(uint32_t& __d, uint32_t __a, uint32_t __b) {
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(__d) : "r"(__a), "r"(__b));
}

__forceinline__ __device__ unsigned int tvm_builtin_cuda_cvta_generic_to_shared(void* p) {
    return __cvta_generic_to_shared(p);
}
__forceinline__ __device__ void tvm_builtin_ptx_st_global_b32(const void* __addr, uint32_t __value) {
  asm volatile("st.global.b32 [%0], %1;" :  : "l"(__addr), "r"(__value) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes(uint32_t __dst_mem, const void* __tmap, int32_t __coords0, int32_t __coords1, uint32_t __mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" :  : "r"(__dst_mem), "l"(__tmap), "r"(__coords0), "r"(__coords1), "r"(__mbar) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_fence_fence_sync_aligned() {
  asm volatile("wgmma.fence.sync.aligned;" :  :  : "memory");
}

__forceinline__ __device__ void ptx_wgmma_fence_float(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes_multicast__cluster(uint32_t __dst_mem, const void* __tmap, int32_t __coords0, int32_t __coords1, uint32_t __mbar, uint16_t __cta_mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;" :  : "r"(__dst_mem), "l"(__tmap), "r"(__coords0), "r"(__coords1), "r"(__mbar), "h"(__cta_mask) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_init_init_shared_b64(uint32_t __addr, uint32_t __count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :  : "r"(__addr), "r"(__count) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_fence_mbarrier_init_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_setmaxnreg_dec_sync_aligned_u32_40() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" :  :  : "memory");
}

__forceinline__ __device__ void tvm_builtin_cuda_cluster_sync() {
    asm("barrier.cluster.arrive.aligned;");
    asm("barrier.cluster.wait.aligned;");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_wait_group_wait_group_sync_aligned_0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" :  :  : "memory");
}

__forceinline__ __device__ void tvm_builtin_cuda_mbarrier_wait(void* barrier, int phase) {
    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
    unsigned int ticks = 0x989680;
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_setmaxnreg_inc_sync_aligned_u32_232() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_arrive_expect_tx_arrive_expect_tx_shared_b64(uint32_t __addr, uint32_t __tx_count) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :  : "r"(__addr), "r"(__tx_count) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned() {
  asm volatile("wgmma.commit_group.sync.aligned;" :  :  : "memory");
}

__forceinline__ __device__ void ptx_wgmma_encode_matrix_descriptor(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {
  GmmaDescriptor _desc{};  // value-init: reading uncovered pad bits is UB

  switch (swizzle) {
    case 0: _desc.bitfield.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.bitfield.layout_type_ = uint8_t(3); break; // 32B swizzle
    case 2: _desc.bitfield.layout_type_ = uint8_t(2); break; // 64B swizzle
    case 3: _desc.bitfield.layout_type_ = uint8_t(1); break; // 128B swizzle
  }

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.bitfield.base_offset_ = base_offset;

  _desc.bitfield.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.bitfield.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, nv_bfloat16* __restrict__ C_ptr);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, nv_bfloat16* __restrict__ C_ptr) {
  int tx = ((int)threadIdx.x);
  int bid = ((int)blockIdx.x);
  __shared__ alignas(1024) nv_bfloat16 A_smem_ptr[32768];
  __shared__ alignas(1024) nv_bfloat16 B_smem_ptr[65536];
  __shared__ alignas(8) uint64_t full_ptr[4];
  __shared__ alignas(8) uint64_t empty_ptr[4];
  int m_blk = (((int)blockIdx.x) & 63);
  int n_blk = (((int)blockIdx.x) >> 6);
  int rank = tvm_builtin_ptx_fetch_register_cluster_ctaid_x();
  if (((int)threadIdx.x) == 0) {
    tvm_builtin_ptx_prefetch_tensormap(((void*)((unsigned long long)(&(A_map)))));
    tvm_builtin_ptx_prefetch_tensormap(((void*)((unsigned long long)(&(B_map)))));
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[0]))), (uint)1);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(empty_ptr[0]))), (uint)4);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[1]))), (uint)1);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(empty_ptr[1]))), (uint)4);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[2]))), (uint)1);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(empty_ptr[2]))), (uint)4);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[3]))), (uint)1);
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(empty_ptr[3]))), (uint)4);
  }
  tvm_builtin_ptx_fence_mbarrier_init_mbarrier_init_release_cluster();
  tvm_builtin_cuda_cluster_sync();
  if (((int)threadIdx.x) < 128) {
    tvm_builtin_ptx_setmaxnreg_dec_sync_aligned_u32_40();
    if (((int)threadIdx.x) == 0) {
      for (int kt = 0; kt < 128; ++kt) {
        int stage = (kt & 3);
        tvm_builtin_cuda_mbarrier_wait((&(empty_ptr[(kt & 3)])), (((kt >> 2) & 1) ^ 1));
        tvm_builtin_ptx_mbarrier_arrive_expect_tx_arrive_expect_tx_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[(kt & 3)]))), (uint)49152);
        tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes(tvm_builtin_cuda_cvta_generic_to_shared((&(A_smem_ptr[((kt & 3) * 8192)]))), ((void*)((unsigned long long)(&(A_map)))), (kt * 64), ((((int)blockIdx.x) & 63) * 128), tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[(kt & 3)]))));
        tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes_multicast__cluster(tvm_builtin_cuda_cvta_generic_to_shared((&(B_smem_ptr[(((kt & 3) * 16384) + (tvm_builtin_ptx_fetch_register_cluster_ctaid_x() * 8192))]))), ((void*)((unsigned long long)(&(B_map)))), (kt * 64), (((((int)blockIdx.x) >> 6) * 256) + (tvm_builtin_ptx_fetch_register_cluster_ctaid_x() * 128)), tvm_builtin_cuda_cvta_generic_to_shared((&(full_ptr[(kt & 3)]))), (ushort)3);
      }
    }
  } else {
    tvm_builtin_ptx_setmaxnreg_inc_sync_aligned_u32_232();
    int wg = ((((int)threadIdx.x) >> 7) - 1);
    int ct = (((int)threadIdx.x) & 127);
    alignas(64) float acc_ptr[128];
    alignas(64) uint64_t descA_ptr[1];
    alignas(64) uint64_t descB_ptr[1];
    alignas(64) uint mapped_ptr[1];
    alignas(64) uint packed_ptr[1];
    acc_ptr[0] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[1] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[2] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[3] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[4] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[5] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[6] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[7] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[8] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[9] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[10] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[11] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[12] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[13] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[14] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[15] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[16] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[17] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[18] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[19] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[20] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[21] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[22] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[23] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[24] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[25] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[26] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[27] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[28] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[29] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[30] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[31] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[32] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[33] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[34] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[35] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[36] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[37] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[38] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[39] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[40] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[41] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[42] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[43] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[44] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[45] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[46] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[47] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[48] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[49] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[50] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[51] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[52] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[53] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[54] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[55] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[56] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[57] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[58] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[59] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[60] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[61] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[62] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[63] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[64] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[65] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[66] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[67] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[68] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[69] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[70] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[71] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[72] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[73] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[74] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[75] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[76] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[77] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[78] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[79] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[80] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[81] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[82] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[83] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[84] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[85] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[86] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[87] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[88] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[89] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[90] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[91] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[92] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[93] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[94] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[95] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[96] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[97] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[98] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[99] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[100] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[101] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[102] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[103] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[104] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[105] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[106] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[107] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[108] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[109] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[110] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[111] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[112] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[113] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[114] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[115] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[116] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[117] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[118] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[119] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[120] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[121] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[122] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[123] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[124] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[125] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[126] = 0x0p+0f/*0.000000e+00*/;
    acc_ptr[127] = 0x0p+0f/*0.000000e+00*/;
    ptx_wgmma_encode_matrix_descriptor((&(descA_ptr[0])), (&(A_smem_ptr[(((((int)threadIdx.x) >> 7) * 4096) - 4096)])), 1, 64, 3);
    ptx_wgmma_encode_matrix_descriptor((&(descB_ptr[0])), (&(B_smem_ptr[0])), 1, 64, 3);
    for (int kt_1 = 0; kt_1 < 128; ++kt_1) {
      int stage_1 = (kt_1 & 3);
      tvm_builtin_cuda_mbarrier_wait((&(full_ptr[(kt_1 & 3)])), ((kt_1 >> 2) & 1));
      ptx_wgmma_fence_float(acc_ptr[0]);
      ptx_wgmma_fence_float(acc_ptr[1]);
      ptx_wgmma_fence_float(acc_ptr[2]);
      ptx_wgmma_fence_float(acc_ptr[3]);
      ptx_wgmma_fence_float(acc_ptr[4]);
      ptx_wgmma_fence_float(acc_ptr[5]);
      ptx_wgmma_fence_float(acc_ptr[6]);
      ptx_wgmma_fence_float(acc_ptr[7]);
      ptx_wgmma_fence_float(acc_ptr[8]);
      ptx_wgmma_fence_float(acc_ptr[9]);
      ptx_wgmma_fence_float(acc_ptr[10]);
      ptx_wgmma_fence_float(acc_ptr[11]);
      ptx_wgmma_fence_float(acc_ptr[12]);
      ptx_wgmma_fence_float(acc_ptr[13]);
      ptx_wgmma_fence_float(acc_ptr[14]);
      ptx_wgmma_fence_float(acc_ptr[15]);
      ptx_wgmma_fence_float(acc_ptr[16]);
      ptx_wgmma_fence_float(acc_ptr[17]);
      ptx_wgmma_fence_float(acc_ptr[18]);
      ptx_wgmma_fence_float(acc_ptr[19]);
      ptx_wgmma_fence_float(acc_ptr[20]);
      ptx_wgmma_fence_float(acc_ptr[21]);
      ptx_wgmma_fence_float(acc_ptr[22]);
      ptx_wgmma_fence_float(acc_ptr[23]);
      ptx_wgmma_fence_float(acc_ptr[24]);
      ptx_wgmma_fence_float(acc_ptr[25]);
      ptx_wgmma_fence_float(acc_ptr[26]);
      ptx_wgmma_fence_float(acc_ptr[27]);
      ptx_wgmma_fence_float(acc_ptr[28]);
      ptx_wgmma_fence_float(acc_ptr[29]);
      ptx_wgmma_fence_float(acc_ptr[30]);
      ptx_wgmma_fence_float(acc_ptr[31]);
      ptx_wgmma_fence_float(acc_ptr[32]);
      ptx_wgmma_fence_float(acc_ptr[33]);
      ptx_wgmma_fence_float(acc_ptr[34]);
      ptx_wgmma_fence_float(acc_ptr[35]);
      ptx_wgmma_fence_float(acc_ptr[36]);
      ptx_wgmma_fence_float(acc_ptr[37]);
      ptx_wgmma_fence_float(acc_ptr[38]);
      ptx_wgmma_fence_float(acc_ptr[39]);
      ptx_wgmma_fence_float(acc_ptr[40]);
      ptx_wgmma_fence_float(acc_ptr[41]);
      ptx_wgmma_fence_float(acc_ptr[42]);
      ptx_wgmma_fence_float(acc_ptr[43]);
      ptx_wgmma_fence_float(acc_ptr[44]);
      ptx_wgmma_fence_float(acc_ptr[45]);
      ptx_wgmma_fence_float(acc_ptr[46]);
      ptx_wgmma_fence_float(acc_ptr[47]);
      ptx_wgmma_fence_float(acc_ptr[48]);
      ptx_wgmma_fence_float(acc_ptr[49]);
      ptx_wgmma_fence_float(acc_ptr[50]);
      ptx_wgmma_fence_float(acc_ptr[51]);
      ptx_wgmma_fence_float(acc_ptr[52]);
      ptx_wgmma_fence_float(acc_ptr[53]);
      ptx_wgmma_fence_float(acc_ptr[54]);
      ptx_wgmma_fence_float(acc_ptr[55]);
      ptx_wgmma_fence_float(acc_ptr[56]);
      ptx_wgmma_fence_float(acc_ptr[57]);
      ptx_wgmma_fence_float(acc_ptr[58]);
      ptx_wgmma_fence_float(acc_ptr[59]);
      ptx_wgmma_fence_float(acc_ptr[60]);
      ptx_wgmma_fence_float(acc_ptr[61]);
      ptx_wgmma_fence_float(acc_ptr[62]);
      ptx_wgmma_fence_float(acc_ptr[63]);
      ptx_wgmma_fence_float(acc_ptr[64]);
      ptx_wgmma_fence_float(acc_ptr[65]);
      ptx_wgmma_fence_float(acc_ptr[66]);
      ptx_wgmma_fence_float(acc_ptr[67]);
      ptx_wgmma_fence_float(acc_ptr[68]);
      ptx_wgmma_fence_float(acc_ptr[69]);
      ptx_wgmma_fence_float(acc_ptr[70]);
      ptx_wgmma_fence_float(acc_ptr[71]);
      ptx_wgmma_fence_float(acc_ptr[72]);
      ptx_wgmma_fence_float(acc_ptr[73]);
      ptx_wgmma_fence_float(acc_ptr[74]);
      ptx_wgmma_fence_float(acc_ptr[75]);
      ptx_wgmma_fence_float(acc_ptr[76]);
      ptx_wgmma_fence_float(acc_ptr[77]);
      ptx_wgmma_fence_float(acc_ptr[78]);
      ptx_wgmma_fence_float(acc_ptr[79]);
      ptx_wgmma_fence_float(acc_ptr[80]);
      ptx_wgmma_fence_float(acc_ptr[81]);
      ptx_wgmma_fence_float(acc_ptr[82]);
      ptx_wgmma_fence_float(acc_ptr[83]);
      ptx_wgmma_fence_float(acc_ptr[84]);
      ptx_wgmma_fence_float(acc_ptr[85]);
      ptx_wgmma_fence_float(acc_ptr[86]);
      ptx_wgmma_fence_float(acc_ptr[87]);
      ptx_wgmma_fence_float(acc_ptr[88]);
      ptx_wgmma_fence_float(acc_ptr[89]);
      ptx_wgmma_fence_float(acc_ptr[90]);
      ptx_wgmma_fence_float(acc_ptr[91]);
      ptx_wgmma_fence_float(acc_ptr[92]);
      ptx_wgmma_fence_float(acc_ptr[93]);
      ptx_wgmma_fence_float(acc_ptr[94]);
      ptx_wgmma_fence_float(acc_ptr[95]);
      ptx_wgmma_fence_float(acc_ptr[96]);
      ptx_wgmma_fence_float(acc_ptr[97]);
      ptx_wgmma_fence_float(acc_ptr[98]);
      ptx_wgmma_fence_float(acc_ptr[99]);
      ptx_wgmma_fence_float(acc_ptr[100]);
      ptx_wgmma_fence_float(acc_ptr[101]);
      ptx_wgmma_fence_float(acc_ptr[102]);
      ptx_wgmma_fence_float(acc_ptr[103]);
      ptx_wgmma_fence_float(acc_ptr[104]);
      ptx_wgmma_fence_float(acc_ptr[105]);
      ptx_wgmma_fence_float(acc_ptr[106]);
      ptx_wgmma_fence_float(acc_ptr[107]);
      ptx_wgmma_fence_float(acc_ptr[108]);
      ptx_wgmma_fence_float(acc_ptr[109]);
      ptx_wgmma_fence_float(acc_ptr[110]);
      ptx_wgmma_fence_float(acc_ptr[111]);
      ptx_wgmma_fence_float(acc_ptr[112]);
      ptx_wgmma_fence_float(acc_ptr[113]);
      ptx_wgmma_fence_float(acc_ptr[114]);
      ptx_wgmma_fence_float(acc_ptr[115]);
      ptx_wgmma_fence_float(acc_ptr[116]);
      ptx_wgmma_fence_float(acc_ptr[117]);
      ptx_wgmma_fence_float(acc_ptr[118]);
      ptx_wgmma_fence_float(acc_ptr[119]);
      ptx_wgmma_fence_float(acc_ptr[120]);
      ptx_wgmma_fence_float(acc_ptr[121]);
      ptx_wgmma_fence_float(acc_ptr[122]);
      ptx_wgmma_fence_float(acc_ptr[123]);
      ptx_wgmma_fence_float(acc_ptr[124]);
      ptx_wgmma_fence_float(acc_ptr[125]);
      ptx_wgmma_fence_float(acc_ptr[126]);
      ptx_wgmma_fence_float(acc_ptr[127]);
      tvm_builtin_ptx_wgmma_fence_fence_sync_aligned();
      tvm_builtin_ptx_wgmma_bf16_ss_mma_async_sync_aligned_m64n256k16_f32_bf16_bf16_1_1_0_0(acc_ptr[0], acc_ptr[1], acc_ptr[2], acc_ptr[3], acc_ptr[4], acc_ptr[5], acc_ptr[6], acc_ptr[7], acc_ptr[8], acc_ptr[9], acc_ptr[10], acc_ptr[11], acc_ptr[12], acc_ptr[13], acc_ptr[14], acc_ptr[15], acc_ptr[16], acc_ptr[17], acc_ptr[18], acc_ptr[19], acc_ptr[20], acc_ptr[21], acc_ptr[22], acc_ptr[23], acc_ptr[24], acc_ptr[25], acc_ptr[26], acc_ptr[27], acc_ptr[28], acc_ptr[29], acc_ptr[30], acc_ptr[31], acc_ptr[32], acc_ptr[33], acc_ptr[34], acc_ptr[35], acc_ptr[36], acc_ptr[37], acc_ptr[38], acc_ptr[39], acc_ptr[40], acc_ptr[41], acc_ptr[42], acc_ptr[43], acc_ptr[44], acc_ptr[45], acc_ptr[46], acc_ptr[47], acc_ptr[48], acc_ptr[49], acc_ptr[50], acc_ptr[51], acc_ptr[52], acc_ptr[53], acc_ptr[54], acc_ptr[55], acc_ptr[56], acc_ptr[57], acc_ptr[58], acc_ptr[59], acc_ptr[60], acc_ptr[61], acc_ptr[62], acc_ptr[63], acc_ptr[64], acc_ptr[65], acc_ptr[66], acc_ptr[67], acc_ptr[68], acc_ptr[69], acc_ptr[70], acc_ptr[71], acc_ptr[72], acc_ptr[73], acc_ptr[74], acc_ptr[75], acc_ptr[76], acc_ptr[77], acc_ptr[78], acc_ptr[79], acc_ptr[80], acc_ptr[81], acc_ptr[82], acc_ptr[83], acc_ptr[84], acc_ptr[85], acc_ptr[86], acc_ptr[87], acc_ptr[88], acc_ptr[89], acc_ptr[90], acc_ptr[91], acc_ptr[92], acc_ptr[93], acc_ptr[94], acc_ptr[95], acc_ptr[96], acc_ptr[97], acc_ptr[98], acc_ptr[99], acc_ptr[100], acc_ptr[101], acc_ptr[102], acc_ptr[103], acc_ptr[104], acc_ptr[105], acc_ptr[106], acc_ptr[107], acc_ptr[108], acc_ptr[109], acc_ptr[110], acc_ptr[111], acc_ptr[112], acc_ptr[113], acc_ptr[114], acc_ptr[115], acc_ptr[116], acc_ptr[117], acc_ptr[118], acc_ptr[119], acc_ptr[120], acc_ptr[121], acc_ptr[122], acc_ptr[123], acc_ptr[124], acc_ptr[125], acc_ptr[126], acc_ptr[127], (descA_ptr[0] + ((uint64_t)((kt_1 & 3) * 1024))), (descB_ptr[0] + ((uint64_t)((kt_1 & 3) * 2048))), (uint)1);
      tvm_builtin_ptx_wgmma_bf16_ss_mma_async_sync_aligned_m64n256k16_f32_bf16_bf16_1_1_0_0(acc_ptr[0], acc_ptr[1], acc_ptr[2], acc_ptr[3], acc_ptr[4], acc_ptr[5], acc_ptr[6], acc_ptr[7], acc_ptr[8], acc_ptr[9], acc_ptr[10], acc_ptr[11], acc_ptr[12], acc_ptr[13], acc_ptr[14], acc_ptr[15], acc_ptr[16], acc_ptr[17], acc_ptr[18], acc_ptr[19], acc_ptr[20], acc_ptr[21], acc_ptr[22], acc_ptr[23], acc_ptr[24], acc_ptr[25], acc_ptr[26], acc_ptr[27], acc_ptr[28], acc_ptr[29], acc_ptr[30], acc_ptr[31], acc_ptr[32], acc_ptr[33], acc_ptr[34], acc_ptr[35], acc_ptr[36], acc_ptr[37], acc_ptr[38], acc_ptr[39], acc_ptr[40], acc_ptr[41], acc_ptr[42], acc_ptr[43], acc_ptr[44], acc_ptr[45], acc_ptr[46], acc_ptr[47], acc_ptr[48], acc_ptr[49], acc_ptr[50], acc_ptr[51], acc_ptr[52], acc_ptr[53], acc_ptr[54], acc_ptr[55], acc_ptr[56], acc_ptr[57], acc_ptr[58], acc_ptr[59], acc_ptr[60], acc_ptr[61], acc_ptr[62], acc_ptr[63], acc_ptr[64], acc_ptr[65], acc_ptr[66], acc_ptr[67], acc_ptr[68], acc_ptr[69], acc_ptr[70], acc_ptr[71], acc_ptr[72], acc_ptr[73], acc_ptr[74], acc_ptr[75], acc_ptr[76], acc_ptr[77], acc_ptr[78], acc_ptr[79], acc_ptr[80], acc_ptr[81], acc_ptr[82], acc_ptr[83], acc_ptr[84], acc_ptr[85], acc_ptr[86], acc_ptr[87], acc_ptr[88], acc_ptr[89], acc_ptr[90], acc_ptr[91], acc_ptr[92], acc_ptr[93], acc_ptr[94], acc_ptr[95], acc_ptr[96], acc_ptr[97], acc_ptr[98], acc_ptr[99], acc_ptr[100], acc_ptr[101], acc_ptr[102], acc_ptr[103], acc_ptr[104], acc_ptr[105], acc_ptr[106], acc_ptr[107], acc_ptr[108], acc_ptr[109], acc_ptr[110], acc_ptr[111], acc_ptr[112], acc_ptr[113], acc_ptr[114], acc_ptr[115], acc_ptr[116], acc_ptr[117], acc_ptr[118], acc_ptr[119], acc_ptr[120], acc_ptr[121], acc_ptr[122], acc_ptr[123], acc_ptr[124], acc_ptr[125], acc_ptr[126], acc_ptr[127], (descA_ptr[0] + ((uint64_t)(((kt_1 & 3) * 1024) + 2))), (descB_ptr[0] + ((uint64_t)(((kt_1 & 3) * 2048) + 2))), (uint)1);
      tvm_builtin_ptx_wgmma_bf16_ss_mma_async_sync_aligned_m64n256k16_f32_bf16_bf16_1_1_0_0(acc_ptr[0], acc_ptr[1], acc_ptr[2], acc_ptr[3], acc_ptr[4], acc_ptr[5], acc_ptr[6], acc_ptr[7], acc_ptr[8], acc_ptr[9], acc_ptr[10], acc_ptr[11], acc_ptr[12], acc_ptr[13], acc_ptr[14], acc_ptr[15], acc_ptr[16], acc_ptr[17], acc_ptr[18], acc_ptr[19], acc_ptr[20], acc_ptr[21], acc_ptr[22], acc_ptr[23], acc_ptr[24], acc_ptr[25], acc_ptr[26], acc_ptr[27], acc_ptr[28], acc_ptr[29], acc_ptr[30], acc_ptr[31], acc_ptr[32], acc_ptr[33], acc_ptr[34], acc_ptr[35], acc_ptr[36], acc_ptr[37], acc_ptr[38], acc_ptr[39], acc_ptr[40], acc_ptr[41], acc_ptr[42], acc_ptr[43], acc_ptr[44], acc_ptr[45], acc_ptr[46], acc_ptr[47], acc_ptr[48], acc_ptr[49], acc_ptr[50], acc_ptr[51], acc_ptr[52], acc_ptr[53], acc_ptr[54], acc_ptr[55], acc_ptr[56], acc_ptr[57], acc_ptr[58], acc_ptr[59], acc_ptr[60], acc_ptr[61], acc_ptr[62], acc_ptr[63], acc_ptr[64], acc_ptr[65], acc_ptr[66], acc_ptr[67], acc_ptr[68], acc_ptr[69], acc_ptr[70], acc_ptr[71], acc_ptr[72], acc_ptr[73], acc_ptr[74], acc_ptr[75], acc_ptr[76], acc_ptr[77], acc_ptr[78], acc_ptr[79], acc_ptr[80], acc_ptr[81], acc_ptr[82], acc_ptr[83], acc_ptr[84], acc_ptr[85], acc_ptr[86], acc_ptr[87], acc_ptr[88], acc_ptr[89], acc_ptr[90], acc_ptr[91], acc_ptr[92], acc_ptr[93], acc_ptr[94], acc_ptr[95], acc_ptr[96], acc_ptr[97], acc_ptr[98], acc_ptr[99], acc_ptr[100], acc_ptr[101], acc_ptr[102], acc_ptr[103], acc_ptr[104], acc_ptr[105], acc_ptr[106], acc_ptr[107], acc_ptr[108], acc_ptr[109], acc_ptr[110], acc_ptr[111], acc_ptr[112], acc_ptr[113], acc_ptr[114], acc_ptr[115], acc_ptr[116], acc_ptr[117], acc_ptr[118], acc_ptr[119], acc_ptr[120], acc_ptr[121], acc_ptr[122], acc_ptr[123], acc_ptr[124], acc_ptr[125], acc_ptr[126], acc_ptr[127], (descA_ptr[0] + ((uint64_t)(((kt_1 & 3) * 1024) + 4))), (descB_ptr[0] + ((uint64_t)(((kt_1 & 3) * 2048) + 4))), (uint)1);
      tvm_builtin_ptx_wgmma_bf16_ss_mma_async_sync_aligned_m64n256k16_f32_bf16_bf16_1_1_0_0(acc_ptr[0], acc_ptr[1], acc_ptr[2], acc_ptr[3], acc_ptr[4], acc_ptr[5], acc_ptr[6], acc_ptr[7], acc_ptr[8], acc_ptr[9], acc_ptr[10], acc_ptr[11], acc_ptr[12], acc_ptr[13], acc_ptr[14], acc_ptr[15], acc_ptr[16], acc_ptr[17], acc_ptr[18], acc_ptr[19], acc_ptr[20], acc_ptr[21], acc_ptr[22], acc_ptr[23], acc_ptr[24], acc_ptr[25], acc_ptr[26], acc_ptr[27], acc_ptr[28], acc_ptr[29], acc_ptr[30], acc_ptr[31], acc_ptr[32], acc_ptr[33], acc_ptr[34], acc_ptr[35], acc_ptr[36], acc_ptr[37], acc_ptr[38], acc_ptr[39], acc_ptr[40], acc_ptr[41], acc_ptr[42], acc_ptr[43], acc_ptr[44], acc_ptr[45], acc_ptr[46], acc_ptr[47], acc_ptr[48], acc_ptr[49], acc_ptr[50], acc_ptr[51], acc_ptr[52], acc_ptr[53], acc_ptr[54], acc_ptr[55], acc_ptr[56], acc_ptr[57], acc_ptr[58], acc_ptr[59], acc_ptr[60], acc_ptr[61], acc_ptr[62], acc_ptr[63], acc_ptr[64], acc_ptr[65], acc_ptr[66], acc_ptr[67], acc_ptr[68], acc_ptr[69], acc_ptr[70], acc_ptr[71], acc_ptr[72], acc_ptr[73], acc_ptr[74], acc_ptr[75], acc_ptr[76], acc_ptr[77], acc_ptr[78], acc_ptr[79], acc_ptr[80], acc_ptr[81], acc_ptr[82], acc_ptr[83], acc_ptr[84], acc_ptr[85], acc_ptr[86], acc_ptr[87], acc_ptr[88], acc_ptr[89], acc_ptr[90], acc_ptr[91], acc_ptr[92], acc_ptr[93], acc_ptr[94], acc_ptr[95], acc_ptr[96], acc_ptr[97], acc_ptr[98], acc_ptr[99], acc_ptr[100], acc_ptr[101], acc_ptr[102], acc_ptr[103], acc_ptr[104], acc_ptr[105], acc_ptr[106], acc_ptr[107], acc_ptr[108], acc_ptr[109], acc_ptr[110], acc_ptr[111], acc_ptr[112], acc_ptr[113], acc_ptr[114], acc_ptr[115], acc_ptr[116], acc_ptr[117], acc_ptr[118], acc_ptr[119], acc_ptr[120], acc_ptr[121], acc_ptr[122], acc_ptr[123], acc_ptr[124], acc_ptr[125], acc_ptr[126], acc_ptr[127], (descA_ptr[0] + ((uint64_t)(((kt_1 & 3) * 1024) + 6))), (descB_ptr[0] + ((uint64_t)(((kt_1 & 3) * 2048) + 6))), (uint)1);
      tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned();
      tvm_builtin_ptx_wgmma_wait_group_wait_group_sync_aligned_1();
      ptx_wgmma_fence_float(acc_ptr[0]);
      ptx_wgmma_fence_float(acc_ptr[1]);
      ptx_wgmma_fence_float(acc_ptr[2]);
      ptx_wgmma_fence_float(acc_ptr[3]);
      ptx_wgmma_fence_float(acc_ptr[4]);
      ptx_wgmma_fence_float(acc_ptr[5]);
      ptx_wgmma_fence_float(acc_ptr[6]);
      ptx_wgmma_fence_float(acc_ptr[7]);
      ptx_wgmma_fence_float(acc_ptr[8]);
      ptx_wgmma_fence_float(acc_ptr[9]);
      ptx_wgmma_fence_float(acc_ptr[10]);
      ptx_wgmma_fence_float(acc_ptr[11]);
      ptx_wgmma_fence_float(acc_ptr[12]);
      ptx_wgmma_fence_float(acc_ptr[13]);
      ptx_wgmma_fence_float(acc_ptr[14]);
      ptx_wgmma_fence_float(acc_ptr[15]);
      ptx_wgmma_fence_float(acc_ptr[16]);
      ptx_wgmma_fence_float(acc_ptr[17]);
      ptx_wgmma_fence_float(acc_ptr[18]);
      ptx_wgmma_fence_float(acc_ptr[19]);
      ptx_wgmma_fence_float(acc_ptr[20]);
      ptx_wgmma_fence_float(acc_ptr[21]);
      ptx_wgmma_fence_float(acc_ptr[22]);
      ptx_wgmma_fence_float(acc_ptr[23]);
      ptx_wgmma_fence_float(acc_ptr[24]);
      ptx_wgmma_fence_float(acc_ptr[25]);
      ptx_wgmma_fence_float(acc_ptr[26]);
      ptx_wgmma_fence_float(acc_ptr[27]);
      ptx_wgmma_fence_float(acc_ptr[28]);
      ptx_wgmma_fence_float(acc_ptr[29]);
      ptx_wgmma_fence_float(acc_ptr[30]);
      ptx_wgmma_fence_float(acc_ptr[31]);
      ptx_wgmma_fence_float(acc_ptr[32]);
      ptx_wgmma_fence_float(acc_ptr[33]);
      ptx_wgmma_fence_float(acc_ptr[34]);
      ptx_wgmma_fence_float(acc_ptr[35]);
      ptx_wgmma_fence_float(acc_ptr[36]);
      ptx_wgmma_fence_float(acc_ptr[37]);
      ptx_wgmma_fence_float(acc_ptr[38]);
      ptx_wgmma_fence_float(acc_ptr[39]);
      ptx_wgmma_fence_float(acc_ptr[40]);
      ptx_wgmma_fence_float(acc_ptr[41]);
      ptx_wgmma_fence_float(acc_ptr[42]);
      ptx_wgmma_fence_float(acc_ptr[43]);
      ptx_wgmma_fence_float(acc_ptr[44]);
      ptx_wgmma_fence_float(acc_ptr[45]);
      ptx_wgmma_fence_float(acc_ptr[46]);
      ptx_wgmma_fence_float(acc_ptr[47]);
      ptx_wgmma_fence_float(acc_ptr[48]);
      ptx_wgmma_fence_float(acc_ptr[49]);
      ptx_wgmma_fence_float(acc_ptr[50]);
      ptx_wgmma_fence_float(acc_ptr[51]);
      ptx_wgmma_fence_float(acc_ptr[52]);
      ptx_wgmma_fence_float(acc_ptr[53]);
      ptx_wgmma_fence_float(acc_ptr[54]);
      ptx_wgmma_fence_float(acc_ptr[55]);
      ptx_wgmma_fence_float(acc_ptr[56]);
      ptx_wgmma_fence_float(acc_ptr[57]);
      ptx_wgmma_fence_float(acc_ptr[58]);
      ptx_wgmma_fence_float(acc_ptr[59]);
      ptx_wgmma_fence_float(acc_ptr[60]);
      ptx_wgmma_fence_float(acc_ptr[61]);
      ptx_wgmma_fence_float(acc_ptr[62]);
      ptx_wgmma_fence_float(acc_ptr[63]);
      ptx_wgmma_fence_float(acc_ptr[64]);
      ptx_wgmma_fence_float(acc_ptr[65]);
      ptx_wgmma_fence_float(acc_ptr[66]);
      ptx_wgmma_fence_float(acc_ptr[67]);
      ptx_wgmma_fence_float(acc_ptr[68]);
      ptx_wgmma_fence_float(acc_ptr[69]);
      ptx_wgmma_fence_float(acc_ptr[70]);
      ptx_wgmma_fence_float(acc_ptr[71]);
      ptx_wgmma_fence_float(acc_ptr[72]);
      ptx_wgmma_fence_float(acc_ptr[73]);
      ptx_wgmma_fence_float(acc_ptr[74]);
      ptx_wgmma_fence_float(acc_ptr[75]);
      ptx_wgmma_fence_float(acc_ptr[76]);
      ptx_wgmma_fence_float(acc_ptr[77]);
      ptx_wgmma_fence_float(acc_ptr[78]);
      ptx_wgmma_fence_float(acc_ptr[79]);
      ptx_wgmma_fence_float(acc_ptr[80]);
      ptx_wgmma_fence_float(acc_ptr[81]);
      ptx_wgmma_fence_float(acc_ptr[82]);
      ptx_wgmma_fence_float(acc_ptr[83]);
      ptx_wgmma_fence_float(acc_ptr[84]);
      ptx_wgmma_fence_float(acc_ptr[85]);
      ptx_wgmma_fence_float(acc_ptr[86]);
      ptx_wgmma_fence_float(acc_ptr[87]);
      ptx_wgmma_fence_float(acc_ptr[88]);
      ptx_wgmma_fence_float(acc_ptr[89]);
      ptx_wgmma_fence_float(acc_ptr[90]);
      ptx_wgmma_fence_float(acc_ptr[91]);
      ptx_wgmma_fence_float(acc_ptr[92]);
      ptx_wgmma_fence_float(acc_ptr[93]);
      ptx_wgmma_fence_float(acc_ptr[94]);
      ptx_wgmma_fence_float(acc_ptr[95]);
      ptx_wgmma_fence_float(acc_ptr[96]);
      ptx_wgmma_fence_float(acc_ptr[97]);
      ptx_wgmma_fence_float(acc_ptr[98]);
      ptx_wgmma_fence_float(acc_ptr[99]);
      ptx_wgmma_fence_float(acc_ptr[100]);
      ptx_wgmma_fence_float(acc_ptr[101]);
      ptx_wgmma_fence_float(acc_ptr[102]);
      ptx_wgmma_fence_float(acc_ptr[103]);
      ptx_wgmma_fence_float(acc_ptr[104]);
      ptx_wgmma_fence_float(acc_ptr[105]);
      ptx_wgmma_fence_float(acc_ptr[106]);
      ptx_wgmma_fence_float(acc_ptr[107]);
      ptx_wgmma_fence_float(acc_ptr[108]);
      ptx_wgmma_fence_float(acc_ptr[109]);
      ptx_wgmma_fence_float(acc_ptr[110]);
      ptx_wgmma_fence_float(acc_ptr[111]);
      ptx_wgmma_fence_float(acc_ptr[112]);
      ptx_wgmma_fence_float(acc_ptr[113]);
      ptx_wgmma_fence_float(acc_ptr[114]);
      ptx_wgmma_fence_float(acc_ptr[115]);
      ptx_wgmma_fence_float(acc_ptr[116]);
      ptx_wgmma_fence_float(acc_ptr[117]);
      ptx_wgmma_fence_float(acc_ptr[118]);
      ptx_wgmma_fence_float(acc_ptr[119]);
      ptx_wgmma_fence_float(acc_ptr[120]);
      ptx_wgmma_fence_float(acc_ptr[121]);
      ptx_wgmma_fence_float(acc_ptr[122]);
      ptx_wgmma_fence_float(acc_ptr[123]);
      ptx_wgmma_fence_float(acc_ptr[124]);
      ptx_wgmma_fence_float(acc_ptr[125]);
      ptx_wgmma_fence_float(acc_ptr[126]);
      ptx_wgmma_fence_float(acc_ptr[127]);
      if (0 < kt_1) {
        if ((((int)threadIdx.x) % 128) == 0) {
          int prev = ((kt_1 + 3) & 3);
          tvm_builtin_ptx_mapa_u32_shared__cluster_u32(mapped_ptr[0], tvm_builtin_cuda_cvta_generic_to_shared((&(empty_ptr[((kt_1 + 3) & 3)]))), (uint)0);
          tvm_builtin_ptx_mbarrier_arrive_nocount_arrive_shared__cluster_b64(mapped_ptr[0]);
          tvm_builtin_ptx_mapa_u32_shared__cluster_u32(mapped_ptr[0], tvm_builtin_cuda_cvta_generic_to_shared((&(empty_ptr[((kt_1 + 3) & 3)]))), (uint)1);
          tvm_builtin_ptx_mbarrier_arrive_nocount_arrive_shared__cluster_b64(mapped_ptr[0]);
        }
      }
    }
    tvm_builtin_ptx_wgmma_wait_group_wait_group_sync_aligned_0();
    ptx_wgmma_fence_float(acc_ptr[0]);
    ptx_wgmma_fence_float(acc_ptr[1]);
    ptx_wgmma_fence_float(acc_ptr[2]);
    ptx_wgmma_fence_float(acc_ptr[3]);
    ptx_wgmma_fence_float(acc_ptr[4]);
    ptx_wgmma_fence_float(acc_ptr[5]);
    ptx_wgmma_fence_float(acc_ptr[6]);
    ptx_wgmma_fence_float(acc_ptr[7]);
    ptx_wgmma_fence_float(acc_ptr[8]);
    ptx_wgmma_fence_float(acc_ptr[9]);
    ptx_wgmma_fence_float(acc_ptr[10]);
    ptx_wgmma_fence_float(acc_ptr[11]);
    ptx_wgmma_fence_float(acc_ptr[12]);
    ptx_wgmma_fence_float(acc_ptr[13]);
    ptx_wgmma_fence_float(acc_ptr[14]);
    ptx_wgmma_fence_float(acc_ptr[15]);
    ptx_wgmma_fence_float(acc_ptr[16]);
    ptx_wgmma_fence_float(acc_ptr[17]);
    ptx_wgmma_fence_float(acc_ptr[18]);
    ptx_wgmma_fence_float(acc_ptr[19]);
    ptx_wgmma_fence_float(acc_ptr[20]);
    ptx_wgmma_fence_float(acc_ptr[21]);
    ptx_wgmma_fence_float(acc_ptr[22]);
    ptx_wgmma_fence_float(acc_ptr[23]);
    ptx_wgmma_fence_float(acc_ptr[24]);
    ptx_wgmma_fence_float(acc_ptr[25]);
    ptx_wgmma_fence_float(acc_ptr[26]);
    ptx_wgmma_fence_float(acc_ptr[27]);
    ptx_wgmma_fence_float(acc_ptr[28]);
    ptx_wgmma_fence_float(acc_ptr[29]);
    ptx_wgmma_fence_float(acc_ptr[30]);
    ptx_wgmma_fence_float(acc_ptr[31]);
    ptx_wgmma_fence_float(acc_ptr[32]);
    ptx_wgmma_fence_float(acc_ptr[33]);
    ptx_wgmma_fence_float(acc_ptr[34]);
    ptx_wgmma_fence_float(acc_ptr[35]);
    ptx_wgmma_fence_float(acc_ptr[36]);
    ptx_wgmma_fence_float(acc_ptr[37]);
    ptx_wgmma_fence_float(acc_ptr[38]);
    ptx_wgmma_fence_float(acc_ptr[39]);
    ptx_wgmma_fence_float(acc_ptr[40]);
    ptx_wgmma_fence_float(acc_ptr[41]);
    ptx_wgmma_fence_float(acc_ptr[42]);
    ptx_wgmma_fence_float(acc_ptr[43]);
    ptx_wgmma_fence_float(acc_ptr[44]);
    ptx_wgmma_fence_float(acc_ptr[45]);
    ptx_wgmma_fence_float(acc_ptr[46]);
    ptx_wgmma_fence_float(acc_ptr[47]);
    ptx_wgmma_fence_float(acc_ptr[48]);
    ptx_wgmma_fence_float(acc_ptr[49]);
    ptx_wgmma_fence_float(acc_ptr[50]);
    ptx_wgmma_fence_float(acc_ptr[51]);
    ptx_wgmma_fence_float(acc_ptr[52]);
    ptx_wgmma_fence_float(acc_ptr[53]);
    ptx_wgmma_fence_float(acc_ptr[54]);
    ptx_wgmma_fence_float(acc_ptr[55]);
    ptx_wgmma_fence_float(acc_ptr[56]);
    ptx_wgmma_fence_float(acc_ptr[57]);
    ptx_wgmma_fence_float(acc_ptr[58]);
    ptx_wgmma_fence_float(acc_ptr[59]);
    ptx_wgmma_fence_float(acc_ptr[60]);
    ptx_wgmma_fence_float(acc_ptr[61]);
    ptx_wgmma_fence_float(acc_ptr[62]);
    ptx_wgmma_fence_float(acc_ptr[63]);
    ptx_wgmma_fence_float(acc_ptr[64]);
    ptx_wgmma_fence_float(acc_ptr[65]);
    ptx_wgmma_fence_float(acc_ptr[66]);
    ptx_wgmma_fence_float(acc_ptr[67]);
    ptx_wgmma_fence_float(acc_ptr[68]);
    ptx_wgmma_fence_float(acc_ptr[69]);
    ptx_wgmma_fence_float(acc_ptr[70]);
    ptx_wgmma_fence_float(acc_ptr[71]);
    ptx_wgmma_fence_float(acc_ptr[72]);
    ptx_wgmma_fence_float(acc_ptr[73]);
    ptx_wgmma_fence_float(acc_ptr[74]);
    ptx_wgmma_fence_float(acc_ptr[75]);
    ptx_wgmma_fence_float(acc_ptr[76]);
    ptx_wgmma_fence_float(acc_ptr[77]);
    ptx_wgmma_fence_float(acc_ptr[78]);
    ptx_wgmma_fence_float(acc_ptr[79]);
    ptx_wgmma_fence_float(acc_ptr[80]);
    ptx_wgmma_fence_float(acc_ptr[81]);
    ptx_wgmma_fence_float(acc_ptr[82]);
    ptx_wgmma_fence_float(acc_ptr[83]);
    ptx_wgmma_fence_float(acc_ptr[84]);
    ptx_wgmma_fence_float(acc_ptr[85]);
    ptx_wgmma_fence_float(acc_ptr[86]);
    ptx_wgmma_fence_float(acc_ptr[87]);
    ptx_wgmma_fence_float(acc_ptr[88]);
    ptx_wgmma_fence_float(acc_ptr[89]);
    ptx_wgmma_fence_float(acc_ptr[90]);
    ptx_wgmma_fence_float(acc_ptr[91]);
    ptx_wgmma_fence_float(acc_ptr[92]);
    ptx_wgmma_fence_float(acc_ptr[93]);
    ptx_wgmma_fence_float(acc_ptr[94]);
    ptx_wgmma_fence_float(acc_ptr[95]);
    ptx_wgmma_fence_float(acc_ptr[96]);
    ptx_wgmma_fence_float(acc_ptr[97]);
    ptx_wgmma_fence_float(acc_ptr[98]);
    ptx_wgmma_fence_float(acc_ptr[99]);
    ptx_wgmma_fence_float(acc_ptr[100]);
    ptx_wgmma_fence_float(acc_ptr[101]);
    ptx_wgmma_fence_float(acc_ptr[102]);
    ptx_wgmma_fence_float(acc_ptr[103]);
    ptx_wgmma_fence_float(acc_ptr[104]);
    ptx_wgmma_fence_float(acc_ptr[105]);
    ptx_wgmma_fence_float(acc_ptr[106]);
    ptx_wgmma_fence_float(acc_ptr[107]);
    ptx_wgmma_fence_float(acc_ptr[108]);
    ptx_wgmma_fence_float(acc_ptr[109]);
    ptx_wgmma_fence_float(acc_ptr[110]);
    ptx_wgmma_fence_float(acc_ptr[111]);
    ptx_wgmma_fence_float(acc_ptr[112]);
    ptx_wgmma_fence_float(acc_ptr[113]);
    ptx_wgmma_fence_float(acc_ptr[114]);
    ptx_wgmma_fence_float(acc_ptr[115]);
    ptx_wgmma_fence_float(acc_ptr[116]);
    ptx_wgmma_fence_float(acc_ptr[117]);
    ptx_wgmma_fence_float(acc_ptr[118]);
    ptx_wgmma_fence_float(acc_ptr[119]);
    ptx_wgmma_fence_float(acc_ptr[120]);
    ptx_wgmma_fence_float(acc_ptr[121]);
    ptx_wgmma_fence_float(acc_ptr[122]);
    ptx_wgmma_fence_float(acc_ptr[123]);
    ptx_wgmma_fence_float(acc_ptr[124]);
    ptx_wgmma_fence_float(acc_ptr[125]);
    ptx_wgmma_fence_float(acc_ptr[126]);
    ptx_wgmma_fence_float(acc_ptr[127]);
    int w = ((((int)threadIdx.x) & 127) >> 5);
    int lane = (((int)threadIdx.x) & 31);
    int row = (((((((int)blockIdx.x) & 63) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 31) >> 2)) - 64);
    int col = (((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2));
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[1], acc_ptr[0]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524288)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[3], acc_ptr[2]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458752)])), packed_ptr[0]);
    int col_1 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 8);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[5], acc_ptr[4]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524280)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[7], acc_ptr[6]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458744)])), packed_ptr[0]);
    int col_2 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 16);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[9], acc_ptr[8]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524272)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[11], acc_ptr[10]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458736)])), packed_ptr[0]);
    int col_3 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 24);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[13], acc_ptr[12]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524264)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[15], acc_ptr[14]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458728)])), packed_ptr[0]);
    int col_4 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 32);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[17], acc_ptr[16]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524256)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[19], acc_ptr[18]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458720)])), packed_ptr[0]);
    int col_5 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 40);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[21], acc_ptr[20]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524248)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[23], acc_ptr[22]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458712)])), packed_ptr[0]);
    int col_6 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 48);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[25], acc_ptr[24]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524240)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[27], acc_ptr[26]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458704)])), packed_ptr[0]);
    int col_7 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 56);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[29], acc_ptr[28]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524232)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[31], acc_ptr[30]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458696)])), packed_ptr[0]);
    int col_8 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 64);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[33], acc_ptr[32]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524224)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[35], acc_ptr[34]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458688)])), packed_ptr[0]);
    int col_9 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 72);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[37], acc_ptr[36]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524216)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[39], acc_ptr[38]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458680)])), packed_ptr[0]);
    int col_10 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 80);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[41], acc_ptr[40]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524208)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[43], acc_ptr[42]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458672)])), packed_ptr[0]);
    int col_11 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 88);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[45], acc_ptr[44]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524200)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[47], acc_ptr[46]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458664)])), packed_ptr[0]);
    int col_12 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 96);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[49], acc_ptr[48]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524192)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[51], acc_ptr[50]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458656)])), packed_ptr[0]);
    int col_13 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 104);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[53], acc_ptr[52]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524184)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[55], acc_ptr[54]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458648)])), packed_ptr[0]);
    int col_14 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 112);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[57], acc_ptr[56]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524176)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[59], acc_ptr[58]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458640)])), packed_ptr[0]);
    int col_15 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 120);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[61], acc_ptr[60]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524168)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[63], acc_ptr[62]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458632)])), packed_ptr[0]);
    int col_16 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 128);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[65], acc_ptr[64]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524160)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[67], acc_ptr[66]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458624)])), packed_ptr[0]);
    int col_17 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 136);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[69], acc_ptr[68]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524152)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[71], acc_ptr[70]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458616)])), packed_ptr[0]);
    int col_18 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 144);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[73], acc_ptr[72]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524144)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[75], acc_ptr[74]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458608)])), packed_ptr[0]);
    int col_19 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 152);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[77], acc_ptr[76]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524136)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[79], acc_ptr[78]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458600)])), packed_ptr[0]);
    int col_20 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 160);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[81], acc_ptr[80]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524128)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[83], acc_ptr[82]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458592)])), packed_ptr[0]);
    int col_21 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 168);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[85], acc_ptr[84]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524120)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[87], acc_ptr[86]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458584)])), packed_ptr[0]);
    int col_22 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 176);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[89], acc_ptr[88]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524112)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[91], acc_ptr[90]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458576)])), packed_ptr[0]);
    int col_23 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 184);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[93], acc_ptr[92]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524104)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[95], acc_ptr[94]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458568)])), packed_ptr[0]);
    int col_24 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 192);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[97], acc_ptr[96]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524096)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[99], acc_ptr[98]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458560)])), packed_ptr[0]);
    int col_25 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 200);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[101], acc_ptr[100]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524088)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[103], acc_ptr[102]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458552)])), packed_ptr[0]);
    int col_26 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 208);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[105], acc_ptr[104]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524080)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[107], acc_ptr[106]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458544)])), packed_ptr[0]);
    int col_27 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 216);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[109], acc_ptr[108]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524072)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[111], acc_ptr[110]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458536)])), packed_ptr[0]);
    int col_28 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 224);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[113], acc_ptr[112]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524064)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[115], acc_ptr[114]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458528)])), packed_ptr[0]);
    int col_29 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 232);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[117], acc_ptr[116]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524056)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[119], acc_ptr[118]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458520)])), packed_ptr[0]);
    int col_30 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 240);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[121], acc_ptr[120]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524048)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[123], acc_ptr[122]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458512)])), packed_ptr[0]);
    int col_31 = ((((((int)blockIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 3) * 2)) + 248);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[125], acc_ptr[124]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 524040)])), packed_ptr[0]);
    tvm_builtin_ptx_cvt_bf16x2_f32_rn_bf16x2_f32(packed_ptr[0], acc_ptr[127], acc_ptr[126]);
    tvm_builtin_ptx_st_global_b32((&(C_ptr[(((((((((int)blockIdx.x) & 63) * 1048576) + ((((int)threadIdx.x) >> 5) * 131072)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + ((((int)blockIdx.x) >> 6) * 256)) + ((((int)threadIdx.x) & 3) * 2)) - 458504)])), packed_ptr[0]);
  }
  tvm_builtin_cuda_cluster_sync();
}

