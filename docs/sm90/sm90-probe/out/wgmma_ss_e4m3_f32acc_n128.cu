
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

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
#include <cuda_fp8.h>
using fp8_e4_t = __nv_fp8_e4m3;
using fp8_e4x2_t = __nv_fp8x2_e4m3;
using fp8_e4x4_t = __nv_fp8x4_e4m3;
struct fp8_e4x8_t {
 fp8_e4_t data[8];
};
struct fp8_e4x16_t {
 fp8_e4_t data[16];
};
using fp8_e5_t = __nv_fp8_e5m2;
using fp8_e5x2_t = __nv_fp8x2_e5m2;
using fp8_e5x4_t = __nv_fp8x4_e5m2;
struct fp8_e5x8_t {
 fp8_e5_t data[8];
};
struct fp8_e5x16_t {
 fp8_e5_t data[16];
};
using fp8_e8_t = __nv_fp8_e8m0;
using fp8_e8x2_t = __nv_fp8x2_e8m0;
using fp8_e8x4_t = __nv_fp8x4_e8m0;
struct fp8_e8x8_t {
 fp8_e8_t data[8];
};
struct fp8_e8x16_t {
 fp8_e8_t data[16];
};
#endif // __CUDA_ARCH__ >= 890

__device__ __nv_fp8x2_e5m2 make___nv_fp8x2_e5m2(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y) {
    __nv_fp8x2_e5m2 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e5m2 make___nv_fp8x4_e5m2(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b, __nv_fp8_e5m2 c, __nv_fp8_e5m2 d) {
    __nv_fp8x4_e5m2 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
__device__ __nv_fp8x2_e4m3 make___nv_fp8x2_e4m3(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y) {
    __nv_fp8x2_e4m3 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e4m3 make___nv_fp8x4_e4m3(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b, __nv_fp8_e4m3 c, __nv_fp8_e4m3 d) {
    __nv_fp8x4_e4m3 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
}
__device__ __nv_fp8x2_e8m0 make___nv_fp8x2_e8m0(__nv_fp8_e8m0 x, __nv_fp8_e8m0 y) {
    __nv_fp8x2_e8m0 result;
    result.__x = (x.__x) | (y.__x << 8);
    return result;
}
__device__ __nv_fp8x4_e8m0 make___nv_fp8x4_e8m0(__nv_fp8_e8m0 a, __nv_fp8_e8m0 b, __nv_fp8_e8m0 c, __nv_fp8_e8m0 d) {
    __nv_fp8x4_e8m0 result;
    result.__x = (a.__x) | (b.__x << 8) | (c.__x << 16) | (d.__x << 24);
    return result;
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
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_wait_group_wait_group_sync_aligned_0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned() {
  asm volatile("wgmma.commit_group.sync.aligned;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_fence_fence_sync_aligned() {
  asm volatile("wgmma.fence.sync.aligned;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_wgmma_fp8_ss_mma_async_sync_aligned_m64n128k32_f32_e4m3_e4m3_1_1(float& __d0, float& __d1, float& __d2, float& __d3, float& __d4, float& __d5, float& __d6, float& __d7, float& __d8, float& __d9, float& __d10, float& __d11, float& __d12, float& __d13, float& __d14, float& __d15, float& __d16, float& __d17, float& __d18, float& __d19, float& __d20, float& __d21, float& __d22, float& __d23, float& __d24, float& __d25, float& __d26, float& __d27, float& __d28, float& __d29, float& __d30, float& __d31, float& __d32, float& __d33, float& __d34, float& __d35, float& __d36, float& __d37, float& __d38, float& __d39, float& __d40, float& __d41, float& __d42, float& __d43, float& __d44, float& __d45, float& __d46, float& __d47, float& __d48, float& __d49, float& __d50, float& __d51, float& __d52, float& __d53, float& __d54, float& __d55, float& __d56, float& __d57, float& __d58, float& __d59, float& __d60, float& __d61, float& __d62, float& __d63, uint64_t __a_desc, uint64_t __b_desc, uint32_t __scale_d) {
  asm volatile("{ .reg .pred ps0; setp.ne.b32 ps0, %66, 0; wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, ps0, 1, 1; }" : "+f"(__d0), "+f"(__d1), "+f"(__d2), "+f"(__d3), "+f"(__d4), "+f"(__d5), "+f"(__d6), "+f"(__d7), "+f"(__d8), "+f"(__d9), "+f"(__d10), "+f"(__d11), "+f"(__d12), "+f"(__d13), "+f"(__d14), "+f"(__d15), "+f"(__d16), "+f"(__d17), "+f"(__d18), "+f"(__d19), "+f"(__d20), "+f"(__d21), "+f"(__d22), "+f"(__d23), "+f"(__d24), "+f"(__d25), "+f"(__d26), "+f"(__d27), "+f"(__d28), "+f"(__d29), "+f"(__d30), "+f"(__d31), "+f"(__d32), "+f"(__d33), "+f"(__d34), "+f"(__d35), "+f"(__d36), "+f"(__d37), "+f"(__d38), "+f"(__d39), "+f"(__d40), "+f"(__d41), "+f"(__d42), "+f"(__d43), "+f"(__d44), "+f"(__d45), "+f"(__d46), "+f"(__d47), "+f"(__d48), "+f"(__d49), "+f"(__d50), "+f"(__d51), "+f"(__d52), "+f"(__d53), "+f"(__d54), "+f"(__d55), "+f"(__d56), "+f"(__d57), "+f"(__d58), "+f"(__d59), "+f"(__d60), "+f"(__d61), "+f"(__d62), "+f"(__d63) : "l"(__a_desc), "l"(__b_desc), "r"(__scale_d));
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
__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_arrive_expect_tx_arrive_expect_tx_shared_b64(uint32_t __addr, uint32_t __tx_count) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :  : "r"(__addr), "r"(__tx_count) : "memory");
}

__forceinline__ __device__ void ptx_wgmma_fence_float(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

__forceinline__ __device__ void tvm_builtin_cuda_cta_sync() {
    __syncthreads();
}
__forceinline__ __device__ void tvm_builtin_ptx_fence_proxy_proxy_async_shared__cta() {
  asm volatile("fence.proxy.async.shared::cta;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_init_init_shared_b64(uint32_t __addr, uint32_t __count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :  : "r"(__addr), "r"(__count) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes(uint32_t __dst_mem, const void* __tmap, int32_t __coords0, int32_t __coords1, uint32_t __mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" :  : "r"(__dst_mem), "l"(__tmap), "r"(__coords0), "r"(__coords1), "r"(__mbar) : "memory");
}

__forceinline__ __device__ unsigned int tvm_builtin_cuda_cvta_generic_to_shared(void* p) {
    return __cvta_generic_to_shared(p);
}
extern "C" __global__ void __launch_bounds__(128) main_kernel(const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, float* __restrict__ C_ptr);
extern "C" __global__ void __launch_bounds__(128) main_kernel(const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, float* __restrict__ C_ptr) {
  int tx = ((int)threadIdx.x);
  __shared__ alignas(1024) __nv_fp8_e4m3 A_smem_ptr[8192];
  __shared__ alignas(1024) __nv_fp8_e4m3 B_smem_ptr[16384];
  __shared__ alignas(64) uint64_t bar_ptr[1];
  alignas(64) uint64_t descA_ptr[1];
  alignas(64) uint64_t descB_ptr[1];
  alignas(64) float C_local_ptr[64];
  alignas(64) uint A_reg_ptr[4];
  if (((int)threadIdx.x) == 0) {
    tvm_builtin_ptx_mbarrier_init_init_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(bar_ptr[0]))), (uint)1);
  }
  tvm_builtin_ptx_fence_proxy_proxy_async_shared__cta();
  tvm_builtin_cuda_cta_sync();
  for (int i = 0; i < 64; ++i) {
    C_local_ptr[i] = 0x0p+0f/*0.000000e+00*/;
    ptx_wgmma_fence_float(C_local_ptr[i]);
  }
  for (int kb = 0; kb < 4; ++kb) {
    if (((int)threadIdx.x) == 0) {
      tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes(tvm_builtin_cuda_cvta_generic_to_shared(A_smem_ptr), ((void*)((unsigned long long)(&(A_map)))), (kb * 128), 0, tvm_builtin_cuda_cvta_generic_to_shared((&(bar_ptr[0]))));
      tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_mbarrier__complete_tx__bytes(tvm_builtin_cuda_cvta_generic_to_shared(B_smem_ptr), ((void*)((unsigned long long)(&(B_map)))), (kb * 128), 0, tvm_builtin_cuda_cvta_generic_to_shared((&(bar_ptr[0]))));
      tvm_builtin_ptx_mbarrier_arrive_expect_tx_arrive_expect_tx_shared_b64(tvm_builtin_cuda_cvta_generic_to_shared((&(bar_ptr[0]))), (uint)24576);
    }
    tvm_builtin_cuda_mbarrier_wait((&(bar_ptr[0])), (kb & 1));
    ptx_wgmma_encode_matrix_descriptor((&(descA_ptr[0])), A_smem_ptr, 1, 64, 3);
    ptx_wgmma_encode_matrix_descriptor((&(descB_ptr[0])), B_smem_ptr, 1, 64, 3);
    tvm_builtin_ptx_wgmma_fence_fence_sync_aligned();
    tvm_builtin_ptx_wgmma_fp8_ss_mma_async_sync_aligned_m64n128k32_f32_e4m3_e4m3_1_1(C_local_ptr[0], C_local_ptr[1], C_local_ptr[2], C_local_ptr[3], C_local_ptr[4], C_local_ptr[5], C_local_ptr[6], C_local_ptr[7], C_local_ptr[8], C_local_ptr[9], C_local_ptr[10], C_local_ptr[11], C_local_ptr[12], C_local_ptr[13], C_local_ptr[14], C_local_ptr[15], C_local_ptr[16], C_local_ptr[17], C_local_ptr[18], C_local_ptr[19], C_local_ptr[20], C_local_ptr[21], C_local_ptr[22], C_local_ptr[23], C_local_ptr[24], C_local_ptr[25], C_local_ptr[26], C_local_ptr[27], C_local_ptr[28], C_local_ptr[29], C_local_ptr[30], C_local_ptr[31], C_local_ptr[32], C_local_ptr[33], C_local_ptr[34], C_local_ptr[35], C_local_ptr[36], C_local_ptr[37], C_local_ptr[38], C_local_ptr[39], C_local_ptr[40], C_local_ptr[41], C_local_ptr[42], C_local_ptr[43], C_local_ptr[44], C_local_ptr[45], C_local_ptr[46], C_local_ptr[47], C_local_ptr[48], C_local_ptr[49], C_local_ptr[50], C_local_ptr[51], C_local_ptr[52], C_local_ptr[53], C_local_ptr[54], C_local_ptr[55], C_local_ptr[56], C_local_ptr[57], C_local_ptr[58], C_local_ptr[59], C_local_ptr[60], C_local_ptr[61], C_local_ptr[62], C_local_ptr[63], descA_ptr[0], descB_ptr[0], (uint)1);
    tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned();
    tvm_builtin_ptx_wgmma_fence_fence_sync_aligned();
    tvm_builtin_ptx_wgmma_fp8_ss_mma_async_sync_aligned_m64n128k32_f32_e4m3_e4m3_1_1(C_local_ptr[0], C_local_ptr[1], C_local_ptr[2], C_local_ptr[3], C_local_ptr[4], C_local_ptr[5], C_local_ptr[6], C_local_ptr[7], C_local_ptr[8], C_local_ptr[9], C_local_ptr[10], C_local_ptr[11], C_local_ptr[12], C_local_ptr[13], C_local_ptr[14], C_local_ptr[15], C_local_ptr[16], C_local_ptr[17], C_local_ptr[18], C_local_ptr[19], C_local_ptr[20], C_local_ptr[21], C_local_ptr[22], C_local_ptr[23], C_local_ptr[24], C_local_ptr[25], C_local_ptr[26], C_local_ptr[27], C_local_ptr[28], C_local_ptr[29], C_local_ptr[30], C_local_ptr[31], C_local_ptr[32], C_local_ptr[33], C_local_ptr[34], C_local_ptr[35], C_local_ptr[36], C_local_ptr[37], C_local_ptr[38], C_local_ptr[39], C_local_ptr[40], C_local_ptr[41], C_local_ptr[42], C_local_ptr[43], C_local_ptr[44], C_local_ptr[45], C_local_ptr[46], C_local_ptr[47], C_local_ptr[48], C_local_ptr[49], C_local_ptr[50], C_local_ptr[51], C_local_ptr[52], C_local_ptr[53], C_local_ptr[54], C_local_ptr[55], C_local_ptr[56], C_local_ptr[57], C_local_ptr[58], C_local_ptr[59], C_local_ptr[60], C_local_ptr[61], C_local_ptr[62], C_local_ptr[63], (descA_ptr[0] + (uint64_t)2ULL), (descB_ptr[0] + (uint64_t)2ULL), (uint)1);
    tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned();
    tvm_builtin_ptx_wgmma_fence_fence_sync_aligned();
    tvm_builtin_ptx_wgmma_fp8_ss_mma_async_sync_aligned_m64n128k32_f32_e4m3_e4m3_1_1(C_local_ptr[0], C_local_ptr[1], C_local_ptr[2], C_local_ptr[3], C_local_ptr[4], C_local_ptr[5], C_local_ptr[6], C_local_ptr[7], C_local_ptr[8], C_local_ptr[9], C_local_ptr[10], C_local_ptr[11], C_local_ptr[12], C_local_ptr[13], C_local_ptr[14], C_local_ptr[15], C_local_ptr[16], C_local_ptr[17], C_local_ptr[18], C_local_ptr[19], C_local_ptr[20], C_local_ptr[21], C_local_ptr[22], C_local_ptr[23], C_local_ptr[24], C_local_ptr[25], C_local_ptr[26], C_local_ptr[27], C_local_ptr[28], C_local_ptr[29], C_local_ptr[30], C_local_ptr[31], C_local_ptr[32], C_local_ptr[33], C_local_ptr[34], C_local_ptr[35], C_local_ptr[36], C_local_ptr[37], C_local_ptr[38], C_local_ptr[39], C_local_ptr[40], C_local_ptr[41], C_local_ptr[42], C_local_ptr[43], C_local_ptr[44], C_local_ptr[45], C_local_ptr[46], C_local_ptr[47], C_local_ptr[48], C_local_ptr[49], C_local_ptr[50], C_local_ptr[51], C_local_ptr[52], C_local_ptr[53], C_local_ptr[54], C_local_ptr[55], C_local_ptr[56], C_local_ptr[57], C_local_ptr[58], C_local_ptr[59], C_local_ptr[60], C_local_ptr[61], C_local_ptr[62], C_local_ptr[63], (descA_ptr[0] + (uint64_t)4ULL), (descB_ptr[0] + (uint64_t)4ULL), (uint)1);
    tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned();
    tvm_builtin_ptx_wgmma_fence_fence_sync_aligned();
    tvm_builtin_ptx_wgmma_fp8_ss_mma_async_sync_aligned_m64n128k32_f32_e4m3_e4m3_1_1(C_local_ptr[0], C_local_ptr[1], C_local_ptr[2], C_local_ptr[3], C_local_ptr[4], C_local_ptr[5], C_local_ptr[6], C_local_ptr[7], C_local_ptr[8], C_local_ptr[9], C_local_ptr[10], C_local_ptr[11], C_local_ptr[12], C_local_ptr[13], C_local_ptr[14], C_local_ptr[15], C_local_ptr[16], C_local_ptr[17], C_local_ptr[18], C_local_ptr[19], C_local_ptr[20], C_local_ptr[21], C_local_ptr[22], C_local_ptr[23], C_local_ptr[24], C_local_ptr[25], C_local_ptr[26], C_local_ptr[27], C_local_ptr[28], C_local_ptr[29], C_local_ptr[30], C_local_ptr[31], C_local_ptr[32], C_local_ptr[33], C_local_ptr[34], C_local_ptr[35], C_local_ptr[36], C_local_ptr[37], C_local_ptr[38], C_local_ptr[39], C_local_ptr[40], C_local_ptr[41], C_local_ptr[42], C_local_ptr[43], C_local_ptr[44], C_local_ptr[45], C_local_ptr[46], C_local_ptr[47], C_local_ptr[48], C_local_ptr[49], C_local_ptr[50], C_local_ptr[51], C_local_ptr[52], C_local_ptr[53], C_local_ptr[54], C_local_ptr[55], C_local_ptr[56], C_local_ptr[57], C_local_ptr[58], C_local_ptr[59], C_local_ptr[60], C_local_ptr[61], C_local_ptr[62], C_local_ptr[63], (descA_ptr[0] + (uint64_t)6ULL), (descB_ptr[0] + (uint64_t)6ULL), (uint)1);
    tvm_builtin_ptx_wgmma_commit_group_commit_group_sync_aligned();
    tvm_builtin_ptx_wgmma_wait_group_wait_group_sync_aligned_0();
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      ptx_wgmma_fence_float(C_local_ptr[i_1]);
    }
    tvm_builtin_cuda_cta_sync();
  }
  int w = (((int)threadIdx.x) >> 5);
  int lane = (((int)threadIdx.x) & 31);
  int row = (((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2));
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    int col = ((i_2 * 8) + ((((int)threadIdx.x) & 3) * 2));
    C_ptr[(((((((int)threadIdx.x) >> 5) * 2048) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2))] = C_local_ptr[(i_2 * 4)];
    C_ptr[((((((((int)threadIdx.x) >> 5) * 2048) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)] = C_local_ptr[((i_2 * 4) + 1)];
    C_ptr[((((((((int)threadIdx.x) >> 5) * 2048) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1024)] = C_local_ptr[((i_2 * 4) + 2)];
    C_ptr[((((((((int)threadIdx.x) >> 5) * 2048) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (i_2 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1025)] = C_local_ptr[((i_2 * 4) + 3)];
  }
}

