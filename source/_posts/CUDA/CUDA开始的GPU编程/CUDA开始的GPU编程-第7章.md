---
title: CUDA开始的GPU编程-第七章
date: 2024-10-31
categories: 
  - GPU编程
tags: [CUDA, GPU编程]
---

## 第七章：原子操作

### 经典案例：数组求和

如何并行地对数组进行求和操作？

首先让我们试着用串行的思路来解题。

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "CudaAllocator.h"
#include "helper_cuda.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    sum[0] += arr[i];
  }
}

int main() {
  int n = 65536;
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);

  for (int i = 0; i < n; i++) {
    arr[i] = std::rand() % 4;
  }

  TICK(parallel_sum);
  parallel_sum<<<n / 128, 128>>>(sum.data(), arr.data(),
                                 n);  // 对数组arr求和，结果保存在sum[0]
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_sum);

  printf("result: %d\n", sum[0]);

  return 0;
}
// parallel_sum: 0.931437s
// result: 3
```

因为 __global__ 函数不能返回值，只能通过指针。因此我们先分配一个大小为 1 的 sum 数组，用来存储数组元素的和。这样我们同步之后就可以通过 sum[0] 看到求和的结果了。

可是算出来的结果却明显不对，为什么？

在并行计算中，`sum[0] += arr[i]` 被拆解成了以下四步操作：

```cpp
sum[0] += arr[i];
```

- 读取 `sum[0]` 到寄存器 A
- 读取 `arr[i]` 到寄存器 B
- 将寄存器 A 和寄存器 B 的值相加
- 写回寄存器 A 到 `sum[0]`

问题在于，如果两个线程同时访问 `sum[0]`，会出现竞争条件。例如：

- 线程 0：读取 `sum[0]` 到寄存器 A（A = 0）
- 线程 1：读取 `sum[0]` 到寄存器 A（A = 0）
- 线程 0：读取 `arr[0]` 到寄存器 B（B = arr[0]）
- 线程 1：读取 `arr[1]` 到寄存器 B（B = arr[1]）
- 线程 0：将寄存器 A 加上寄存器 B（A = arr[0]）
- 线程 1：将寄存器 A 加上寄存器 B（A = arr[1]）
- 线程 0：将寄存器 A 写回到 `sum[0]`（`sum[0] = arr[0]`）
- 线程 1：将寄存器 A 写回到 `sum[0]`（`sum[0] = arr[1]`）

最终，`sum[0]` 的值会变成 `arr[1]`，而不是期望的 `arr[0] + arr[1]`。这样就导致了错误的结果。

### 解决方案：使用原子操作

所以，熟悉 CPU 上并行编程的同学们可能就明白了，要用 atomic 对吧！

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "CudaAllocator.h"
#include "helper_cuda.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&sum[0], arr[i]);
  }
}

int main() {
  int n = 65536;
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);

  for (int i = 0; i < n; i++) {
    arr[i] = std::rand() % 4;
  }

  TICK(parallel_sum);
  parallel_sum<<<n / 128, 128>>>(sum.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_sum);

  printf("result: %d\n", sum[0]);

  return 0;
}
// parallel_sum: 0.383628s
// result: 98229
```

原子操作的功能就是保证读取/加法/写回三个操作，不会有另一个线程来打扰。

CUDA 也提供了这种函数，即 atomicAdd。效果和 += 一样，不过是原子的。他的第一个参数是个指针，指向要修改的地址。第二个参数是要增加多少。也就是说：

`atomicAdd(dst, src)` 和 `*dst += src` 作用相同，只不过前者是原子操作，能够避免竞争。

### atomicAdd：会返回旧值（划重点！）

`atomicAdd` 会返回旧值——依靠这个特性可以实现一些很有意思的操作

- 一个常见的需求是在并行计算中向一个数组中追加元素。如何在多个线程并行执行时确保每个线程都能够正确地将元素追加到数组的末尾而不会发生冲突呢？

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "CudaAllocator.h"
#include "helper_cuda.h"
#include "ticktock.h"

__global__ void parallel_filter(int *sum, int *res, int const *arr, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    if (arr[i] >= 2) {	// 手动实现一个过滤器功能
      int loc = atomicAdd(&sum[0], 1);
      res[loc] = arr[i];
    }
  }
}

int main() {
  int n = 1 << 24;
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);
  std::vector<int, CudaAllocator<int>> res(n);

  for (int i = 0; i < n; i++) {
    arr[i] = std::rand() % 4;
  }

  TICK(parallel_filter);
  parallel_filter<<<n / 4096, 512>>>(sum.data(), res.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_filter);

  for (int i = 0; i < sum[0]; i++) {
    if (res[i] < 2) {
      printf("Wrong At %d\n", i);
      return -1;
    }
  }

  printf("All Correct!\n");
  return 0;
}
// parallel_filter: 0.848773s
// All Correct!
```

我们有一个全局数组 `res` 和一个用于记录数组当前大小的变量 `sum`，我们希望利用 `atomicAdd` 实现类似 `push_back` 的操作：

1. 每当线程需要将数据插入到 `res` 中时，它首先调用 `atomicAdd` 来获取当前数组的大小，并同时将数组的大小加 1。
2. 然后，线程根据 `atomicAdd` 返回的旧值将数据写入到 `res[old]` 位置，确保每个线程都能在一个唯一的位置上写入数据。

这种方法通过 `atomicAdd` 原子地更新 `sum`，保证了不同线程不会争抢同一个位置，从而避免了数据竞争。

`old = atomicAdd(dst, src)` 其实相当于：

```cpp
old = *dst;
*dst += src;
```

利用这一点可以实现往一个全局的数组 res 里追加数据的效果（push_back），其中 sum 起到了记录当前数组大小的作用。

### 其它原子操作

| 原子操作              | 描述                         |
| --------------------- | ---------------------------- |
| `atomicAdd(dst, src)` | `*dst += src`                |
| `atomicSub(dst, src)` | `*dst -= src`                |
| `atomicOr(dst, src)`  | `*dst                        |
| `atomicAnd(dst, src)` | `*dst &= src`                |
| `atomicXor(dst, src)` | `*dst ^= src`                |
| `atomicMax(dst, src)` | `*dst = std::max(*dst, src)` |
| `atomicMin(dst, src)` | `*dst = std::min(*dst, src)` |

当然，他们也都会返回旧值（如果需要的话）。

### atomicExch：原子交换操作，返回旧值

除了带有运算的原子操作，`atomicExch` 是一种仅进行写入操作而不进行读取的原子操作。

`old = atomicExch(dst, src)` 等同于：

```cpp
old = *dst;
*dst = src;
```

说明：`Exch` 是 `exchange` 的缩写，对应于 `std::atomic` 中的 `exchange` 函数。

### atomicCAS：原子比较与交换的妙用

#### 原子判断是否相等，相等则写入并返回旧值

`old = atomicCAS(dst, cmp, src)` 实现的操作类似于：

```cpp
old = *dst;
if (old == cmp)
    *dst = src;
```

为什么需要这种复杂的原子指令？

#### 实现自定义原子操作

`atomicCAS` 的强大之处在于它能实现 CUDA 未直接提供的任意原子读-修改-写操作。比如，使用 `atomicCAS` 可以模拟整数的原子加法，效果与 `atomicAdd` 相同：

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "CudaAllocator.h"
#include "helper_cuda.h"
#include "ticktock.h"

__device__ __inline__ int my_atomic_add(int *dst, int src) {
  int old = *dst, expect;
  do {
    expect = old;
    old = atomicCAS(dst, expect, expect + src);  // 使用 atomicCAS 实现原子加法
  } while (expect != old);
  return old;
}

__global__ void parallel_sum(int *sum, int const *arr, int n) {
  int local_sum = 0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; 
       i += blockDim.x * gridDim.x) {
    local_sum += arr[i];
  }
  my_atomic_add(&sum[0], local_sum);  // 调用自定义的原子加法
}

int main() {
  int n = 65536;
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);

  for (int i = 0; i < n; i++) {
    arr[i] = std::rand() % 4;
  }

  TICK(parallel_sum);
  parallel_sum<<<n / 4096, 512>>>(sum.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_sum);

  printf("result: %d\n", sum[0]);

  return 0;
}
// parallel_sum: 8.323379s
// result: 98229
```

此外，如果将 `expect + src` 替换为 `expect * src`，就能实现原子乘法（`atomicMul`），尽管 CUDA 原生并不提供这个功能，我们仍然可以通过 `atomicCAS` 来实现：

```cpp
__device__ __inline__ int my_atomic_add(int *dst, int src) {
  int old = *dst, expect;
  do {
    expect = old;
    old = atomicCAS(dst, expect, expect * src);  // 使用 atomicCAS 实现原子乘法
  } while (expect != old);
  return old;
}
```

值得一提的是，在某些较老版本的 CUDA 中，`atomicAdd` 不支持 `float` 类型。此时可以通过 `atomicCAS` 配合按位转换函数（`__float_as_int` 和 `__int_as_float`）来实现浮点数的原子加法：

```cpp
__device__ __inline__ int float_atomic_add(float *dst, float src) {
  int old = __float_as_int(*dst), expect;
  do {
    expect = old;
    old = atomicCAS((int *)dst, expect,
                    __float_as_int(__int_as_float(expect) + src));  // 原子浮点加法
  } while (expect != old);
  return old;
}
```

当然，CUDA 11 已原生支持 `atomicAdd` 对 `float` 类型的支持，此处仅作为展示 `atomicCAS` 灵活性的案例。

**注意：atomicCAS非常非常影响性能，如果CUDA提供了相应的操作，就不要使用atomicCAS去模拟！！！**

### 原子操作的性能影响

原子操作的主要问题在于它们需要保证同一时刻只有一个线程可以修改某个内存地址。当多个线程尝试同时修改同一个地址时，操作会像“排队”一样，一个线程修改完后另一个线程才能进行，导致性能显著下降。

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "CudaAllocator.h"
#include "helper_cuda.h"
#include "ticktock.h"

__global__ void parallel_filter(int *sum, int *res, int const *arr, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    if (arr[i] >= 2) {
      int loc = atomicAdd(&sum[0], 1);  // 使用原子加法
      res[loc] = arr[i];  // 将符合条件的元素写入结果数组
    }
  }
}

int main() {
  int n = 1 << 24;  // 2^24个元素
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);
  std::vector<int, CudaAllocator<int>> res(n);

  for (int i = 0; i < n; i++) {
    arr[i] = std::rand() % 4;
  }

  TICK(parallel_filter);
  parallel_filter<<<n / 4096, 512>>>(sum.data(), res.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_filter);

  for (int i = 0; i < sum[0]; i++) {
    if (res[i] < 2) {
      printf("Wrong At %d\n", i);
      return -1;
    }
  }

  printf("All Correct!\n");
  return 0;
}
// parallel_filter: 0.318569s
// All Correct!
```

#### 性能意外地很快？

尽管数据量达到 `2^24` 个元素，理论上使用原子操作会导致性能瓶颈，但这里的代码运行速度仍然很快，几乎没有明显的性能下降。

**原因：** 这是因为 CUDA 编译器进行了智能优化。它通过某些技术（如线程间的内存访问优化、局部内存的使用等）减少了原子操作的竞争，从而提高了整体性能。

稍后我们会深入探讨 CUDA 如何通过这些优化来提升原子操作性能。

### 解决方案：线程局部变量（TLS）

为了解决原子操作导致的性能瓶颈，一种常见的优化方法是：每个线程先将计算结果累加到一个局部变量 `local_sum` 中，然后在每个线程的工作完成后，再一次性将局部变量的值通过原子操作加到全局变量 `sum` 中。

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "CudaAllocator.h"
#include "helper_cuda.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
  int local_sum = 0;  // 每个线程的局部变量
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    local_sum += arr[i];  // 将结果累加到局部变量
  }
  atomicAdd(&sum[0], local_sum);  // 一次性将局部和全局相加
}

int main() {
  int n = 1 << 24;  // 2^24个元素
  std::vector<int, CudaAllocator<int>> arr(n);
  std::vector<int, CudaAllocator<int>> sum(1);

  for (int i = 0; i < n; i++) {
    arr[i] = std::rand() % 4;
  }

  TICK(parallel_sum);
  parallel_sum<<<n / 4096, 512>>>(sum.data(), arr.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(parallel_sum);

  printf("result: %d\n", sum[0]);

  return 0;
}
// parallel_sum: 0.392188s
// result: 25172683
```

#### 优化效果

这样每个线程仅需要进行一次原子操作，而不像原来那样在每次累加时都进行原子操作，从而大大减少了原子操作的开销。

此外，为了进一步减少原子操作的次数，建议调小 `gridDim` 和 `blockDim`，使其小于数据量 `n`，这样局部变量才有足够的累积次数。例如，上面的代码将 `gridDim * blockDim` 减小了 8 倍，从而减少了原子操作的次数。

这就是 **TLS**（Thread Local Storage，线程本地存储）的基本原理：每个线程使用局部变量来暂存结果，最后才将结果进行汇总。通过这种方式，可以有效减少对全局内存的竞争和原子操作的频繁调用。