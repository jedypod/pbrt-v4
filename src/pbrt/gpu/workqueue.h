// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/pbrt.h>

#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 2)
  #include <cuda/std/std/atomic>
#else
  #include <cuda/std/atomic>
#endif

namespace pbrt {

template <int NumCategories, typename WorkItem, typename QueueIndex>
class MultiWorkQueue {
  public:
    MultiWorkQueue(Allocator alloc, int maxItems) {
        for (auto &q : queues) {
            WorkItem *wi = alloc.allocate_object<WorkItem>(maxItems);
            q.items = pstd::MakeSpan(wi, maxItems);
        }
    }

    PBRT_CPU_GPU
    void Reset() {
        for (auto &q : queues)
            q.count.store(0, cuda::std::memory_order_relaxed);
    }

    PBRT_CPU_GPU
    size_t Size(int category) const {
        return queues[category].count.load(cuda::std::memory_order_relaxed);
    }

    PBRT_CPU_GPU
    QueueIndex Add(int category, WorkItem item) {
        QueueIndex offset(
            queues[category].count.fetch_add(1, cuda::std::memory_order_relaxed));
        queues[category].items[offset] = item;
        return offset;
    }

    PBRT_CPU_GPU
    WorkItem Get(int category, QueueIndex index) { return queues[category].items[index]; }

  private:
    struct ItemQueue {
        cuda::std::atomic<int> count;
        TypedIndexSpan<WorkItem, QueueIndex> items;
    };
    pstd::array<ItemQueue, NumCategories> queues;
};

template <typename WorkItem, typename QueueIndex>
class WorkQueue : private MultiWorkQueue<1, WorkItem, QueueIndex> {
  public:
    using Parent = MultiWorkQueue<1, WorkItem, QueueIndex>;

    WorkQueue(Allocator alloc, int maxItems) : Parent(alloc, maxItems) {}

    PBRT_CPU_GPU
    void Reset() { Parent::Reset(); }

    PBRT_CPU_GPU
    size_t Size() const { return Parent::Size(0); }

    PBRT_CPU_GPU
    QueueIndex Add(WorkItem item) { return Parent::Add(0, item); }

    PBRT_CPU_GPU
    WorkItem Get(QueueIndex index) { return Parent::Get(0, index); }
};

}  // namespace pbrt
