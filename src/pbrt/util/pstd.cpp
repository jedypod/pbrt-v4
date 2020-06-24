
#include <pbrt/util/pstd.h>

#include <pbrt/util/check.h>
#include <pbrt/util/memory.h>

namespace pstd {

namespace pmr {

memory_resource::~memory_resource() { }

class NewDeleteResource : public memory_resource {
    void *do_allocate(size_t bytes, size_t alignment) {
        void *ptr = pbrt::AllocAligned(bytes);
        CHECK_EQ(0, intptr_t(ptr) % alignment);
        return ptr;
    }

    void do_deallocate(void *p, size_t bytes, size_t alignment) {
        pbrt::FreeAligned(p);
    }

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }
};

static NewDeleteResource ndr;

memory_resource *new_delete_resource() noexcept {
    return &ndr;
}

static memory_resource *defaultMemoryResource = new_delete_resource();

memory_resource *set_default_resource(memory_resource *r) noexcept {
    memory_resource *orig = defaultMemoryResource;
    defaultMemoryResource = r;
    return orig;
}

memory_resource* get_default_resource() noexcept {
    return defaultMemoryResource;
}

} // namespace pmr

} // namespace pstd
