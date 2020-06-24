
#ifndef PSTD_H
#define PSTD_H

#include <pbrt/util/check.h>

#include <initializer_list>
#include <utility>
#include <cstddef>
#include <iterator>
#include <list>
#include <new>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <float.h>
#include <limits.h>

namespace pstd {

template <typename T>
PBRT_HOST_DEVICE_INLINE
void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

template <typename T, int N> class array {
public:
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using size_t = std::size_t;

    array() = default;
    PBRT_HOST_DEVICE_INLINE
    array(std::initializer_list<T> v) {
        // c++17? static_assert(v.size() == N, "Incorrect number of initializers provided");
        size_t i = 0;
        for (const T &val : v)
            values[i++] = val;
        for (; i < N; ++i)
            values[i] = T{};
    }

    PBRT_HOST_DEVICE_INLINE
    void fill(const T &v) {
        for (int i = 0; i < N; ++i)
            values[i] = v;
    }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const array<T, N> &a) const {
        for (int i = 0; i < N; ++i)
            if (values[i] != a.values[i])
                return false;
        return true;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const array<T, N> &a) const {
        return !(*this == a);
    }

    PBRT_HOST_DEVICE_INLINE
    iterator begin() { return values; }
    PBRT_HOST_DEVICE_INLINE
    iterator end() { return values + N; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator begin() const { return values; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator end() const { return values + N; }

    PBRT_HOST_DEVICE_INLINE
    size_t size() const { return N; }

    PBRT_HOST_DEVICE_INLINE
    T &operator[](size_t i) { return values[i]; }
    PBRT_HOST_DEVICE_INLINE
    const T &operator[](size_t i) const { return values[i]; }

    PBRT_HOST_DEVICE_INLINE
    T *data() { return values; }
    PBRT_HOST_DEVICE_INLINE
    const T *data() const { return values; }

private:
    T values[N];
};

template <typename T> class optional {
 public:
    PBRT_HOST_DEVICE_INLINE
    optional() : set(false) { }
    PBRT_HOST_DEVICE_INLINE
    optional(const T &v) : optionalValue(v), set(true) { }
    PBRT_HOST_DEVICE_INLINE
    optional(T &&v) : optionalValue(std::move(v)), set(true) { }

    PBRT_HOST_DEVICE_INLINE
    optional &operator=(const T &v) {
        optionalValue = v;
        set = true;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    optional &operator=(T &&v) {
        if (set) {
            optionalValue.~T();
            new (&optionalValue) T(v);
        } else
            optionalValue = std::move(v);

        set = true;
        return *this;
    }

    PBRT_HOST_DEVICE_INLINE
    operator bool() const { return set; }

    PBRT_HOST_DEVICE_INLINE
    T value_or(const T &alt) const { return set ? optionalValue : alt; }

    PBRT_HOST_DEVICE_INLINE
    T *operator->() {
        CHECK(set);
        return &optionalValue;
    }
    PBRT_HOST_DEVICE_INLINE
    const T *operator->() const {
        CHECK(set);
        return &optionalValue;
    }
    PBRT_HOST_DEVICE_INLINE
    T &operator*() {
        CHECK(set);
        return optionalValue;
    }
    PBRT_HOST_DEVICE_INLINE
    const T &operator*() const {
        CHECK(set);
        return optionalValue;
    }
    PBRT_HOST_DEVICE_INLINE
    T &value() {
        CHECK(set);
        return optionalValue;
    }
    PBRT_HOST_DEVICE_INLINE
    const T &value() const {
        CHECK(set);
        return optionalValue;
    }

    PBRT_HOST_DEVICE_INLINE
    bool has_value() const { return set; }

 private:
    T optionalValue;
    bool set;
};

namespace span_internal {

// Wrappers for access to container data pointers.
template <typename C>
PBRT_HOST_DEVICE_INLINE
constexpr auto GetDataImpl(C& c, char) noexcept  // NOLINT(runtime/references)
    -> decltype(c.data()) {
  return c.data();
}

#if 0
// Before C++17, string::data returns a const char* in all cases.
PBRT_HOST_DEVICE_INLINE
char* GetDataImpl(std::string& s,  // NOLINT(runtime/references)
                         int) noexcept {
  return &s[0];
}
#endif

template <typename C>
PBRT_HOST_DEVICE_INLINE
constexpr auto GetData(C& c) noexcept  // NOLINT(runtime/references)
    -> decltype(GetDataImpl(c, 0)) {
  return GetDataImpl(c, 0);
}

// Detection idioms for size() and data().
template <typename C>
using HasSize =
    std::is_integral<typename std::decay<decltype(std::declval<C&>().size())>::type>;

// We want to enable conversion from vector<T*> to span<const T* const> but
// disable conversion from vector<Derived> to span<Base>. Here we use
// the fact that U** is convertible to Q* const* if and only if Q is the same
// type or a more cv-qualified version of U.  We also decay the result type of
// data() to avoid problems with classes which have a member function data()
// which returns a reference.
template <typename T, typename C>
using HasData =
    std::is_convertible<typename std::decay<decltype(GetData(std::declval<C&>()))>::type*,
                        T* const*>;

} // namespace span_internal

template <typename T> class span {
 public:
    // Used to determine whether a Span can be constructed from a container of
    // type C.
    template <typename C>
        using EnableIfConvertibleFrom =
        typename std::enable_if_t<span_internal::HasData<T, C>::value &&
                                  span_internal::HasSize<C>::value>;

    // Used to SFINAE-enable a function when the slice elements are const.
    template <typename U>
        using EnableIfConstView =
        typename std::enable_if_t<std::is_const<T>::value, U>;

    // Used to SFINAE-enable a function when the slice elements are mutable.
    template <typename U>
        using EnableIfMutableView =
        typename std::enable_if_t<!std::is_const<T>::value, U>;

    using value_type = typename std::remove_cv<T>::type;
    using iterator = T *;
    using const_iterator = const T *;

    PBRT_HOST_DEVICE_INLINE
    span() : ptr(nullptr), n(0) { }
    PBRT_HOST_DEVICE_INLINE
    span(T *ptr, size_t n) : ptr(ptr), n(n) { }
    template <size_t N>
    PBRT_HOST_DEVICE_INLINE
    span(T (&a)[N]) : span(a, N) { }
    PBRT_HOST_DEVICE_INLINE
    span(std::initializer_list<value_type> v) : span(v.begin(), v.size()) { }

    // Explicit reference constructor for a mutable `span<T>` type. Can be
    // replaced with Makespan() to infer the type parameter.
    template <typename V, typename = EnableIfConvertibleFrom<V>,
        typename = EnableIfMutableView<V>>
    PBRT_HOST_DEVICE_INLINE
        explicit span(V& v) noexcept
        : span(v.data(), v.size()) {}

    // Implicit reference constructor for a read-only `span<const T>` type
    template <typename V, typename = EnableIfConvertibleFrom<V>,
        typename = EnableIfConstView<V>>
    PBRT_HOST_DEVICE_INLINE
        constexpr span(const V& v) noexcept
        : span(v.data(), v.size()) {}

    PBRT_HOST_DEVICE_INLINE
    iterator begin() { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    iterator end() { return ptr + n; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator begin() const { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator end() const { return ptr + n; }

    PBRT_HOST_DEVICE_INLINE
    T &operator[](size_t i) { return ptr[i]; }
    PBRT_HOST_DEVICE_INLINE
    const T &operator[](size_t i) const { return ptr[i]; }

    PBRT_HOST_DEVICE_INLINE
    size_t size() const { return n; };
    PBRT_HOST_DEVICE_INLINE
    bool empty() const { return size() == 0; }
    PBRT_HOST_DEVICE_INLINE
    T *data() { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    const T *data() const { return ptr; }

    PBRT_HOST_DEVICE_INLINE
    T front() const { return ptr[0]; }
    PBRT_HOST_DEVICE_INLINE
    T back() const { return ptr[n-1]; }

    PBRT_HOST_DEVICE_INLINE
    void remove_prefix(size_t count) {
        // assert(size() >= count);
        ptr += count;
        n -= count;
    }
    PBRT_HOST_DEVICE_INLINE
    void remove_suffix(size_t count) {
        // assert(size() > = count);
        n -= count;
    }

    PBRT_HOST_DEVICE_INLINE
    span subspan(size_t pos, size_t count) {
        size_t np = count < (size() - pos) ? count : (size() - pos);
        return span(ptr + pos, np);
    }

 private:
    T *ptr;
    size_t n;
};

template <int&... ExplicitArgumentBarrier, typename T>
PBRT_HOST_DEVICE_INLINE
constexpr span<T> MakeSpan(T* ptr, size_t size) noexcept {
  return span<T>(ptr, size);
}

template <int&... ExplicitArgumentBarrier, typename T>
PBRT_HOST_DEVICE_INLINE
span<T> MakeSpan(T* begin, T* end) noexcept {
  return span<T>(begin, end - begin);
}

template <int&... ExplicitArgumentBarrier, typename C>
PBRT_HOST_DEVICE_INLINE
constexpr auto MakeSpan(C& c) noexcept
    -> decltype(MakeSpan(span_internal::GetData(c), c.size())) {
  return MakeSpan(span_internal::GetData(c), c.size());
}

template <int&... ExplicitArgumentBarrier, typename T, size_t N>
PBRT_HOST_DEVICE_INLINE
constexpr span<T> MakeSpan(T (&array)[N]) noexcept {
  return span<T>(array, N);
}

template <int&... ExplicitArgumentBarrier, typename T>
PBRT_HOST_DEVICE_INLINE
constexpr span<const T> MakeConstSpan(T* ptr, size_t size) noexcept {
  return span<const T>(ptr, size);
}

template <int&... ExplicitArgumentBarrier, typename T>
PBRT_HOST_DEVICE_INLINE
span<const T> MakeConstSpan(T* begin, T* end) noexcept {
  return span<const T>(begin, end - begin);
}

template <int&... ExplicitArgumentBarrier, typename C>
PBRT_HOST_DEVICE_INLINE
constexpr auto MakeConstSpan(const C& c) noexcept -> decltype(MakeSpan(c)) {
  return MakeSpan(c);
}

template <int&... ExplicitArgumentBarrier, typename T, size_t N>
PBRT_HOST_DEVICE_INLINE
constexpr span<const T> MakeConstSpan(const T (&array)[N]) noexcept {
  return span<const T>(array, N);
}

// memory_resource...

namespace pmr {

class memory_resource {
    static constexpr size_t max_align = alignof(std::max_align_t);

public:
    virtual ~memory_resource();
    void *allocate(size_t bytes, size_t alignment = max_align) {
        return do_allocate(bytes, alignment);
    }
    void deallocate(void *p, size_t bytes, size_t alignment = max_align) {
        return do_deallocate(p, bytes, alignment);
    }
    bool is_equal(const memory_resource &other) const noexcept {
        return do_is_equal(other);
    }

private:
  virtual void *do_allocate(size_t bytes, size_t alignment) = 0;
  virtual void do_deallocate(void *p, size_t bytes, size_t alignment) = 0;
  virtual bool do_is_equal(const memory_resource& other) const noexcept = 0;
};

inline bool operator==(const memory_resource &a, const memory_resource &b) noexcept {
    return a.is_equal(b);
}

inline bool operator!=(const memory_resource &a, const memory_resource &b) noexcept {
    return !(a == b);
}

// TODO
struct pool_options {
    size_t max_blocks_per_chunk = 0;
    size_t largest_required_pool_block = 0;
};
class synchronized_pool_resource;
class unsynchronized_pool_resource;

// global memory resources
memory_resource* new_delete_resource() noexcept;
// TODO: memory_resource* null_memory_resource() noexcept;
memory_resource* set_default_resource(memory_resource* r) noexcept;
memory_resource* get_default_resource() noexcept;

class monotonic_buffer_resource : public memory_resource {
public:
    explicit monotonic_buffer_resource(memory_resource *upstream)
        : upstreamResource(upstream) { }
    monotonic_buffer_resource(size_t blockSize, memory_resource *upstream)
        : blockSize(blockSize), upstreamResource(upstream) { }
#if 0
    // TODO
    monotonic_buffer_resource(void *buffer, size_t buffer_size,
                              memory_resource *upstream);
#endif
    monotonic_buffer_resource()
        : monotonic_buffer_resource(get_default_resource()) {}
    explicit monotonic_buffer_resource(size_t initial_size)
        : monotonic_buffer_resource(initial_size, get_default_resource()) {}
#if 0
    // TODO
    monotonic_buffer_resource(void *buffer, size_t buffer_size)
        : monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) {}
#endif
    monotonic_buffer_resource(const monotonic_buffer_resource&) = delete;

    ~monotonic_buffer_resource() {
        release();
    }

    monotonic_buffer_resource
    operator=(const monotonic_buffer_resource&) = delete;

    void release() {
        for (const auto &block : usedBlocks)
            upstreamResource->deallocate(block.ptr, block.size);
        usedBlocks.clear();

        for (const auto &block : availableBlocks)
            upstreamResource->deallocate(block.ptr, block.size);
        availableBlocks.clear();

        upstreamResource->deallocate(currentBlock.ptr, currentBlock.size);
        currentBlock = MemoryBlock();
    }

    memory_resource* upstream_resource() const { return upstreamResource; }

protected:
    void *do_allocate(size_t bytes, size_t align) override {
        if (bytes > blockSize) {
            // We've got a big allocation; let the current block be so that
            // smaller allocations have a chance at using up more of it.
            usedBlocks.push_back(MemoryBlock{upstreamResource->allocate(bytes, align), bytes});
            return usedBlocks.back().ptr;
        }

        if ((currentBlockPos % align) != 0)
            currentBlockPos += align - (currentBlockPos % align);
        DCHECK_EQ(0, currentBlockPos % align);

        if (currentBlockPos + bytes > currentBlock.size) {
            // Add current block to _usedBlocks_ list
            if (currentBlock.size) {
                usedBlocks.push_back(currentBlock);
                currentBlock = {};
            }

            // Get new block of memory for _MemoryArena_

            // Try to get memory block from _availableBlocks_
            for (auto iter = availableBlocks.begin();
                 iter != availableBlocks.end(); ++iter) {
                if (bytes <= iter->size) {
                    currentBlock = std::move(*iter);
                    availableBlocks.erase(iter);
                    goto success;
                }
            }
            currentBlock = {upstreamResource->allocate(blockSize,alignof(std::max_align_t)),
                            blockSize};
        success:
            currentBlockPos = 0;
        }

        void *ptr = (char *)currentBlock.ptr + currentBlockPos;
        currentBlockPos += bytes;
        return ptr;
    }

    void do_deallocate(void *p, size_t bytes, size_t alignment) override {
        // no-op
    }

    bool do_is_equal(const memory_resource &other) const noexcept override {
        return this == &other;
    }

private:
    struct MemoryBlock {
        void *ptr = nullptr;
        size_t size = 0;
    };

    memory_resource *upstreamResource;
    size_t blockSize = 256 * 1024;
    MemoryBlock currentBlock;
    size_t currentBlockPos = 0;
    // TODO: should use the memory_resource for this list's allocations...
    std::list<MemoryBlock> usedBlocks, availableBlocks;
};

template<class Tp = byte> class polymorphic_allocator {
public:
    using value_type = Tp;

    polymorphic_allocator() noexcept { memoryResource = new_delete_resource(); }
    polymorphic_allocator(memory_resource *r) : memoryResource(r) { }
    polymorphic_allocator(const polymorphic_allocator &other) = default;
    template<class U>
    polymorphic_allocator(const polymorphic_allocator<U> &other) noexcept
        : memoryResource(other.resource()) { }

    polymorphic_allocator& operator=(const polymorphic_allocator &rhs) = delete;

    // member functions
    [[nodiscard]] Tp *allocate(size_t n) {
        return static_cast<Tp *>(resource()->allocate(n * sizeof(Tp), alignof(Tp)));
    }
    void deallocate(Tp *p, size_t n) {
        resource()->deallocate(p, n);
    }

    void *allocate_bytes(size_t nbytes, size_t alignment = alignof(max_align_t)) {
        return resource()->allocate(nbytes, alignment);
    }
    void deallocate_bytes(void *p, size_t nbytes, size_t alignment = alignof(std::max_align_t)) {
        return resource()->deallocate(p, nbytes, alignment);
    }
    template<class T> T *allocate_object(size_t n = 1) {
        return static_cast<T *>(allocate_bytes(n * sizeof(T), alignof(T)));
    }
    template<class T> void deallocate_object(T *p, size_t n = 1) {
        deallocate_bytes(p, n * sizeof(T), alignof(T));
    }
    template<class T, class... Args> T* new_object(Args&&... args) {
        // NOTE: this doesn't handle constructors that throw exceptions...
        T *p = allocate_object<T>();
        construct(p, std::forward<Args>(args)...);
        return p;
    }
    template<class T> void delete_object(T *p) {
        destroy(p);
        deallocate_object(p);
    }

    template<class T, class... Args>
    void construct(T *p, Args&&... args) {
        ::new ((void *)p) T(std::forward<Args>(args)...);
    }

    template<class T>
    void destroy(T *p) { p->~T(); }

    //polymorphic_allocator select_on_container_copy_construction() const;

    memory_resource *resource() const { return memoryResource; }

private:
    memory_resource *memoryResource;
};

template <class T1, class T2>
bool operator==(const polymorphic_allocator<T1> &a,
                const polymorphic_allocator<T2> &b) noexcept {
    return a.resource() == b.resource();
}

template <class T1, class T2>
bool operator!=(const polymorphic_allocator<T1> &a,
                const polymorphic_allocator<T2> &b) noexcept {
    return !(a == b);
}

} // namespace pmr


class string_view {
  public:
    using iterator = const char *;
    using const_iterator = const char *;

    string_view() : ptr(nullptr), length(0) {}
    string_view(const char *start, size_t size)
        : ptr(start), length(size) {}
    string_view(const char *start) : ptr(start), length(0) {
        while (*start) {
            ++length;
            ++start;
        }
    }
    template <typename Allocator>
    string_view(const std::basic_string<char, std::char_traits<char>, Allocator> &str) noexcept
        : string_view(str.data(), str.size()) {}

    const char *data() const { return ptr; }
    size_t size() const { return length; }
    bool empty() const { return length == 0; }

    char operator[](int index) const { return ptr[index]; }
    char back() const { return ptr[length - 1]; }

    const_iterator begin() const { return ptr; }
    const_iterator end() const { return ptr + length; }

    bool operator==(const char *str) const {
        int index;
        for (index = 0; *str; ++index, ++str) {
            if (index >= length) return false;
            if (*str != ptr[index]) return false;
        }
        return index == length;
    }
    bool operator!=(const char *str) const { return !(*this == str); }

    void remove_prefix(int n) {
        ptr += n;
        length -= n;
    }
    void remove_suffix(int n) { length -= n; }

  private:
    const char *ptr;
    size_t length;
};


#if 0

template <typename T>
using vector = std::vector<T, pmr::polymorphic_allocator<T>>;

#else

template <typename T, class Allocator = pmr::polymorphic_allocator<T>>
class vector {
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = T *;
    using const_pointer = const T *;
    using iterator = T *;
    using const_iterator = const T *;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const iterator>;

    vector(const Allocator &alloc = {})
        : alloc(alloc) {}
    vector(size_t count, const T &value, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            this->alloc.template construct<T>(ptr + i, value);
        nStored = count;
    }
    vector(size_t count, const Allocator &alloc = {})
        : vector(count, T{}, alloc) { }
    vector(const vector &other, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            this->alloc.template construct<T>(ptr + i, other[i]);
        nStored = other.size();
    }
    template <class InputIt>
    vector(InputIt first, InputIt last, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(last - first);
        size_t i = 0;
        for (InputIt iter = first; iter != last; ++iter, ++i)
            this->alloc.template construct<T>(ptr + i, *iter);
        nStored = nAlloc;
    }
    vector(vector &&other)
        : alloc(other.alloc) {
        nStored = other.nStored;
        nAlloc = other.nAlloc;
        ptr = other.ptr;

        other.nStored = other.nAlloc = 0;
        other.ptr = nullptr;
    }
    vector(vector &&other, const Allocator &alloc) {
        if (alloc == other.alloc) {
            ptr = other.ptr;
            nAlloc = other.nAlloc;
            nStored = other.nStored;

            other.ptr = nullptr;
            other.nAlloc = other.nStored = 0;
        } else {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(ptr + i, std::move(other[i]));
            nStored = other.size();
        }
    }
    vector(std::initializer_list<T> init, const Allocator &alloc = {})
        : vector(init.begin(), init.end(), alloc) { }

    vector &operator=(const vector &other) {
        if (this == &other)
            return *this;

        clear();
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            alloc.template construct<T>(ptr + i, other[i]);
        nStored = other.size();

        return *this;
    }
    vector &operator=(vector &&other) {
        if (this == &other)
            return *this;

        if (alloc == other.alloc) {
            pstd::swap(ptr, other.ptr);
            pstd::swap(nAlloc, other.nAlloc);
            pstd::swap(nStored, other.nStored);
        } else {
            clear();
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(ptr + i, std::move(other[i]));
            nStored = other.size();
        }

        return *this;
    }
    vector &operator=(std::initializer_list<T> &init) {
        reserve(init.size());
        clear();
        iterator iter = begin();
        for (const auto &value : init) {
            *iter = value;
            ++iter;
        }
        return *this;
    }

    void assign(size_type count, const T &value) {
        clear();
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            push_back(value);
    }
    template <class InputIt>
    void assign(InputIt first, InputIt last) {
        LOG_FATAL("TODO");
        // TODO
    }
    void assign(std::initializer_list<T> &init) {
        assign(init.begin(), init.end());
    }

    ~vector() {
        clear();
        alloc.deallocate_object(ptr, nAlloc);
    }

    PBRT_HOST_DEVICE_INLINE
    iterator begin() { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    iterator end() { return ptr + nStored; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator begin() const { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator end() const { return ptr + nStored; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator cbegin() const { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator cend() const { return ptr + nStored; }

    PBRT_HOST_DEVICE_INLINE
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    PBRT_HOST_DEVICE_INLINE
    reverse_iterator rend() { return reverse_iterator(begin()); }
    PBRT_HOST_DEVICE_INLINE
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    PBRT_HOST_DEVICE_INLINE
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    allocator_type get_allocator() const { return alloc; }
    PBRT_HOST_DEVICE_INLINE
    size_t size() const { return nStored; }
    PBRT_HOST_DEVICE_INLINE
    bool empty() const { return size() == 0; }
    PBRT_HOST_DEVICE_INLINE
    size_t max_size() const { return (size_t)-1; }
    PBRT_HOST_DEVICE_INLINE
    size_t capacity() const { return nAlloc; }
    void reserve(size_t n) {
        if (nAlloc >= n)
            return;

        T *ra = alloc.template allocate_object<T>(n);
        for (int i = 0; i < nStored; ++i) {
            alloc.template construct<T>(ra + i, std::move(begin()[i]));
            alloc.destroy(begin() + i);
        }

        alloc.deallocate_object(ptr, nAlloc);
        nAlloc = n;
        ptr = ra;
    }
    // TODO: shrink_to_fit

    PBRT_HOST_DEVICE_INLINE
    reference operator[](size_type index) {
        DCHECK_LT(index, size());
        return ptr[index];
    }
    PBRT_HOST_DEVICE_INLINE
    const_reference operator[](size_type index) const {
        DCHECK_LT(index, size());
        return ptr[index];
    }
    PBRT_HOST_DEVICE_INLINE
    reference front() { return ptr[0]; }
    PBRT_HOST_DEVICE_INLINE
    const_reference front() const { return ptr[0]; }
    PBRT_HOST_DEVICE_INLINE
    reference back() { return ptr[nStored - 1]; }
    PBRT_HOST_DEVICE_INLINE
    const_reference back() const { return ptr[nStored - 1]; }
    PBRT_HOST_DEVICE_INLINE
    pointer data() { return ptr; }
    PBRT_HOST_DEVICE_INLINE
    const_pointer data() const { return ptr; }

    void clear() {
        for (int i = 0; i < nStored; ++i)
            alloc.destroy(&ptr[i]);
        nStored = 0;
    }

    iterator insert(const_iterator, const T &value) {
        // TODO
        LOG_FATAL("TODO");
    }
    iterator insert(const_iterator, T &&value) {
        // TODO
        LOG_FATAL("TODO");
    }
    iterator insert(const_iterator pos, size_type count, const T &value) {
        // TODO
        LOG_FATAL("TODO");
    }
    template <class InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
        if (pos == end()) {
            size_t firstOffset = size();
            for (auto iter = first; iter != last; ++iter)
                push_back(*iter);
            return begin() + firstOffset;
        }
        else
            LOG_FATAL("TODO");
    }
    iterator insert(const_iterator pos, std::initializer_list<T> init) {
        // TODO
        LOG_FATAL("TODO");
    }

    template <class... Args>
    iterator emplace(const_iterator pos, Args&&... args) {
        // TODO
        LOG_FATAL("TODO");
    }
    template <class... Args>
    void emplace_back(Args&&... args) {
        if (nAlloc == nStored)
            reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

        alloc.construct(ptr + nStored, std::forward<Args>(args)...);
        ++nStored;
    }

    iterator erase(const_iterator pos) {
        // TODO
        LOG_FATAL("TODO");
    }
    iterator erase(const_iterator first, const_iterator last) {
        // TODO
        LOG_FATAL("TODO");
    }

    void push_back(const T &value) {
        if (nAlloc == nStored)
            reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

        alloc.construct(ptr + nStored, value);
        ++nStored;
    }
    void push_back(T &&value) {
        if (nAlloc == nStored)
            reserve(nAlloc == 0 ? 4 : 2 * nAlloc);

        alloc.construct(ptr + nStored, std::move(value));
        ++nStored;
    }
    void pop_back() {
        DCHECK(!empty());
        alloc.destroy(ptr + nStored - 1);
        --nStored;
    }

    void resize(size_type n) {
        if (n < size()) {
            for (size_t i = n; n < size(); ++i)
                alloc.destroy(ptr + i);
        } else if (n > size()) {
            reserve(n);
            for (size_t i = nStored; i < n; ++i)
                alloc.construct(ptr + i);
        }
        nStored = n;
    }
    void resize(size_type count, const value_type &value) {
        // TODO
        LOG_FATAL("TODO");
    }

    void swap(vector &other) {
        // TODO
        LOG_FATAL("TODO");
    }

private:
    Allocator alloc;
    T *ptr = nullptr;
    size_t nAlloc = 0, nStored = 0;
};

#endif

#if 0

template <typename T> using unique_ptr = std::unique_ptr<T>;

#else

template <class T, class Deleter = std::default_delete<T>>
class unique_ptr {
public:
    using pointer = T *;

    unique_ptr() = default;
    unique_ptr(std::nullptr_t);
    explicit unique_ptr(pointer p) : ptr(p) { }
    unique_ptr(pointer p, Deleter deleter) : ptr(p), deleter(deleter) { }
    unique_ptr(unique_ptr &&other) {
        ptr = other.release();
    }
    template <class U, class E>
    // TODO: enable_if U is derived from T
    // TODO: what do we do with deletere??
    unique_ptr(unique_ptr<U, E> &&other) {
        ptr = other.release();
    }

    ~unique_ptr() {
        get_deleter()(get());
    }

    unique_ptr &operator=(unique_ptr &&other) {
        if (this != &other)
            pstd::swap(ptr, other.ptr);
        return *this;
    }
    unique_ptr &operator=(std::nullptr_t) {
        reset();
        return *this;
    }

    pointer release() {
        pointer ret = ptr;
        ptr = nullptr;
        return ret;
    }

    void reset(pointer p = pointer()) {
        get_deleter()(get());
        ptr = p;
    }
    void reset(std::nullptr_t) {
        get_deleter()(get());
        ptr = nullptr;
    }
    PBRT_HOST_DEVICE_INLINE
    void swap(unique_ptr &other) {
        pstd::swap(ptr, other.ptr);
    }

    PBRT_HOST_DEVICE_INLINE
    pointer get() const { return ptr; }

    Deleter &get_deleter() { return deleter; }
    const Deleter &get_deleter() const { return deleter; }

    PBRT_HOST_DEVICE_INLINE
    explicit operator bool() const { return ptr != nullptr; }

    PBRT_HOST_DEVICE_INLINE
    typename std::add_lvalue_reference<T>::type operator *() const {
        DCHECK(ptr != nullptr);
        return *ptr;
    }
    PBRT_HOST_DEVICE_INLINE
    pointer operator->() const {
        return ptr;
    }

  private:
    T *ptr = nullptr;
    Deleter deleter;
};

template<class T, class... Args>
inline unique_ptr<T> make_unique(Args&&... args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool operator==(const unique_ptr<T> &a, const unique_ptr<T> &b) {
    return a.get() == b.get();
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
void swap(unique_ptr<T> &a, unique_ptr<T> &b) {
    pstd::swap(a.ptr, b.ptr);
}

#endif

} // namespace pstd

#endif // PSTD_H
