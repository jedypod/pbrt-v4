
#ifndef PBRT_CONTAINERS_H
#define PBRT_CONTAINERS_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <iterator>
#include <tuple>

namespace pbrt {

namespace detail {

template <size_t Index, typename Elt, typename... Elements> struct ElementOffset;

template <size_t Index, typename Elt, typename... Elements> struct ElementOffset {
    static constexpr size_t offset = sizeof(Elt) + ElementOffset<Index - 1, Elements...>::offset;
};

template <typename Elt, typename... Elements> struct ElementOffset<0, Elt, Elements...> {
    static constexpr size_t offset = 0;
};

} // namespace detail

// Partially inspired by https://github.com/Lunarsong/StructureOfArrays/blob/master/include/soa.h
template <typename... Elements>
class AoSoA {
public:
    template <int N> using TypeOfNth =
        typename std::tuple_element<N, std::tuple<Elements...>>::type;

    AoSoA() = delete;
    AoSoA(size_t n, Allocator alloc)
        : n(n), alloc(alloc) {
        allocSize = ((n + Factor - 1) / Factor) * ChunkSize;
        CHECK_GE(allocSize, n * ElementSize);
        buffer = (uint8_t *)alloc.allocate_bytes(allocSize, Alignment);
    }
    ~AoSoA() {
        alloc.deallocate_bytes(buffer, allocSize, Alignment);
    }

    AoSoA(const AoSoA &) = delete;
    AoSoA &operator=(const AoSoA &) = delete;

    PBRT_HOST_DEVICE
    size_t size() const { return n; }

    template <int Index>
    PBRT_HOST_DEVICE
    TypeOfNth<Index> &at(int offset) {
        DCHECK_LT(offset, size());

        return *ptr<Index>(offset);
    }
    template <int Index>
    PBRT_HOST_DEVICE
    const TypeOfNth<Index> &at(int offset) const {
        DCHECK_LT(offset, size());

        return *ptr<Index>(offset);
    }

    template <int Index>
    PBRT_HOST_DEVICE
    TypeOfNth<Index> *ptr(int offset) {
        DCHECK_LT(offset, size());

        using ElementType = TypeOfNth<Index>;

        // Start of the chunk.
        uint8_t *chunkPtr = buffer + (offset / Factor) * ChunkSize;
        // Start of the Factor-wide element array
        chunkPtr += detail::ElementOffset<Index, Elements...>::offset * Factor;
        // And to the element
        chunkPtr += (offset % Factor) * sizeof(ElementType);
        return ((ElementType *)chunkPtr);
    }
    template <int Index>
    PBRT_HOST_DEVICE
    const TypeOfNth<Index> *ptr(int offset) const {
        DCHECK_LT(offset, size());

        using ElementType = TypeOfNth<Index>;

        // Start of the chunk.
        uint8_t *chunkPtr = buffer + (offset / Factor) * ChunkSize;
        // Start of the Factor-wide element array
        chunkPtr += detail::ElementOffset<Index, Elements...>::offset * Factor;
        // And to the element
        chunkPtr += (offset % Factor) * sizeof(ElementType);
        return (ElementType *)chunkPtr;
    }

private:
    static constexpr int Alignment = 128;
    static constexpr int Factor = 32;
    static constexpr size_t ElementSize = sizeof(std::tuple<Elements...>);
    // Make sure each chunk starts out aligned
    static constexpr size_t ChunkSize = (Factor * ElementSize + Alignment - 1) & ~(Alignment - 1);

    uint8_t *buffer;
    size_t n, allocSize;
    Allocator alloc;
};

template <typename... Elements>
class SoA {
public:
    template <int N> using TypeOfNth =
        typename std::tuple_element<N, std::tuple<Elements...>>::type;

    SoA() = delete;
    SoA(size_t n, Allocator alloc)
        : n(n), alloc(alloc) {
        size_t allocSize = n * ElementSize;  // may be too much if there's padding
        buffer = (uint8_t *)alloc.allocate_bytes(allocSize, Alignment);
    }
    ~SoA() {
        alloc.deallocate_bytes(buffer, n * ElementSize, Alignment);
    }

    SoA(const SoA &) = delete;
    SoA &operator=(const SoA &) = delete;

    PBRT_HOST_DEVICE
    size_t size() const { return n; }

    template <int Index>
    PBRT_HOST_DEVICE_INLINE
    TypeOfNth<Index> &at(int offset) {
        return *ptr<Index>(offset);
    }
    template <int Index>
    PBRT_HOST_DEVICE_INLINE
    const TypeOfNth<Index> &at(int offset) const {
        return *ptr<Index>(offset);
    }

    template <int Index>
    PBRT_HOST_DEVICE_INLINE
    TypeOfNth<Index> *ptr(int offset) {
        DCHECK_LT(offset, size());

        using ElementType = TypeOfNth<Index>;
        uint8_t *chunkPtr = buffer + detail::ElementOffset<Index, Elements...>::offset * n;
        chunkPtr += offset * sizeof(ElementType);
        return (ElementType *)chunkPtr;
    }
    template <int Index>
    PBRT_HOST_DEVICE_INLINE
    const TypeOfNth<Index> *ptr(int offset) const {
        DCHECK_LT(offset, size());

        using ElementType = TypeOfNth<Index>;
        uint8_t *chunkPtr = buffer + detail::ElementOffset<Index, Elements...>::offset * n;
        chunkPtr += offset * sizeof(ElementType);
        return (ElementType *)chunkPtr;
    }

private:
    static constexpr size_t ElementSize = sizeof(std::tuple<Elements...>);

    static constexpr int Alignment = 128;
    uint8_t *buffer;
    size_t n;
    Allocator alloc;
};

template <typename T, int N>
class SOAArray {
public:
    SOAArray() = delete;
    SOAArray(const SOAArray &) = delete;
    SOAArray &operator=(const SOAArray &) = delete;

    SOAArray(size_t n, Allocator alloc)
        : n(n), alloc(alloc) {
        for (int i = 0; i < N; ++i)
            ptrs[i] = alloc.allocate_object<T>(n);
    }
    ~SOAArray() {
        for (int i = 0; i < N; ++i)
            alloc.deallocate_object<T>(ptrs[i], n);
    }

    PBRT_HOST_DEVICE
    std::array<T, N> at(int offset) const {
        DCHECK_LT(offset, n);
        std::array<T, N> result;
        for (int i = 0; i < N; ++i)
            result[i] = ptrs[i][offset];
        return result;
    }

    template <typename Ta, int Na> struct ArrayRef {
        PBRT_HOST_DEVICE
        void operator=(const pstd::array<Ta, Na> &a) {
            for (int i = 0; i < Na; ++i)
                *(ptrs[i]) = a[i];
        }

        PBRT_HOST_DEVICE
        operator pstd::array<Ta, Na>() const {
            pstd::array<Ta, Na> a;
            for (int i = 0; i < Na; ++i)
                a[i] = *(ptrs[i]);
            return a;
        }

        pstd::array<Ta *, Na> ptrs;
    };

    PBRT_HOST_DEVICE
    ArrayRef<T, N> at(int offset) {
        ArrayRef<T, N> ref;
        for (int i = 0; i < N; ++i)
            ref.ptrs[i] = &ptrs[i][offset];
        return ref;
    }

private:
    size_t n;
    Allocator alloc;
    pstd::array<T *, N> ptrs;
};

template <typename T, int N, class Allocator = pstd::pmr::polymorphic_allocator<T>>
class InlinedVector {
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
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    InlinedVector(const Allocator &alloc = {})
        : alloc(alloc) {}
    InlinedVector(size_t count, const T &value, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            this->alloc.template construct<T>(begin() + i, value);
        nStored = count;
    }
    InlinedVector(size_t count, const Allocator &alloc = {})
        : InlinedVector(count, T{}, alloc) { }
    InlinedVector(const InlinedVector &other, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            this->alloc.template construct<T>(begin() + i, other[i]);
        nStored = other.size();
    }
    template <class InputIt>
    InlinedVector(InputIt first, InputIt last, const Allocator &alloc = {})
        : alloc(alloc) {
        reserve(last - first);
        for (InputIt iter = first; iter != last; ++iter, ++nStored)
            this->alloc.template construct<T>(begin() + nStored, *iter);
    }
    InlinedVector(InlinedVector &&other)
        : alloc(other.alloc) {
        nStored = other.nStored;
        nAlloc = other.nAlloc;
        ptr = other.ptr;
        if (other.nStored <= N)
            for (int i = 0; i < other.nStored; ++i)
                alloc.template construct<T>(fixed + i, std::move(other.fixed[i]));
            // Leave other.nStored as is, so that the detrius left after we
            // moved out of fixed has its destructors run...
        else
            other.nStored = 0;

        other.nAlloc = 0;
        other.ptr = nullptr;
    }
    InlinedVector(InlinedVector &&other, const Allocator &alloc) {
        LOG_FATAL("TODO");

        if (alloc == other.alloc) {
            ptr = other.ptr;
            nAlloc = other.nAlloc;
            nStored = other.nStored;
            if (other.nStored <= N)
                for (int i = 0; i < other.nStored; ++i)
                    fixed[i] = std::move(other.fixed[i]);

            other.ptr = nullptr;
            other.nAlloc = other.nStored = 0;
        } else {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(begin() + i, std::move(other[i]));
            nStored = other.size();
        }
    }
    InlinedVector(std::initializer_list<T> init, const Allocator &alloc = {})
        : InlinedVector(init.begin(), init.end(), alloc) { }

    InlinedVector &operator=(const InlinedVector &other) {
        if (this == &other)
            return *this;

        clear();
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
            alloc.template construct<T>(begin() + i, other[i]);
        nStored = other.size();

        return *this;
    }
    InlinedVector &operator=(InlinedVector &&other) {
        if (this == &other)
            return *this;

        clear();
        if (alloc == other.alloc) {
            pstd::swap(ptr, other.ptr);
            pstd::swap(nAlloc, other.nAlloc);
            pstd::swap(nStored, other.nStored);
            if (nStored > 0 && ptr == nullptr) {
                for (int i = 0; i < nStored; ++i)
                    alloc.template construct<T>(fixed + i, std::move(other.fixed[i]));
                other.nStored = nStored; // so that dtors run...
            }
        } else {
            reserve(other.size());
            for (size_t i = 0; i < other.size(); ++i)
                alloc.template construct<T>(begin() + i, std::move(other[i]));
            nStored = other.size();
        }

        return *this;
    }
    InlinedVector &operator=(std::initializer_list<T> &init) {
        clear();
        reserve(init.size());
        for (const auto &value : init) {
            alloc.template construct<T>(begin() + nStored, value);
            ++nStored;
        }
        return *this;
    }

    void assign(size_type count, const T &value) {
        clear();
        reserve(count);
        for (size_t i = 0; i < count; ++i)
            alloc.template construct<T>(begin() + i, value);
        nStored = count;
    }
    template <class InputIt>
    void assign(InputIt first, InputIt last) {
        // TODO
        LOG_FATAL("TODO");
    }
    void assign(std::initializer_list<T> &init) {
        assign(init.begin(), init.end());
    }

    ~InlinedVector() {
        clear();
        alloc.deallocate_object(ptr, nAlloc);
    }

    PBRT_HOST_DEVICE_INLINE
    iterator begin() { return ptr ? ptr : fixed; }
    PBRT_HOST_DEVICE_INLINE
    iterator end() { return begin() + nStored; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator begin() const { return ptr ? ptr : fixed; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator end() const { return begin() + nStored; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator cbegin() const { return ptr ? ptr : fixed; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator cend() const { return begin() + nStored; }

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
    size_t capacity() const { return ptr ? nAlloc : N; }

    void reserve(size_t n) {
        if (capacity() >= n)
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
        return begin()[index];
    }
    PBRT_HOST_DEVICE_INLINE
    const_reference operator[](size_type index) const {
        DCHECK_LT(index, size());
        return begin()[index];
    }
    PBRT_HOST_DEVICE_INLINE
    reference front() { return *begin(); }
    PBRT_HOST_DEVICE_INLINE
    const_reference front() const { return *begin(); }
    PBRT_HOST_DEVICE_INLINE
    reference back() { return *(begin() + nStored - 1); }
    PBRT_HOST_DEVICE_INLINE
    const_reference back() const { return *(begin() + nStored - 1); }
    PBRT_HOST_DEVICE_INLINE
    pointer data() { return ptr ? ptr : fixed; }
    PBRT_HOST_DEVICE_INLINE
    const_pointer data() const { return ptr ? ptr : fixed; }

    void clear() {
        for (int i = 0; i < nStored; ++i)
            alloc.destroy(begin() + i);
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
            reserve(size() + (last - first));
            iterator pos = end(), startPos = end();
            for (auto iter = first; iter != last; ++iter, ++pos)
                alloc.template construct<T>(pos, *iter);
            nStored += last - first;
            return pos;
        } else {
            // TODO
            LOG_FATAL("TODO");
        }
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
        // TODO
        LOG_FATAL("TODO");
    }

    iterator erase(const_iterator cpos) {
        iterator pos = begin() + (cpos - begin());  // non-const iterator, thank you very much
        while (pos != end() - 1) {
            *pos = std::move(*(pos + 1));
            ++pos;
        }
        alloc.destroy(pos);
        --nStored;
        return begin() + (cpos - begin());
    }
    iterator erase(const_iterator first, const_iterator last) {
        // TODO
        LOG_FATAL("TODO");
    }

    void push_back(const T &value) {
        if (size() == capacity())
            reserve(2 * capacity());

        alloc.construct(begin() + nStored, value);
        ++nStored;
    }
    void push_back(T &&value) {
        if (size() == capacity())
            reserve(2 * capacity());

        alloc.construct(begin() + nStored, std::move(value));
        ++nStored;
    }
    void pop_back() {
        DCHECK(!empty());
        alloc.destroy(begin() + nStored - 1);
        --nStored;
    }

    void resize(size_type n) {
        if (n < size()) {
            for (size_t i = n; n < size(); ++i)
                alloc.destroy(begin() + i);
        } else if (n > size()) {
            reserve(n);
            for (size_t i = nStored; i < n; ++i)
                alloc.construct(begin() + i);
        }
        nStored = n;
    }
    void resize(size_type count, const value_type &value) {
        // TODO
        LOG_FATAL("TODO");
    }

    void swap(InlinedVector &other) {
        // TODO
        LOG_FATAL("TODO");
    }

private:
    Allocator alloc;
    // ptr non-null is discriminator for whether fixed[] is valid...
    T *ptr = nullptr;
    union {
        T fixed[N];
    };
    size_t nAlloc = 0, nStored = 0;
};


template <typename Key, typename Value, typename Hash,
          typename Allocator = pstd::pmr::polymorphic_allocator<pstd::optional<std::pair<Key, Value>>>>
class HashMap {
public:
    using TableEntry = pstd::optional<std::pair<Key, Value>>;

    class Iterator {
    public:
        PBRT_HOST_DEVICE
        Iterator &operator++() {
            while (++ptr < end && !ptr->has_value())
                ;
            return *this;
        }

        PBRT_HOST_DEVICE
        Iterator operator++(int) {
            Iterator old = *this;
            operator++();
            return old;
        }

        PBRT_HOST_DEVICE
        bool operator==(const Iterator &iter) const {
            return ptr == iter.ptr;
        }
        PBRT_HOST_DEVICE
        bool operator!=(const Iterator &iter) const {
            return ptr != iter.ptr;
        }

        PBRT_HOST_DEVICE
        std::pair<Key, Value> &operator*() { return ptr->value(); }
        PBRT_HOST_DEVICE
        const std::pair<Key, Value> &operator*() const { return ptr->value(); }

        PBRT_HOST_DEVICE
        std::pair<Key, Value> *operator->() { return &ptr->value(); }
        PBRT_HOST_DEVICE
        const std::pair<Key, Value> *operator->() const { return ptr->value(); }

    private:
        friend class HashMap;
        Iterator(TableEntry *ptr, TableEntry *end)
            : ptr(ptr), end(end) { }
        TableEntry *ptr;
        TableEntry *end;
    };

    using iterator = Iterator;
    using const_iterator = const iterator;

    HashMap(Allocator alloc)
        : table(8, alloc), alloc(alloc) { }
    HashMap(const HashMap &) = delete;
    HashMap &operator=(const HashMap &) = delete;

    void Insert(const Key &key, const Value &value) {
        size_t offset = FindOffset(key);
        if (table[offset].has_value() == false) {
            // Not there already; possibly grow.
            if (3 * ++nStored > capacity()) {
                Grow();
                offset = FindOffset(key);
            }
        }
        table[offset] = std::make_pair(key, value);
    }

    PBRT_HOST_DEVICE
    bool HasKey(const Key &key) const {
        return table[FindOffset(key)].has_value();
    }

    PBRT_HOST_DEVICE
    const Value &operator[](const Key &key) const {
        size_t offset = FindOffset(key);
        CHECK(table[offset].has_value());
        return table[offset]->second;
    }

    PBRT_HOST_DEVICE
    iterator begin() {
        Iterator iter(table.data(), table.data() + table.size());
        while (iter.ptr < iter.end && !iter.ptr->has_value())
            ++iter.ptr;
        return iter;
    }
    PBRT_HOST_DEVICE
    iterator end() {
        return Iterator(table.data() + table.size(), table.data() + table.size());
    }

    PBRT_HOST_DEVICE
    size_t size() const {
        return nStored;
    }
    PBRT_HOST_DEVICE
    size_t capacity() const {
        return table.size();
    }

    void Clear() {
        table.clear();
    }

private:
    PBRT_HOST_DEVICE
    size_t FindOffset(const Key &key) const {
        size_t baseOffset = Hash()(key) & (capacity() - 1);
        for (int nProbes = 0;; ++nProbes) {
            // Quadratic probing.
            size_t offset = (baseOffset + nProbes/2 + nProbes*nProbes/2) & (capacity() - 1);
            if (table[offset].has_value() == false || key == table[offset]->first) {
                return offset;
            }
        }
    }

    void Grow() {
        size_t currentCapacity = capacity();
        pstd::vector<TableEntry> newTable(std::max<size_t>(64, 2 * currentCapacity));

        size_t newCapacity = newTable.size();
        for (size_t i = 0; i < currentCapacity; ++i) {
            if (!table[i].has_value())
                continue;

            size_t baseOffset = Hash()(table[i]->first) & (newCapacity - 1);
            for (int nProbes = 0;; ++nProbes) {
                // Quadratic probing.
                size_t offset = (baseOffset + nProbes/2 + nProbes*nProbes/2) & (newCapacity - 1);
                if (!newTable[offset]) {
                    newTable[offset] = std::move(*table[i]);
                    break;
                }
            }
        }

        table = std::move(newTable);
    }

    pstd::vector<TableEntry> table;
    size_t nStored = 0;
    Allocator alloc;
};

} // namespace pbrt

#endif // PBRT_CONTAINERS_H
