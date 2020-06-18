// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_TAGGEDPTR_H
#define PBRT_UTIL_TAGGEDPTR_H

// util/taggedptr.h*
#include <pbrt/pbrt.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/print.h>

#include <algorithm>
#include <string>
#include <type_traits>

namespace pbrt {

namespace detail {

template <typename Enable, typename T, typename... Ts>
struct TypeIndexHelper_impl;

template <typename T, typename... Ts>
struct TypeIndexHelper_impl<void, T, T, Ts...> {
    static constexpr size_t value = 1;
};

template <typename T, typename U, typename... Ts>
struct TypeIndexHelper_impl<
    typename std::enable_if_t<std::is_base_of<U, T>::value && !std::is_same<T, U>::value>,
    T, U, Ts...> {
    static constexpr size_t value = 1;
};

template <typename T, typename U, typename... Ts>
struct TypeIndexHelper_impl<typename std::enable_if_t<!std::is_base_of<U, T>::value &&
                                                      !std::is_same<T, U>::value>,
                            T, U, Ts...> {
    static constexpr size_t value = 1 + TypeIndexHelper_impl<void, T, Ts...>::value;
};

template <typename T, typename... Ts>
class TypeIndexHelper : public TypeIndexHelper_impl<void, T, Ts...> {};

/**
 * Given a target type and a list of types, return the 1-based index of the
 * type in the list of types.  Fail to compile if the target type doesn't
 * appear in the list.
 *
 * GetIndex<int, void, char, int>::value == 3
 * GetIndex<int, void, char>::value -> fails to compile
 */
template <typename... Types>
struct GetTypeIndex;

// When recursing, we never reach the 0- or 1- template argument base case
// unless the target type is not in the list.  If the target type is in the
// list, we stop recursing when it is at the head of the remaining type
// list via the GetTypeIndex<T, T, Types...> partial specialization.
template <typename T, typename... Types>
struct GetTypeIndex<T, T, Types...> {
    static const size_t value = 1;
};

template <typename T, typename U, typename... Types>
struct GetTypeIndex<T, U, Types...> {
    static const size_t value = 1 + GetTypeIndex<T, Types...>::value;
};

// Generalize std::is_same for variable number of type arguments
template <typename... Types>
struct IsSameType;

template <>
struct IsSameType<> {
    static const bool value = true;
};

template <typename T>
struct IsSameType<T> {
    static const bool value = true;
};

template <typename T, typename U, typename... Types>
struct IsSameType<T, U, Types...> {
    static const bool value = std::is_same<T, U>::value && IsSameType<U, Types...>::value;
};

// Define type as the type of all T in (non-empty) Types..., asserting that
// all types in Types... are the same.
template <typename... Types>
struct SameType;

template <typename T, typename... Types>
struct SameType<T, Types...> {
    typedef T type;
    static_assert(IsSameType<T, Types...>::value, "Not all types in pack are the same");
};

template <int n, typename ReturnType>
struct ApplySplit;

template <typename ReturnType>
struct ApplySplit<1, ReturnType> {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU
    inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) -> ReturnType {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);
        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

template <int n, typename ReturnType>
struct ApplySplit {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU
    inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) -> ReturnType {
        constexpr int mid = n / 2;

        if (tag - 1 < mid)  // 0-based indexing here to be more traditional
            return ApplySplit<mid, ReturnType>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return ApplySplit<n - mid, ReturnType>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

template <int n, typename ReturnType>
struct ConstApplySplit;

template <typename ReturnType>
struct ConstApplySplit<1, ReturnType> {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU
    inline auto operator()(F func, const Tp tp, int tag, TypePack<Ts...> types)
        -> ReturnType {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);
        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

template <int n, typename ReturnType>
struct ConstApplySplit {
    template <typename F, typename Tp, typename... Ts>
    PBRT_CPU_GPU
    inline auto operator()(F func, const Tp tp, int tag, TypePack<Ts...> types)
        -> ReturnType {
        constexpr int mid = n / 2;

        if (tag - 1 < mid)  // 0-based indexing here to be more traditional
            return ConstApplySplit<mid, ReturnType>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return ConstApplySplit<n - mid, ReturnType>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

template <int n, typename ReturnType>
struct ApplySplitCPU;

template <typename ReturnType>
struct ApplySplitCPU<1, ReturnType> {
    template <typename F, typename Tp, typename... Ts>
    inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) -> ReturnType {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);

        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

template <int n, typename ReturnType>
struct ApplySplitCPU {
    template <typename F, typename Tp, typename... Ts>
    inline auto operator()(F func, Tp tp, int tag, TypePack<Ts...> types) -> ReturnType {
        constexpr int mid = n / 2;

        if (tag - 1 < mid)  // 0-based indexing here to be more traditional
            return ApplySplitCPU<mid, ReturnType>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return ApplySplitCPU<n - mid, ReturnType>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

template <int n, typename ReturnType>
struct ConstApplySplitCPU;

template <typename ReturnType>
struct ConstApplySplitCPU<1, ReturnType> {
    template <typename F, typename Tp, typename... Ts>
    inline auto operator()(F func, const Tp tp, int tag, TypePack<Ts...> types)
        -> ReturnType {
        DCHECK_EQ(1, tag);
        static_assert(sizeof...(Ts) == 1);

        using T = typename GetFirst<TypePack<Ts...>>::type;
        return func(tp.template Cast<T>());
    }
};

template <int n, typename ReturnType>
struct ConstApplySplitCPU {
    template <typename F, typename Tp, typename... Ts>
    inline auto operator()(F func, const Tp tp, int tag, TypePack<Ts...> types)
        -> ReturnType {
        constexpr int mid = n / 2;

        if (tag - 1 < mid)  // 0-based indexing here to be more traditional
            return ConstApplySplitCPU<mid, ReturnType>()(
                func, tp, tag, typename TakeFirstN<mid, TypePack<Ts...>>::type());
        else
            return ConstApplySplitCPU<n - mid, ReturnType>()(
                func, tp, tag - mid, typename RemoveFirstN<mid, TypePack<Ts...>>::type());
    }
};

}  // namespace detail

// Derived from DiscriminatedPtr in Facebook's folly library.
template <typename... Ts>
class TaggedPointer {
  public:
    using Types = TypePack<Ts...>;

    TaggedPointer() = default;

    template <typename T>
    PBRT_CPU_GPU TaggedPointer(T *ptr) {
        uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
        // Reminder: if this CHECK hits, it's likely that the class
        // involved needs an alignas(8).
        DCHECK_EQ(iptr & ptrMask, iptr);
        constexpr uint16_t type = TypeIndex<T>();
        bits = iptr | ((uintptr_t)type << tagShift);
    }

    PBRT_CPU_GPU
    TaggedPointer(std::nullptr_t np) {}

    PBRT_CPU_GPU
    TaggedPointer(const TaggedPointer &t) { bits = t.bits; }
    PBRT_CPU_GPU
    TaggedPointer &operator=(const TaggedPointer &t) {
        bits = t.bits;
        return *this;
    }

    template <typename T>
    PBRT_CPU_GPU bool Is() const {
        return Tag() == TypeIndex<T>();
    }

    PBRT_CPU_GPU
    explicit operator bool() const { return (bits & ptrMask) != 0; }

    PBRT_CPU_GPU
    bool operator<(const TaggedPointer &tp) const { return bits < tp.bits; }

    template <typename T>
    PBRT_CPU_GPU T *Cast() {
        DCHECK(Is<T>());
        return reinterpret_cast<T *>(ptr());
    }
    template <typename T>
    PBRT_CPU_GPU const T *Cast() const {
        DCHECK(Is<T>());
        return reinterpret_cast<const T *>(ptr());
    }
    template <typename T>
    PBRT_CPU_GPU T *CastOrNullptr() {
        if (Is<T>())
            return reinterpret_cast<T *>(ptr());
        else
            return nullptr;
    }
    template <typename T>
    PBRT_CPU_GPU const T *CastOrNullptr() const {
        if (Is<T>())
            return reinterpret_cast<const T *>(ptr());
        else
            return nullptr;
    }

    PBRT_CPU_GPU
    uint16_t Tag() const { return uint16_t((bits & tagMask) >> tagShift); }
    PBRT_CPU_GPU
    static constexpr uint16_t MaxTag() { return sizeof...(Ts); }
    PBRT_CPU_GPU
    static constexpr uint16_t NumTags() { return MaxTag() + 1; }

    template <typename T>
    PBRT_CPU_GPU static constexpr uint16_t TypeIndex() {
        return uint16_t(detail::TypeIndexHelper<T, Ts...>::value);
    }

    std::string ToString() const {
        return StringPrintf("[ TaggedPointer ptr: 0x%p tag: %d ]", ptr(), Tag());
    }

    PBRT_CPU_GPU
    bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }
    PBRT_CPU_GPU
    bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }

    PBRT_CPU_GPU
    void *ptr() { return reinterpret_cast<void *>(bits & ptrMask); }
    PBRT_CPU_GPU
    const void *ptr() const { return reinterpret_cast<const void *>(bits & ptrMask); }

    template <typename ReturnType, typename F>
    PBRT_CPU_GPU
    inline auto Apply(F func) -> ReturnType {
        DCHECK(ptr() != nullptr);
        int tag = Tag();
        constexpr int n = MaxTag();
        return detail::ApplySplit<n, ReturnType>()(func, *this, tag, Types());
    }

    template <typename ReturnType, typename F>
    PBRT_CPU_GPU
    inline auto Apply(F func) const -> ReturnType {
        DCHECK(ptr() != nullptr);
        int tag = Tag();
        constexpr int n = MaxTag();
        return detail::ConstApplySplit<n, ReturnType>()(func, *this, tag, Types());
    }

    template <typename ReturnType, typename F>
    inline auto ApplyCPU(F func) -> ReturnType {
        DCHECK(ptr() != nullptr);
        int tag = Tag();
        constexpr int n = MaxTag();
        return detail::ApplySplitCPU<n, ReturnType>()(func, *this, tag, Types());
    }

    template <typename ReturnType, typename F>
    inline auto ApplyCPU(F func) const -> ReturnType {
        DCHECK(ptr() != nullptr);
        int tag = Tag();
        constexpr int n = MaxTag();
        return detail::ConstApplySplitCPU<n, ReturnType>()(func, *this, tag, Types());
    }
  private:
    static_assert(sizeof(uintptr_t) == 8, "Expected uintptr_t to be 64 bits");

    static constexpr bool useLowBits = sizeof...(Ts) < 7;  // 0 used for null
    static constexpr int tagShift = useLowBits ? 0 : 48;
    static constexpr uint64_t tagMask =
        useLowBits ? 0x7 : (((1ull << 16) - 1) << tagShift);
    static constexpr uint64_t ptrMask = ~tagMask;

    uintptr_t bits = 0;
};

namespace detail {

template <typename... Ts>
struct DeleteTaggedPointer;

template <>
struct DeleteTaggedPointer<> {
    void operator()(uint16_t tag, void *ptr) {
        LOG_FATAL("This shouldn't happen. Tag = %d", tag);
    }
};

template <typename T, typename... Ts>
struct DeleteTaggedPointer<T, Ts...> {
    void operator()(uint16_t tag, void *ptr) {
        if (tag == 1)
            delete (T *)ptr;
        else
            DeleteTaggedPointer<Ts...>()(tag - 1, ptr);
    }
};

}  // namespace detail

}  // namespace pbrt

#endif  // PBRT_UTIL_TAGGEDPTR_H
