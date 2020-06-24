
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_TAGGEDPTR_H
#define PBRT_UTIL_TAGGEDPTR_H

// util/taggedptr.h*
#include <pbrt/pbrt.h>
#include <pbrt/util/check.h>
#include <pbrt/util/print.h>

#include <algorithm>
#include <string>
#include <type_traits>

namespace pbrt {

// TODO: provide an ~apply sort of function that does the iteration and calls a method?

namespace detail {

template <typename Enable, typename T, typename... Ts> struct TypeIndexHelper_impl;

template <typename T, typename... Ts>
struct TypeIndexHelper_impl<void, T, T, Ts...> {
    static constexpr size_t value = 1;
};

template <typename T, typename U, typename... Ts>
struct TypeIndexHelper_impl<typename std::enable_if_t<std::is_base_of<U, T>::value &&
                                                      !std::is_same<T, U>::value>,
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
  static const bool value =
      std::is_same<T, U>::value && IsSameType<U, Types...>::value;
};

// Define type as the type of all T in (non-empty) Types..., asserting that
// all types in Types... are the same.
template <typename... Types>
struct SameType;

template <typename T, typename... Types>
struct SameType<T, Types...> {
  typedef T type;
  static_assert(
      IsSameType<T, Types...>::value,
      "Not all types in pack are the same");
};

#if 0
// Determine the result type of applying a visitor of type V on a pointer
// to type T.
template <typename V, typename T>
struct VisitorResult1 {
    typedef std::invoke_result_t<V, T*> type;
};

// Determine the result type of applying a visitor of type V on a const pointer
// to type T.
template <typename V, typename T>
struct ConstVisitorResult1 {
    typedef std::invoke_result_t<V, const T*> type;
};

// Determine the result type of applying a visitor of type V on pointers of
// all types in Types..., asserting that the type is the same for all types
// in Types...
template <typename V, typename... Types>
struct VisitorResult {
  typedef
      typename SameType<typename VisitorResult1<V, Types>::type...>::type type;
};

// Determine the result type of applying a visitor of type V on const pointers
// of all types in Types..., asserting that the type is the same for all types
// in Types...
template <typename V, typename... Types>
struct ConstVisitorResult {
  typedef
      typename SameType<typename ConstVisitorResult1<V, Types>::type...>::type
          type;
};

template <size_t index, typename V, typename R, typename... Types>
struct ApplyVisitor1;

template <typename V, typename R, typename T, typename... Types>
struct ApplyVisitor1<1, V, R, T, Types...> {
  R operator()(size_t, V&& visitor, void* ptr) const {
    return visitor(static_cast<T*>(ptr));
  }
};

template <size_t index, typename V, typename R, typename T, typename... Types>
struct ApplyVisitor1<index, V, R, T, Types...> {
  R operator()(size_t runtimeIndex, V&& visitor, void* ptr) const {
    return runtimeIndex == 1
        ? visitor(static_cast<T*>(ptr))
        : ApplyVisitor1<index - 1, V, R, Types...>()(
              runtimeIndex - 1, std::forward<V>(visitor), ptr);
  }
};

template <size_t index, typename V, typename R, typename... Types>
struct ApplyConstVisitor1;

template <typename V, typename R, typename T, typename... Types>
struct ApplyConstVisitor1<1, V, R, T, Types...> {
  R operator()(size_t, V&& visitor, void* ptr) const {
    return visitor(static_cast<const T*>(ptr));
  }
};

template <size_t index, typename V, typename R, typename T, typename... Types>
struct ApplyConstVisitor1<index, V, R, T, Types...> {
  R operator()(size_t runtimeIndex, V&& visitor, void* ptr) const {
    return runtimeIndex == 1
        ? visitor(static_cast<const T*>(ptr))
        : ApplyConstVisitor1<index - 1, V, R, Types...>()(
              runtimeIndex - 1, std::forward<V>(visitor), ptr);
  }
};

template <typename V, typename... Types>
using ApplyVisitor = ApplyVisitor1<
    sizeof...(Types),
    V,
    typename VisitorResult<V, Types...>::type,
    Types...>;

template <typename V, typename... Types>
using ApplyConstVisitor = ApplyConstVisitor1<
    sizeof...(Types),
    V,
    typename ConstVisitorResult<V, Types...>::type,
    Types...>;

#endif

} // namespace detail

template <typename... Ts> class TaggedPointer;

class RawTaggedPointer {
public:
    RawTaggedPointer() = default;
    RawTaggedPointer(std::nullptr_t np) { bits = 0; }

    template <typename... Ts>
    RawTaggedPointer &operator=(TaggedPointer<Ts...> tp);

    operator bool() const { return bits != 0; }

private:
    template <typename... Ts> friend class TaggedPointer;

    RawTaggedPointer(uintptr_t bits)
        : bits(bits) { }

    uintptr_t bits = 0;
};

// Based on/extracted from DiscriminatedPtr in Facebook's folly library.
template <typename... Ts>
class TaggedPointer {
 public:
    TaggedPointer() = default;

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    TaggedPointer(T *ptr) {
        uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
        // Reminder: if this CHECK hits, it's likely that the class
        // involved needs an alignas(8).
        CHECK_EQ(iptr & ptrMask, iptr);
        constexpr uint16_t type = TypeIndex<T>();
        bits = iptr | ((uintptr_t)type << tagShift);
    }
    PBRT_HOST_DEVICE_INLINE
    TaggedPointer(RawTaggedPointer rawPtr) {
        bits = rawPtr.bits;
        // Will catch some cases of different typed tagged pointer being
        // passed in...
        CHECK_LT(Tag(), MaxTag());
    }

    PBRT_HOST_DEVICE_INLINE
    TaggedPointer(std::nullptr_t np) { }

    PBRT_HOST_DEVICE_INLINE
    TaggedPointer(const TaggedPointer &t) {
        bits = t.bits;
    }
    PBRT_HOST_DEVICE_INLINE
    TaggedPointer &operator=(const TaggedPointer &t) {
        bits = t.bits;
        return *this;
    }

    PBRT_HOST_DEVICE_INLINE
    operator RawTaggedPointer() const {
        return RawTaggedPointer(bits);
    }

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    bool Is() const { return Tag() == TypeIndex<T>(); }

    PBRT_HOST_DEVICE_INLINE
    explicit operator bool() const { return (bits & ptrMask) != 0; }

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    T *Cast() {
        CHECK(Is<T>());
        return reinterpret_cast<T *>(ptr());
    }
    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    const T *Cast() const {
        CHECK(Is<T>());
        return reinterpret_cast<const T *>(ptr());
    }
    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    T *CastOrNullptr() {
        if (Is<T>()) return reinterpret_cast<T *>(ptr());
        else return nullptr;
    }
    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    const T *CastOrNullptr() const {
        if (Is<T>()) return reinterpret_cast<const T *>(ptr());
        else return nullptr;
    }

    PBRT_HOST_DEVICE_INLINE
    uint16_t Tag() const {
        return uint16_t((bits & tagMask) >> tagShift);
    }
    PBRT_HOST_DEVICE_INLINE
    static constexpr uint16_t MaxTag() {
        return sizeof...(Ts);
    }

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    static constexpr uint16_t TypeIndex() {
        return uint16_t(detail::TypeIndexHelper<T, Ts...>::value);
    }

    std::string ToString() const {
        return StringPrintf("[ TaggedPointer ptr: 0x%p tag: %d ]", ptr(), Tag());
    }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }

    PBRT_HOST_DEVICE_INLINE
    void *ptr() {
        return reinterpret_cast<void *>(bits & ptrMask);
    }
    PBRT_HOST_DEVICE_INLINE
    const void *ptr() const {
        return reinterpret_cast<const void *>(bits & ptrMask);
    }

#if 0
    /**
   * Apply a visitor to this object, calling the appropriate overload for
   * the type currently stored in DiscriminatedPtr.  Throws invalid_argument
   * if the DiscriminatedPtr is empty.
   *
   * The visitor must meet the following requirements:
   *
   * - The visitor must allow invocation as a function by overloading
   *   operator(), unambiguously accepting all values of type T* (or const T*)
   *   for all T in Types...
   * - All operations of the function object on T* (or const T*) must
   *   return the same type (or a static_assert will fire).
   */
  template <typename V>
  typename detail::VisitorResult<V, Ts...>::type Apply(V&& visitor) {
    size_t n = Tag();
    if (n == 0) {
      throw std::invalid_argument("Empty TaggedPointer");
    }
    return detail::ApplyVisitor<V, Ts...>()(
        n, std::forward<V>(visitor), ptr());
  }

  template <typename V>
  typename detail::ConstVisitorResult<V, Ts...>::type Apply(
      V&& visitor) const {
    size_t n = Tag();
    if (n == 0) {
      throw std::invalid_argument("Empty TaggedPointer");
    }
    return detail::ApplyConstVisitor<V, Ts...>()(
        n, std::forward<V>(visitor), const_cast<void *>(ptr()));
  }
#endif

 private:
    friend class RawTaggedPointer;

    static_assert(sizeof(uintptr_t) == 8, "Expected uintptr_t to be 64 bits");

    static constexpr bool useLowBits = sizeof...(Ts) < 7;  // 0 used for null
    static constexpr int tagShift = useLowBits ? 0 : 48;
    static constexpr uint64_t tagMask = useLowBits ? 0x7 : (((1ull << 16) - 1) << tagShift);
    static constexpr uint64_t ptrMask = ~tagMask;

    uintptr_t bits = 0;
};

template <typename Visitor, typename... Args>
decltype(auto) apply_visitor(
    Visitor&& visitor,
    const TaggedPointer<Args...>& variant) {
  return variant.apply(std::forward<Visitor>(visitor));
}

template <typename Visitor, typename... Args>
decltype(auto) apply_visitor(
    Visitor&& visitor,
    TaggedPointer<Args...>& variant) {
  return variant.apply(std::forward<Visitor>(visitor));
}

template <typename Visitor, typename... Args>
decltype(auto) apply_visitor(
    Visitor&& visitor,
    TaggedPointer<Args...>&& variant) {
  return variant.apply(std::forward<Visitor>(visitor));
}

namespace detail {

template <typename... Ts> struct DeleteTaggedPointer;

template <> struct DeleteTaggedPointer<> {
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

} // namespace detail

template <typename... Ts>
inline RawTaggedPointer &RawTaggedPointer::operator=(TaggedPointer<Ts...> tp) {
    bits = tp.bits;
    return *this;
}

} // namespace pbrt

#endif  // PBRT_UTIL_TAGGEDPTR_H
