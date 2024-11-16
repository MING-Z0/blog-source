---
title: 全面理解STL-set系列容器与迭代器分类
date: 2024-10-23
categories: 
  - C++
tags: [C++, STL, 开发语言]
---

## 迭代器的分类

| 迭代器类型     | 描述                                           | **提供的运算符重载**                | **具有此迭代器的容器**                | **相应的 C++20 concept** |
| -------------- | ---------------------------------------------- | ----------------------------------- | ------------------------------------- | ------------------------ |
| 输入迭代器     | 只读访问序列，允许单次读取每个元素             | *（可读取），!=，==，++（一次性）   | `istream_iterator`                    | `input_iterator`         |
| 输出迭代器     | 只写访问序列，允许将数据写入到序列中           | *（可写入），!=，==，++（一次性）   | `back_insert_iterator`                | `output_iterator`        |
| 前向迭代器     | 允许读写访问，支持多次读取同一元素             | *，!=，==，++                       | `forward_list`                        | `forward_iterator`       |
| 双向迭代器     | 支持前向和后向遍历                             | *，!=，==，++，--                   | `set`，`map`，`list`                  | `bidirectional_iterator` |
| 随机访问迭代器 | 支持直接访问序列中的任意元素，提供随机访问能力 | *，!=，==，++，--，+，-，+=，-=，[] | `vector`，`array`，`deque`            | `random_access_iterator` |
| 迭代器外包装   | 封装原始迭代器，增强功能（如反向遍历、过滤等） | 与所包装的迭代器保持一致            | `reverse_iterator`，`filter_iterator` | 与所包装的迭代器一致     |

- "一次性"的理解：单向性和不可逆性
- 每个元素只能被访问一次，调用`++`会使这个迭代器指向下一个元素

包含关系：前向迭代器＞双向迭代器＞随机访问迭代器

这意味着如果一个STL模板函数（比如std::find）要求迭代器是**前向迭代器**即可，那么也可以给他随机访问迭代器，因为**前向迭代器**是**随机访问迭代器**的子集。

例如，vector 和 list 都可以调用 std::find（set 则直接提供了 find 作为成员函数）

## 小技巧

打印**任意** 包含`begin`、`end`的STL 容器的黑科技

```cpp
#pragma once

#include <iostream>
#include <utility>
#include <type_traits>

namespace std {

template <class T, class = const char *>
struct __printer_test_c_str {
    using type = void;
};

template <class T>
struct __printer_test_c_str<T, decltype(std::declval<T>().c_str())> {};

template <class T, int = 0, int = 0, int = 0,
         class = decltype(std::declval<std::ostream &>() << *++std::declval<T>().begin()),
         class = decltype(std::declval<T>().begin() != std::declval<T>().end()),
         class = typename __printer_test_c_str<T>::type>
std::ostream &operator<<(std::ostream &os, T const &v) {
    os << '{';
    auto it = v.begin();
    if (it != v.end()) {
        os << *it;
        for (++it; it != v.end(); ++it)
            os << ',' << *it;
    }
    os << '}';
    return os;
}

}
```

## set容器

`set`容器是一个存储不重复元素的**集合**。

- **唯一性**：`set`中的元素是唯一的，不能重复。

- **自动排序**：`set`中的元素根据键值自动进行排序，默认使用升序排列。

- **动态大小**：`set`可以根据需要动态调整大小，自动管理内存。

### `set`和`vector`的区别

都是能存储一连串数据的容器。

- 区别1：set会自动给其中的元素**从小到大排序**，而vector会保持插入时的顺序。

- 区别2：set会把重复的元素去除，只保留一个，即**去重**。

- 区别3：vector中的元素在内存中是连续的，可以高效地按**索引**随机访问，set则不行。

- 区别4：set中的元素可以高效地按**值**查找，而 vector 则低效。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> a = {9, 8, 5, 2, 1, 1};
    cout << "vector=" << a << endl;
    set<int> b = {9, 8, 5, 2, 1, 1};
    cout << "set=" << b << endl;
    return 0;
}
// 输出：
// vector={9,8,5,2,1,1}
// set={1,2,5,8,9}	排序并去重
```

### `set`的排序：

#### string会按“字典序”排列

set 会从小到大排序，对 int 来说就是数值的大小比较。那么对字符串类型 string 要怎么排序呢？

```cpp
#include <vector>
#include <set>
#include <string>
#include "printer.h"

using namespace std;

int main() {
    vector<string> a = {"arch", "any", "zero", "Linux"};
    cout << "vector=" << a << endl;
    set<string> b = {"arch", "any", "zero", "Linux"};
    cout << "set=" << b << endl;
    return 0;
}
//输出
// vector={arch,any,zero,Linux}
// set={Linux,any,arch,zero}
```

其实 string 类定义了运算符重载 <，他会按**字典序**比较两个字符串。所谓字典序就是优先比较两者第一个字符（按 ASCII 码比较），如果相等则继续比较下一个，不相等则直接以这个比较的结果返回。如果比到末尾都相等且字符串长度一样，则视为相等。

- 警告：千万别用 set<char *> 做字符串集合。这样只会按字符串指针的地址去判断相等，而不是所指向字符串的内容。

#### 自定义排序函数

```cpp
#include <vector>
#include <set>
#include <string>
#include "printer.h"

using namespace std;

struct MyComp {
    bool operator()(string const &a, string const &b) const {
        return a < b;
    }
};

int main() {
    set<string, MyComp> b = {"arch", "any", "zero", "Linux"};
    cout << "set=" << b << endl;
    return 0;
}
//输出：
// set={Linux,any,arch,zero}
```

set 作为模板类，其实有两个模板参数：set<T, CompT>

第一个 T 是容器内元素的类型，例如 int 或 string 等。

第二个 CompT 定义了你想要的**比较函子**，set 内部会调用这个函数来决定怎么排序。

如果 CompT 不指定，默认会直接用运算符 `<` 来比较。

这里我们定义个 MyComp 作为比较函子，和默认的一样用 < 来比较，所以没有变化。

**相等怎么判断？**

恶搞一下，这里我们把比较函子 MyComp 定义成只比较字符串第一个字符 a[0] < b[0]。

```cpp
#include <vector>
#include <set>
#include <string>
#include "printer.h"

using namespace std;

struct MyComp {
    bool operator()(string const &a, string const &b) const {
        return a[0] < b[0]; //只比较第一个字符
    }
};

int main() {
    set<string, MyComp> b = {"arch", "any", "zero", "Linux"};
    cout << "set=" << b << endl;
    return 0;
}
//输出：
// set={Linux,arch,zero}
```

神奇的一幕发生了，“any” 不见了！为什么？因为去重！

为什么 set 会把 “arch” 和 “any” 视为相等的元素？明明内容都不一样？

首先搞懂 set 内部是怎么确定两个元素 a 和 b 相等的：

`!(a < b) && !(b < a)`

也就是说他 set 内部没有用到 `==` 运算符，而是调用了两次比较函子来判断的。逻辑是：

若 **a不小于b且b不小于a**，则视为**a等于b**，所以这就是为什么 set 只需要一个比较函子，不需要相等函子的原因。

所以我们这里写了 `a[0] < b[0]` 就相当于让相等条件变成了 `a[0] == b[0]`。也就是说只要第一个字符相等就视为字符串相等，所以 “arch” 和 “any” 会被视为相等的元素，从而被 set 给去重了！

- 扩展知识：其实，`map<K, T>` 无非就是个只比较 K 无视 T 的 `set<pair<K, T>>`，顺手还加了一些方便的函数，比如 `[]` 和 `at`

**小技巧：大小写不敏感的set容器**

既然已经知道了set容器进行比较的原理，我们就可以在比较函子中进行一些操作，让"linux"和"Linux"被视为相同的元素，只保存一个，即大小写不敏感。

```cpp
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include "printer.h"

using namespace std;

struct MyComp {
    bool operator()(string a, string b) const {
        // 转换为小写
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        cout << "a=" << a << ", b=" << b << endl;
        return a < b;
    }
};

int main() {
    set<string, MyComp> b = {"arch", "any", "zero", "Linux", "linUX"};
    cout << "set=" << b << endl;
    return 0;
}
//输出：
//set={any,arch,Linux,zero}
```

### `set`的迭代器

##### **set和vector迭代器的相同点：**

上一篇我们讨论了迭代器：vector 具有 `begin()` 和 `end()` 两个成员函数，他们分别返回指向数组**头部元素**和**尾部再之后一格元素**的迭代器对象。

vector 作为连续数组，他的迭代器基本等效于指针。

set 也有 **begin()** 和 **end()** 函数，他返回的迭代器对象重载了 ***** 来访问指向的地址。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    vector<int>::iterator a_it = a.begin();
    cout << "vector[0]=" << *a_it << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    set<int>::iterator b_it = b.begin();
    cout << "set[0]=" << *b_it << endl;

    return 0;
}
// 输出：
// vector[0]=1
// set[0]=1
```

##### **set和vector迭代器的不同点：**

set 的迭代器对象也重载了 **++** 为红黑树的遍历。

vector 提供了 **+** 和 **+=** 的重载，而 set 没有。这是因为 vector 中的元素在内存中是连续的，可以**随机访问**。而 set 是不连续的，所以不能随机访问，只能**顺序访问**。

set容器随机访问的开销很大，为了防止滥用，set直接放弃重载`+`运算符，因此无法通过`迭代器 + 偏移量`的方式来访问元素。

所以这里调用 b.begin() + 3，就出错了。

```cpp
set<int>::iterator b_it = b.begin() + 3;	//error: no match for 'operator+'
```

##### **多次调用`++`实现`+`同样的效果**

set 迭代器没有重载 + 运算符，因为他不是随机迭代器。

那如果我确实需要让 set 迭代器向前移动 3 格怎么办？

可以调用三次 ++ 运算，实现和 + 3 同样的效果。

```cpp
int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    vector<int>::iterator a_it = a.begin() + 3;
    cout << "vector[3]=" << *a_it << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    //set<int>::iterator b_it = b.begin() + 3;
    set<int>::iterator b_it = b.begin();
    ++b_it;
    ++b_it;
    ++b_it;
    cout << "set[3]=" << *b_it << endl;

    return 0;
}
// 输出
// vector[3]=4
// set[3]=4
```

vector 迭代器的 + n 复杂度是 O(1)。而 set 迭代器模拟出来的 + n 复杂度为 O(n)。虽然低效，但至少可以用了。

### 迭代器的帮手函数

#### 1. `std::next`：获取迭代器的下一个位置

`std::next` 函数用于获取指定迭代器向前移动指定步数后的新迭代器，而不会改变原迭代器。它适用于所有类型的迭代器，包括输入迭代器、前向迭代器、双向迭代器和随机访问迭代器。

- 作用：
  - 对于随机访问迭代器，它会直接使用 `+` 运算符。
  - 对于不支持 `+` 运算符的迭代器（如前向迭代器和双向迭代器），则会依次调用 `++` 运算符，直至移动完成。

```cpp
#include <vector>
#include <set>
#include <iostream>

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    auto a_it = std::next(a.begin(), 3);  // 相当于 a_it + 3
    cout << "vector[3] = " << *a_it << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    auto b_it = std::next(b.begin(), 3);  // 相当于 ++b_it 三次
    cout << "set[3] = " << *b_it << endl;

    return 0;
}
// 输出：
// vector[3] = 4
// set[3] = 4
```

- **实现思想**： `std::next` 判断迭代器类型，若是随机访问迭代器则直接使用 `+`，否则循环调用 `++`：

```cpp
auto next(auto it, int n = 1) {
    if (it is random_access) {
        return it + n;
    }
    while (n--) {
        ++it;
    }
    return it;
}
```

#### 2. `std::advance`：就地移动迭代器

与 `std::next` 不同，`std::advance` 是**就地修改**传入的迭代器，而不是返回一个新的迭代器。它适用于需要修改迭代器本身的场景。

- 作用：

  - 如果迭代器支持 `+=` 运算符，则直接使用该运算符。

  - 否则会像 `std::next` 一样，逐步调用 `++`。

```cpp
#include <vector>
#include <set>
#include <iostream>

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    auto a_it = a.begin();
    std::advance(a_it, 3);  // 等价于 a_it += 3
    cout << "vector[3] = " << *a_it << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    auto b_it = b.begin();
    std::advance(b_it, 3);  // 会调用三次 ++b_it
    cout << "set[3] = " << *b_it << endl;

    return 0;
}
// 输出：
// vector[3] = 4
// set[3] = 4
```

- **实现思想**： `std::advance` 会直接修改传入的迭代器，而不返回新的迭代器：

```cpp
void advance(auto &it, int n) {
    if (it is random_access) {
        it += n;
    } else {
        while (n--) {
            ++it;
        }
    }
}
```

#### 3. `next` 和 `advance` 对负数的支持

- **负数支持**：如果迭代器类型是**双向迭代器**或更高级别，`next` 和 `advance` 支持负数，表示迭代器向前移动。例如：

```cpp
auto it = std::next(iter, -3);  // 相当于 iter - 3
```

对于更简便的使用，`std::prev` 提供了向前移动的功能。

#### 4. `std::prev`：向后移动迭代器

`std::prev` 是 `std::next` 的逆操作，用于向后移动迭代器。其内部实现类似于 `std::next(it, -n)`，可以有效替代手动的负数操作。

- 实现：

  ```cpp
  auto prev(auto it, int n = 1) {
      return std::next(it, -n);
  }
  ```

#### 5. `std::distance`：计算两个迭代器之间的距离

`std::distance` 用于计算两个迭代器之间的距离。对于随机访问迭代器，它的计算是常数时间；对于其他类型的迭代器，它会逐步计算两者的差距。

- **示例**：

```cpp
#include <vector>
#include <set>
#include <iostream>

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    auto a_size = std::distance(a.begin(), a.end());  // 调用 a.end() - a.begin()
    cout << "vector size = " << a_size << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    auto b_size = std::distance(b.begin(), b.end());  // 逐步计算距离
    cout << "set size = " << b_size << endl;

    return 0;
}
// 输出：
// vector size = 7
// set size = 7
```

- **实现思想**： `std::distance` 会根据迭代器的类型，使用高效或逐步的方式计算迭代器之间的距离。

#### **注意上面所有提到的实现思想都是伪代码**

感兴趣的可以查看的源码，下面是`std::distance`的实现，它使用了一个辅助函数`std::__distance`以及`__iterator_category`来获取迭代器的类别

```cpp
template<typename _InputIterator>
inline _GLIBCXX17_CONSTEXPR
typename iterator_traits<_InputIterator>::difference_type
distance(_InputIterator __first, _InputIterator __last)
{
    // concept requirements -- taken care of in __distance
    return std::__distance(__first, __last,
                           std::__iterator_category(__first));
}

template<typename _Iter>
inline _GLIBCXX_CONSTEXPR
typename iterator_traits<_Iter>::iterator_category
__iterator_category(const _Iter&)
{
    return typename iterator_traits<_Iter>::iterator_category();
}
```

### `inset()`函数

#### 向`set`中插入元素

```cpp
pair<iterator, bool> insert(int val);
```

可以通过调用 insert 往 set 中添加一个元素。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 4, 2, 1};
    cout << "插入之前: " << b << endl;
    b.insert(3);
    cout << "插入之后: " << b << endl;

    return 0;
}
// 输出:
// 插入之前: {1, 2, 4}
// 插入之后: {1, 2, 3, 4}
```

用户无需关心插入的位置，例如插入元素 3 时，set 会自动插入到 2 和 4 之间，从而使元素总是从小到大排列。

刚刚说过 set 具有**自动去重**的功能，如果插入的元素已经在 set 中存在，则不会完成插入。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 4, 2, 1};
    cout << "插入之前: " << b << endl;
    b.insert(4);
    cout << "插入之后: " << b << endl;

    return 0;
}
// 输出:
// 插入之前: {1, 2, 4}
// 插入之后: {1, 2, 4}
```

例如往集合 {1,2,4} 中插入 4 则什么也不会发生，因为 4 已经在集合中了。

#### insert 的第一个返回值：指向插入/现有元素的迭代器

**第一个返回值是一个迭代器**，分两种情况讨论。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};
    auto res1 = b.insert(3);
    cout << "插入3成功吗：" << res1.second << endl;
    cout << "3所在的位置：" << *res1.first << endl;
    auto res2 = b.insert(3);
    cout << "再次插入3成功吗：" << res2.second << endl;
    cout << "3所在的位置：" << *res2.first << endl;

    return 0;
}
// 输出:
// 插入3成功吗：1
// 3所在的位置：3
// 再次插入3成功吗：0
// 3所在的位置：3
```

当向 set 容器添加元素成功时，该迭代器指向 set 容器新添加的元素，bool 类型的值为 true；

如果添加失败，即证明原 set 容器中已存有相同的元素，此时返回的迭代器就指向容器中相同的此元素，同时 bool 类型的值为 false。

#### insert的第二个返回值：是否插入成功

```cpp
pair<iterator, bool> insert(int val);
```

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 4, 2, 1};
    auto res4 = b.insert(4);
    cout << "插入4成功吗：" << res4.second << endl;
    auto res3 = b.insert(3);
    cout << "插入3成功吗：" << res3.second << endl;

    return 0;
}
// 输出:
// 插入4成功吗：0
// 插入3成功吗：1
```

insert 函数的返回值是一个 `pair` 类型，也就是说他同时返回了两个值。其中**第二个返回值是** **bool** **类型，指示了插入是否成功**。

若元素在 set 容器中已存有相同的元素，则插入失败，这个 bool 值为 false；如果元素在 set 中不存在，则插入成功，这个 bool 值为 true。

#### 补充内容：

- pair 类似于 python 里的元组，不过固定只能有两个元素，自从 C++11 引入了能支持任意多元素的 tuple 以来，就没 pair 什么事了……但是为了兼容 pair 还是继续存在着。pair 是个模板类，根据尖括号里你给定的类型来替换这里的 _T1 和 _T2。例如 pair<iterator, bool> 就会变成：

```cpp
struct pair {
iterator first;
bool second;
};
```

- C++17 提供了结构化绑定(structual binding)的语法，可以取出一个 POD 结构体的所有成员，pair 也不例外。

```cpp
auto [ok, it] = b.insert(3);
// 等价于
auto tmp = b.insert(3);
auto ok = tmp.first;
auto it = tmp.second;
```

```cpp
#include <vector>
#include <set>
#include <iostream>

using namespace std;

int main()
{
    set<int> b = {1, 4, 2, 1};
    auto [it1, ok1] = b.insert(3);	//使用结构化绑定语法
    cout << "插入3成功吗：" << ok1 << endl;
    cout << "3所在的位置：" << *it1 << endl;
    auto [it2, ok2] = b.insert(3);
    cout << "再次插入3成功吗：" << ok2 << endl;
    cout << "3所在的位置：" << *it2 << endl;

    return 0;
}
// 输出:
// 插入3成功吗：1
// 3所在的位置：3
// 再次插入3成功吗：0
// 3所在的位置：3
```

### `find()`函数

```cpp
iterator find(int const &val) const;
```

set 有一个 find函数。只需给定一个参数，他会寻找 set 中与之相等的元素。

如果找到，则返回指向找到元素的迭代器。

如果找不到，则返回 end() 迭代器。

刚刚说过，end() 指向的是 set 的**尾部再之后一格**元素，他指向的是一个不存在的地址，不可能有任何元素在那里！因此 end() 常被标准库用作一个标记，来表示找不到的情况。

- Python 中的 find 找不到元素时会返回 -1 来表示，也是这个思想。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 4, 2, 1};
    auto it = b.find(2);
    cout << "2所在位置：" << *it << endl;
    cout << "比2小的数：" << *prev(it) << endl;
    cout << "比2大的数：" << *next(it) << endl;

    return 0;
}
// 输出:
// 2所在位置：2
// 比2小的数：1
// 比2大的数：4
```

**判断某个元素是否存在**

因此，可以用这个写法：

`set.find(x) !=set.end()`

来判断集合 set 中是否存在元素 x。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 4, 2, 1};

    if (b.find(2) != b.end())
    {
        cout << "集合中存在2" << endl;
    }
    else
    {
        cout << "集合中没有2" << endl;
    }

    if (b.find(8) != b.end())
    {
        cout << "集合中存在8" << endl;
    }
    else
    {
        cout << "集合中没有8" << endl;
    }

    return 0;
}
// 输出:
// 集合中存在2
// 集合中没有8
```

这是个固定的写法，虽然要调用两个函数看起来好像挺麻烦，但是大家都在用。

还有一种更直观的写法：

`set.count(x) != 0`

count 返回的是一个 int 类型，表示**集合中相等元素的个数**。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};

    if (b.count(2)) {
        cout << "集合中存在2" << endl;
    } else {
        cout << "集合中没有2" << endl;
    }

    if (b.count(8)) {
        cout << "集合中存在8" << endl;
    } else {
        cout << "集合中没有8" << endl;
    }

    return 0;
}
// 输出：
// 集合中存在2
// 集合中没有8
```

等等，不是说 set 具有**去重**的功能，不会有重复的元素吗？为什么标准库让 count 计算个数而不是直接返回 bool…因为他们考虑到接口的泛用性，毕竟 multiset 就不去重。对于能去重的 set，**count只可能返回0或1**。

个数为 **0** 就说明集合中**没有**该元素。个数为 **1** 就说明集合中**存在**该元素。

因为 int 类型能隐式转换为 bool，所以 != 0 可以省略不写。

### `erase()`函数

#### 删除指定元素

以**元素值**作为参数传入

```cpp
size_t erase(int const &val);
```

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "删之前：" << b << endl;
    int num = b.erase(4);
    cout << "删之后：" << b << endl;
    cout << "删了 " << num << " 个元素" << endl;

    return 0;
}
// 输出:
// 删之前：{1, 2, 3, 4, 5}
// 删之后：{1, 2, 3, 5}
// 删了 1 个元素
```

set.erase(x) 可以删除集合中值为 x 的元素。

erase 返回一个整数，表示被他删除元素的个数。

- 个数为 **0** 就说明集合中**没有**该元素，删除失败。

- 个数为 **1** 就说明集合中**存在**该元素，删除成功。

这里的“个数”和 count 的情况很像，因为 set 中不会有重复的元素，所以 **erase** **只可能返回** **0** **或** **1**，表示是否删除成功。

erase 还**支持迭代器**作为参数。

```cpp
iterator erase(iterator pos);
```

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    b.erase(b.find(4));
    cout << "删除元素4：" << b << endl;
    b.erase(b.begin());
    cout << "删最小元素：" << b << endl;
    b.erase(std::prev(b.end()));
    cout << "删最大元素：" << b << endl;

    return 0;
}
// 输出:
// 原始集合：{1, 2, 3, 4, 5}
// 删除元素4：{1, 2, 3, 5}
// 删最小元素：{2, 3, 5}
// 删最大元素：{2, 3}
```

`set.erase(it)` 可以删除集合位于 it 处的元素。用法举例：

- `set.erase(set.find(x))` 会**删除集合中值为** **x** **的元素**，和 set.erase(x) 等价。
- `set.erase(set.begin())` 会**删除集合中最小的元素**（因为 set 具有自动排序的特性，排在最前面的元素一定是最小的那个）
- `set.erase(std::prev(set.end()))` 会**删除集合中最大的元素**（因为自动排序的特性，排在最后面的元素一定是最大的那个）

```cpp
// ！错误操作
#include <iostream>
#include <set>
#include <string>
#include "printer.h"

struct MyComp
{
    bool operator()(const std::string &lhs, const std::string &rhs) const
    {
        return lhs < rhs;
    }
};

int main()
{
    std::set<std::string, MyComp> b = {"arch", "any", "zero", "Linux", "linUX"};

    std::cout << "set = " << b << std::endl;

    // 错误的修改方式：直接使用const_cast强制转换后修改
    *const_cast<std::string *>(&(*b.find("any"))) = "zebra";

    std::cout << "修改后的set = " << b << std::endl;

    std::cout << "found = " << b.count("zebra") << std::endl;

    return 0;
}
// 输出:
// set = {Linux, any, arch, linUX, zero} 
// 修改后的set = {Linux, zebra, arch, linUX, zero} 
// found = 0
```

#### 删除指定区间（隐患操作）

`erase` 方法支持输入**两个迭代器**作为参数：

```
iterator erase(iterator first, iterator last);
```

使用 `set.erase(beg, end)` 可以删除集合中从 `beg` 到 `end` 之间的元素，包含 `beg`，不包含 `end`。即它是一个**前开后闭区间 [beg, end)**，这符合标准库的一贯设计风格。

**示例代码：**

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    b.erase(b.find(2), b.find(4));
    cout << "删除[2, 4)之间的元素：" << b << endl;

    return 0;
}
// 输出:
// 原始集合：{1, 2, 3, 4, 5}
// 删除[2, 4)之间的元素：{1, 4, 5}
```

上面的代码会**删除 set 中所有满足 2 ≤ x < 4 的元素**。由于 `set` 有自动排序的特性，删除 `2` 和 `4` 之间的元素即为删除 `2 ≤ x < 4` 的元素。

##### 隐患警告：

- **注意**：`beg` 必须在 `end` 之前，否则会导致崩溃。
- 如果集合中没有 `2`，`find(2)` 将返回 `end()`，而 `find(4)` 会返回指向 `4` 的迭代器。此时 `find(2)` 的迭代器在 `find(4)` 之后，违反了“**beg 必须在 end 之前**”的规则，可能导致程序崩溃。

**示例代码（隐患演示）：**

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {0, 1, 3, 4, 5}; // 没有2

    cout << "原始集合：" << b << endl;
    b.erase(b.find(2), b.find(4));
    cout << "删除[2, 4)之间的元素：" << b << endl;

    return 0;
}
// 输出:
// 原始集合：{0, 1, 3, 4, 5}
// free() : invalid pointer
// Aborted(core dumped)
```

##### 使用 `lower_bound()` 和 `upper_bound()` 函数

为了安全地删除指定区间的元素，可以使用以下两个函数：

```cpp
iterator lower_bound(int const &val) const;
iterator upper_bound(int const &val) const;
```

**示例代码：**

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

static void check(bool success) {
    if (!success)
        throw;
    cout << "通过测试" << endl;
}

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    check(b.find(2) == b.end());
    check(b.lower_bound(2) == b.find(3));
    check(b.upper_bound(2) == b.find(3));

    return 0;
}
// 输出:
// 原始集合：{0, 1, 3, 4, 5}
// 通过测试
// 通过测试
// 通过测试
```

- `find(x)` 找到第一个**等于** `x` 的元素。
- `lower_bound(x)` 找到第一个**大于等于** `x` 的元素。
- `upper_bound(x)` 找到第一个**大于** `x` 的元素。

当找不到时，它们都会返回 `end()`。

##### 正确从 `set` 中删除指定范围的元素

可以使用 `lower_bound()` 和 `upper_bound()` 来安全删除指定区间：

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    b.erase(b.lower_bound(2), b.upper_bound(4));
    cout << "删除[2, 4]之间的元素：" << b << endl;

    return 0;
}
// 输出:
// 原始集合：{0, 1, 3, 4, 5}
// 删除[2, 4]之间的元素：{0, 1, 5}
```

在此代码中，`a.erase(a.lower_bound(2), a.upper_bound(4));` 会**删除 set 中所有满足 2 ≤ x ≤ 4 的元素**，使得区间变为 [2, 4]，即**闭区间**。

### set增删改查总结

| **操作** | **实现方法**                           |
| -------- | -------------------------------------- |
| 增       | a.insert(x)                            |
| 删       | a.erase(x)  或者  a.erase(a.find(x))   |
| 改       | 一旦插入就无法修改，只能先删再增       |
| 查       | a.find(x)  != a.end() 或者  a.count(x) |

可以看到没有直接的修改，为什么？

正如上面提到的，set内部是需要排序的，如果你直接修改某个元素，就会破坏顺序

###  `set()`的遍历

```cpp
set<int> b= {0,1,2,3,4,5};

cout << "原始结合：" << b << endl;
for(auto it = b.begin(); it != b.end(); ++it){
    int value = *it;
    cout << value << endl;
}
```

遍历方法和vector一样，只需要背板上面的操作。我们需要讨论的是这种设计的思想是什么。

#### 从c语言指针到迭代器

上一张我们提到，迭代器就是在**模仿** C 语言指针。

回想一下 C 语言咋遍历数组的：

```c
int arr[n];

for (int i = 0; i < n; i++) {

 int value = arr[i];

}
```

循环的范围是 [0, n)。

因为这里 arr[i] 等价于 *(arr + i)，所以索性用 arr + i 作为迭代的变量，避免一次加法的开销。

```c
int arr[n];
for(int *p = arr; p < arr + n;p++){
    int value = *p;
}
```

循环的范围变成 [arr, arr + n)。

n 总是大于 0 的。p 的初值 arr 总是小于末值 arr + n，所以把 `p < arr + n` 改成 `p != arr + n` 是一样的，还高效一点。

```c
int arr[n];
for (int *p = arr; p != arr + n; p++) {
  int value = *p;
}
```

之前我们提过建议用**前置 ++** 运算符，区别我们上一篇说过（后置++先保存变量用于返回，再原地自增）。前置比较高效且符合逻辑，后置对 C 语言八股文考试有用（个人拙见）。

```c
int arr[n];
for(int *p = arr; p < arr + n;++p){
    int value = *p;
}
```

终于，到了这一步c++的**迭代器模式**也就呼之欲出了:

```cpp
set<int> arr;
for (set<int>::iterator p = arr.begin(); p != arr.end(); ++p) {
  int value = *p;
}
```

begin 和 end 返回了迭代器类，这个类具有运算符重载，使得他能模仿指针的行为，从而尽可能在不同容器之间重用算法（例如 std::find 和 std::reverse），而不必修改算法的代码本身，是 STL 库解耦思想的体现。

#### 基于范围的for循环

为了减少重复打代码的痛苦，C++11 引入了个语法糖：**基于范围的for循环**(range-based for loop)。

```cpp
for (类型 变量名 : 可迭代对象)
```

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    for (int value: b) {
        cout << value << endl;
    }

    return 0;
}
// 输出：
// 原始集合 : {0,1,2,3,4,5}
// 0
// 1
// 2
// 3
// 4
// 5
```

这种写法，无非就是刚才那一大堆代码的简写。

基于范围的 for 循环只是一个简写，他会遍历整个区间 [begin, end)。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    for (auto it = b.lower_bound(2); it != b.upper_bound(4); ++it)
    {
        int value = *it;
        cout << value << endl;
    }

    return 0;
}
// 输出:
// 原始集合：{0,1,3,4,5}
// 3
// 4
```

有时写完整版会有更大的自由度，也就是说这里的 begin 和 end 可以替换为其他位置的迭代器（如 find/lower_bound/upper_bound）

比如用 lower_bound 和 upper_bound 返回的迭代器，选择满足 `2 ≤ x ≤ 4` 的元素来打印。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    vector<int> arr(b.lower_bound(2), b.upper_bound(4));

    cout << "结果数组：" << arr << endl;

    return 0;
}
// 输出:
// 原始集合：{0, 1, 3, 4, 5}
// 结果数组：{3, 4}
```

### set 和其他容器之间的转换

我们说过 **vector 的构造函数**也能接受两个前向迭代器作为参数，set 的迭代器符合这个要求。

- `template <class ForwardIt>`

  `explicit vector(ForwardIt beg, ForwardIt end);`

- 没错，vector 的构造函数可以接受任何前向迭代器。不一定是 vector 自己的迭代器哦，任何前向迭代器！
- 而 set 是双向迭代器，覆盖了前向迭代器，满足要求。

所以可以把 set 中的一个区间（**2 ≤ x ≤ 4**）拷贝到 vector 中去。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    vector<int> arr(b.begin(), b.end());

    cout << "结果数组：" << arr << endl;

    return 0;
}
// 输出:
// 原始集合：{0, 1, 3, 4, 5}
// 结果数组：{0, 1, 3, 4, 5}
```

相当于过滤出所有 **2 ≤ x ≤ 4** 的元素了。

如果是 `vector(b.begin(),b.end())`那就毫无保留地把 set 的全部元素都拷贝进 vector 了

```cpp
int main() {
    vector<int> b = {0, 1, 3, 4, 5};

    cout << "原始数组：" << b << endl;

    set<int> arr(b.begin(), b.end());

    cout << "结果集合：" << arr << endl;

    return 0;
}
```

也可以反过来，把 vector 转成 set。

```cpp
template <class ForwardIt>
explicit set(ForwardIt beg, ForwardIt end);
```

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    vector<int> arr = {9, 8, 5, 2, 1, 1};

    cout << "原始数组：" << arr << endl;

    set<int> b(arr.begin(), arr.end());

    cout << "结果集合：" << b << endl;

    return 0;
}
// 输出:
// 原始数组：{9, 8, 5, 2, 1, 1}
// 结果集合：{1, 2, 5, 8, 9}
```

##### set的妙用：排序

*请先忘掉`std::sort()`（bushi*

把 vector 转成 set 会让元素自动**排序**和**去重**。

我们其实可以利用这一点，把 vector 转成 set 再转回 vector，这样就实现去重了。

```cpp
#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    vector<int> arr = {9, 8, 5, 2, 1, 1};

    cout << "原始数组：" << arr << endl;

    set<int> b(arr.begin(), arr.end());
    arr.assign(b.begin(), b.end());

    cout << "排序&去重后的数组：" << arr << endl;

    return 0;
}
// 输出:
// 原始数组：[9, 8, 5, 2, 1, 1]
// 排序&去重后的数组：[1, 2, 5, 8, 9]
```

当然这个操作是远不如`std::sort`效率高，只有偶尔的场景可能遇到。

### 清空set所有元素

如下，清空set有三种方式

- `b = {};`
- `b.erase(b.begin(), b.end());`
- `b.clear();`

最常用的是调用 clear 函数。

这和 vector 的 clear 函数名字是一样的，方便记忆，也是STL的设计哲学。

```cpp
#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    b.clear();
    // b = {};
    // b.erase(b.begin(), b.end());

    cout << "清空后集合：" << b << endl;

    return 0;
}
// 输出:
// 原始集合：{1, 2, 3, 4, 5}
// 清空后集合：{}
```

### set的大小（元素个数）

和 vector 一样，set 也有个 size() 函数查询其中元素个数。

```cpp
#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include "printer.h"

using namespace std;

int main()
{
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    cout << "元素个数：" << b.size() << endl;

    return 0;
}
// 输出:
// 原始集合：{1, 2, 3, 4, 5}
// 元素个数：5
```

## multiset：set的不去重版本

### 不去重的set

set 具有**自动排序，自动去重，能高效地查询**的特点。其中**去重**和数学的**集合**很像。

还有一种不会去重的版本，那就是 `multiset`，他允许重复的元素，但仍保留**自动排序，能高效地查询**的特点。

```cpp
#include <set> // multiset的头文件也是set
#include "printer.h"

using namespace std;

int main()
{
    set<int> a = {1, 1, 2, 2, 3};
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "set: " << a << endl;
    cout << "multiset: " << b << endl;

    return 0;
}
// 输出：
// set: {1 2 3}
// multiset: {1 1 2 2 3}
```

特点：因为 multiset **不会去重**，但又**自动排序**，所以其中所有相等的元素都会紧挨着，例如 {1, 2, 2, 4, 6}。

### 查找multiset中的等值区间

刚刚说了 multiset 里相等的元素都是紧挨着排列的。

所以可以用 upper_bound 和 lower_bound 函数获取所有相等值的区间，并进行相应操作。

[lower_bound, upper_bound)

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    b.erase(b.lower_bound(2), b.upper_bound(2));
    cout << "删除2以后：" << b << endl;

    return 0;
}
// 输出：
// 原始集合：{1, 1, 2, 2, 3}
// 删除2以后：{1, 1, 3}
```

对于 lower_bound 和 upper_bound 的参数相同的情况，可以用 **equal_range** 一次性求出两个边界，获得等值区间，更高效。

```cpp
pair<iterator, iterator> equal_range(int const &val) const;
```

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    auto r = b.equal_range(2);
    b.erase(r.first, r.second);
    cout << "删除2以后：" << b << endl;

    return 0;
}
// 输出：
// 原始集合：{1, 1, 2, 2, 3}
// 删除2以后：{1, 1, 3}
```

equal_range（等值区间）和调用两次 lower_bound（大于等于起点）upper_bound（大于起点）的不同：

- 当指定的值找不到时，equal_range 返回两个 end() 迭代器，代表空区间。

- lower/upper_bound 却会正常返回指向大于等于/大于指定值的迭代器。

原因：equal_range 的用途都是返回一个用来遍历的区间，两个迭代器是一起用的，不会单独用。所以为了高效，找不到等值元素会直接返回空区间。

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    auto r = b.equal_range(6);

    cout << boolalpha; // 输出bool值时，用true和false而不是1和0
    cout << (r.first == b.end()) << endl;
    cout << (r.second == b.end()) << endl;

    return 0;
}
// 输出：
// 原始集合：{1,1,2,2,3}
// true
// true
```

equal_range 返回的等值区间，可以求长度，也可以遍历。

- 对 multiset 而言遍历似乎没什么用，反正都是一堆相等的元素。

- 求长度也没什么用，可以用 count 替代，总之就是非常尴尬。

但之后说到 multimap 的时候这个函数就会很有用了。

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;

    auto r = b.equal_range(2);
    size_t n = std::distance(r.first, r.second);
    cout << "等于2的元素个数：" << n << endl;

    for (auto it = r.first; it != r.second; ++it)
    {
        int value = *it;
        cout << value << endl;
    }

    return 0;
}
// 输出：
// 原始集合：{1, 1, 2, 2, 3}
// 等于2的元素个数：2
// 2
// 2
```

### 删除 multiset 中的等值元素

erase 只有一个参数的版本，会把所有等于 2 的元素删除。

```cpp
iterator erase(int const &val) const;
```

例如：b.erase(2) 等价于b.erase(b.lower_bound(2), b.upper_bound(2));

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    multiset<int> b = {1, 1, 2, 2, 3, 2};

    cout << "原始集合：" << b << endl;
    b.erase(2);
    cout << "删除2以后：" << b << endl;

    return 0;
}
// 输出：
// 原始集合：{1, 1, 2, 2, 3, 2}
// 删除2以后：{1, 1, 3}
```

### multiset中等值元素个数

count(x) 返回 multiset 中等于 x 的元素个数（如果找不到则返回 0）。

```cpp
size_t count(int const &val) const;
```

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 1, 2, 2, 3};

    b.size();
    cout << "原始集合：" << b << endl;
    cout << "等于2的元素个数：" << b.count(2) << endl;
    cout << "等于1的元素个数：" << b.count(1) << endl;

    return 0;
}
// 输出：
// 原始集合：{1,1,1,2,2,3}
// 等于2的元素个数：2
// 等于1的元素个数：3
```

- 刚刚说 set（具有去重功能）的 count 只会返回 0 或 1。而 multiset（没有去重功能）的 count 可以返回任何 ≥ 0 的数。

### multiset 中的`find()`函数

multiset 允许多个重复的元素存在，那么 find 会返回哪一个？第一个！

```cpp
#include <set>
#include "printer.h"

using namespace std;

int main()
{
    multiset<int> b = {1, 1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    cout << boolalpha;
    cout << "集合中存在2：" << (b.find(2) != b.end()) << endl;
    cout << "集合中存在1：" << (b.find(1) != b.end()) << endl;
    cout << "第一个1在头部：" << (b.find(1) == b.begin()) << endl;

    return 0;
}
// 输出：
// 原始集合：{1,1,1,2,2,3}
// 集合中存在2：true
// 集合中存在1：true
// 第一个1在头部：true
```

find(x) 会返回第一个等于 x 的元素的迭代器。找不到也是返回 end()

### multiset增删改查操作总结

| **操作** | **实现方法**                                                 |
| :------- | :----------------------------------------------------------- |
| 增       | a.insert(x)                                                  |
| 删       | a.erase(x)  或者  a.erase(a.lower_bound(x), a.upper_bound(x)) |
| 改       | 一旦插入就无法修改，只能先删再增                             |
| 查       | a.find(x)  != a.end() 或者  a.count(x)                       |

依旧没有直接的修改操作，原因与set容器相同。

## unordered_set 无序集合

C++11 新增了一个 `unordered_set` 容器。

```cpp
#include <set>
#include <unordered_set>
#include "printer.h"

using namespace std;

int main()
{
    set<int> a = {1, 4, 2, 8, 5, 7};
    unordered_set<int> b = {1, 4, 2, 8, 5, 7};

    cout << "set: " << a << endl;
    cout << "unordered_set: " << b << endl;

    return 0;
}
// 输出：
// set: {1, 2, 4, 5, 7, 8}
// unordered_set: {7, 5, 8, 2, 4, 1}
```

`set` 会让元素从小到大排序，而 `unordered_set` **不会排序**，里面的元素都是**完全随机的顺序**，与插入的顺序也不一样。

- 虽然你可能观察到有时刚好与插入的顺序相反，但这只是巧合，具体顺序与 glibc 实现有关。

`set` 基于红黑树实现，相当于二分查找树；而 `unordered_set` 基于散列哈希表实现，正是哈希函数导致了随机的顺序。

## set 系列成员函数总结

| **函数**       | **含义**                      | **set**   | **multiset** | **unordered_set** |
| -------------- | ----------------------------- | --------- | ------------ | ----------------- |
| insert(x)      | 插入一个元素 x                | √         | √            | √                 |
| erase(x)       | 删除所有等于 x 的元素         | √         | √            | √                 |
| count(x)       | 有多少个等于 x 的元素         | √（0或1） | √            | √（0或1）         |
| find(x)        | 指向第一个等于 x 的元素       | √         | √            | √                 |
| lower_bound(x) | 指向第一个大于等于 x 的元素   | √         | √            | ×                 |
| upper_bound(x) | 指向第一个大于 x 的元素       | √         | √            | ×                 |
| equal_range(x) | 所有等于 x 的元素所组成的区间 | √         | √            | √                 |

## 不同版本的 set 容器比较

| **类型**           | **去重** | **有序** | **查找** | **插入**    |
| ------------------ | -------- | -------- | -------- | ----------- |
| vector             | ×        | ×        | O(n)     | O(1) ~ O(n) |
| set                | √        | √        | O(logn)  | O(logn)     |
| multiset           | ×        | √        | O(logn)  | O(logn)     |
| unordered_set      | √        | ×        | O(1)     | O(1)        |
| unordered_multiset | ×        | ×        | O(1)     | O(1)        |

| **类型**           | **头文件**        | **lower/upper_bound** | **equal_range** | **find**   |
| ------------------ | ----------------- | --------------------- | --------------- | ---------- |
| vector             | `<vector>`        | √，O(logn)            | √，O(logn)      | √，O(n)    |
| set                | `<set>`           | √，O(logn)            | √，O(logn)      | √，O(logn) |
| multiset           | `<set>`           | √，O(logn)            | √，O(logn)      | √，O(logn) |
| unordered_set      | `<unordered_set>` | ×，因为是无序的       | √，O(1)         | √，O(1)    |
| unordered_multiset | `<unordered_set>` | ×，因为是无序的       | √，O(1)         | √，O(1)    |

`vector` 适合按**索引**查找，通过运算符 `[]`。

`set` 适合按**值相等**查找，以及按**值大于/小于**查找，分别通过函数 `find`、`lower_bound` 和 `upper_bound`。

`unordered_set` 只适合按**值相等**查找，通过函数 `find`。

小贴士：`unordered_set` 的性能在数据量足够大（>1000）时，平均查找时间比 `set` 短，但不保证稳定。

我个人推荐使用久经考验的 `set`，在数据量小时更高效。
