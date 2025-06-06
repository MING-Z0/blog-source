---
title: 全面理解STL- std::map和它的朋友们
date: 2024-11-4
categories: 
  - C++
tags: [C++, STL, 开发语言]
---

## ！！本文内容搬运小彭老师现代C++大典内容，仅为个人学习备份使用，请大家支持原作者

## 链接：[✝️小彭大典✝️](https://142857.red/book/stl_map/)



### 数据结构与复杂度

不论什么语言，容器（或者用学校里的话说：数据结构）的正确使用，能够在复杂度层面上，大幅提升性能。

C++ 中也是如此，有数组（vector），字典（map），还有上一课讲过的集合（set）。

今天我们要介绍的就是 C++ 的字典容器 map，以及 C++11 引入的另一个字典容器 unordered_map，他们的区别我们最后会详细讨论。我们先学习较为简单的 map，随后学习 unordered_map 时也可以举一反三、融会贯通。

> 介绍完这两个标准库自带的字典容器后，我们还将介绍一些常用的第三方库容器，例如 absl::flat_hash_map、tbb::concurrent_hash_map、google::dense_hash_map、robin_hood::unordered_map、tsl::robin_pg_map 等，需要根据应用场景选择适合的容器。

map/set 家族都是高效查找的专家：

- vector 容器用 std::find 查找：O(N)O(N)
- map 或 set 容器用 .find 查找：O(logN)O(log⁡N)
- unordered_map 或 unordered_set 容器用 .find 查找：O(1)+O(1)+

不仅是查找，map 们还支持高效的增删改查等操作。

## map 的逻辑结构

![logicmap](五、map和他的朋友们/logicmap.png)

特点：

- 由一系列 **键值对** 组成
- 一个键只能对应一个值
- 键不得重复，值可以重复



> std::map, std::unordered_map, absl::flat_hash_map, tbb::concurrent_hash_map 都满足“键值对”这一基本逻辑结构，只是物理实现不同。

在编程中我们常常需要用到“映射”的关系，这就非常需要用到以 map 为首的“键值对”这类容器了。

### 为什么要学习 std::map

map 的具体实现可以是红黑树、AVL 树、线性哈希表、链表哈希表、跳表……不同的实现在不同操作上的复杂度不同，分别适用于不同的场景。

用法上几乎是差不多的，他们都有着几乎相同的接口（除了部分扩展功能）。当你觉得红黑树的 std::map 不合适时，可以轻松把对象类型就地替换为链表哈希表 std::unordered_map 或是是线性哈希表 absl::flat_hash_map，而不用对其他代码有任何更改。

这就是所有 map 类容器都有着相同的**逻辑结构**：都是一个键-值映射，不同的只是他们的**物理结构**而已。

所有的 map 实现，都会模仿提供和 std::map 一样的 API。这就是为什么虽然 std::map 实现的很低效，我们还是要学他的原因。std::map 本身并不是完美的，但却提供了一个所有第三方都会遵循的统一接口。学会了 std::map，任何第三方库的 map 类容器你都可以轻易举一反三。

> 不仅是各种第三方的 map 库，比如 rapidjson 库中的 JSON 对象，也提供了类似 std::map 的 `find` 和 `end` 迭代器接口：`MemberFind` 和 `MemberEnd`，来查找一个字典的子键；几何处理库 cgal 中的“顶点查找”功能也是基于类似的迭代器接口。总之，学会 std::map 将大大有助于你看懂这类业界公认的接口规范。

------

### 标准库中的 map 容器

标准库中，map[1](https://142857.red/book/stl_map/#fn:1) 是一个**模板类**，他的键类型，值类型，可以由尖括号内的参数指定，便于适应不同的用户需求。

> 由于 C++ 标准库的容器大多都是模板类，提供的算法也大多是模板函数，因此 C++ 标准库常被称为标准模板库 (Standard-Template-Library, STL)。

键类型和值类型可以是任意类型，包括基本类型，用户自定义的类，其他 STL 容器等，体现了容器的泛用性。

唯一的要求是：键必须支持比较，这里 map 要求的是小于运算符 `<`。

- 例如 `map<string, int>` 是一个键类型为 string，值类型为 int 的 map 容器。
- 例如 `map<int, Student>` 是一个键类型为 int，值类型为 Student 的 map 容器。
- 例如 `map<char, vector<int>>` 是一个键类型为 char，值类型为 `vector<int>` 的 map 容器。

后面为了方便研究，以 `map<K, V>` 形式书写得出的结论，对于任何实际键和值类型，只需代入 K 和 V 即可。

> 已知：要想使用 `map<K, V>`，就得满足 `K` 必须支持比较运算符 `<`。
>
> 可得：要想使用 `map<string, int>`，就得满足 `string` 必须支持比较运算符 `<`[2](https://142857.red/book/stl_map/#fn:2)。
>
> 已知：遍历 `map<K, V>` 时，是以键 `K` 部分从小到大的顺序遍历的。
>
> 可得：遍历 `map<int, string>` 时，是以键 `int` 部分从小到大的顺序遍历的。

## map 的物理结构

![physmap](五、map和他的朋友们/physmap.png)

map 和 set 一样，都是基于红黑树的二叉排序树，实现 O(logN)O(log⁡N) 复杂度的高效查找。

vector 就是因为元素没有固定的顺序，所以才需要暴力遍历查找。

在持续的插入和删除操作下，始终维持元素的有序性，正是 map 实现高效查找的关键所在。

### 二叉排序树与二分法

始终保存元素按键排序的好处是，如果需要寻找指定键值的元素，就可以采用二分法：

1. 从根节点开始查找。
2. 如果当前节点的键小于要找的键，则往左子节点移动；
3. 如果当前节点的键大于要找的键，则往右子节点移动；
4. 如果当前节点的键等于要找的键，则该节点就是要找的节点，返回该节点。
5. 如果当前节点已经是最后一层叶子节点，也没找到相等的键，则说明该键不存在。
6. 把左/右子节点设为新的当前节点，然后回到第 2 步，重复这一查找过程。

------

### 二叉排序树

由于 map 的实现基于二叉排序树，map 额外有一个特点：**有序**。

map (或 set) 中的键 K 总是从小到大排列，方便进行二分查找，在 O(logN)O(log⁡N) 时间内找到对应元素。

每次插入新的键时，会找到适当的插入位置，使得插入后的 map 仍然有序。

> 注：基于哈希散列表实现的 unordered_map (和 unordered_set)，就不具备**有序**这一特点。

![sortedset](五、map和他的朋友们/sortedset.png)

------

![setvsmap](五、map和他的朋友们/setvsmap.png)

两者的区别在于：map 在 K 之外，额外外挂了一个 V 类型。

map 中的 V 类型不参与排序，只按照 K 进行排序。

这样当用户根据 K 找到的是 K-V 对，然后可以取出 K 对应的 V。

这就实现了从 K 到 V 的映射。

### 二叉树退化问题

二叉排序树只解决了查找的问题，但是他并不能保证经历一通插入后的树不会“退化”。

如果插入的时候不小心，可能会让树的形状变得非常诡异！

例如，若插入数据的顺序是从小到大的，那就会一直在往右插入，清一色的一边倒，以至于几乎成了一根往右跑的链表。

如果插入顺序是从大到小，就变成一直往左边倒。即使插入的顺序不那么刻意，依然可能产生非常变态的形状，违背了二叉树的初衷。

![binary_tree_best_worst_cases](五、map和他的朋友们/binary_tree_best_worst_cases.png)

这样“退化”的二叉排序树，虽然能保持有序，但二分查找时就起不到加速作用了。

如果要找一个中间的元素，几乎就和链表一样，需要遍历整个右枝干。

为了限制二叉排序树不要长成畸形，我们引入一个指标：“深度”，表示从根节点到最底层叶子节点的距离。

要最大化二分查找的效率，就需要二叉树的深度尽可能的低。

因为二分查找的次数就取决于每个叶子节点的平均深度，要尽可能减少平均需要访问的次数，就是要减少二叉树的深度。

也就是说要让大家都尽可能贴近根部，但我们不可能让所有叶子都最贴近根部。

例如右侧只有一个叶子节点，他自己是深度最低了，但代价是左边全部挤在一条链表上了！这不公平。

![binary_tree_almost_worst_case](五、map和他的朋友们/binary_tree_almost_worst_case.png)

所以要最大化二分查找的效率，我们真正需要的是让所有叶子节点都尽可能“平等”！

### 红黑树 vs 平衡树

为了避免二叉树长成畸形，陷入一边倒的情况。我们需要在每次插入后，检查二叉树是否深度差距过大。

如果差的太多了，就需要进行一系列矫正操作，“劫富济贫”，把太长的枝干砍断，接在短的地方，尽可能保持所有叶子路径的深度差不多，这个“劫富济贫”的动作就是**平衡操作 (balancing)**。

问题是，最大能容忍叶子节点之间多大的深度差才开始矫正？针对这个问题，二叉排序树分为两派：

#### 平衡树

最理想的情况下，一颗含有 NN 个节点的二叉树，至少需要有 ⌈logN⌉⌈log⁡N⌉ 深度。

这就是平衡树（AVL），他强制保证整个树处于完美的平衡状态，每个叶子节点之间的深度差距不会超过 1（当节点数量 NN 不是 2 的整数倍时，这是不得不存在的 1 格差距）。


![balanced_binary_tree](五、map和他的朋友们/balanced_binary_tree.png)

- 优点：始终保持最完美的平衡，平均复杂度和最坏复杂度最低。所以平衡树的查找性能是最好的。
- 缺点：然而始终保持完美的平衡意味着，几乎每插入一个元素（可能会突然产生深度差距超过 1 的情况），就立即需要平衡一次。平衡一次的开销是比较大的，所以平衡树的性能是插入性能是比较差的。

平衡树实现平衡的方式是“旋转”，他能始终保持最低的深度差：

![avltree_right_rotate_with_grandchild](五、map和他的朋友们/avltree_right_rotate_with_grandchild.png)

> 这里的细节我们不会深究，那是数据结构课的内容，届时会带大家手搓平衡树和红黑树，本期只是稍微了解 map 常见的底层实现，帮助你理解为什么 map 是有序容器。

#### 红黑树

而红黑树认为，我们不需要总是保持深度差距为 1 那么小，我们只需要保证最深叶子和最浅叶子的深度差不超过 2 倍即可。

例如最浅的一个叶子是 6 深度，另一个最深的叶子可以是 12 深度。只有当最深的叶子超过 12 深度时，红黑树才会开始主动干预平衡，避免继续畸形发展下去。

- 缺点：树可能有一定的一边倒情况，平均复杂度稍微降低，最坏复杂度可以达到原来的 2 倍！
- 优点：因为对不平衡现象更加宽松，正常插入时基本不需要平衡，只有特别扭曲了才会下场“救急”。所以红黑树是牺牲了一部分查找性能，换取了更好的插入和删除性能。

总之，如果你的用况是插入比较少，但是查询非常多，那就适合用平衡树。

由于换来的这部分插入和删除性能实际上比损失的查找性能多，而 map 常见的用况确实需要经常增删改查，所以现在 C++ 标准库的 map 底层都是基于红黑树实现的。

> 如果你的需求是大量查找的话，完全可以考虑用查找平均复杂度低至 O(1)O(1) 的哈希表 unordered_map。
>
> 如果是一次性插入完毕后不会再修改，还可以用完美哈希表（frozen_map），他会为你的键值序列专门生成一个专用的哈希函数，编译期确定，且保证完全无冲突。例如你在做一种语言编译器，有很多“关键字”，比如“if”、“while”，你需要运行时频繁的查找这些关键字，而关键字有哪些在编译期是固定的，那就很适合用完美哈希。

#### 红黑树实现平衡的秘密

红黑树是如何保证最深叶子和最浅叶子的深度差不超过 2 倍的呢？

他设定了这样 5 条规则：

1. 节点可以是红色或黑色的。
2. 根节点总是黑色的。
3. 所有叶子节点都是黑色（叶子节点就是 NULL）。
4. 红色节点的两个子节点必须都是黑色的。
5. 从任一节点到其所有叶子节点的路径都包含相同数量的黑色节点。

看起来好像很复杂，但实际上大多是废话，有用的只是 4 和 5 这两条。

规则 4 翻译一下就是：不得出现相邻的红色节点（相邻指两个节点是父子关系）。这条规则还有一个隐含的信息：黑色节点可以相邻！

规则 5 翻译一下就是：从根节点到所有底层叶子的距离（以黑色节点数量计），必须相等。

因为规则 4 的存在，红色节点不可能相邻，也就是说最深的枝干只能是：红-黑-红-黑-红-黑-红-黑。

结合规则 5 来看，也就是说每条枝干上的黑色节点数量必须相同，因为最深的枝干是 4 个黑节点了，所以最浅的枝干至少也得有 4 个节点全是黑色的：黑-黑-黑-黑。

可以看到，规则 4 和规则 5 联合起来实际上就保证了：最深枝干的深度不会超过最浅枝干的 2 倍。

![Red-black_tree_example](五、map和他的朋友们/Red-black_tree_example.svg.png)

如果超出了 2 倍，就不得不破坏红黑树的规则 4 或 5，从而触发“劫富济贫”的平衡操作，从而阻止了二叉树过于畸形化。

红黑树如何实现“劫富济贫”的细节我们就不再多谈了，点到为止，接下来直接进入正题：

## 开始使用 map 容器

创建一个 map 对象：

```cpp
map<string, int> config;
```

一开始 map 初始是空的，如何插入一些初始数据？

```cpp
config["timeout"] = 985;
config["delay"] = 211;
```

数据插入成功了，根据键查询对应的值？

```cpp
print(config["timeout"]);
print(config["delay"]);
```

查询时建议用 `.at(key)` 而不是 `[key]`：

```cpp
print(config.at("timeout"));
print(config.at("delay"));
```

------

老生常谈的问题：map 中存 string 还是 const char *？

```cpp
map<const char *, const char *> m;
m["hello"] = "old";    // 常量区的 "hello"
char key[] = "hello";  // key 的地址在栈上
print(key == "hello"); // false
m[key] = "new";        // 栈上变量的 key = "hello"
print(m);              // 两个重复的键 "hello"
false
{hello: old, hello: new}
```

在 C++ 中，任何时候都务必用 string！别用 C 语言老掉牙的 const char *，太危险了。

const char * 危险的原因：

1. const char * 的 == 判断的是指针的相等，两个 const char * 只要地址不同，即使实际的字符串相同，也不会被视为同一个元素（如上代码案例所示）。导致 map 里会出现重复的键，以及按键查找可能找不到等。
2. 保存的是弱引用，如果你把局部的 char [] 或 string.c_str() 返回的 const char * 存入 map，等这些局部释放了，map 中的 const char * 就是一个空悬指针了，会造成 segfault。

------

请用安全的 string：

```cpp
map<string, string> m;
m["hello"] = "old";
string key = "hello";
m[key] = "new";
print(m);
print(key == "hello");  // string 的 == 运算符是经过重载的，比较的是字符串里面的内容相等，而不是地址相等
{"hello": "new"}
true
```

| 描述     | C++                                  | Java                      | Python                       |
| -------- | ------------------------------------ | ------------------------- | ---------------------------- |
| 内容相等 | `string("hello") == string("hello")` | `"hello".equals("hello")` | `'hello' == 'hello'`         |
| 地址相等 | `"hello" == "hello"`                 | `"hello" == "hello"`      | `id('hello') == id('hello')` |

------

如果你精通对象生命周期分析，能保证 key 指向的字符串活的比 m 久，想要避免拷贝，节省性能。

string 的弱引用版本：string_view，同样可以用封装了正确的 == 运算符，会比较字符串内容而不是地址：

```cpp
map<string_view, string_view> m;
m["hello"] = "old";
string_view key = "hello";
m[key] = "new";
print(m);
print(key == "hello");
// 此处 m 是栈上变量，key 是弱引用指向全局常量区（rodata），key 比 m 活得久，没有空悬指针问题
{"hello": "new"}
true
```

⚠️ string_view 属于不建议初学者使用的优化小寄巧：有手之前，非常好用。

> 注：map 实际上完全没有用到 ==，用到的只有 < 运算符，当需要判定 `a == b` 时，他会转而用 `!(a < b || b < a)` 来判定。



> string_view 也具有正确的 `hash<string_view>` 特化，因此也可以用做 unordered_map 的键类型。string_view 试图和 string 表现得完全一样，区别在于他是个弱引用，不持有对象，拷贝构造函数是浅拷贝。string_view 大小只有 16 个字节，内部是一个 const char * 和 size_t，但封装了正确的 ==，<，> 和 hash。

------

C++11 新特性——花括号初始化列表，允许创建 map 时直接指定初始数据：

```cpp
map<string, int> config = { {"timeout", 985}, {"delay", 211} };
```

通常我们会换行写，一行一个键值对，看起来条理更清晰：

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};

print(config.at("timeout"));  // 985
```

------

总结花括号初始化语法：

```cpp
map<K, V> m = {
    {k1, v1},
    {k2, v2},
    ...,
};
```

让 map 初始就具有这些数据。

------

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
```

等号可以省略（这其实相当于是在调用 map 的构造函数）：

```cpp
map<string, int> config{
    {"timeout", 985},
    {"delay", 211},
};
```

也可以先构造再赋值给 auto 变量：

```cpp
auto config = map<string, int>{
    {"timeout", 985},
    {"delay", 211},
};
```

都是等价的。

作为函数参数时，可以用花括号初始化列表就地构造一个 map 对象：

```cpp
void myfunc(map<string, int> config);  // 函数声明

myfunc(map<string, int>{               // 直接创建一个 map 传入
    {"timeout", 985},
    {"delay", 211},
});
```

由于 `myfunc` 函数具有唯一确定的重载，要构造的参数类型 `map<string, int>` 可以省略不写：

```cpp
myfunc({
    {"timeout", 985},
    {"delay", 211},
});
```

函数这边，通常还会加上 `const &` 修饰避免不必要的拷贝。

```cpp
void myfunc(map<string, int> const &config);
```

------

从 vector 中批量导入键值对：

```cpp
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
};
map<string, int> config(kvs.begin(), kvs.end());
```

与刚刚花括号初始化的写法等价，只不过是从现有的 vector 中导入。同样的写法也适用于从 array 导入。

> 如果记不住这个写法，也可以自己手写 for 循环遍历 vector 逐个逐个插入 map，效果是一样的。

冷知识，如果不是 vector 或 array，而是想从传统的 C 语言数组中导入：

```cpp
pair<string, int> kvs[] = {  // C 语言原始数组
    {"timeout", 985},
    {"delay", 211},
};
map<string, int> config(kvs, kvs + 2);                    // C++98
map<string, int> config(std::begin(kvs), std::end(kvs));  // C++17
```

> 其中 `std::begin` 和 `std::end` 为 C++17 新增函数，专门用于照顾没法有成员函数 `.begin()` 的 C 语言数组。类似的全局函数还有 `std::size` 和 `std::data` 等……他们都是既兼容 STL 容器也兼容 C 数组的。

------

重点来了：如何根据键查询相应的值？

很多同学都知道 map 具有 [] 运算符重载，[] 里写要查询的键就可以返回对应值，也可以用 = 往里面赋值，和某些脚本语言一样直观易懂。

```cpp
config["timeout"] = 985;       // 把 config 中键 timeout 对应值设为 985
auto val = config["timeout"];  // 读取 config 中键 timeout 对应值
print(val);                    // 985
```

但其实用 [] 去**读取元素**是很不安全的，下面我会做实验演示这一点。

------

沉默的 []，无言的危险：当键不存在时，会返回 0 而不会出错！

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config["timeout"]); // 985
print(config["tmeout"]);  // 默默返回 0
985
0
```

当查询的键值不存在时，[] 会默默创建并返回 0，而不会爆出任何错误。

这非常危险，例如一个简简单单的拼写错误，就会导致 map 的查询默默返回 0，你还在那里找了半天摸不着头脑，根本没发现错误原来在 map 这里。

------

爱哭爱闹的 at()，反而更讨人喜欢

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config.at("timeout"));  // 985
print(config.at("tmeout"));   // 该键不存在！响亮地出错
985
terminate called after throwing an instance of 'std::out_of_range'
  what():  map::at
Aborted (core dumped)
```

有经验的老手都明白一个道理：**及时奔溃**比**容忍错误**更有利于调试。即 fail-early, fail-loudly[1](https://142857.red/book/stl_map/#fn:1) 原则。

例如 JS 和 Lua 的 [] 访问越界不报错而是返回 undefined / nil，导致实际出错的位置在好几十行之后，无法定位到真正出错的位置，这就是为什么后来发明了错误检查更严格的 TS。

使用 at() 可以帮助你更容易定位到错误，是好事。

------

> 在官方文档和各种教学课件中，都会展示一个函数的“原型”来讲解。
>
> 原型展现了一个函数的名称，参数类型，返回类型等信息，掌握了函数的原型就等于掌握了函数的调用方法。
>
> 本课程后面也会大量使用，现在来教你如何看懂成员函数的原型。

假设要研究的类型为 `map<K, V>`，其中 K 和 V 是模板参数，可以替换成你具体的类型。

例如当我使用 `map<string, int>` 时，就把下面所有的 K 替换成 string，V 替换成 int。

`map<K, V>` 的 [] 和 at 员函数，原型如下：

```cpp
V &operator[](K const &k);
V &at(K const &k);                   // 第一个版本的 at
V const &at(K const &k) const;       // 第二个版本的 at
```

可见 operator[] 只有一个版本，at 居然有名字相同的两个！这样不会发生冲突吗？

这是利用了 C++ 的“重载”功能，重载就是同一个函数有多个不同的版本，各个版本的参数类型不同。

------

同理，编译器也是会根据调用时你传入的参数类型，决定要调用重载的哪一个具体版本。

- C 语言没有重载，函数名字相同就会发生冲突，编译器会当场报错。
- C++ 支持重载，只有当函数名字相同，参数列表也相同时，才会发生冲突。
- 返回值类型不影响重载，重载只看参数列表。

菜鸟教程上对 C++ 重载的解释[1](https://142857.red/book/stl_map/#fn:1)：

> C++ 允许在同一作用域中的某个函数和运算符指定多个定义，分别称为函数重载和运算符重载。
>
> 重载声明是指一个与之前已经在该作用域内声明过的函数或方法具有相同名称的声明，但是它们的参数列表和定义（实现）不相同。
>
> 当您调用一个重载函数或重载运算符时，编译器通过把您所使用的参数类型与定义中的参数类型进行比较，决定选用最合适的定义。选择最合适的重载函数或重载运算符的过程，称为重载决策。
>
> 在同一个作用域内，可以声明几个功能类似的同名函数，但是这些同名函数的形式参数（指参数的个数、类型或者顺序）必须不同。您不能仅通过返回类型的不同来重载函数。

------

```cpp
V &at(K const &k);                   // 第一个版本的 at
V const &at(K const &k) const;       // 第二个版本的 at
```

但是上面这两个 at 函数的参数类型都是 `K const &`，为什么可以重载呢？

注意看第二个版本最后面多了一个 const 关键字，这种写法是什么意思？对其进行祛魅化：

```cpp
V &at(map<K, V> *this, K const &k);                   // 第一个版本的 at
V const &at(map<K, V> const *this, K const &k);       // 第二个版本的 at
```

原来加在函数括号后面的 const，实际上是用于修饰 this 指针的！

> 该写法仅供示意，并不是真的可以把 this 写成参数

所以两个 at 的参数列表不同，不同在于传入 this 指针的类型，所以可以重载，不会冲突。

- 当 map 对象为 const 时，传入的 this 指针为 `map<K, V> const *`，所以只能调用第二个版本的 at。
- 当 map 对象不为 const 时，传入的 this 指针为 `map<K, V> *`，两个重载都可以调用，但由于第一个重载更加符合，所以会调用第一个版本的 at。

> 有趣的是，C++23 支持了显式对象形参（deducing-this），this 也能像普通参数一样定义了！上面的代码可以写成：

```cpp
class map {
    ...

    V &at(this map &self, K const &k) {
        // 函数体内可以使用self代替原来的this（this将不再可用）
        ...
    }

    V const &at(this map const &self, K const &k) {
        ...
    }
};
```

------

刚刚解释了函数重载，那么运算符重载呢？

因为原本 C 语言就有 [] 运算符，不过那只适用于原始指针和原始数组。而 C++ 允许也 [] 运算符支持其他用户自定义类型（比如 std::map），和 C 语言自带的相比就只有参数类型不同（一个是原始数组，一个是 std::map），所以和函数重载很相似，这就是运算符重载。

```cpp
m["key"];
```

会被编译器“翻译”成：

```cpp
m.operator[]("key");
```

以上代码并非仅供示意，是可以通过编译运行的。

> operator[] 虽然看起来很复杂一个关键字加特殊符号，其实无非就是个特殊的函数名，学过 Python 的童鞋可以把他想象成 `__getitem__`。

```cpp
V &operator[](K const &k);
```

结论：[] 运算符实际上是在调用 operator[] 函数。

> 所有的所谓“运算符重载函数”实际上都是一个特殊的标识符，以`operator` + 运算符的形式，他们两个组成一个整体，你还可以试试 `string("hel").operator+("lo")`，和 `string("hel") + "lo"` 是等价的。

------

因为 operator[] 这个成员函数后面没有 const 修饰，因此当 map 修饰为 const 时编译会不通过[1](https://142857.red/book/stl_map/#fn:1)：

```cpp
const map<string, int> config = {  // 此处如果是带 const & 修饰的函数参数也是同理
    {"timeout", 985},
    {"delay", 211},
};
print(config["timeout"]);          // 编译出错
/home/bate/Codes/course/stlseries/stl_map/experiment/main.cpp: In function ‘int main()’:
/home/bate/Codes/course/stlseries/stl_map/experiment/main.cpp:10:23: error: passing ‘const std::map<std::__cxx11::basic_string<char>, int>’ as ‘this’ argument discards qualifiers [-fpermissive]
   10 | print(config["timeout"]);
```

编译器说 discards qualifiers，意思是 map 有 const 修饰，但是 operator[] 没有。

这实际上就是在说：`map<K, V> const *this` 不能转换成 `map<K, V> *this`。

有 const 修饰的 map 作为 this 指针传入没 const 修饰的 operator[] 函数，是减少了修饰（discards qualifers）。

C++ 规定传参时只能增加修饰不能减少修饰：只能从 `map *` 转换到 `map const *` 而不能反之。

所以对着一个 const map 调用非 const 的成员函数 operator[] 就出错了，相比之下 at() 就可以在 const 修饰下编译通过。

------

为什么 operator[] 是非 const 修饰的呢？通常来说，一个成员函数不是 const，意味着他会**就地修改 this 对象**。

其实，operator[] 发现所查询的键值不存在时：

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config);
print(config["tmeout"]);  // 有副作用！
print(config);
{"delay": 211, "timeout": 985}
0
{"delay": 211, "timeout": 985, "tmeout": 0}
```

**会自动创建那个不存在的键值！**

你以为你只是观察了一下 map 里的 “tmeout” 元素，却意外改变了 map 的内容，薛定谔直呼内行。

------

为什么把 [] 设计的这么危险？

既然已经有更安全的 .at()，为什么还要让 [] 继续存在呢？

```cpp
map<string, int> config = {
    {"delay", 211},
};
config.at("timeout") = 985;  // 键值不存在，报错！
config["timeout"] = 985;     // 成功创建并写入 985
```

由上可见，当我们写入一个本不存在的键值的时候，恰恰需要 [] 的“自动创建”这一特性，这是 at() 所不具有的。

总结：读取时应该用 at() 更安全，写入时才需要用带有自动创建功能的 []。

> 许多第三方库，例如 jsoncpp，他们的字典类型也使用类似的接口，at() 负责读，[] 负责写，分工明确！

------

# 总结

- 
  读取元素时，统一用 at()
- 写入元素时，统一用 []

```cpp
auto val = m.at("key");
m["key"] = val;
```

为什么其他语言比如 Python，只有一个 [] 就行了呢？而 C++ 需要两个？

- 因为 Python 会检测 [] 位于等号左侧还是右侧，根据情况分别调用 `__getitem__` 或者 `__setitem__`。
- C++ 编译器没有这个特殊检测，也检测不了，因为 C++ 的 [] 只是返回了个引用，并不知道 [] 函数返回以后，你是拿这个引用写入还是读取。为了保险起见他默认你是写入，所以先帮你创建了元素，返回这个元素的引用，让你写入。
- 而 Python 的引用是不能用 = 覆盖原值的，那样只会让变量指向新的引用，只能用 .func() 引用成员函数或者 += 才能就地修改原变量，这是 Python 这类脚本语言和 C++ 最本质的不同。
- 总而言之，我们用 C++ 的 map 读取元素时，需要显式地用 at() 告诉编译器我是打算读取。

------

[] 找不到就返回个“默认值”，其实也是很多语言的传统异能了，只有刚好 Python 比较对初学者友好，会自动判断你的 [] 是读取还是写入，如果是读取，当找不到键值时能友善的给你报错。

| 语言及其关联容器名 | C++ map    | Python dict   | Lua table    | JS HashMap         | Java HashMap          |
| ------------------ | ---------- | ------------- | ------------ | ------------------ | --------------------- |
| 找不到键时的行为   | 默默返回 0 | 报错 KeyError | 默默返回 nil | 默默返回 undefined | .get()，默默返回 null |

其中 C++ 的 [] 最为恶劣，因为古代 C++ 中并没有一个 null 或 nil 之类的额外特殊常量。

[] 返回的必须是个具体的类型，由于 [] 不能报错，值的类型又千变万化，`map<K, V>` 的 [] 只能返回“V 类型默认构造函数创建的值”：对于 int 而言是 0，对于 string 而言是 “”（空字符串）。

> 也正因如此，如果一个 `map<K, V>` 中的 V 类型没有默认构造函数，就无法使用 [] 了。看似美好的 [] 只是骗骗小朋友的面子工程，模棱两可，充满危险。高手都使用更专业的写入函数：insert 或 insert_or_assign 代替。这两个函数不需要默认构造函数，还更高效一些，稍后会详细介绍。

------

at 与 [] 实战演练

我们现在的甲方是一个学校的大老板，他希望让我们管理学生信息，因此需要建立一个映射表，能够快速通过学生名字查询到相应的学生信息。思来想去 C++ 标准库中的 map 容器最合适。决定设计如下：

- 键为学生的名字，string 类型。
- 值为一个自定义结构体，Student 类型，里面存放各种学生信息。

然后自定义一下 Student 结构体，现在把除了名字以外的学生信息都塞到这个结构体里。

创建 `map<string, Student>` 对象，变量名为 `stus`，这个 map 就是甲方要求的学生表，成功交差。

```cpp
struct Student {
    int id;             // 学号
    int age;            // 年龄
    string sex;         // 性别
    int money;          // 存款
    set<string> skills; // 技能
};

map<string, Student> stus;
```

------

现在让我们用 `[]` 大法插入他的个人信息：

```cpp
stus["彭于斌"] = Student{20220301, 22, "自定义", {"C", "C++"}};
stus["相依"] = Student{20220301, 21, "男", 2000, {"Java", "C"}};
stus["樱花粉蜜糖"] = Student{20220301, 20, "女", 3000, {"Python", "CUDA"}};
stus["Sputnik02"] = Student{20220301, 19, "男", 4000, {"C++"}};
```

由于 C++11 允许省略花括号前的类型不写，所以 Student 可以省略，简写成：

```cpp
stus["彭于斌"] = {20220301, 22, "自定义", {"C", "C++"}};
stus["相依"] = {20220301, 21, "男", 2000, {"Java", "C"}};
stus["樱花粉蜜糖"] = {20220301, 20, "女", 3000, {"Python", "CUDA"}};
stus["Sputnik02"] = {20220301, 19, "男", 4000, {"C++"}};
```

又由于 map 支持在初始化时就指定所有元素，我们直接写：

```cpp
map<string, Student> stus = {
    {"彭于斌", {20220301, 22, "自定义", 1000, {"C", "C++"}}},
    {"相依", {20220301, 21, "男", 2000, {"Java", "C"}}},
    {"樱花粉蜜糖", {20220301, 20, "女", 3000, {"Python", "CUDA"}}},
    {"Sputnik02", {20220301, 19, "男", 4000, {"C++"}}},
};
```

------

现在甲方要求添加一个“培训”函数，用于他们的 C++ 培训课。

培训函数的参数为字符串，表示要消费学生的名字。如果该名字学生不存在，则应该及时报错。

每次培训需要消费 2650 元，消费成功后，往技能 skills 集合中加入 “C++”。

```cpp
void PeiXunCpp(string stuName) {
    auto stu = stus.at(stuName);  // 这是在栈上拷贝了一份完整的 Student 对象
    stu.money -= 2650;
    stu.skills.insert("C++");
}
```

然而，这样写是不对的！

`stus.at(stuName)` 返回的是一个引用 `Student &` 指向 map 中的学生对象。但是等号左侧，却是个不带任何修饰的 `auto`，他会被推导为 `Student`。如何从一个引用 `Student &` 转换为具体的 `Student`？找不到 `Student(Student &)`，但是找到了最接近的 `Student(Student const &)` 函数（这是编译器自动生成的拷贝构造函数），因此我们拷贝了一份 map 中的学生对象，到栈上的 stu 变量，之后不论如何修改，修改的都是这个栈上对象，而不会对 map 中的学生对象产生任何影响。

结论：把引用保存到普通变量中，则引用会退化，造成深拷贝！不仅影响性能，还影响功能！stu 已经是一个独立的 Student 对象，对 stu 的修改已经不会影响到 stus.at(stuName) 指向的那个 Student 对象了。

此时你对这个普通变量的所有修改，都不会同步到 map 中的那个 Student 中去！

------

我们现在对相依童鞋进行 C++ 培训：

```cpp
PeiXunCpp("相依");
print(stus.at("相依"));
```

结果发现他的存款一分没少，也没学会 C++：

```yaml
{id: 20220302, age: 21, sex: "男", money: 2000, skills: {"C", "Java"}}
```

看来我们的修改没有在 map 中生效？原来是因为我们在 PeiXunCpp 函数里：

```cpp
auto stu = stus.at(stuName);  // 在栈上拷贝了一份完整的 Student 对象
```

一不小心就用了“克隆人”技术！从学生表里的“相依1号”，克隆了一份放到栈上的“相依2号”！

然后我们扣了这个临时克隆人“相依2号”的钱，并给他培训 C++ 技术。

然而我们培训的是栈上的临时变量“相依2号”，克隆前的“相依1号”并没有受到培训，也没有扣钱。

然后呢？残忍的事情发生了！在一通操作培训完“相依2号”后，我们把他送上断头台——析构了！

而这一切“相依1号”完全不知情，他只知道有人喊他做克隆，然后就回家玩 Java 去了，并没有培训 C++ 的记忆。

------

要防止引用退化成普通变量，需要把变量类型也改成引用！这种是浅拷贝，stu 和 stus.at(stuName) 指向的仍然是同一个 Student 对象。用 `auto` 捕获的话，改成 `auto &` 就行。

```cpp
void PeiXunCpp(string stuName) {
    auto &stu = stus.at(stuName);  // 在栈上创建一个指向原 Student 对象的引用
    stu.money -= 2650;
    stu.skills.insert("C++");
}
{id: 20220302, age: 21, sex: "男", money: -650, skills: {"C", "C++", "Java"}}
```

终于，正版“相依1号”本体鞋废了 C++！

之后如果再从“相依1号”身上克隆，克隆出来的“相依n号”也都会具有培训过 C++ 的记忆了。

引用相当于身份证，我们复印了“相依”的身份证，身份证不仅复印起来比克隆一个大活人容易（拷贝开销）从而提升性能，而且通过身份证可以找到本人，对身份证的修改会被编译器自动改为对本人的修改，例如通过“相依”的身份证在银行开卡等，银行要的是身份证，不是克隆人哦。

------

引用是一个烫手的香香面包，普通变量就像一个臭臭的答辩马桶，把面包放到马桶（auto）里，面包就臭掉，腐烂掉，不能吃了！要让面包转移阵地了以后依然好吃，需要放到保鲜盒（auto &）里。

这就是 C++ 的 decay（中文刚好是“退化”、“变质”的意思）规则。

以下都是“香香面包”，放进马桶里会变质：

- `T &` 会变质成 `T`（引用变质成普通变量）
- `T []` 会变质成 `T *`（数组变质成首地址指针）
- `T ()` 会变质成 `T (*)()`（函数变质成函数指针）

在函数的参数中、函数的返回值中、auto 捕获的变量中，放入这些“香香面包”都会发生变质！

如何避免变质？那就不要用马桶（普通变量）装面包呗！用保鲜盒（引用变量）装！

- 避免引用 `T &t` 变质，就得把函数参数类型改成引用，或者用 `auto &`，`auto const &` 捕获才行。
- 避免原生数组 `T t[N]` 变质，也可以改成引用 `T (&t)[N]`，但比较繁琐，不如直接改用 C++11 封装的安全静态数组 `array<T, N>` 或 C++98 就有的安全动态数组 `vector<T>`。
- 避免函数 `T f()` 变质，可以 `T (&f)()`，但繁琐，不如直接改用 C++11 的函数对象 `function<T()>`。


![autodecays](五、map和他的朋友们/autodecays.png)

------

### C 语言的退化规则真是害人不浅

题外话：邪恶的退化规则造成空悬指针的案例

```cpp
typedef double arr_t[10];

auto func(arr_t val) {
    arr_t ret;
    memcpy(ret, val, sizeof(arr_t));  // 对 val 做一些运算, 把计算结果保存到 ret
    return ret;     // double [10] 自动变质成 double *
}

int main() {
    arr_t val = {1, 2, 3, 4};
    auto ret = func(val);             // 此处 auto 会被推导为 double *
    print(std::span(ret, ret + 10));
    return 0;
}
Segmentation fault (core dumped)
```

修复方法：别再用 C 语言的煞笔原始人数组了！用 C++ 封装好的 array，无隐患

```cpp
typedef std::array<double, 10> arr_t;  // 如需动态长度，改用 vector 亦可

auto func(arr_t val) {
    arr_t ret;
    ret = val;  // 对 val 做一些运算, 把计算结果保存到 ret
    return ret;
}

int main() {
    arr_t val = {1, 2, 3, 4};
    auto ret = func(val);
    print(ret);
    return 0;
}
{1, 2, 3, 4, 0, 0, 0, 0, 0, 0}
```

------

如果你还是学不会怎么保留香香引用的话，土办法：也可以在修改后再次用 [] 写回学生表。这样学生表里不会 C++ 的“相依1号”就会被我们栈上培训过 C++ 的“相依1号”覆盖，现在学生表里的也是有 C++ 技能的“相依”辣！只不过需要翻来覆去克隆了好几次比较低效而已，至少能用了，建议只有学不懂引用的童鞋再用这种保底写法。

```cpp
void PeiXunCpp(string stuName) {
    auto stu = stus.at(stuName);  // 克隆了一份“相依2号”
    stu.money -= 2650;
    stu.skills.insert("C++");
    stus[stuName] = stu;          // “相依2号”夺舍，把“相依1号”给覆盖掉了
}
```

如果要根据学号进行查找呢？那就以学号为键，然后把学生姓名放到 Student 结构体中。

如果同时有根据学号进行查找和根据姓名查找两种需求呢？

同时高效地根据多个键进行查找，甚至指定各种条件，比如查询所有会 C++ 的学生等，这可不是 map 能搞定的，或者说能搞定但不高效（最后往往只能暴力遍历查找，时间复杂度太高）。这是个专门的研究领域，称为：关系数据库。

关系数据库的实现有 MySQL，SQLite，MongoDB 等。C++ 等编程语言只需调用他们提供的 API 即可，不必自己手动实现这些复杂的查找和插入算法。

这就是为什么专业的“学生管理系统”都会用关系数据库，而不是自己手动维护一个 map。关系数据库底层的数据结构更复杂，但经过高度封装，效率更高，提供的功能也更全面，用起来也比较无感。何况 map 存在内存中，电脑一关机，学生数据就没了！而数据库可以把数据持久化到磁盘中，相当于在磁盘里构建出了一颗查找树，关机后数据依然保持。

查询 map 中元素的数量

```cpp
size_t size() const noexcept;
```

使用 `m.size()` 获得的 map 大小，或者说其中元素的数量。

```cpp
map<string, int> m;
print(m.size()); // 0
m["fuck"] = 985;
print(m.size()); // 1
m["dick"] = 211;
print(m.size()); // 2
```

------

应用举例：给每个键一个独一无二的计数

```cpp
map<string, int> m;
m["fuck"] = m.size();
m["dick"] = m.size();
```

> 需要 C++17 以上的版本，才能保证等号右边的 `m.size()` 先于 `m["fuck"]` 求值。C++14 中上面这段代码行为未定义，需要改用 `m.insert({"fuck", m.size()})` 的写法（函数参数总是优先于函数求值，这保证 `m.size()` 先求值，然后才发生元素插入）。

------

判断一个键是否存在：count 函数

```cpp
size_t count(K const &k) const;
```

count 返回容器中键和参数 k 相等的元素个数，类型为 size_t（无符号 64 位整数）。

由于 map 中同一个键最多只可能有一个元素，取值只能为 0 或 1。

并且 size_t 可以隐式转换为 bool 类型，0 则 false，1 则 true。

------

因此可以直接通过 count 的返回值是否为 0 判断一个键在 map 中是否存在：

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
if (msg.count("fuck")) {
    print("存在fuck，其值为", msg.at("fuck"));
} else {
    print("找不到fuck");
}
if (msg.count("dick")) {
    print("存在dick，其值为", msg.at("suck"));
} else {
    print("找不到dick");
}
{"fuck": "rust", "hello": "world"}
存在fuck，其值为 "rust"
找不到dick
```

C++20 中建议改用返回类型为 `bool` 的 `contains` 函数，函数名和类型更加一目了然，但实际效果和 `count` 是一样的。

```cpp
if (msg.contains("fuck")) {
    print("存在fuck，其值为", msg.at("fuck"));
} else {
    print("找不到fuck");
}
```

## 你知道吗？[] 的妙用

除了写入元素需要用 [] 以外，还有一些案例中合理运用 [] 会非常的方便。

[] 的效果：当所查询的键值不存在时，会调用默认构造函数创建一个元素[1](https://142857.red/book/stl_map/#fn:1)。

- 对于 int, float 等数值类型而言，默认值是 0。
- 对于指针（包括智能指针）而言，默认值是 nullptr。
- 对于 string 而言，默认值是空字符串 “”。
- 对于 vector 而言，默认值是空数组 {}。
- 对于自定义类而言，会调用你写的默认构造函数，如果没有，则每个成员都取默认值。

### [] 妙用举例：出现次数统计

```cpp
vector<string> input = {"hello", "world", "hello"};
map<string, int> counter;
for (auto const &key: input) {
    counter[key]++;
}
print(counter);
{"hello": 2, "world": 1}
```

#### 对比

活用 [] 自动创建 0 元素的特性

```cpp
map<string, int> counter;
for (auto const &key: input) {
    counter[key]++;
}
```

古板的写法

```cpp
map<string, int> counter;
for (auto const &key: input) {
    if (!counter.count(key)) {
        counter[key] = 1;
    } else {
        counter[key] = counter.at(key) + 1;
    }
}
```

### [] 妙用举例：归类

```cpp
vector<string> input = {"happy", "world", "hello", "weak", "strong"};
map<char, vector<string>> categories;
for (auto const &str: input) {
    char key = str[0];
    categories[key].push_back(str);
}
print(categories);
{'h': {"happy", "hello"}, 'w': {"world", "weak"}, 's': {"strong"}}
```

#### 对比

活用 [] 自动创建”默认值”元素的特性

```cpp
map<char, vector<string>> categories;
for (auto const &str: input) {
    char key = str[0];
    categories[key].push_back(str);
}
print(categories);
```

古板的写法

```cpp
map<char, vector<string>> categories;
for (auto const &str: input) {
    char key = str[0];
    if (!categories.count(key)) {
        categories[key] = {str};
    } else {
        categories[key].push_back(str);
    }
}
```

![Elegence](五、map和他的朋友们/v2-f2560f634b1e09f81522f29f363827f7_720w.jpg)

### [] 妙用举例：线程局部变量

```cpp
concurrent_map<std::thread::id, Data> tls;
parallel_for([] {
    Data &data = tls[std::this_thread::get_id()];
    ...;
});
```

不过 `thread_local` 关键字，可以取代。

## 为什么需要反向查找表

反面典型：查找特定元素在 vector 中的位置（下标）

```cpp
size_t array_find(vector<string> const &arr, string const &val) {
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == val) return i;
    }
    return (size_t)-1;
}
vector<string> arr = {"hello", "world", "nice", "day", "fucker"};
print("hello在数组中的下标是：", array_find(arr, "fucker"));    // O(N) 低效
print("nice在数组中的下标是：", array_find(arr, "nice"));       // O(N) 低效
```

每次调用 `array_find`，都需要 O(N)O(N) 复杂度。

```bash
fucker在数组中的下标是：0
nice在数组中的下标是：2
```

如果查询 N 次，则复杂度就是 O(N2)O(N2)。

> 注：假设 vector 中不存在重复的元素

### map 构建下标查找表

正确做法：构建 vector 的反向查找表，以后查找更高效

```cpp
vector<string> arr = {"hello", "world", "nice", "day", "fucker"};
map<string, size_t> arrinv;
for (size_t i = 0; i < arr.size(); i++) {                // O(N) 一次性受苦
    arrinv[arr[i]] = i;
}
print("反向查找表构建成功：", arrinv);
print("fucker在数组中的下标是：", arrinv.at("fucker"));  // O(log N) 高效
print("nice在数组中的下标是：", arrinv.at("nice"));      // O(log N) 高效
```

只有第一次构造反向查找表时，需要 O(N)O(N) 复杂度。

以后每次调用 `map.at`，只需要 O(logN)O(log⁡N) 复杂度。

```bash
反向查找表构建成功：{"day": 3, "fucker", 4, "hello": 0, "nice": 2, "world": 1}
fucker在数组中的下标是：4
nice在数组中的下标是：2
```

> 轶事：在数据库中，这种反向查找表被称为“倒序索引”

```cpp
for (size_t i = 0; i < arr.size(); i++) {
    arrinv[arr[i]] = i;
}
```

提前构造好查找表 O(N)O(N)，以后每次查找只需要 O(logN)O(log⁡N) 复杂度就行。

- （正向查找）已知下标 i，求元素 v：`v = arr[i]`
- （反向查找）已知元素 v，求下标 i：`i = arrinv[v]`

如果查询 N 次，则复杂度就是 O(NlogN)O(Nlog⁡N)，比优化前高效。

>
> 只有当 vector 更新时，才需要重新构建 map。如果 vector 的删除采用 back-swap-erase（见 [C++ 小妙招](https://142857.red/book/cpp_tricks/)），那么无需完全重构 map，只需更新 swap 的两个元素即可，总复杂度 O(logN)O(log⁡N)，这样就实现了一个 O(logN)O(log⁡N) 的有下标又能快速查找数组，兼具 map 和 vector 的优势。在之后的数据结构进阶课中我们会详细介绍此类复合数据结构。

### map 构建另一个 map 的反向查找表

map 只能通过值映射到键，能不能反过来通过键查找值？

案例：构建另一个 map 的反向查找表

```cpp
map<string, string> tab = {
    {"hello", "world"},
    {"fuck", "rust"},
};
map<string, string> tabinv;
for (auto const &[k, v]: tab) {
    tabinv[v] = k;
}
print(tabinv);
```

效果就是，键变值，值变键，反一反，两个 map 互为逆运算：

```json
{"rust": "fuck", "world": "hello"}
```

> 注意：要求 tab 中不能存在重复的值，键和值必须是一一对应关系，才能用这种方式构建双向查找表。否则一个值可能对应到两个键，反向表必须是 `map<string, vector<string>>` 了。

------

## 元编程查询成员类型：`value_type`

STL 容器的元素类型都可以通过成员 `value_type` 查询，常用于泛型编程（又称元编程）。

```cpp
set<int>::value_type      // int
vector<int>::value_type   // int
string::value_type        // char
```

此外还有引用类型 `reference`，迭代器类型 `iterator`，常迭代器类型 `const_iterator` 等。

曾经在 C++98 中很常用，不过自从 C++11 有了 auto 和 decltype 以后，就不怎么用了，反正能自动推导返回类型。

C++23:

```cpp
std::vector<int> arr;
for (auto const &elem: arr) {
    std::println("{}", elem);
}
```

C++17:

```cpp
std::vector<int> arr;
for (auto const &elem: arr) {
    std::cout << elem << '\n';
}
```

C++11:

```cpp
std::vector<int> arr;
for (auto it = arr.begin(); it != arr.end(); ++it) {
    std::cout << *it << '\n';
}
```

C++98:

```cpp
std::vector<int> arr;
for (std::vector<int>::iterator it = arr.begin(); it != arr.end(); ++it) {
    std::cout << *it << '\n';
}
```

### typename 修饰

当容器有至少一个不确定的类型 T 作为模板参数时，就需要前面加上 `typename` 修饰了：

```cpp
set<int>::value_type;               // 没有不定类型，不需要
typename set<T>::value_type;        // 包含有 T 是不定类型
typename set<set<T>>::value_type;   // 包含有 T 是不定类型
typename map<int, T>::value_type;   // 包含有 T 是不定类型
typename map<K, T>::value_type;     // 包含有 K、T 是不定类型
map<int, string>::value_type;       // 没有不定类型，不需要
```

如果你搞不清楚，始终加 `typename` 就行了，反正加多肯定不会有错。你就认为：这就是一个平时可以省略，偶尔不能省略的东西。

```cpp
typename set<int>::value_type;    // 可以省略，但你加了也没关系
typename set<T>::value_type;      // 不能省略
typename set<set<T>>::value_type; // 不能省略
typename map<int, T>::value_type; // 不能省略
typename map<K, T>::value_type;   // 不能省略
typename map<int, string>::value_type; // 可以省略，但你加了也没关系
```

> 含有 T 的类型表达式称为 dependent-type，根本原因是因为在不知道具体是类型表达式还是值表达式的情况下，编译器无法区分模板的 `<` 和小于符号 `<`，以及类型的指针 `*` 和数值乘法 `*`。默认会认为是小于符号和数值乘法，加上 `typename` 后明确前面这一串是类型表达式，才知道这是模板的 `<` 和指针的 `*`。

### decltype 大法好

也有更直观的获取 STL 容器元素类型的方法：

```cpp
std::vector<int> arr;

using T = std::decay_t<decltype(arr[0])>; // T = int
```

> `decltype` 必须配合 `std::decay_t` 才能用！否则会得到引用类型 `int &`，后续使用中就坑到你！（因为 arr 的 [] 返回的是一个引用类型）

```cpp
// 错误示范
using T = decltype(arr[0]); // T = int &

T i = 0; // int &i = 0; 后续使用中编译出错！
```

------

### 查询类名小工具

在本课程的案例代码中附带的 “cppdemangle.h”，可以实现根据指定的类型查询类型名称并打印出来。

跨平台，需要 C++11，支持 MSVC，Clang，GCC 三大编译器，例如：

```cpp
int i;
print(cppdemangle<decltype(std::move(i))>());
print(cppdemangle<std::string>());
print(cppdemangle<std::wstring::value_type>());
```

在我的 GCC 12.2.1 上得到：

```bash
"int &&"
"std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >"
"wchar_t"
```

------

### map 真正的元素类型究竟是什么？

map 具有三个成员类型[1](https://142857.red/book/stl_map/#fn:1)：

- 元素类型：`value_type`
- 键类型：`key_type`
- 值类型：`mapped_type`

后面，将会一直以“元素”称呼官方的“value”，“键”称呼官方的“key”，“值”称呼官方的“mapped”

用 cppdemangle 做实验，看看这些成员类型具体是什么吧：

```cpp
map<int, float>::value_type   // pair<const int, float>
map<int, float>::key_type     // int
map<int, float>::mapped_type  // float
```

结论：`map<K, V>` 的元素类型是 `pair<const K, V>` 而不是 `V`。

------

疑惑：`pair<const K, V>` 中，为什么 K 要加 const？

我们在 set 课中说过，set 内部采用红黑树数据结构保持有序，这样才能实现在 O(logN)O(log⁡N) 时间内高效查找。

键值改变的话会需要重新排序，如果只修改键值而不重新排序，会破坏有序性，导致二分查找结果错误！所以 set 只提供了不可变迭代器（const_iterator），没有可变的迭代器，不允许用户修改任何元素的值。

map 和 set 一样也是红黑树，不同在于：map 只有键 K 的部分会参与排序，V 是个旁观者，随便修改也没关系。

所以 map 有可变迭代器，只是在其值类型 value_type 中给键的部分，K，加上了 const 修饰：不允许修改 K，但可以随意修改 V。

如果你确实需要修改键值，那么请先取出旧值，把这个键删了，然后再以同样的值重新插入一遍到新的键。相当于重新构建了一个 `pair<const K, V>` 对象。

> C++17 开始也可以用更高效 `node_handle` 系列 API，避免数据发生移动，稍后介绍。

```cpp
iterator begin();
const_iterator begin() const;
iterator end();
const_iterator end() const;
```

begin() 和 end() 迭代器分别指向 map 的首个元素和最后一个元素的后一位。

其中 end() 迭代器指向的地址为虚空索敌，不可解引用，仅仅作为一个“标志”存在（回顾之前 vector 课）。

------

- 迭代器可以通过 `*it` 或 `it->` 解引用，获取其指向的元素。
- 由于 map 内部总是保持有序，map 的首个元素一定是键最小的元素。
- 由于 map 内部总是保持有序，map 的最后一个元素一定是键最大的元素。

例如要查询成绩最好和最坏的学生，可以把成绩当做 key，学生名做 value 依次插入 map，他会帮我们排序：

```cpp
map<int, string> score = {
    {100, "彭于斌"},
    {80, "樱花粉蜜糖"},
    {0, "相依"},
    {60, "Sputnik02"},
};
string poorestStudent = score.begin()->second;   // 成绩最差学生的姓名
string bestStudent = prev(score.end())->second;  // 成绩最好学生的姓名
print("最低分:", poorestStudent);
print("最高分:", bestStudent);
最低分: "相依"
最高分: "彭于斌"
```

> 注：仅当确保 `score.size() != 0` 时才可以解引用，否则 begin() 和 end() 都是虚空迭代器，这时解引用会奔溃。

------

map 的遍历：古代 C++98 的迭代器大法

```cpp
for (map<string, int>::iterator it = m.begin(); it != m.end(); ++it) {
    print("Key:", it->first);
    print("Value:", it->second);
}
```

要特别注意迭代器是一个指向元素的指针，不是元素本身！要用 `->` 而不是 `.`。

------

运用 C++11 的 auto 简写一下：

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    print("Key:", it->first);
    print("Value:", it->second);
}
```

运用 C++17 结构化绑定（structured-binding）语法[1](https://142857.red/book/stl_map/#fn:1)直接拆开 pair 类型：

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    auto [k, v] = *it;
    print("Key:", k);
    print("Value:", v);
}
```

------

map 的遍历：现代 C++17 基于范围的循环（range-based loop）

```cpp
for (auto kv: m) {
    print("Key:", kv.first);
    print("Value:", kv.second);
}
```

同时运用 C++17 结构化绑定语法[1](https://142857.red/book/stl_map/#fn:1)：

```cpp
for (auto [k, v]: m) {
    print("Key:", k);
    print("Value:", v);
}
```

------

如何在遍历的过程中修改值？

古代：

```cpp
map<string, int> m = {
    {"fuck", 985},
    {"rust", 211},
};
for (auto it = m.begin(); it != m.end(); ++it) {
    it->second = it->second + 1;
}
print(m);
{"fuck": 986, "rust": 212}
```

------

如何在遍历的过程中修改值？

现代：

```cpp
map<string, int> m = {
    {"fuck", 985},
    {"rust", 211},
};
for (auto [k, v]: m) {
    v = v + 1;
}
print(m);
{"fuck": 985, "rust": 211}
```

没有成功修改！为什么？

------

```cpp
for (auto [k, v]: m) {
    v = v + 1;
}
```

Range-based loop 只是个花哨语法糖，他相当于：

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    auto [k, v] = *it;
    v = v + 1;
}
```

Structured-binding 也只是个花哨语法糖，他相当于：

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    auto tmp = *it;
    auto k = tmp.first;
    auto v = tmp.second;
    v = v + 1;
}
```

这样保存下来的 v 是个栈上变量，是对原值的一份拷贝，不仅浪费性能，且对 v 的修改不会反映到原 map 中去！

------

```cpp
for (auto &[k, v]: m) {  // 解决方案是在这里加一个小小的 &，让 range-based loop 捕获引用而不是拷贝
    v = v + 1;
}
```

同样是拆除 Range-based loop 的花哨语法糖，相当于：

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    auto &[k, v] = *it;
    v = v + 1;
}
```

继续拆除 Structured-binding 的花哨语法糖，相当于：

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    auto &tmp = *it;
    auto &k = tmp.first;
    auto &v = tmp.second;
    v = v + 1;
}
```

这样保存下来的 v 是个引用，是对原值的引用（用 Rust 的话说叫 borrowed）。不仅避免拷贝的开销节省了性能，而且对 v 的修改会实时反映到原 map 中去。

------

总结，当需要在遍历的同时修改 map 中的值时，要用 `auto &` 捕获引用：

```cpp
for (auto &[k, v]: m) {  // 捕获一个引用，写入这个引用会立即作用在原值上
    v = v + 1;
}
```

即使不需要修改 map 中的值时，也建议用 `auto const &` 避免拷贝的开销：

```cpp
for (auto const &[k, v]: m) {   // 捕获只读的 const 引用，引用避免拷贝开销，const 避免不小心手滑写入
    print(v);
}
```

------

注：即使捕获为 `auto &`，由于 map 的元素类型是 `pair<const K, V>` 所以 K 部分还是会捕获为 `K const &`，无法写入。

```cpp
for (auto &[k, v]: m) {
    k = "key";    // 编译期报错：const 引用不可写入！
    v = 985211;   // OK
}
```

只是如果捕获为 `auto const &` 就两个都不允许写入了。

```cpp
for (auto const &[k, v]: m) {
    k = "key";    // 编译期报错：const 引用不可写入！
    v = 985211;   // 编译期报错：const 引用不可写入！
}
```

------

```cpp
iterator find(K const &k);
const_iterator find(K const &k) const;
```

m.find(key) 函数，根据指定的键 key 查找元素[1](https://142857.red/book/stl_map/#fn:1)。

- 成功找到，则返回指向找到元素的迭代器
- 找不到，则返回 m.end()

由于 STL 传统异能之 end() 虚空索敌，他不可能指向任何值，所以经常作为找不到时候缺省的返回值。

可以用 `m.find(key) != m.end()` 判断一个元素是否存在，等价于 `m.count(key) != 0`。

第二个版本的原型作用是：如果 map 本身有 const 修饰，则返回的也是 const 迭代器。

为的是防止你在一个 const map 里 find 了以后利用迭代器变相修改 map 里的值。

### count 和 contains 没区别

实际上 count 和 contains 函数就是基于 find 实现的，性能没有区别，glibc 源码：

```cpp
#if __cplusplus > 201703L
      /**
       *  @brief  Finds whether an element with the given key exists.
       *  @param  __x  Key of (key, value) pairs to be located.
       *  @return  True if there is an element with the specified key.
       */
      bool
      contains(const key_type& __x) const
      { return _M_t.find(__x) != _M_t.end(); }

      template<typename _Kt>
      auto
      contains(const _Kt& __x) const
      -> decltype(_M_t._M_find_tr(__x), void(), true)
      { return _M_t._M_find_tr(__x) != _M_t.end(); }
#endif
      /**
       *  @brief  Finds the number of elements with given key.
       *  @param  __x  Key of (key, value) pairs to be located.
       *  @return  Number of elements with specified key.
       *
       *  This function only makes sense for multimaps; for map the result will
       *  either be 0 (not present) or 1 (present).
       */
      size_type
      count(const key_type& __x) const
      { return _M_t.find(__x) == _M_t.end() ? 0 : 1; }
// 以下三者等价
m.contains(key)
m.count(key)
m.find(key) != m.end()
```

### end 不能解引用

检查过不是 m.end()，以确认成功找到后，就可以通过 * 运算符解引用获取迭代器指向的值：

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto it = m.find("fuck");  // 寻找 K 为 "fuck" 的元素
if (it != m.end()) {
    auto kv = *it;     // 解引用得到 K-V 对
    print(kv);         // {"fuck", 985}
    print(kv.first);   // "fuck"
    print(kv.second);  // 985
} else {
    print("找不到 fuck！");
}
```

### find 的好处

find 的高效在于，可以把两次查询合并成一次。

保底写法：开销 2logN2log⁡N

```cpp
if (m.count("key")) {    // 第一次查询，只包含"是否找到"的信息
    print(m.at("key"));  // 第二次查询，只包含"找到了什么"的信息
}
```

高效写法：开销 logNlog⁡N

```cpp
auto it = m.find("key"); // 一次性查询
if (it != m.end()) {     // 查询的结果，既包含"是否找到"的信息
    print(it->second);   // 也包含"找到了什么"的信息
}
```

#### C++17 语法糖

C++17 的 if-auto 语法糖如何简化 find 的迭代器判断

```cpp
auto it = m.find("key1");
if (it != m.end()) {
    print(it->second);
}
auto it = m.find("key2");  // 编译器报错：变量 it 重复定义！
if (it != m.end()) {
    print(it->second);
}
```

虽然删去前面的 auto 可以解决问题，但是如果这里是不同类型的 map 就尬了，得另外想一个变量名。

而 C++17 的 if-auto 语法糖捕获的 it 是限制在当前 if 作用域的，不会跑出去和别人发生冲突。

```cpp
if (auto it = m.find("key1"); it != m.end()) {
    print(it->second);
}
if (auto it = m.find("key2"); it != m.end()) {  // 这个变量 it 是局域的，不会和上一个局域的 it 产生名字冲突
    print(it->second);
}
```

等价于：

```cpp
{
    auto it = m.find("key1");
    if (it != m.end()) {
        print(it->second);
    }
}
```

#### 题外话

我给 C++ 标准委员会提一个建议，能不能给迭代器加一个 `operator bool` 代替烦人的 `!= m.end()`？

```cpp
struct iterator {
    _RbTreeNode *node;

    bool operator!=(iterator const &other) const noexcept {
        return node == other.node;
    }

    operator bool() const noexcept {
        return node;
    }
};
```

那样的话就可以直接：

```cpp
if (auto it = m.find("key")) {
    print(it->second);
}
```

因为 if-auto 省略分号后面的条件时，默认就是 `if (auto it = m.find("key"); (bool)it)`

### 对 map 而言，迭代器解引用得到的是 pair

注意 `*it` 解引用得到的是 `pair<const K, V>` 类型的键值对，需要 `(*it).second` 才能获取单独的值 V。

好在 C 语言就有 `->` 运算符作为语法糖，我们可以简写成 `it->second`，与 `(*it).second` 等价。

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto it = m.find("fuck");   // 寻找 K 为 "fuck" 的元素
if (it != m.end()) {
    print(it->second);      // 迭代器有效，可以直接获得值部分 985
} else {
    print("找不到 fuck！");  // 这个分支里不得用 * 和 -> 运算符解引用 it
}
```

大多数情况下我们查询只需要获取值 V 的部分就行了，直接 `it->second` 就可以了✅

> 注意：find 找不到键时，会返回 `m.end()`，这是个无效迭代器，只作为标识符使用（类比 Python 中的 find 有时会返回 -1）。
>
> 没有确认 `it != m.end()` 前，不可以访问 `it->second`！那相当于解引用一个空指针，会造成 segfault（更专业一点说是 UB）。
>
> 记住，一定要在 `it != m.end()` 的分支里才能访问 `it->second` 哦！你得先检查过饭碗里没有老鼠💩之后，才能安心吃饭！
>
> 如果你想让老妈（标准库）自动帮你检查有没有老鼠💩，那就用会自动报错的 at（类比 Python 中的 index 找不到直接报错）。
>
> 之所以用 find，是因为有时饭碗里出老鼠💩，是计划的一部分！例如当有老鼠💩时你可以改吃别的零食。而 at 这个良心老妈呢？一发现老鼠💩就拖着你去警察局报案，零食（默认值）也不让你吃了。今日行程全部取消，维权（异常处理，找上层 try-catch 块）设为第一要务。

------

```cpp
iterator find(K const &k);
const_iterator find(K const &k) const;
```

如果 map 没有 const 修饰，则其 find 返回的 it 也是非 const 迭代器。

```cpp
const map<string, int> cm;
map<string, int>::const_iterator cit = cm.find("key");
print(cit->second);  // OK: 可以读取
cit->second = 1;     // 编译期报错: 不允许写入 const 迭代器指向的值

map<string, int> m;
map<string, int>::iterator it = m.find("key");
print(it->second);   // OK: 可以读取
it->second = 1;      // OK: 可以写入
```

`it->second` 可以写入，it 是迭代器，迭代器类似于指针，写入迭代器指向的 second 就可以修改 map 里的值部分。

`it->first` 是键部分，由于 map 的真正元素类型是 `pair<const K, V>` 所以这部分无法被修改。

------

带默认值的查询

众所周知，Python 中的 dict 有一个 m.get(key, defl) 的功能，效果是当 key 不存在时，返回 defl 这个默认值代替 m[key]，而 C++ 的 map 却没有，只能用一套组合拳代替：

```cpp
m.count(key) ? m.at(key) : defl
```

但上面这样写是比较低效的，相当于查询了 map 两遍，at 里还额外做了一次多余的异常判断。

正常来说是用通用 find 去找，返回一个迭代器，然后判断是不是 end() 决定要不要采用默认值。

```cpp
auto it = m.find(key);
return it != m.end() ? it->second : defl;
```

> 饭碗里发现了老鼠💩？别急着报警，这也在我的预料之中：启用 B 计划，改吃 defl 这款美味零食即可！
>
> 如果是良心老妈 at，就直接启用 C 计划：![Plan C](https://142857.red/book/img/stl/planc.png) 抛出异常然后奔溃了，虽然这很方便我们程序员调试。

------

由于自带默认值的查询这一功能实在是太常用了，为了把这个操作浓缩到一行，我建议同学们封装成函数放到自己的项目公共头文件（一般是 utils.h 之类的名称）里方便以后使用：

```cpp
template <class M>
typename M::mapped_type map_get
( M const &m
, typename M::key_type const &key
, typename M::mapped_type const &defl
) {
  typename M::const_iterator it = m.find(key);
  if (it != m.end()) {
    return it->second;
  } else {
    return defl;
  }
}
int val = map_get(config, "timeout", -1);  // 如果配置文件里不指定，则默认 timeout 为 -1
```

------

这样还不够优雅，我们还可以更优雅地运用 C++17 的函数式容器 optional：

```cpp
template <class M>
std::optional<typename M::mapped_type> map_get
( M const &m
, typename M::key_type const &key
) {
  typename M::const_iterator it = m.find(key);
  if (it != m.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}
```

当找不到时就返回 nullopt，找到就返回含有值的 optional。

调用者可以自行运用 optional 的 value_or 函数[1](https://142857.red/book/stl_map/#fn:1)指定找不到时采用的默认值：

```cpp
int val = map_get(config, "timeout").value_or(-1);
```

如果要实现 at 同样的找不到就自动报错功能，那就改用 value 函数：

```cpp
int val = map_get(config, "timeout").value();
```

optional 具有 `operator bool` 和无异常的 `operator*`，所以也可以配合 if-auto 语法糖使用：

```cpp
if (auto o_val = map_get(config, "timeout")) {
    int val = *o_val;
    print("找到了", val);
} else {
    print("找不到时的处理方案...");
}
```

等价于：

```cpp
auto o_val = map_get(config, "timeout");
if (o_val) {
    int val = *o_val;
    print("找到了", val);
} else {
    print("找不到时的处理方案...");
}
```

------

以上是典型的函数式编程范式 (FP)，C++20 还引入了更多这样的玩意[2](https://142857.red/book/stl_map/#fn:2)，等有空会专门开节课为大家一一介绍。

```cpp
auto even = [] (int i) { return 0 == i % 2; };
auto square = [] (int i) { return i * i; };
for (int i: std::views::iota(0, 6)
          | std::views::filter(even)
          | std::views::transform(square))
    print(i);  // 0 4 16
```

------

现在学习删除元素用的 erase 函数，其原型如下[1](https://142857.red/book/stl_map/#fn:1)：

```cpp
size_t erase(K const &key);
```

指定键值 key，erase 会删除这个键值对应的元素。

返回一个整数，表示删除了多少个元素（只能是 0 或 1）。

------

```cpp
size_t erase(K const &key);
```

erase 运用举例：删除一个元素

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
msg.erase("fuck");
print(msg);
{"fuck": "rust", "hello": "world"}
{"hello": "world"}
```

------

```cpp
size_t erase(K const &key);
```

erase 的返回值和 count 一样，返回成功删除的元素个数，类型为 size_t（无符号 64 位整数）。

由于 map 中同一个键最多只可能有一个元素，取值只能为 0 或 1。

并且 size_t 可以隐式转换为 bool 类型，0 则 false，1 则 true。

------

因此可以直接通过 erase 的返回值是否为 0 判断是否删除成功：

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
if (msg.erase("fuck")) {
    print("删除fuck成功");
} else {
    print("删除fuck失败，键不存在");
}
if (msg.erase("dick")) {
    print("删除dick成功");
} else {
    print("删除dick失败，键不存在");
}
print(msg);
{"fuck": "rust", "hello": "world"}
删除fuck成功
删除dick失败，键不存在
{"hello": "world"}
```

------

```cpp
size_t erase(K const &key);  // 指定键版
iterator erase(iterator it);   // 已知位置版
```

区别：

- 指定键版 erase(key) 实际上需要先调用 find(key) 找到元素位置，然后才能删除，而且还有找不到的可能性。
- 而已知位置的话（比如你已经事先用 find 找到了元素位置），可以用 erase(it) 直接用迭代器作为参数

复杂度不同：

- 指定键版 erase(key) 的时间复杂度：O(logN)O(log⁡N)。
- 已知位置版 erase(it) 的时间复杂度：O(1)+O(1)+，更高效。

其中 ++ 代表这是平摊（Amortized）下来的时间复杂度。

这是因为即使已知位置，erase 有可能涉及树的更新，需要 O(logN)O(log⁡N) 复杂度。

但是大多数情况下需要的更新很少，平均下来是 O(1)O(1) 的。

这种情况就会用记号 O(1)+O(1)+ 来表示。

------

erase(key) 可能是基于 erase(it) 实现的：

```cpp
size_t erase(K const &key) {  // 小彭老师猜想标准库内部
    auto it = this->find(key);  // O(log N)
    if (it != this->end()) {
        this->erase(it);        // O(1)+
        return 1;  // 找到了，删除成功
    } else {
        return 0;  // 找不到，没有删除
    }
}  // 开销大的 find(key) 会覆盖小的 erase(it)，所以 erase(key) 的总复杂度为 O(log N)
```

------

指定位置版 erase(it) 返回的是删除元素的下一个元素位置。

由于 map 内部保持键从小到大升序排列，所谓的下一个就是键比当前键大一个的元素，例如：

```json
{"answer": 42, "hello": 985, "world": 211}
```

- erase(find(“answer”)) 会返回指向 “hello” 的迭代器，因为 “hello” 最接近且大于 “answer”。
- erase(find(“hello”)) 会返回指向 “world” 的迭代器，因为 “world” 最接近且大于 “hello”。
- erase(find(“world”)) 会返回 end()，因为 “world” 已经是最大键，没有下一个。

此外 erase(it) 还有性能上的优势：

- 指定位置版 erase(it) 的复杂度是 O(1)+O(1)+
- 指定键版 erase(key) 的复杂度是 O(logN)O(log⁡N)

当已知指向要删除元素的迭代器时（例如先通过 find 找到），直接指定那个迭代器比指定键参数更高效。

删除成绩最差的学生：

```cpp
score.erase(score.begin());
```

------

## 一边遍历一边删除部分元素

常见需求场景：一边遍历一边删除部分元素（错误示范）

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
for (auto const &[k, v]: msg) {
    if (k.starts_with("fuck")) {
        msg.erase(k);  // 遍历过程中删除当前元素，会导致正在遍历中的迭代器失效，奔溃
    }
}
print(msg);
Segmentation fault (core dumped)
```

------

引出问题：迭代器失效

- 每当往 map 中插入新元素时，原先保存的迭代器不会失效。
- 删除 map 中的其他元素时，也不会失效。
- **只有当删除的刚好是迭代器指向的那个元素时，才会失效**。

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto it = m.find("fuck");
m["dick"] = 211;
print(it->second);  // 没有失效，打印 985
m.erase("dick");
print(it->second);  // 没有失效，打印 985
m.erase("fuck");
print(it->second);  // 没有失效，打印 985
```

------

map 比起 unordered_map 来，已经是非常稳定，随便增删改查都不会迭代器失效。

只有一个例外：删除的元素刚好是迭代器指向的。

你拿着个你朋友家的地址，结果你一发 RPG 导弹把他家炸了，还摸不着头脑“奇怪，明明就是这个地址呀”，这时确实无论如何都不能避免失效，不能怪 map。

而刚刚的案例中，我们删除的恰好就是当前正在遍历的迭代器正在指向的那个元素（即使你用了 range-based loop 语法糖他背后还是迭代器遍历）。

而当你对着一个失效的迭代器执行 `++it` 时，就产生了 segfault 错误。因为红黑树的迭代器要找到“下一个”节点，需要访问这个红黑树节点中存的 `next` 指针，而这个红黑树节点都已经删除了已经析构了已经释放内存了，里面存的 `next` 指针也已经释放，被其他系统数据覆盖，这时会访问到错误的指针——野指针。

------

所以《好友清除计划》完整的剧情是：

你有好多朋友，今天你要把他们全炸了。

1号朋友家里有一个字条，写着2号朋友家的地址。

2号朋友家里有一个字条，写着3号朋友家的地址。

…

你拿着1号朋友家的地址，一发 RPG 导弹把他家炸了。然后你现在突然意识到需要2号朋友家的地址，但是1号朋友家已经被你炸了，你傻乎乎进入燃烧的1号朋友家，被火烧死了。

```cpp
for (auto it = m.begin(); it != m.end(); ++it /* 进入燃烧中的1号朋友家 */) {
    m.erase(it);  // 一发 RPG 导弹炸毁1号朋友家
}
```

------

你拿着1号朋友家的地址，一发 RPG 导弹把他家炸了。然后你现在突然意识到需要2号朋友家的地址，但是1号朋友家已经被你炸了，你傻乎乎进入燃烧的1号朋友家，被火烧死了。

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    m.erase(it);
    // it 已经失效！
}
```

正确的做法是，先进入1号朋友家，安全取出写着2号朋友家地址的字条后，再来一发 RPG 把1号朋友家炸掉。这样才能顺利找到2号朋友家，以此类推继续拆3号……

```cpp
for (auto it = m.begin(); it != m.end(); ) {
    auto next_it = it;  // 先进入1号朋友的家
    ++next_it;          // 拿出写有2号朋友家地址的字条
    m.erase(it);        // 再发射 RPG 导弹
    it = next_it;       // 前往2号朋友家
}
```

------

注意到 erase 会返回删除元素的下一个元素的迭代器，也就是说这个 RPG 导弹非常智能，好像他就是专为《好友清除计划》设计的一样：他能在炸毁你朋友的房屋前，自动拿到其中的字条，并把他通过“弹射座椅”弹出来送到门外的你手上，把纸条安全送出来后，再爆炸摧毁你朋友的房屋。这样你就不用冒险进入燃烧的房屋拿字条（迭代器失效导致 segfault），也不用先劳烦您自己先进去一趟房屋拿字条了（上一页中那样提前保存 next_it）。

```cpp
for (auto it = m.begin(); it != m.end(); ) {
    it = m.erase(it);        // 这款 RPG 导弹“智能地”在摧毁你朋友的房屋同时把其中的字条拿出来了!?
}
```

> 只是注意这里 for 循环的步进条件 `++it` 要删掉，因为智能的 RPG 导弹 `it = m.erase(it)` 已经帮你步进了。

------

一边遍历一边删除部分元素（正解[1](https://142857.red/book/stl_map/#fn:1)）

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
for (auto it = m.begin(); it != m.end(); ) {  // 没有 ++it
    auto const &[k, v] = *it;
    if (k.starts_with("fuck")) {
        it = msg.erase(it);
    } else {
        ++it;
    }
}
print(msg);
{"good": "job", "hello": "world"}
```

------

不奔溃

```cpp
for (auto it = m.begin(); it != m.end(); ) {
    auto const &[k, v] = *it;
    if (k.starts_with("fuck")) {
        it = msg.erase(it);
    } else {
        ++it;
    }
}
```

奔溃

```cpp
for (auto it = m.begin(); it != m.end(); ++it) {
    auto const &[k, v] = *it;
    if (k.starts_with("fuck")) {
        msg.erase(it);
        // 或者 msg.erase(k);
    }
}
```

------

### C++20 更好的写法：erase_if

批量删除符合条件的元素（C++20[1](https://142857.red/book/stl_map/#fn:1)）

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
std::erase_if(msg, [&] (auto const &kv) {
    auto &[k, v] = kv;
    return k.starts_with("fuck");
});
print(msg);
{"good": "job", "hello": "world"}
```

------

如果你搞不懂迭代器这些，这里我提供一个保底写法，先把键提前保存到一个 vector 中去：

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
vector<string> keys;             // vector 或者 set 都可以
for (auto const &[k, v]: msg) {  // 先把所有键提前拷贝到临时 vector 里
    keys.push_back(k);
}
for (auto const &k: keys) {      // 遍历刚才保存的键
    if (k.starts_with("fuck")) {
        msg.erase(k);            // 键值对已经提前深拷贝到临时 vector 里，这时删除 map 里的键不会奔溃
    }
}
```

> 小彭老师，永远的祛魅大师。

------

还是搞不懂的话，也可以新建一个 map，条件反之，把不需要删除的元素插入新 map，过滤出需要保留的元素，最后再一次性用新 map 覆盖旧 map。

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
map<string, string> newmsg;
for (auto const &[k, v]: msg) {
    if (!k.starts_with("fuck")) {   // 注意这里条件反了，不需要删除的才插入 newmsg
        newmsg[k] = v;
    }
}
msg = std::move(newmsg);        // 覆盖旧的 map，用更高效的移动赋值函数，O(1) 复杂度
```

> 小彭老师，永远的保底大师。

------

接下来开始学习如何插入元素，map 的成员 insert 函数原型如下[1](https://142857.red/book/stl_map/#fn:1)：

```cpp
pair<iterator, bool> insert(pair<const K, V> const &kv);
pair<iterator, bool> insert(pair<const K, V> &&kv);
```

他的参数类型就是刚刚介绍的 `value_type`，也就是 `pair<const K, V>`。

pair 是一个 STL 中常见的模板类型，`pair<K, V>` 有两个成员变量：

- first：K 类型，表示要插入元素的键
- second：V 类型，表示要插入元素的值

我称之为”键值对”。

------

试着用 insert 插入键值对：

```cpp
map<string, int> m;
pair<string, int> p;
p.first = "fuck";  // 键
p.second = 985;    // 值
m.insert(p);  // pair<string, int> 可以隐式转换为 insert 参数所需的 pair<const string, int>
print(m);
```

结果：

```json
{"fuck": 985}
```

------

简化 insert

1. 直接使用 pair 的构造函数，初始化 first 和 second

```cpp
pair<string, int> p("fuck", 985);
m.insert(p);
```

1. 不用创建一个临时变量，pair 表达式直接作为 insert 函数的参数

```cpp
m.insert(pair<string, int>("fuck", 985));
```

1. 可以用 `std::make_pair` 这个函数，自动帮你推导模板参数类型，省略 `<string, int>`

```cpp
m.insert(make_pair("fuck", 985));  // 虽然会推导为 pair<const char *, int> 但还是能隐式转换为 pair<const string, int>
```

1. 由于 insert 函数原型已知参数类型，可以直接用 C++11 的花括号初始化列表 {…}，无需指定类型

```cpp
m.insert({"fuck", 985});           // ✅
```

------

因此，insert 的最佳用法是：

```cpp
map<K, V> m;
m.insert({"key", "val"});
```

insert 插入和 [] 写入的异同：

- 同：当键 K 不存在时，insert 和 [] 都会创建键值对。
- 异：当键 K 已经存在时，insert 不会覆盖，默默离开；而 [] 会覆盖旧的值。

例子：

```cpp
map<string, string> m;
m.insert({"key", "old"});
m.insert({"key", "new"});  // 插入失败，默默放弃不出错
print(m);
{"key": "old"}
map<string, string> m;
m["key"] = "old";
m["key"] = "new";        // 已经存在？我踏马强行覆盖！
print(m);
{"key": "new"}
```

------

insert 的返回值是 `pair<iterator, bool>` 类型，~~STL 的尿性：在需要一次性返回两个值时喜欢用 pair~~。

这又是一个 pair 类型，其具有两个成员：

- first：iterator 类型，是个迭代器
- second：bool 类型，表示插入成功与否，如果发生键冲突则为 false

其中 first 这个迭代器指向的是：

- 如果插入成功（second 为 true），指向刚刚成功插入的元素位置
- 如果插入失败（second 为 false），说明已经有相同的键 K 存在，发生了键冲突，指向已经存在的那个元素

------

其实 insert 返回的 first 迭代器等价于插入以后再重新用 find 找到刚刚插入的那个键，只是效率更高：

```cpp
auto it = m.insert({k, v}).first;  // 高效，只需遍历一次
m.insert({k, v});     // 插入完就忘事了
auto it = m.find(k);  // 重新遍历第二次，但结果一样
```

参考 C 编程网[1](https://142857.red/book/stl_map/#fn:1)对 insert 返回值的解释：

> 当该方法将新键值对成功添加到容器中时，返回的迭代器指向新添加的键值对；
>
> 反之，如果添加失败，该迭代器指向的是容器中和要添加键值对键相同的那个键值对。

------

可以用 insert 返回的 second 判断插入多次是否成功：

```cpp
map<string, string> m;
print(m.insert({"key", "old"}).second);  // true
print(m.insert({"key", "new"}).second);  // false
m.erase("key");     // 把原来的 {"key", "old"} 删了
print(m.insert({"key", "new"}).second);  // true
```

也可以用 structured-binding 语法拆解他返回的 `pair<iterator, bool>`：

```cpp
map<string, int> counter;
auto [it, success] = counter.insert("key", 1);  // 直接用
if (!success) {  // 如果已经存在，则修改其值+1
    it->second = it->second + 1;
} else {  // 如果不存在，则打印以下信息
    print("created a new entry!");
}
```

以上这一长串代码和之前“优雅”的计数 [] 等价：

```cpp
counter["key"]++;
```

### insert_or_assign

在 C++17 中，[] 写入有了个更高效的替代品 insert_or_assign[1](https://142857.red/book/stl_map/#fn:1)：

```cpp
pair<iterator, bool> insert_or_assign(K const &k, V v);
pair<iterator, bool> insert_or_assign(K &&k, V v);
```

正如他名字的含义，“插入或者写入”：

- 如果 K 不存在则创建（插入）
- 如果 K 已经存在则覆盖（写入）

用法如下：

```cpp
m.insert_or_assign("key", "new");  // 与 insert 不同，他不需要 {...}，他的参数就是两个单独的 K 和 V
```

返回值依旧是 `pair<iterator, bool>`。由于这函数在键冲突时会覆盖，按理说是必定成功了，因此这个 bool 的含义从“是否插入成功”变为“是否创建了元素”，如果是创建的新元素返回true，如果覆盖了旧元素返回false。

------

#### insert_or_assign 的优势

看来 insert_or_assign 和 [] 的效果完全相同！都是在键值冲突时覆盖旧值。

既然 [] 已经可以做到同样的效果，为什么还要发明个 insert_or_assign 呢？

insert_or_assign 的优点是**不需要调用默认构造函数**，可以提升性能。

其应用场景有以下三种情况：

- ⏱ 您特别在乎性能
- ❌ 有时 V 类型没有默认构造函数，用 [] 编译器会报错
- 🥵 强迫症发作

否则用 [] 写入也是没问题的。

而且 insert_or_assign 能取代 [] 的岗位仅限于纯写入，之前 `counter[key]++` 这种“优雅”写法依然是需要用 [] 的。

#### 效率问题

创建新键时，insert_or_assign 更高效。

##### []

```cpp
map<string, string> m;
m["key"] = "old";
m["key"] = "new";
print(m);
{"key": "new"}
```

覆盖旧键时，使用 [] 造成的开销：

- 调用移动赋值函数 `V &operator=(V &&)`

创建新键时，使用 [] 造成的开销：

- 调用默认构造函数 `V()`
- 调用移动赋值函数 `V &operator=(V &&)`

##### insert_or_assign

```cpp
map<string, string> m;
m.insert_or_assign("key", "old");
m.insert_or_assign("key", "new");
print(m);
{"key": "new"}
```

覆盖旧键时，使用 insert_or_assign 造成的开销：

- 调用移动赋值函数 `V &operator=(V &&)`

创建新键时，使用 insert_or_assign 造成的开销：

- 调用移动构造函数 `V(V &&)`

#### 那我应该用什么

总结，如果你有性能强迫症，并且是 C++17 标准：

- 写入用 insert_or_assign
- 读取用 at

如果没有性能强迫症，或者你的编译器不支持 C++17 标准：

- 写入用 []
- 读取用 at

最后，如果你是还原论者，只需要 find 和 insert 函数就是完备的了，别的函数都不用去记。所有 at、[]、insert_or_assign 之类的操作都可以通过 find 和 insert 的组合拳实现，例如刚刚我们自定义的 map_get。

#### insert_or_assign vs insert：顺序问题

回顾之前的反向查找表，如果有重复，如何区分找第一个还是最后一个？

构建反向查找表，找到最后一个的下标：

```cpp
for (size_t i = 0; i < arr.size(); i++) {
    arrinv.insert_or_assign(arr[i], i);
    // 等价于 arrinv[arr[i]] = i;
}
```

构建反向查找表，找到第一个的下标：

```cpp
for (size_t i = 0; i < arr.size(); i++) {
    arrinv.insert({arr[i], i});
}
```

## 批量 insert

刚刚介绍的那些 insert 一次只能插入一个元素，insert 还有一个特殊的版本，用于批量插入一系列元素。

```cpp
template <class InputIt>
void insert(InputIt beg, InputIt end);
```

参数[1](https://142857.red/book/stl_map/#fn:1)是两个迭代器 beg 和 end，组成一个区间，之间是你要插入的数据。

该区间可以是任何其他容器的 begin() 和 end() 迭代器——那会把该容器中所有的元素都插入到本 map 中去。

例如，把 vector 中的键值对批量插入 map：

```cpp
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
};
map<string, int> config;
config.insert(kvs.begin(), kvs.end());
print(config);  // {"delay": 211, "timeout": 985}
```

### 批量 insert 同样遵循不覆盖原则

注：由于 insert 不覆盖的特性，如果 vector 中有重复的键，则会以键第一次出现时的值为准，之后重复出现的键会被忽视。

```cpp
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
    {"delay", 666},
    {"delay", 233},
    {"timeout", 996},
};
map<string, int> config;
config.insert(kvs.begin(), kvs.end());
print(config);
{"delay": 211, "timeout": 985}
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
    {"delay", 666},
    {"delay", 233},
    {"timeout", 996},
};
map<string, int> config = {
    {"timeout", 404},
};
config.insert(kvs.begin(), kvs.end());
print(config);

vector<unique_ptr<int>> v;
{"delay": 211, "timeout": 404}
```

### 批量 insert 实现 map 合并

批量 insert 运用案例：两个 map 合并

这个批量 insert 输入的迭代器可以是任何容器，甚至可以是另一个 map 容器。

运用这一点可以实现两个 map 的并集操作。

```cpp
map<string, int> m1 = {  // 第一个 map
    {"answer", 42},
    {"timeout", 7},
};
map<string, int> m2 = {  // 第二个 map
    {"timeout", 985},
    {"delay", 211},
};
m1.insert(m2.begin(), m2.end());  // 把 m2 的内容与 m1 合并，结果写回到 m1
print(m1);
{"answer": 42, "delay": 211, "timeout": 7}
```

注：还是由于 insert 不覆盖的特性，当遇到重复的键时（例如上面的 “timeout”），会以 m1 中的值为准。

#### 就地写入！

使用 `m1.insert(m2.begin(), m2.end())` 后，合并的结果会就地写入 m1。

如果希望合并结果放到一个新的 map 容器中而不是就地修改 m1，请先自行生成一份 m1 的深拷贝：

```cpp
const map<string, int> m1 = {  // 第一个 map，修饰有 const 禁止修改
    {"answer", 42},
    {"timeout", 7},
};
const map<string, int> m2 = {  // 第二个 map，修饰有 const 禁止修改
    {"timeout", 985},
    {"delay", 211},
};
auto m12 = m1;  // 生成一份 m1 的深拷贝 m12，避免 insert 就地修改 m1
m12.insert(m2.begin(), m2.end());
print(m12);     // m1 和 m2 的合并结果
{"answer": 42, "delay": 211, "timeout": 7}
```

------

#### 批量 insert 优先保留已经有的

```cpp
auto m12 = m1;
m12.insert(m2.begin(), m2.end());
print(m12);     // m1 和 m2 的合并结果，键冲突时优先取 m1 的值
{"answer": 42, "delay": 211, "timeout": 7}
```

刚刚写的 m1 和 m2 合并，遇到重复时会优先采取 m1 里的值，如果希望优先采取 m2 的呢？反一反就可以了：

```cpp
auto m12 = m2;
m12.insert(m1.begin(), m1.end());
print(m12);     // m1 和 m2 的合并结果，键冲突时优先取 m2 的值
{"answer": 42, "delay": 211, "timeout": 985}
```

要是学不会批量 insert，那手写一个 for 循环遍历 m2，然后 m1.insert_or_assign(k2, v2) 也是可以的，总之要懂得变通，动动脑，总是有保底写法的。

#### 其他操作：交集、并集、差集等

有同学就问了，这个 insert 实现了 map 的并集操作，那交集操作呢？这其实是 set 的常规操作而不是 map 的：

- set_intersection（取集合交集）
- set_union（取集合并集）
- set_difference（取集合差集）
- set_symmetric_difference（取集合对称差集）

非常抱歉在之前的 set 课中完全没有提及，因为我认为那是 `<algorithm>` 头文件里的东西。

不过别担心，之后我们会专门有一节 algorithm 课详解 STL 中这些全局函数——我称之为算法模板，因为他提供了很多常用的算法，对小彭老师这种算法弱鸡而言，实在非常好用，妈妈再也不用担心我的 ACM 奖杯。

在小彭老师制作完 algorithm 课之前，同学们可以自行参考 https://blog.csdn.net/u013095333/article/details/89322501 提前进行学习这四个函数。

```cpp
std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::inserter(C, C.begin()));  // C = A U B
```

> 注：set_union 仅仅要求输入的两个区间有序，可以是 set，也可以是排序过的 vector。而且通过重载运算符或者指定 compare 函数，同样可以模拟 map 只对 key 部分排序的效果——参考 thrust::sort_by_key，但很可惜 STL 没有这函数，需要自定义 compare 函数模拟。

同样地，这些操作也是很容易基于 map 的 contains、erase、insert 等接口“动动脑”写出保底写法：

```cpp
map<string, string> m12;
for (const auto &[k, v] : m2) {
    if (m1.contains(k)) { // 此处为 count 也可以
        // 交集操作：如果 m1 和 m2 都有这个键，则插入他俩的交集 m12
        m12.insert({k, v});
    }
}
```

### insert 一个初始化列表

C++11 还引入了一个以初始化列表（initializer_list）为参数的 insert 版本：

```cpp
void insert(initializer_list<pair<const K, V>> ilist);
```

用法和 map 的构造函数一样，还是用花括号列表：

```cpp
map<string, int> m = {  // 初始化时就插入两个元素
    {"answer", 42},
    {"timeout", 7},
};
m.insert({              // 批量再插入两个新元素
    {"timeout", 985},   // "timeout" 发生键冲突，根据 insert 的特性，不会覆盖
    {"delay", 211},
});
{"answer": 42, "delay": 211, "timeout": 7}
```

注：这里还是和逐个 insert 一样，重复的键 “timeout” 没有被覆盖，依旧了保留原值。

------

#### 小彭老师锐评批量 insert 有什么用

```cpp
m.insert({
    {"timeout", 985},
    {"delay", 211},
});
```

总之这玩意和分别调用两次 insert 等价：

```cpp
m.insert({"timeout", 985});
m.insert({"delay", 211});
```

如果需要覆盖原值的批量写入，还是得乖乖写个 for 循环调用 [] 或 insert_or_assign。

问：既然和批量插入没什么区别，复杂度也一样是 O(logN)O(log⁡N)，那批量 insert 究竟还有什么存在的必要呢？map 又不像 vector 一个个分别插入会变成 O(N2)O(N2) 复杂度，确实需要提供个批量插入的方法。

答：

1. 是为了统一，既然 vector 都有批量 insert，那 set 和 map 也得有才符合完美主义美学，而且用他来合并两个 map 也很方便。
2. 复杂度并不一样，当输入已经有序时，批量 insert 会比逐个 insert 更快，只需 O(N)O(N) 而不是 O(NlogN)O(Nlog⁡N)；如果输入无序，那么依然是 O(NlogN)O(Nlog⁡N)，稍后会讲原理。

### operator= 也支持初始化列表

```cpp
map &operator=(initializer_list<pair<const K, V>> ilist);
```

map 也支持赋值函数，不仅有 map 自己给自己赋值的移动赋值和拷贝赋值函数，还有从列表初始化的函数。

用法是等号右边一个花括号列表，作用是清空原有内容，直接设为一个全新的 map：

```cpp
map<string, int> m = {  // 初始化时就插入两个元素
    {"answer", 42},
    {"timeout", 7},
};
m = {                   // 原有内容全部清空！重新插入两个新元素
    {"timeout", 985},
    {"delay", 211},
};
{"delay": 211, "timeout": 985}
```

> 相当于 clear 了再重新 insert，原有的 “answer” 键也被删掉了。

#### 赋值函数和构造函数概念辨析

要注意赋值函数 `operator=(initializer_list)` 和构造函数 `map(initializer_list)` 是不同的。

构造函数是初始化时调用的（无论有没有 = 号），赋值函数是后期重新赋值时调用的。

```cpp
map<string, int> m{    // 构造函数 map(initializer_list)
    {"answer", 42},
    {"timeout", 7},
};
map<string, int> m = {  // 虽然有等号，但这里是初始化语境，调用的依然是构造函数 map(initializer_list)
    {"answer", 42},
    {"timeout", 7},
};
m = {                   // m 已经初始化过，这里是重新赋值，才是赋值函数 operator=(initializer_list)
    {"timeout", 985},
    {"delay", 211},
};
```

如果一个类要支持初始化，又要支持后期重新赋值，那么构造函数和赋值函数都要实现。

但也可以选择只定义 `operator=(map &&)` 移动赋值函数而不定义 `operator=(initializer_list)`。这样当试图 `operator=(initializer_list)` 时，会匹配到 `map(initializer_list)` 这个隐式构造函数来转换，然后调用到 `operator=(map &&)`。标准库选择将两个都定义可能是处于避免一次 map 移动的效率考量。

#### assign 函数

map 还有一个 assign 函数，他和 `operator=` 等价：

```cpp
void assign(initializer_list<pair<const K, V>> ilist);
```

assign 的额外好处是他拥有两个迭代器参数组成区间的版本，和批量 insert 类似，只不过 assign 会清除已有的元素。

```cpp
template <class InputIt>
void assign(InputIt first, InputIt last);
```

和 `operator=(map(first, last))` 等价。

## 带插入位置提示的 insert

```cpp
iterator insert(const_iterator pos, pair<K, V> const &kv);
```

这又是 insert 函数的一个重载版，增加了 pos 参数提示插入位置，官方文档称[1](https://142857.red/book/stl_map/#fn:1)：

> Inserts value in the position as close as possible to the position just prior to pos.
>
> 把元素（键值对）插入到位于 pos 之前，又离 pos 尽可能近的地方。

然而 map 作为红黑树应该始终保持有序，插入位置可以由 K 唯一确定，为啥还要提示？

是为了在已知要插入的大致位置时，能够提升性能。

> （带提示的 insert 版本）中传入的迭代器，仅是给 map 容器提供一个建议，并不一定会被容器采纳。该迭代器表明将新键值对添加到容器中的位置。需要注意的是，新键值对添加到容器中的位置，并不是此迭代器说了算，最终仍取决于该键值对的键的值。[2](https://142857.red/book/stl_map/#fn:2)
>
> 也就是说这玩意还不一定管用，只是提示性质的（和 mmap 函数的 start 参数很像，你可以指定，但只是个提示，指定了不一定有什么软用，具体什么地址还是操作系统说了算，他从返回值里给你的地址才是正确答案）。例如已知指向 “key” 的迭代器，想要插入 “kea”，那么指定指向 “key” 的迭代器就会让 insert 能更容易定位到 “kea” 要插入的位置。

### 复杂度分类讨论

```cpp
iterator insert(const_iterator pos, pair<K, V> const &kv);
```

- 当插入位置 pos 提示的准确时，insert 的复杂度可以低至 O(1)+O(1)+。
- 当插入位置 pos 提示不准确时，和普通的 insert 一样，还是 O(logN)O(log⁡N)。
- 返回指向成功插入元素位置的迭代器。

想想看，这三个看似不相干的特性，能有什么用呢？

可以让已经有序数据的批量插入更高效！

众所周知，普通的批量 insert 复杂度为 O(NlogN)O(Nlog⁡N)。

```cpp
vector<pair<int, int>> arr;
map<int, int> tab;
for (auto const &[k, v]: arr) {
    tab.insert({k, v});               // O(log N)
}  // 总共 O(N log N)
```

假如输入本就有序，带提示的批量 insert 复杂度可以降低到 O(N)O(N)。

如果输入无序，带提示的批量 insert 复杂度依然是 O(NlogN)O(Nlog⁡N) 不变。

```cpp
vector<pair<int, int>> arr;
map<int, int> tab;
auto hint = tab.begin();
for (auto const &[k, v]: arr) {
    hint = tab.insert(hint, {k, v});  // 平均 O(1)
}  // 总共 O(N)
```

想一想，为什么？

### 小学生的趣味早操

你是一名小学老师，马上就要出早操了，为应付领导面子，你需要给你的学生排队，根据个子从矮到高排列。

不过这所小学的学生都比较懒散，有的来得早有的来得晚，而且来的早晚顺序和他们的高矮无关。

你本来打算等所有人到齐之后再一次性完成排序（std::sort）的，但是同学来的时间实在是太分散了：明明 8 点就要出早操，最晚的同学却 7 点 59 分才到达。意味着你需要一直干等着这个懒散的同学，最后在 1 分钟时间内临时抱佛脚，完成快速排序。这是不可能的，只能在同学陆续抵达的同时进行排序，这就是堆排序，一边插入一边排序，每次插入后都保证有序，与插入排序不同他使用堆内存中的节点而不是数组避免昂贵的数组平移操作。

每当来一个学生，你就得把他插入到现有的一个已经排好的队伍中去。

如何确定插入的位置？二分法。先从现有队伍的最中间（1/2 处）开始，比较中间这个学生和新来的学生哪个高哪个矮，如果发现新来的学生高，则继续从队列的 3/4 处那个同学开始比高矮，如果新来的学生矮就从队列的 1/4 处继续比较。以此类推，最终唯一确定新同学要插入的位置。因为每次确定的范围就小一半，所以最多只需要 logNlog⁡N 次比较就可以成功插入，其中 NN 是当前已有学生的数量。

总共要来 NN 名学生，则你总共需要比较 NlogNNlog⁡N 次。能不能优化？让我们小彭老师省力点？

### 小学生来的顺序已经有序的情况

后来你发现一个规律，似乎学生来的早晚顺序和高矮有关：矮小的同学喜欢起的早，高大的同学喜欢起的晚。

知道这个规律后，你改变你的策略：二分法时，不是先从最中间（1/2 处）开始查找，而是从最末尾开始查找。因为矮小同学会早到，导致每次新来的同学往往总是队列中最高的那一个。所以可以从队伍的末尾（最高的地方）开始找，例如有 64 名同学则优先和 65/64 处比较，找不到再往上一级和 31/32 处比较。

这个策略也有缺点：对于早晚顺序和高矮无关、甚至负相关的情况，每次插入的消耗就会变成 2logN2log⁡N 了。

最终我们决定采用的策略是：不是从中间，也不是从开头，也不是从末尾，而是**记住上一次成功插入的位置**，下一次从上一次成功插入的位置开始找。这个记忆的插入位置，就是刚刚代码中那个位置提示迭代器 hint。

这正是我们代码的写法：

```cpp
hint = tab.insert(hint, {k, v});
```

实际上，insert 的批量插入版 `insert(arr.begin(), arr.end())` 内部就会使用这种带提示的方式，逐个插入。

```cpp
vector<pair<int, int>> arr;
```

## 分奴 emplace

insert 的究极分奴版（不推荐）：emplace

```cpp
template <class ...Args>
pair<iterator, bool> emplace(Args &&...args);
```

虽然变长参数列表 `Args &&...args` 看起来很酷，然而由于 map 的特殊性，其元素类型是 `pair<const K, V>`，而 pair 的构造函数只有两个参数，导致实际上这个看似炫酷的变长参数列表往往只能接受两个参数，因此这个函数的调用方法实际上只能是：

```cpp
pair<iterator, bool> emplace(K k, V v);
```

写法：

```cpp
m.emplace(key, val);
```

等价于：

```cpp
m.insert({key, val});
```

返回值还是 `pair<iterator, bool>`，其意义和 insert 一样，不再赘述。

------

### emplace_hint

insert 的宇宙无敌分奴版（不推荐）：emplace_hint[1](https://142857.red/book/stl_map/#fn:1)

```cpp
template <class ...Args>
iterator emplace_hint(const_iterator pos, Args &&...args);
```

写法：

```cpp
m.emplace_hint(pos, key, val);
```

等价于：

```cpp
m.insert(pos, {key, val});
```

之所以要分两个函数名 emplace 和 emplace_hint 而不是利用重载区分，是因为直接传入 pos 会被 emplace 当做 pair 的构造参数，而不是插入位置提示。

- emplace 对应于普通的 `insert(pair<const K, V>)` 这一重载。
- emplace_hint 对应于带插入位置提示的 `insert(const_iterator, pair<const K, V>)` 这一重载。
- emplace_hint 的返回类型也和带插入位置提示的 insert 一样，是单独一个 iterator。

### emplace 的原理和优点

```cpp
template <class ...Args>
pair<iterator, bool> emplace(Args &&...args);
```

emplace 对于 set，元素类型是比较大的类型时，例如 `set<array<int, 100>>`，可能确实能起到减少移动构造函数开销的作用。

但是这个 map 他的元素类型不是直接的 V 而是一个 pair，他分的是 pair 的构造函数，没有用，V 部分还是会造成一次额外的移动开销，所以这玩意除了妨碍类型安全和可读性以外，没有任何收益。

- set 可以用 emplace/emplace_hint。
- vector 可以用 emplace_back。
- 不建议在 map 上使用 emplace/emplace_hint，请改用 try_emplace。

## try_emplace 更好

emplace 只支持 pair 的就地构造，这有什么用？我们要的是 pair 中值类型的就地构造！这就是 try_emplace 的作用了，他对 key 部分依然是传统的移动，只对 value 部分采用就地构造。

> 这是观察到大多是值类型很大，急需就地构造，而键类型没用多少就地构造的需求。例如 `map<string, array<int, 1000>>`
>
> 如果想不用 try_emplace，完全基于 emplace 实现针对值 value 的就地构造需要用到 std::piecewise_construct 和 std::forward_as_tuple，非常麻烦。

insert 的托马斯黄金大回旋分奴版：try_emplace（C++17 引入）

```cpp
template <class ...Args>
pair<iterator, bool> try_emplace(K const &k, Args &&...args);
```

写法：

```cpp
m.try_emplace(key, arg1, arg2, ...);
```

他等价于：

```cpp
m.insert({key, V(arg1, arg2, ...)});
```

后面的变长参数也可以完全没有：

```cpp
m.try_emplace(key);
```

他等价于调用 V 的默认构造函数：

```cpp
m.insert({key, V()});
```

由于 emplace 实在是憨憨，他变长参数列表就地构造的是 pair，然而 pair 的构造函数正常不就是只有两个参数吗，变长没有用。实际有用的往往是我们希望用变长参数列表就地构造值类型 V，对 K 部分并不关系。因此 C++17 引入了 try_emplace，其键部分保持 `K const &`，值部分采用变长参数列表。

我的评价是：这个比 emplace 实用多了，如果要与 vector 的 emplace_back 对标，那么 map 与之对应的一定是 try_emplace。同学们如果要分奴的话还是建议用 try_emplace。

### try_emplace 可以避免移动！

insert 类函数总是不可避免的需要移动构造：先在函数中构造出临时对象，然后构造到真正的 pair 上。

而 try_emplace 可以允许你就地构造值对象，避免移动造成开销。

try_emplace 第一个参数是键，第二个开始是传给构造函数的参数，如只有第一个参数则是调用无参构造函数。

```cpp
struct MyClass {
    MyClass() { printf("MyClass()\n"); }
    MyClass(int i) { printf("MyClass(int)\n"); }
    MyClass(const char *p, float x) { printf("MyClass(const char *, float)\n"); }
};

map<string, MyClass> m;
m.try_emplace("key");                 // MyClass()
m.try_emplace("key", 42);             // MyClass(int)
m.try_emplace("key", "hell", 3.14f);  // MyClass(const char *, float)
// 等价于：
m.insert({"key", MyClass()});                // MyClass()
m.insert({"key", MyClass(42)});              // MyClass(int)
m.insert({"key", MyClass("hell", 3.14f)});   // MyClass(const char *, float)
```

对于移动开销较大的类型（例如 `array<int, 1000>`），try_emplace 可以避免移动；对于不支持移动构造函数的值类型，就必须使用 try_emplace 了。

### 谈谈 try_emplace 的优缺点

```cpp
// 以下两种方式效果等价，只有性能不同
m.try_emplace(key, arg1, arg2, ...);           // 开销：1次构造函数
m.insert({key, V(arg1, arg2, ...)});           // 开销：1次构造函数 + 2次移动函数
m.insert(make_pair(key, V(arg1, arg2, ...)));  // 开销：1次构造函数 + 3次移动函数
```

但是由于 try_emplace 是用圆括号帮你调用的构造函数，而不是花括号初始化。

导致你要么无法省略类型，要么你得手动定义类的构造函数：

```cpp
struct Student {  // 没有构造函数，只能用花括号语法进行初始化
    string sex;
    int age;
};
map<string, Student> m;
m.insert({"彭于斌", {"自定义", 22}});            // OK: insert 参数类型已知，Student 可以省略不写，但是会造成 2 次移动
m.try_emplace("彭于斌", "自定义", 22);           // ERROR: 不存在构造函数 Student(string, int)；C++20 开始则 OK: C++20 起聚合初始化同时支持花括号和圆括号
m.try_emplace("彭于斌", {"自定义", 22});         // ERROR: 参数类型是模板类型，未知，无法省略花括号前的类型
m.try_emplace("彭于斌", Student{"自定义", 22});  // OK: 明确指定类型的花括号初始化；但这样又会造成 1 次移动，失去了 try_emplace 避免移动的意义
```

> 此外还要注意不论 insert、emplace、emplace_hint、try_emplace，都是一个尿性：键冲突时不会覆盖已有元素。
>
> 如果需要覆盖性的插入，还得乖乖用 [] 或者 insert_or_assign 函数。

由于 try_emplace 里写死了圆括号，我们只好手动定义的构造函数才能劳驾 try_emplace 就地构造。

```cpp
struct Student {
    string sex;
    int age;
    Student(string sex, int age)
        : sex(std::move(sex))
        , age(age)
    {}
    // 由于 try_emplace 会就地构造对象，其值类型可以没有移动构造函数，而 insert 会出错
    Student(Student &&) = delete;
    Student &operator=(Student &&) = delete;
    Student(Student const &) = delete;
    Student &operator=(Student const &) = delete;
};

map<string, Student> m;
m.try_emplace("彭于斌", "自定义", 22);           // OK: 会调用构造函数 Student(string, int) 就地构造对象
m.insert({"彭于斌", Student("自定义", 22)});     // ERROR: insert 需要移动 Student 而 Student 的移动被 delete 了！
```

### 什么是聚合初始化

无构造函数时，C++11 支持花括号初始化（官方名: 聚合初始化[1](https://142857.red/book/stl_map/#fn:1)），C++20 开始聚合初始化也能用圆括号（所以 emplace / try_emplace 这类函数变得更好用了）：

```cpp
struct Student {
    string sex;
    int age;
};
auto s1 = Student{"自定义", 22};  // C++11 起 OK: 无构造函数时的花括号初始化语法
auto s2 = Student("自定义", 22);  // C++20 起 OK: 编译器会自动生成圆括号构造函数 Student(string, int)
```

和花括号初始化时一样，可以省略一部分参数，这部分参数会用他们的默认值：

```cpp
auto s1 = Student("自定义", 22);     // OK: sex 为 "自定义"，age 为 22
auto s2 = Student("自定义");         // OK: 省略 age 自动为 0
auto s3 = Student();                 // OK: 省略 sex 自动为 ""
```

不过他和花括号不一样的是，作为已知参数类型的函数参数时，类型名不能省略了：

```cpp
void func(Student const &stu);    // 已知函数签名
func(Student{"自定义", 22});      // OK: C++11 语法
func({"自定义", 22});             // OK: C++11 语法，已知函数具有唯一重载的情况下类名可以省略
func(Student("自定义", 22));      // OK: C++20 语法
func(("自定义", 22));             // ERROR: 无法从 int 转换为 Student
```

### C++20 修复了聚合初始化不支持圆括号的缺点

所以现在 try_emplace 也可以就地构造无构造函数的类型了：

```cpp
map<string, Student> m;
m.try_emplace("彭于斌", "自定义", 22);       // OK: 等价于 m["彭于斌"] = Student{"自定义", 22}
m.try_emplace("彭于斌", "自定义");           // OK: 等价于 m["彭于斌"] = Student{"自定义", 0}
m.try_emplace("彭于斌");                    // OK: 等价于 m["彭于斌"] = Student{"", 0}
```

方便！

> 关于更多 C++20 的聚合初始化小知识，可以看这期 CppCon 视频：https://www.youtube.com/watch?v=flLNi0aejew
>
> 为方便你在比站搜索搬运，他的标题是：Lightning Talk: Direct Aggregate Initialisation - Timur Doumler - CppCon 2021

### 调用开销分析

```cpp
struct MyClass {
    MyClass() { printf("MyClass()\n"); }
    MyClass(MyClass &&) noexcept { printf("MyClass(MyClass &&)\n"); }
    MyClass &operator=(MyClass &&) noexcept { printf("MyClass &operator=(MyClass &&)\n"); return *this; }
};

map<int, MyClass> tab;
printf("insert的开销:\n");
tab.insert({1, MyClass()});
printf("try_emplace的开销:\n");
tab.try_emplace(2);  // try_emplace 只有一个 key 参数时，相当于调用无参构造函数 MyClass()
```

insert 调用了两次移动函数，一次发生在 pair 的构造函数，一次发生在 insert 把参数 pair 移进红黑树节点里。

而 try_emplace 内部使用了现代 C++ 的就地构造（placement new），直接在红黑树节点的内存中构造 MyClass，无需反复移动，对于尺寸较大的值类型会更高效。

```vbnet
insert的开销:
MyClass()
MyClass(MyClass &&)
MyClass(MyClass &&)
try_emplace的开销:
MyClass()
```

### try_emplace 成功提升性能的案例

提升了 1.42 倍性能，不能说是惊天地泣鬼神吧，至少也可以说是聊胜于无了。这里的值类型 string 只有 32 字节还不够明显，可能更大的自定义类型会有明显的优势。这种优化的理论上限是 3 倍，最多能从 try_emplace 获得 3 倍性能提升。

```cpp
template <class K, class V>
static void test_insert(map<K, V> &tab) {
    DefScopeProfiler;
    for (int i = 0; i < 1000; i++) {
        // 1次string(const char *) 2次string(string &&)
        tab.insert({i, "hello"});
    }
}

template <class K, class V>
static void test_try_emplace(map<K, V> &tab) {
    DefScopeProfiler;
    for (int i = 0; i < 1000; i++) {
        // 1次string(const char *)
        tab.try_emplace(i, "hello");
    }
}
int main() {
    for (int i = 0; i < 1000; i++) {
        map<int, string> tab;
        test_insert(tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < 1000; i++) {
        map<int, string> tab;
        test_try_emplace(tab);
        doNotOptimize(tab);
    }
    printScopeProfiler();
}
   avg   |   min   |   max   |  total  | cnt | tag
       39|       34|      218|    39927| 1000| test_insert
       28|       27|       91|    28181| 1000| test_try_emplace
```

------

如果改成更大的自定义类型，可以提升 2.3 倍。

```cpp
struct MyClass {
    int arr[4096];
};
   avg   |   min   |   max   |  total  | cnt | tag
     1312|     1193|    18298|  1312871| 1000| test_insert
      573|      537|     1064|   573965| 1000| test_try_emplace
```

------

### 带插入位置提示的 try_emplace

insert 的炫彩中二摇摆混沌大魔王分奴版：带插入位置提示的 try_emplace

```cpp
template <class ...Args>
iterator try_emplace(const_iterator pos, K const &k, Args &&...args);
```

写法：

```cpp
hint = m.try_emplace(hint, key, arg1, arg2, ...);
```

等价于：

```cpp
hint = m.insert(hint, {key, V(arg1, arg2, ...)});
```

> 这次不需要再分一个什么 try_emplace_hint 出来了，是因为 try_emplace 的第一个参数是 K 类型而不是泛型，不可能和 const_iterator 类型混淆，因此 C++ 委员会最终决定直接共用同一个名字，让编译器自动重载了。

### emplace 家族总结

总结，如何用 emplace 家族优化？分直接插入和带提示插入两种用法，和你是否需要高性能两种需求，这里标了“推荐”的是建议采用的：

```cpp
// 直接插入版
m.insert({"key", MyClass(1, 2, 3)});              // 可读性推荐
m.try_emplace("key", 1, 2, 3);                    // 高性能推荐
m.emplace("key", MyClass(1, 2, 3));               // 没意义
m.emplace(std::piecewise_construct, std::forward_as_tuple("key"), std::forward_as_tuple(1, 2, 3));  // C++17 以前的高性能写法
// 带插入位置提示版
hint = m.insert(hint, {"key", MyClass(1, 2, 3)});       // 可读性推荐
hint = m.try_emplace(hint, "key", 1, 2, 3);             // 高性能推荐
hint = m.emplace_hint(hint, "key", MyClass(1, 2, 3));   // 没意义
hint = m.emplace_hint(hint, std::piecewise_construct, std::forward_as_tuple("key"), std::forward_as_tuple(1, 2, 3));  // C++17 以前的高性能写法
```

## map 与 RAII

梦幻联动：map 容器与 RAII 的双向奔赴

如果 map 中元素的值类型是 RAII 类型，其析构函数会在元素被删除时自动调用。

map 被移动时，不会调用元素的移动函数，因为 map 里只存着指向红黑树根节点的指针，只需指针移动即可。

map 被拷贝时，会调用元素的拷贝函数，如果元素不支持拷贝，则 map 的拷贝也会被禁用（delete）掉。

map 被析构时，其所有元素都会被析构。

### 案例 1：资源类可以移动

```cpp
struct RAII {
    int i;

    explicit RAII(int i_) : i(i_) {
        printf("%d号资源初始化\n", i);
    }

    RAII(RAII &&) noexcept {
        printf("%d号资源移动\n", i);
    }

    RAII &operator=(RAII &&) noexcept {
        printf("%d号资源移动赋值\n", i);
        return *this;
    }

    ~RAII() {
        printf("%d号资源释放\n", i);
    }
};
int main() {
    {
        map<string, RAII> m;
        m.try_emplace("资源1号", 1);
        m.try_emplace("资源2号", 2);
        m.erase("资源1号");
        m.try_emplace("资源3号", 3);
    }
    printf("此时所有资源都应该已经释放\n");
    return 0;
}
1号资源初始化
2号资源初始化
1号资源释放
3号资源初始化
3号资源释放
2号资源释放
此时所有资源都应该已经释放
```

### 案例 2：资源类禁止移动

```cpp
struct RAII {
    int i;

    explicit RAII(int i_) : i(i_) {
        printf("%d号资源初始化\n", i);
    }

    RAII(RAII &&) = delete;
    RAII &operator=(RAII &&) = delete;
    RAII(RAII const &) = delete;
    RAII &operator=(RAII const &) = delete;

    ~RAII() {
        printf("%d号资源释放\n", i);
    }
};
```

新手定义 RAII 类时，记得把移动和拷贝 4 个函数全部删除。没错，**移动也要删除**，很多新手会觉得资源类应该可以移动的呀？要是想保留移动，就得预留一个 i == 0 的空状态，那种处理很复杂的。总之一旦定义了析构函数，全部 4 个函数都得删除，除非你有相关经验。参见 [C++ 生命周期与析构函数专题](https://142857.red/book/cpp_lifetime/)

```cpp
int main() {
    {
        map<string, RAII> m;
        m.try_emplace("资源1号", 1);
        m.try_emplace("资源2号", 2);
        m.erase("资源1号");
        m.try_emplace("资源3号", 3);
    }
    printf("此时所有资源都应该已经释放\n");
    return 0;
}
1号资源初始化
2号资源初始化
1号资源释放
3号资源初始化
3号资源释放
2号资源释放
此时所有资源都应该已经释放
```

这时就体现出 try_emplace 的好处了：值类型不需要有移动构造函数也可以插入。

### 记得删除移动构造函数

```cpp
struct RAII {
    int i;

    explicit RAII(int i_) : i(i_) {
        printf("%d号资源初始化\n", i);
    }

    RAII(RAII &&) = delete;

    ~RAII() {
        printf("%d号资源释放\n", i);
    }
};
```

冷知识：只需要删除移动构造函数，编译器就会自动帮你删除剩下 3 个，这是因为看到你用了 `&&` 就知道你是懂 C++11 的，所以不用照顾 C++98 兼容性保留烦人的拷贝构造函数，自动帮你删了，这是个标准，所有 C++ 编译器都是这样的（要我说，建议改成定义了析构函数就自动删全 4 个函数，可惜标准委员会要照顾兼容性…）

以后 RAII 类只需要一行 `C(C &&) = delete` 就够了。

```cpp
int main() {
    {
        map<string, RAII> m;
        m.try_emplace("资源1号", 1);
        m.try_emplace("资源2号", 2);
        m.erase("资源1号");
        m.try_emplace("资源3号", 3);
    }
    printf("此时所有资源都应该已经释放\n");
    return 0;
}
1号资源初始化
2号资源初始化
1号资源释放
3号资源初始化
3号资源释放
2号资源释放
此时所有资源都应该已经释放
```

### 统一交给智能指针管理

如果你想用更可读的 insert，RAII 资源类又不支持移动，可以用 `unique_ptr<RAII>` 包装一下：

~~~cpp
```cpp
int main() {
    {
        map<string, std::unique_ptr<RAII>> m;
        m.insert("资源1号", std::make_unique<RAII>(1));
        m.insert("资源2号", std::make_unique<RAII>(2));
        m.erase("资源1号");
        m.insert("资源3号", std::make_unique<RAII>(3));
    }
    printf("此时所有资源都应该已经释放\n");
    return 0;
}
~~~

#### 智能指针帮你避免移动

对于很大的 V 类型，也可以改用 `map<T, unique_ptr<V>>` 避免反复移动元素本体。（用在需要反复扩容的 vector 中也有奇效）

因为包括 map 在内的所有容器都完美支持 RAII 类型，所以也可以用智能指针作为这些容器的元素。

```cpp
struct MyData {
    int value;  // 假设这个很大
    explicit MyData(int value_) : value(value_) {}
};
map<string, unique_ptr<MyData>> m;
m.insert({"answer", make_unique<MyData>(42)});  // 只有 8 字节的 unique_ptr 被移动 2 次
m.insert({"fuck", make_unique<MyData>(985)});
print(m.at("answer")->value);  // 42
// ↑等价于：print((*m.at("answer")).value);
```

- `map<T, unique_ptr<V>>` 中，智能指针指向的对象会在元素被删除时自动释放。
- `map<T, V *>` 中，C 语言原始指针不具备 RAII 功能，除非该指针被其他智能指针打理着，或者用户删除元素之前手动 delete，否则当元素删除时内存会泄露！

我推荐完全采用智能指针来自动管理内存，智能指针和同样符合 RAII 思想的各大容器也是相性很好的。

如果需要浅拷贝的话，则可以改用 `map<T, shared_ptr<V>>`，小彭老师在他的 Zeno 项目中就是这样用的。

## 增删改查总结

### 增删

| 写法                             | 效果         | 版本  | 推荐 |
| -------------------------------- | ------------ | ----- | ---- |
| `m.insert(make_pair(key, val))`  | 插入但不覆盖 | C++98 | 💩    |
| `m.insert({key, val})`           | 插入但不覆盖 | C++11 | ❤    |
| `m.emplace(key, val)`            | 插入但不覆盖 | C++11 | 💩    |
| `m.try_emplace(key, valargs...)` | 插入但不覆盖 | C++17 | 💣    |
| `m.insert_or_assign(key, val)`   | 插入或覆盖   | C++17 | ❤    |
| `m[key] = val`                   | 插入或覆盖   | C++98 | 💣    |
| `m.erase(key)`                   | 删除指定元素 | C++98 | ❤    |

### 改查

| 写法                             | 效果                            | 版本  | 推荐 |
| -------------------------------- | ------------------------------- | ----- | ---- |
| `m.at(key)`                      | 找不到则出错，找到则返回引用    | C++98 | ❤    |
| `m[key]`                         | 找不到则自动创建`0`值，返回引用 | C++98 | 💣    |
| `myutils::map_get(m, key, defl)` | 找不到则返回默认值              | C++98 | ❤    |
| `m.find(key) == m.end()`         | 检查键 `key` 是否存在           | C++98 | 💣    |
| `m.count(key)`                   | 检查键 `key` 是否存在           | C++98 | ❤    |
| `m.contains(key)`                | 检查键 `key` 是否存在           | C++20 | 💩    |

#### 初始化

| 写法                                     | 效果                   | 版本  | 推荐 |
| ---------------------------------------- | ---------------------- | ----- | ---- |
| `map<K, V> m = {{k1, v1}, {k2, v2}}`     | 初始化为一系列键值对   | C++11 | ❤    |
| `auto m = map<K, V>{{k1, v1}, {k2, v2}}` | 初始化为一系列键值对   | C++11 | 💩    |
| `func({{k1, v1}, {k2, v2}})`             | 给函数参数传入一个 map | C++11 | ❤    |
| `m = {{k1, v1}, {k2, v2}}`               | 重置为一系列键值对     | C++11 | ❤    |
| `m.clear()`                              | 清空所有表项           | C++98 | ❤    |
| `m = {}`                                 | 清空所有表项           | C++11 | 💣    |

## 节点句柄系列接口

### extract

C++17 新增的 extract 函数[1](https://142857.red/book/stl_map/#fn:1) 可以“剥离”出单个节点：

```cpp
node_type extract(K const &key);
node_type extract(const_iterator pos);
auto node = m.extract("fuck");
auto &k = node.key();    // 键（引用）
auto &v = node.mapped(); // 值（引用）
```

其功能与 erase 类似，都会将元素从 map 中删除，但 extract 只是把节点从 map 中移走，并不会直接销毁节点。

extract 会返回这个刚被“剥离”出来节点的句柄，类型为 node_type，节点的生杀大权就这样返回给了用户来处置。

node_type 是指向游离红黑树节点的特殊智能指针，称为节点句柄[2](https://142857.red/book/stl_map/#fn:2)。只可移动不可拷贝，类似一个指向节点的 unique_ptr。

当调用 extract(key) 时会把 key 对应的键值对所在的红黑树节点“脱离”出来——不是直接释放节点内存并销毁键值对象，而是把删除的节点的所有权移交给了调用者，以返回一个特殊智能指针 node_type 的形式。

调用 extract 后，节点句柄指向的这个红黑树节点已经从 map 中移除（其 left、right、parent 等指针为 NULL），处于游离状态。

> 节点中不仅存储着我们感兴趣的键和值，还有 left、right、parent、color 等用于维护数据结构的成员变量，对用户不可见。

只是因为节点句柄类似于 unique_ptr，维持着节点的生命周期，保护着键 key() 和值 mapped() 没有被销毁，内存没有被释放。

如果调用者接下来不做操作，那么当离开调用者所在的函数体时，这个特殊的 unique_ptr 会自动释放其指向节点。

- 对于第一个按键取出节点句柄的 extract 重载：如果键值不存在，那么 extract 会返回一个特殊的空节点句柄，类似于空指针。可以通过 `(bool)node` 来判断一个节点句柄是否为空。
- 对于第二个按迭代器取出句柄的 extract：总是成功，因为既然你已经获得了迭代器，肯定是 find 获得的，而 find 找不到返回的 end 传入 extract 是未定义行为。正如 erase 迭代器版重载 erase(it) 总是成功一样。

#### 用途举例

调用者稍后可以直接销毁这个特殊智能指针：

```cpp
{
    auto node = m.extract("fuck");
    print(node.key(), node.mapped());
} // node 在此自动销毁
```

也可以做一些修改后（例如修改键值），稍后重新用 insert(node) 重新把他插入回去：

```cpp
auto node = m.extract("fuck");
node.key() = "love";
m.insert(std::move(node));
```

> 过去，通过迭代器来修改键值是不允许的：

```cpp
map<string, int> m;
auto it = m.find("fuck");
assert(it != m.end());
// *it 是 pair<const string, int>
it->first = "love"; // 错误！first 是 const string 类型
m.insert(*it);
```

> 因为直接修改在 map 里面的一个节点的键，会导致排序失效，破坏红黑树的有序。而 extract 取出来的游离态节点，可以修改 `.key()`，不会影响任何红黑树的顺序，他已经不在树里面了。

或者插入到另一个不同的 map 对象（但键和值类型相同）里：

```cpp
// 从 m1 挪到 m2
auto node = m1.extract("fuck");
m2.insert(std::move(node));
```

优点在于，extract 和节点版 insert 不涉及内存的重新分配与释放，不涉及元素类型的移动（因为节点句柄类似于智能指针，智能指针的移动并不会导致其指向对象的移动），所以会比下面这种传统写法更高效：

```cpp
// 从 m1 挪到 m2：传统写法
if (m1.count("fuck")) {
    auto value = std::move(m1.at("fuck"));
    m2["fuck"] = std::move(value);
    m1.erase(it);
}
```

不用 auto 完整写出全部类型的形式（古代 C++98 作风）：

```cpp
typename map<K, V>::node_type node = m.extract("fuck");
K &k = node.key();
V &v = node.mapped();
```

set 也有 extract 函数，其节点句柄没有 key() 和 mapped() 了，而是只有一个 value()，获取其中的值

```cpp
set<V> s = {"fuck", "suck", "dick"};
set<V>::node_type node = s.extract("fuck");
V &v = node.value();
```

### insert 节点版

insert 函数：插入游离节点的版本

```cpp
insert_return_type insert(node_type &&node);
iterator insert(const_iterator pos, node_type &&node); // 带提示的版本
```

可以用 insert(move(node)) 直接插入一个节点。

```cpp
map<string, int> m1 = {
    {"fuck", 985},
    {"dick", 211},
};
map<string, int> m2;
auto node = m1.extract("fuck");
m2.insert(std::move(node));  // 节点句柄类似于 unique_ptr，不可拷贝，需要用移动语义进行插入
```

调用 insert(move(node)) 后由于所有权被移走，node 将会处于“空指针”状态，可以用 `node.empty()` 查询节点句柄是否为“空”状态，即节点所有权是否已经移走。

#### insert_return_type

这个版本的 insert 返回值类型 insert_return_type 是一个结构体（我的天他们终于肯用结构体而不是 pair 了）：

```cpp
struct insert_return_type {
    iterator position;
    bool inserted;
    node_type node;
};
insert_return_type insert(node_type &&nh);
```

官方说法是[1](https://142857.red/book/stl_map/#fn:1)：

> If nh is empty, inserted is false, position is end(), and node is empty.
>
> Otherwise if the insertion took place, inserted is true, position points to the inserted element, and node is empty.
>
> If the insertion failed, inserted is false, node has the previous value of nh, and position points to an element with a key equivalent to nh.key().

### extract + insert 运用案例

```cpp
map<int, string> hells = {
    {666, "devil"},
};
map<int, string> schools = {
    {985, "professor"},
    {211, "doctor"},
    {996, "fucker"},
};
auto node = schools.extract(996);
hells.insert(std::move(node));
print(schools);
print(hells);
{211: "doctor", 985: "professor"}
{666: "devil", 996: "fucker"}
```

extract + insert(move(node)) 对比 find + insert({key, val})，可以避免键和值类型移动构造函数的开销，至始至终移动的只是一个红黑树节点的指针，元素没有被移动，也没有造成内存空间不必要的分配和释放。

但是 insert(move(node)) 仅适用于从 extract 中取出现有节点的情况，如果要新建节点还得靠 insert({key, val}) 或者 try_emplace(key, val) 的。

### extract 性能优化案例

已知两个映射表 tab1 和 tab2，和一个接受 K 类型做参数的仿函数 cond。

要求把 tab1 中键符合 cond 条件的元素移动到 tab2 中去，其余保留在 tab1 中。

我们编写四份同样功能的程序，分别采用：

- extract + 带提示的 insert
- erase + 带提示的 insert
- extract + 直接 insert
- erase + 直接 insert

```cpp
template <class K, class V, class Cond>
void filter_with_extract(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            auto next_it = it;
            ++next_it;
            auto node = tab1.extract(it);
            tab2.insert(std::move(node));
            it = next_it;
        } else ++it;
    }
}

template <class K, class V, class Cond>
void filter_with_erase(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            it = tab1.erase(it);
            auto kv = std::move(*it);
            tab2.insert(std::move(kv));
        } else ++it;
    }
}

template <class K, class V, class Cond>
void filter_with_extract_with_hint(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    auto hint = tab2.begin();
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            auto next_it = it;
            ++next_it;
            auto node = tab1.extract(it);
            hint = tab2.insert(hint, std::move(node));
            it = next_it;
        } else ++it;
    }
}
template <class K, class V, class Cond>
void filter_with_erase_with_hint(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    auto hint = tab2.begin();
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            it = tab1.erase(it);
            auto kv = std::move(*it);
            hint = tab2.insert(hint, std::move(kv));
        } else ++it;
    }
}
```

extract vs erase 性能测试结果 (testextractvserase.cpp)：

```yaml
   avg   |   min   |   max   |  total  | cnt | tag
      889|      803|     2388|   889271| 1000| filter_with_erase
      642|      595|     1238|   642542| 1000| filter_with_extract
      525|      491|     1398|   525137| 1000| filter_with_erase_with_hint
      305|      289|      842|   305472| 1000| filter_with_extract_with_hint
```

extract + 带提示的 insert 获胜，即函数 `filter_with_extract_with_hint` 是性能最好的那一个。

### 游离节点可以修改键值

由于处于游离状态，游离节点不属于任何 map 中，不需要满足排序性质，因此 `node.key()` 可修改。

先用 extract 取出游离态节点，修改完节点的键后再重新插入，利用这一点做到以前做不到的修改键值。

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto node = m.extract("fuck");  // 移出 "fuck" 键对应的节点，此时 m 会变为空
node.key() = "fxxk";  // 修改键（以前用迭代器时 it->first 是不允许修改键的，因为迭代器指向的节点不是游离状态，修改键会破坏排序）
node.mapped() = 211;  // 修改值（这个以前用迭代器时 it->second 也可以修改）
m.insert(move(node)); // 把修改好的节点插入回去
print(m);             // {{"fxxk": 211}}
```

相当于你给小学生排队时，有一个小学生突然瞬间不知道吃了什么激素长高了，你的队伍就会乱掉。

所以需要让这个小学生先出列，让他单独一个人长高，等他长高完了再插入回队列。

### 带提示的节点版 insert

但是小学生长高的量可能是有限的（新的键可能和老键很接近）。

这时插入可以优先从他长高之前的位置开始二分法，也就是用 extract 之前，这个小学生后一位同学的位置，作为 insert 的提示，让 insert 更快定位到这个小学生应该插入的位置。

```cpp
auto it = m.find("fuck");
assert(it != m.end()); // 假定 "fuck" 必须存在（如果不存在会返回 end）
auto next_it = std::next(it); // 下一位同学（可能会得到 end，但没关系，因为 insert 的提示也允许为 end 迭代器）
auto node = m.extract(it);
node.key() = "fxxk";   // 修改键值，变化不大
m.insert(next_it, move(node)); // 如果键值变动不大，优先尝试在老位置插入
```

> 这里的 `std::next(it)` 对于等价于 it + 1。但是 map 属于双向迭代器（而不是随机迭代器），不支持加法操作，只支持就地 ++。所以 `std::next` 内部等价于：

```cpp
auto next(auto it) {
    auto next_it = it; // 先拷贝一份，防止原迭代器被破坏（迭代器都支持拷贝，性质上是浅拷贝）
    ++next_it;         // 再让 next_it 就地自增到下一位
    return next_it;    // 返回现在已经相当于 it + 1 的 next_it
}
```

如果键不变，或者键变了以后，插入位置不变的话，那么这次 insert 可以低至 O(1)O(1) 复杂度。

```cpp
map<string, int> m = {
    {"dick", 211},
    {"fuck", 985}, // "fuck" -> "fxxk" 后，重新插入，其依字典序的“大小”依然是介于 "dick" 和 "suck"
    {"suck", 996},
};
```

### merge：map 的合并操作（并集）

C++17 新增的 merge 函数[1](https://142857.red/book/stl_map/#fn:1)

```cpp
template <class Cmp2>
void merge(map<K, V, Cmp2> &__source);
```

> 注：set 也有 merge 函数

注意到 merge 的参数是另一个 map，可变引用，必须和本 map 同类型（这是为了保证节点句柄类型相同），但允许有不同的比较函数

- `merge(source)` 会把 source 中的所有节点都**移动**并合并到本 map，注意是**移动**而不是拷贝，source 将会被清空，这样是为了更高效。
- `insert(source.begin(), source.end())` 则是把 source 里的元素拷贝后插入到本 map，更低效，因为需要拷贝，还得新建红黑树节点，额外分配内存空间。

对于键存在冲突的情况：

- merge: 如果 source 中有与本 map 重复的键，则该元素不会被移动，保留在 source 里。
- insert: 如果 source 中有与本 map 重复的键，则该元素不会被插入本 map。无论有没有插入本 map，原 source 中的键都不会被清除。

> 因此，merge 也并不总是完全清空 source，当 source 和本 map 有冲突时，冲突的键就保留在 source 里了。

merge 等价于以下手动用 extract 和 insert 来移动节点的代码：

```cpp
// m1.merge(m2) 等价于：
auto hint = m1.begin();
for (auto it = m2.begin(); it != m2.end(); ++it) {
    if (!m1.contains(it->first)) {
        auto node = m2.extract(it);
        hint = m1.insert(hint, node);
    }
}
```

#### 批量 insert vs merge

同样做到两个 map 合并，`m1.merge(m2)` 与 `m1.insert(m2.begin(), m2.end())` 性能比较：

```cpp
#include <map>
#include <string>
#include "benchmark/benchmark.h"

using namespace std;

static void BM_Insert(benchmark::State &state) {
    map<string, int> m1_init;
    map<string, int> m2_init;
    for (int i = 0; i < state.range(0); i++) {
        m1_init[to_string(i)] = i;
        m2_init[to_string(i + state.range(0))] = i;
    }
    for (auto _ : state) {
        auto m1 = m1_init;
        auto m2 = m2_init;
        m2.insert(m1.begin(), m1.end());
        benchmark::DoNotOptimize(m2);
    }
}
BENCHMARK(BM_Insert)->Arg(1000);

static void BM_Merge(benchmark::State &state) {
    map<string, int> m1_init;
    map<string, int> m2_init;
    for (int i = 0; i < state.range(0); i++) {
        m1_init[to_string(i)] = i;
        m2_init[to_string(i + state.range(0))] = i;
    }
    for (auto _ : state) {
        auto m1 = m1_init;
        auto m2 = m2_init;
        m2.merge(m1);
        benchmark::DoNotOptimize(m2);
    }
}
BENCHMARK(BM_Merge)->Arg(1000);
```

merge 函数不会产生不必要的内存分配导致内存碎片化，所以更高效。但作为代价，他会清空 m2！

- merge 相当于把 m2 的元素“移动”到 m1 中去了。
- insert 则是把 m2 的元素“拷贝”了一份插入到 m1 中去，效率自然低下。

如果不想破坏掉 m2，或者你用不上 C++17，则仍需要传统的 insert。

#### merge 和 insert 一样不覆盖旧值

merge(m2) 和 insert(m2.begin(), m2.end()) 一样尿性：如果 m2 中的键在 m1 中已经存在，则不会 extract 该 m2 中的节点，仍然留在 m2 中。

```cpp
int main()
{
  std::map<int, std::string> ma {{1, "apple"}, {5, "pear"}, {10, "banana"}};
  std::map<int, std::string> mb {{2, "zorro"}, {4, "batman"}, {5, "X"}, {8, "alpaca"}};
  std::map<int, std::string> u;
  u.merge(ma);
  std::cout << "ma.size(): " << ma.size() << '\n';
  u.merge(mb);
  std::cout << "mb.size(): " << mb.size() << '\n';
  std::cout << "mb.at(5): " << mb.at(5) << '\n';
  for(auto const &kv: u)
    std::cout << kv.first << ", " << kv.second << '\n';
}
```

## map 自定义比较器

map 容器的全部参数为：

```cpp
std::map<K, V, Cmp, Alloc>
```

其中第 3、4 个参数 Cmp 和 Alloc 可以省略。

- Cmp 默认为 `std::less<K>`
- Alloc 默认为 `std::allocator<std::pair<K, V>>`

因此 `map<K, V>` 的完整模板参数是：

```cpp
std::map<K, V, std::less<K>, std::allocator<std::pair<K, V>>>
```

我们可以简写成 `map<K, V>`。

其中 allocator 我们以后专门开一节课讲，其他很多容器都有 allocator。

今天只研究 Cmp 这个参数，他决定了 map 如何排序，判断相等。

```cpp
std::map<K, V, std::less<K>>
```

这个 `std::less` 是个什么呢？是一个仿函数(functor)。

```cpp
template <class T>
struct less {
    constexpr bool operator()(T const &x, T const &y) const {
        return x < y;
    }
};
```

具有成员函数 `operator()` 的类型，都被称之为仿函数。

### `std::less` 的作用

仿函数对象，可以直接用圆括号当做普通函数调用，这就是“仿函数”的得名原因，例如：

```cpp
less<int> cmp;
print(cmp(1, 2));  // 1 < 2: true
print(cmp(5, 2));  // 5 < 2: false
less<string> cmp;
print(cmp("hello", "world"));   // "hello" < "world": false
print(cmp("cmake", "cppcon"));  // "cmake" < "cppcon": true
less<string_view> cmp;
print(cmp("hello", "world"));   // "hello" < "world": false
print(cmp("cmake", "cppcon"));  // "cmake" < "cppcon": true
```

#### `operator()`

注意仿函数的成员函数 `operator()` 是两个括号：

```cpp
operator()(...)
```

- 第一个括号是 `operator()` 的一部分，表示这是对圆括号 `()` 的运算符重载。
- 第二个括号是函数的参数列表，里面是 `operator()` 这个函数的形参。

> `operator()` 相当于 Python 中的 `__call__`。正如 `operator<` 相当于 Python 中的 `__lt__`。这里 `operator` 和 `()` 是一个整体，连在一起，形成了一个标识符。

### 自定义排序方式

```cpp
std::map<K, V, std::less<K>>
```

我们之前提到 map 内部的元素始终按照键 K 从小到大的顺序排列。

map 决定大小顺序的，并不是直接调用 K 类型的比较运算符 `operator<`。

而是调用他的模板参数 Cmp 类型的 `operator()`。

这是为了允许用户通过修改这个参数，控制 map 内部的行为，防止 map 数据结构与具体的比较方法耦合。

由于默认的 Cmp 是 `less<K>`，调用 `Cmp()(x, y)` 就相当于 `x < y`，由此实现从小到大排序。

接下来我们将修改这一默认行为。

#### 只需要小于号

一个类型要想作为 map 的键，只需要他支持 `<` 运算符即可，不必定义其他 `>`、`==` 运算符。

当 map 需要判断两个键是否相等时 `x == y`，会用 `!(x < y) && !(y < x)` 来等价地计算。

string, string_view, int, float, void *, shared_ptr, pair, tuple, array…

这些类型都支持比较运算符，都可以作为 map 的键。

### 自定义小于号的三种方式

如果你写了个自定义类 Student，要让他作为 map 的键类型，有三种方法：

一、在 Student 类中添加 `operator<`

```cpp
struct Student {
    string name;
    int id;
    string sex;

    bool operator<(Student const &that) const {
        return x.name < y.name || (x.name == y.name && (x.id < y.id || (x.id == y.id && x.sex < y.sex)));
        // 等价于：
        return std::tie(x.name, x.id, y.sex) < std::tie(x.name, x.id, y.sex); // tuple 实现了正确的 operator< 运算符
    }
};

map<Student, int> stutab;
```

二、特化 `less<Student>`，添加 `operator()`

```cpp
struct Student {
    string name;
    int id;
    string sex;
};

template <>
struct std::less<Student> {  // 用户可以特化标准库中的 trait
    bool operator()(Student const &x, Student const &y) const {
        return std::tie(x.name, x.id, y.sex) < std::tie(x.name, x.id, y.sex);
    }
};

map<Student, int> stutab;
```

> 缺点：以后如果 map 外面要用用到这个类的大小比较，也只能用 `less<Student>()(stu1, stu2)` 代替 `stu1 < stu2`。

三、重新自定义一个仿函数类 `LessStudent`，添加 `operator()`，然后把这个 `LessStudent` 作为 map 的比较器传入模板

```cpp
struct Student {
    string name;
    int id;
    string sex;
};

struct LessStudent {
    bool operator()(Student const &x, Student const &y) const {
        return std::tie(x.name, x.id, y.sex) < std::tie(x.name, x.id, y.sex);
    }
};

map<Student, int, LessStudent> stutab;
```

> 缺点：以后每次创建新的 map 时，都需要加一个 LessStudent 参数。

### 自定义按哪个字段来索引

如果希望 map 在查找时只根据学生姓名索引，则只需要改一下比较器的实现，让他只比较姓名字段即可。

```cpp
struct LessStudent {
    bool operator()(Student const &x, Student const &y) const {
        return x.name < y.name;
    }
};
```

上面这样的比较器，map 会认为姓名 name 相同的 Student 就是相等的，并去重。即使 id 和 sex 不同，只要名字相等就会视为重复，利用这一点可以实现针对特定字段的去重。

> 结论：map 的排序和去重，都取决于于你的比较器如何实现！比较器里没比较的字段，就会被忽略而不参与排序、索引、和去重。

### C++20 三路运算符 `<=>`

四（同一）、利用 C++20 新特性，三路比较运算符 `<=>`：如果自定义类的每个成员都支持比较运算符，可以把 `operator<=>` 函数声明为 `default`，然后编译器会自动添加自定义类的所有比较运算符。

```cpp
struct Student {
    string name;
    int id;
    string sex;

    auto operator<=>(Student const &) const = default;
};
```

此时默认的 `operator<` 实现等价于 `x.name < y.name || (x.name == y.name && (x.id < y.id || (x.id == y.id && x.sex < y.sex)))`。

> `<=>` 的返回类型是 `std::strong_ordering`，这是一种有三种取值的强枚举类型
>
> ```
> <=>` 对应的仿函数为 `std::compare_three_way
> ```

### 仿函数运算符全家桶

libstdc++ 头文件中的 less 和 greater 实现参考：

```cpp
template<typename _Tp>
struct less : public binary_function<_Tp, _Tp, bool>
{
  _GLIBCXX14_CONSTEXPR
  bool
  operator()(const _Tp& __x, const _Tp& __y) const
  { return __x < __y; }
};

template<typename _Tp>
struct greater : public binary_function<_Tp, _Tp, bool>
{
  _GLIBCXX14_CONSTEXPR
  bool
  operator()(const _Tp& __x, const _Tp& __y) const
  { return __x > __y; }
};
```

类似的运算符仿函数还有：

| 运算符 | 仿函数类           |
| ------ | ------------------ |
| x == y | std::equal_to      |
| x != y | std::not_equal_to  |
| x < y  | std::less          |
| x > y  | std::greater       |
| x <= y | std::less_equal    |
| x >= y | std::greater_equal |
| x + y  | std::plus          |
| x - y  | std::minus         |
| x * y  | std::multiplies    |
| x / y  | std::divides       |
| x % y  | std::modulus       |
| -x     | std::negate        |

他们都在 `#include <functional>` 头文件中定义。

### greater 实现反向排序

案例：使用 greater 仿函数，让 map 反过来从大到小排序：

```cpp
auto ilist = {
    {985, "拳打"},
    {211, "脚踢"},
};
map<int, string> m1 = ilist;                // 从小到大排序
map<int, string, greater<int>> m2 = ilist;  // 从大到小排序
print(m1); // {{211, "脚踢"}, {985, "拳打"}}
print(m2); // {{985, "拳打"}, {211, "脚踢"}}
```

### 大小写不敏感的字符串比较器

自定义比较仿函数，实现无视键大小写的 map 容器：

```cpp
struct LessIgnoreCase {
    bool operator()(std::string const &lhs, std::string const &rhs) const {
        return std::lexicographical_compare // 位于 <algorithm> 头文件，和 std::string 同款的字典序比较
        ( lhs.begin(), lhs.end()
        , rhs.begin(), rhs.end()
        , [] (char lhs, char rhs) {
            return std::toupper(lhs) < std::toupper(rhs);
        });
    }
};
int main() {
    map<string, string, LessIgnoreCase> m = {
        {{"Fuck"}, "rust"},
        {{"fUCK"}, "java"},
        {{"STUdy"}, "cpp"},
        {{"stUDy"}, "js"},
    };
    print(m);
    print("fuck对应的值为:", m.at("fuck"));
    return 0;
}
{"Fuck": "rust", "STUdy": "cpp"}
fuck对应的值为: "rust"
```

### 传入 lambda 做比较器

C++11 的 lambda 表达式也是仿函数，配合 decltype 后就可以传入 map 作为比较器：

```cpp
auto cmp = [] (std::string const &lhs, std::string const &rhs) {
    return std::lexicographical_compare
    ( lhs.begin(), lhs.end()
    , rhs.begin(), rhs.end()
    , [] (char lhs, char rhs) {
        return std::toupper(lhs) < std::toupper(rhs);
    });
};
map<string, string, decltype(cmp)> m({
    {{"Fuck"}, "rust"},
    {{"fUCK"}, "java"},
    {{"Study"}, "cpp"},
}, cmp);
print(m);
auto val = m.at({"fuck"});
print(val);
```

写的更清晰一点：

```cpp
auto cmp = [] (std::string const &lhs, std::string const &rhs) {
    return std::lexicographical_compare
    ( lhs.begin(), lhs.end()
    , rhs.begin(), rhs.end()
    , [] (char lhs, char rhs) {
        return std::toupper(lhs) < std::toupper(rhs);
    });
};
map<string, string, decltype(cmp)> m(cmp);
m = {
    {{"Fuck"}, "rust"},
    {{"fUCK"}, "java"},
    {{"Study"}, "cpp"},
};
print(m);
auto val = m.at({"fuck"});
print(val);
```

#### map 构造函数是如何传入比较器的

刚刚用到的两个 map 构造函数：

```cpp
template <class K, class V, class Cmp = std::less<K>>
class map {
    explicit map(Cmp cmp);
    map(initializer_list<pair<K, V>> ilist, Cmp cmp);
};
```

基本每个 map 的构造函数都有一个提供额外 cmp 参数的版本，统一都是在最后一个参数后面追加。

#### 有状态（捕获变量）的比较器

传入的仿函数 cmp 甚至可以捕获其他变量，这种捕获了变量的仿函数称之为有状态仿函数 - stateful functor，和无状态仿函数 - stateless functor 相对：

```cpp
vector<int> arr = {1, 4, 2, 8, 5, 7};
auto cmp = [&] (int i, int j) {
    return arr[i] < arr[j];
};
map<int, int, decltype(cmp)> m(cmp);
```

利用有状态仿函数可以实现 argsort 等操作，例如上面代码就是根据在 arr 里对应索引的值来排序。

> 由于 map 需要比较仿函数为纯函数(pure function)，在上面例子中，请保证 map 存在期间 arr 的内容不发生变化，否则 map 基于排序的二分查找功能会得到错误的结果。

传入比较器仿函数是设计模式中典型的策略模式，通过依赖注入，允许我们控制 map 内部的行为。

#### 建议用 function

如果嫌 decltype 麻烦（难以在全局或类内部用），function 容器作为比较运算符，就可以统一了：

```cpp
auto cmp = [] (int i, int j) {
    return i < j;
};
map<int, int, function<bool(int, int)>> m;
```

稍后还可以通过 `key_comp()` 获取到用于键比较的仿函数，这个就是你刚刚传入的 cmp 参数：

```cpp
m.key_comp()(1, 2);              // 等价于 cmp(1, 2)
```

`value_comp()` 获取到用于元素（键-值对）比较的仿函数（他帮你适配参数类型了）：

```cpp
m.value_comp()({1, 0}, {2, 0});  // 等价于 cmp(1, 2)
```

## 透明 map

### 什么是透明仿函数

C++14 新增了“透明(transparent)”运算符仿函数。

对于 less、greater 这类标准库提供的仿函数，指定模板参数为 void 即可让一个运算符仿函数变成“透明”的。例如对 less 而言，他的透明版就是 `less<void>`。

C++14 之前用的都是“不透明”版的仿函数，必须指定一个具体的类型，例如 `less<int>` 就只能用于 int 类型的比较，`less<string>` 就只能用于 string 类型的比较。

无法用 `less<int>` 仿函数比较 string 类型。

而 `less<void>` 是通用的，他的 `operator()` 函数是泛型的，可以接受任意类型。

```cpp
template <>
struct less<void> {   // 针对 void 的特化
    // 标准委员会想：由于 void 类型不可能有 < 运算符的需求，所以他们干脆拿 void 作为透明版的模板参数“占位符”了
    template <class Tx, class Ty>
    constexpr decltype(auto) operator()(Tx &&x, Ty &&y) const {
        return forward<Tx>(x) < forward<Ty>(y);
    }

    struct is_transparent;  // 空类，仅供 SFINAE 元编程时检测一个仿函数是否透明时使用
};
```

> ![img](https://142857.red/book/img/question.png) 我的思考：不透明版的 `less<T>` 泛型体现在类的模板参数上，而透明版的体现在了成员函数 `operator()` 的模板参数上。
>
> ![img](https://142857.red/book/img/bulb.png) 这里用 `void` 特化只是一个偷懒，`void` 并没有什么特殊的，实际上应该单独定义一个没有模板的 `transparent_less` 类，但他们就是懒得引入新标识符。

### 为什么需要透明仿函数

“透明”版的好处是可以同一个兼容任意类型，而不必创建多个 cmp 对象。而不透明版的好处是方便特化 traits，但毕竟 < 运算符是可以用户自定义(运算符重载)的，没必要用 traits 特化，所以他们逐步发现透明版香了，还能支持左右参数为不同类型。

```cpp
less<void> cmp;
print(cmp(1, 2));  // 1 < 2: true
print(cmp(5, 2));  // 5 < 2: false
print(cmp(string("hello"), "world"));  // "hello" < "world": false
print(cmp(string("cmake"), "cppcon"));  // "cmake" < "cppcon": true
```

> 但也要特别注意不能再依赖参数类型自动的隐式转换了，必须至少写完整其中一个 `string("hello")` 才能触发 `string` 的 `operator<` 而不是 `const char *` 的指针比大小。如果只写 `cmp("cmake", "cppcon")` 则是在比较指针的地址大小，结果是不一定的。

由于 C++14 的 less 模板参数 T 默认为 void，所以 `less<void>` 还可以简写成 `less<>`。

```cpp
less<> cmp;
print(cmp(1, 2));  // 1 < 2: true
print(cmp(5, 2));  // 5 < 2: false
print(cmp(string("hello"), "world"));  // "hello" < "world": false
print(cmp(string("cmake"), "cppcon"));  // "cmake" < "cppcon": true
```

### 泛型版的 find 函数

普通 find 函数：键类型作为参数

```cpp
iterator find(K const &k);
const_iterator find(K const &k) const;
```

C++14 新增泛型版的 find 函数[1](https://142857.red/book/stl_map/#fn:1)：任意类型作为参数，只要该类型支持与和键比大小。

```cpp
template <class Kt>
iterator find(Kt &&k);
template <class Kt>
const_iterator find(Kt &&k) const;
```

> 这里的 Kt 是模板参数类型，可以接受任意类型，此处 `&&` 是万能引用不是右值引用。

相同点：找到了就返回一个迭代器指向与该参数相等的元素，如果找不到还是返回 end()。

不同点：泛型版本的参数类型 Kt 不必和键类型 K 一致，只要 Kt 和 K 可以比较大小（< 运算符）即可。

不仅

### 泛型 find 的要求：透明

要想用泛型版的 find 函数有一个条件：

map 的比较器必须是“透明(transparent)”的，也就是 `less<void>` 这种。否则泛型版的 `find(Kt &&)` 不会参与重载，也就是只能调用传统的 `find(K const &)`。

但是 `map<K, V>` 默认的比较器是 `less<K>`，他是不透明的，比较的两边必须都是 `K` 类型。如果其中一边不是的话，就得先隐式转换为 `K` 才能用。

这是早期 C++98 设计的失败，当时他们没想到 `find` 还可以接受 `string_view` 和 `const char *` 这类可以和 `string` 比较，但构造会廉价得多的弱引用类型。

只好后来引入了透明比较器企图力挽狂澜，然而为了历史兼容，`map<K, V>` 默认仍然是 `map<K, V, less<K>>`。

如果我们同学的编译器支持 C++14，建议全部改用这种写法 `map<K, V, less<>>`，从而用上更高效的 find、at、erase、count、contains 等需要按键查找元素的函数。

#### 应用：字符串为键的字典

除非传入的刚好就是一个 `string` 的 const 引用，否则就会发生隐式构造 `string` 的操作。

如果传入的是一个 `string_view` 或 `const char *`，那么需要从他们构造出一个 `string`，然后才能传入传统的 `find(string const &)` 函数。而 `string` 的构造会发生拷贝，且可能产生内存分配。

对于比较大的字符串做键值，每次查找都需要重新构造一个 `string` 对象，开销会比较大。

```cpp
map<string, int> lut;

lut.at("a-very-very-very-very-long-key");
// 等价于:
lut.at(string("a-very-very-very-very-long-key")); // 隐式构造了一个 string，导致深拷贝了整个字符串！
```

而启用了透明比较后，就不需要每次都拷贝整个字符串来构造 `string` 了。因为 find、at 这类函数会启用一个泛型的版本 `at(Kt &&)`，Kt 可以是任何类型，只要他支持与 `string` 比较。可以是 `const char *`，`string_view` 或另一个 `string`。

```cpp
map<string, int, less<>> lut;

lut.at("a-very-very-very-very-long-key");
// 等价于:
lut.at<const char *>("a-very-very-very-very-long-key");
```

因为不用拷贝了，更加高效，特别是对于键字符串非常长的情况。

at 内部也不会构造任何新的 `string`，他会拿着 `const char *` 和红黑树中的每个节点调用 `==` 比较。

> `string == const char *` 是安全的，会比较字符串的内容而不是地址。

#### 应用：智能指针为键的字典

某有某些特殊情况下，我们需要把指针，甚至智能指针！放进 map 或 set 的键中，用于快速按指针的值查找到元素。（是的你没听错，是放在**键类型**里！）

> 轶事：把指针放在键里并不罕见，常见的一个用法是 `set<Node *>`。好处是当 `Node` 析构时，他可以直接调用 `set.erase(this)` 把自己剔除掉。而普通的 `set<Node>` 就很难做到这一点了，你无法通过 Node 的 this 指针获得他在 set 中的迭代器，也无法知道自己位于哪个 set 中。侵入式红黑树完美解决了这一痛点，LLVM 和 Linux 内核中都大量运用了侵入式链表/LRU/红黑树，以后的高级数据结构课程中会为你讲解。

```cpp
map<Node *, int> lut;

Node *raw_ptr = get_some_ptr();
lut.find(raw_ptr);
```

如果是智能指针，就比较困难了，特别是 `unique_ptr`。如果你已知一个原始指针，想要在 map 中查找指向同样的智能指针键。

```cpp
map<unique_ptr<Node>, int> lut;
Node *raw_ptr = get_some_ptr();
lut.find(raw_ptr); // 错误：无法从 Node * 隐式构造 unique_ptr<Node>
```

过去，人们不得不用一种称为 stale-ptr（变质指针）的黑科技，来构造一个不掌握生命周期的伪 unique_ptr 出来：

```cpp
map<unique_ptr<Node>, int> lut;
Node *raw_ptr = get_some_ptr();
unique_ptr<Node> stale_ptr(raw_ptr);  // 一个并不掌握生命周期的“变质智能指针”
lut.find(stale_ptr); // OK: 匹配到 find(unique_ptr<Node> const &) 重载
stale_ptr.release(); // 必须！否则会出现双重释放 (double-free) 错误
```

而 C++14 中，我们只需定义一个透明的比较函数，支持 `Node *` 与 `unique_ptr<Node>` 互相比较即可：

```cpp
struct transparent_ptr_less {
    template <class T>
    bool operator()(T *const &p1, T const &p2) const {
        return p1 < p2;
    }

    template <class T>
    bool operator()(T *const &p1, unique_ptr<T> const &p2) const {
        return p1 < p2.get();
    }

    template <class T>
    bool operator()(unique_ptr<T> const &p1, T *const &p2) const {
        return p1.get() < p2;
    }

    template <class T>
    bool operator()(unique_ptr<T> const &p1, unique_ptr<T> const &p2) const {
        return p1.get() < p2.get();
    }

    using is_transparent = std::true_type;
};

map<unique_ptr<Node>, int, transparent_ptr_less> lut;
Node *raw_ptr = get_some_ptr();
lut.find(raw_ptr); // OK: 匹配到泛型的 find(Kt &&) 重载，其中 Kt 推导为 Node *const &
```

#### 应用：超大对象为键的字典

以下摘自 cppreference 上泛型 find 的官方案例：

```cpp
struct FatKey   { int x; int data[1000]; };
struct LightKey { int x; };
// Note: as detailed above, the container must use std::less<> (or other 
//   transparent Comparator) to access these overloads.
// This includes standard overloads, such as between std::string and std::string_view.
bool operator<(const FatKey& fk, const LightKey& lk) { return fk.x < lk.x; }
bool operator<(const LightKey& lk, const FatKey& fk) { return lk.x < fk.x; }
bool operator<(const FatKey& fk1, const FatKey& fk2) { return fk1.x < fk2.x; }

int main() {
    // transparent comparison demo
    std::map<FatKey, char, std::less<>> example = {{{1, {}}, 'a'}, {{2, {}}, 'b'}};

    LightKey lk = {2};
    if (auto search = example.find(lk); search != example.end())
        std::cout << "Found " << search->first.x << " " << search->second << '\n';
    else
        std::cout << "Not found\n";
}
Found 2 b
```

## 神奇的 multimap

允许重复键值的 multimap

map 中一个键对应一个值，而 multimap 一个键可以对应多个值。

- map：排序 + 去重；
- multimap：只排序，不去重。

```cpp
// map<K, V> 的插入函数：
pair<iterator, bool> insert(pair<const K, V> const &kv);
pair<iterator, bool> insert(pair<const K, V> &&kv);
// multimap<K, V> 的插入函数：
iterator insert(pair<K, V> const &kv);
iterator insert(pair<K, V> &&kv);
```

因为 multimap 允许重复键值，所以插入总是成功，与普通 map 相比不用返回 bool 表示是否成功了。

### 元素的排列顺序

```cpp
multimap<string, string> tab;
tab.insert({"rust", "silly"});
tab.insert({"rust", "trash"});
tab.insert({"rust", "trash"});
tab.insert({"cpp", "smart"});
tab.insert({"rust", "lazy"});
tab.insert({"cpp", "fast"});
tab.insert({"java", "pig"});
print(tab);
{"cpp": "smart", "cpp": "fast", "java": "pig", "rust": "silly", "rust": "trash", "rust": "trash", "rust": "lazy"}
```

插入进 multimap 的重复键会紧挨着，他们之间的顺序取决于插入的顺序。例如上面键同样是 “cpp” 的两个元素，”smart” 先于 “fast” 插入，所以 “smart” 靠前了。

### 用途：动态排序！

multimap / multiset 的作用通常就不是键值映射了，而是利用红黑树会保持元素有序的特性（任何二叉搜索树都这样）实现一边插入一边动态排序。

传统排序方式：

```cpp
std::vector<int> arr;
int i;
while (cin >> i) {
    arr.push_back(i);
}
std::sort(arr.begin(), arr.end(), std::less<int>());
```

multiset 排序方式：

```cpp
std::multiset<int> tab;
int i;
while (cin >> i) {
    tab.insert(i);
}
// 无需再排序，tab 中的键已经是有序的了！
// 如需取出到 vector:
std::vector<int> arr(tab.begin(), tab.end());
```

利用 multimap 键-值对的特点，还能轻易实现只对键排序，值的部分不参与排序的效果。

multimap 排序的好处是：

- 动态排序，在插入的过程中就保持整个红黑树的有序性，最后任何无需额外操作。
- 在一次次插入的过程中，每时每刻都是有序的，而不必等到最后才变得有序。
- 可以随时动态删除一个元素，同样不会破坏有序性。
- 还很方便随时按键值查找到和我相等的元素。
- 如果还额外需要去重，则只需改用普通 map

普通 map 轻松实现去重 + 动态排序，如何处置重复的键随你决定：

- 普通 map 的 insert 只接受第一次出现的键-值对。
- 普通 map 的 insert_or_assign 只保留最后一次出现的键-值对。

### 查询某个键对应的多个值

因为 multimap 中，一个键不再对于单个值了；所以 multimap 没有 `[]` 和 `at` 了，也没有 `insert_or_assign`（反正 `insert` 永远不会发生键冲突！）

```cpp
pair<iterator, iterator> equal_range(K const &k);

template <class Kt>
pair<iterator, iterator> equal_range(Kt &&k);
```

要查询 multimap 中的一个键对应了哪些值，可以用 `equal_range` 获取一前一后两个迭代器，他们形成一个区间。这个区间内所有的元素都是同样的键。

```cpp
multimap<string, string> tab;
tab.insert({"rust", "silly"});
tab.insert({"rust", "trash"});
tab.insert({"rust", "trash"});
tab.insert({"cpp", "smart"});
tab.insert({"rust", "lazy"});
tab.insert({"cpp", "fast"});
tab.insert({"java", "pig"});

auto range = tab.equal_range("cpp");
for (auto it = range.first; it != range.second; ++it) {
    print(it->first, it->second);
}
cpp smart
cpp fast
```

`equal_range` 返回两个迭代器相等时（即区间大小为 0），就代表找不到该键值。

```cpp
auto range = tab.equal_range("html");
if (range.first == range.second) {
    print("找不到该元素！");
} else {
    for (auto it = range.first; it != range.second; ++it) {
        print(it->first, it->second);
    }
}
```

`equal_range` 返回的两个迭代器，也可以用 `lower_bound` 和 `upper_bound` 分别获得：

```cpp
auto begin_it = tab.lower_bound("html");
auto end_it = tab.upper_bound("html");
if (begin_it == end_it) {
    print("找不到该元素！");
} else {
    for (auto it = begin_it; it != end_it; ++it) {
        print(it->first, it->second);
    }
}
```

### lower/upper_bound 实现范围查询

- `lower_bound(key)` 到 `end()` 迭代器之间的元素，都是大于等于（>=）当前 key 的元素。
- `upper_bound(key)` 到 `end()` 迭代器之间的元素，都是大于（>）当前 key 的元素。
- `begin()` 到 `lower_bound(key)` 迭代器之间的元素，都是小于（<）当前 key 的元素。
- `begin()` 到 `upper_bound(key)` 迭代器之间的元素，都是小于等于（<=）当前 key 的元素。

例如我要对一系列小彭友的成绩数据进行排序，要求查出大于等于 60 分的所有同学，发放“小红花”：

```cpp
struct Student {
    string name;
    int score;
};

vector<Student> students;
```

就可以把成绩 int 作为键，学生名字作为值，插入 multimap。

插入的过程中 multimap 就自动为你动态排序了。

```cpp
multimap<int, string> sorted;
for (auto const &stu: students) {
    sorted.insert({stu.score, stu.name});
}
```

然后，要找出所有大于等于 60 分的同学，也就是 `lower_bound(60)` 到 `end()` 这个区间：

```cpp
// where score >= 60
for (auto it = sorted.lower_bound(60); it != sorted.end(); ++it) {
    print("恭喜 {} 同学，考出了 {} 分，奖励你一朵小红花", it->second, it->first);
}
```

找出 30（含）到 60（不含）分的同学也很容易：

```cpp
// where 30 <= score and score < 60
for (auto it = sorted.upper_bound(30); it != sorted.lower_bound(60); ++it) {
    print("{} 同学考出了 {} 分，不要灰心！小彭老师奖励你一朵小黄花，表示黄牌警告", it->second, it->first);
}
```

## 时间复杂度总结说明

| 函数或写法    | 解释说明 | 时间复杂度 |
| ------------- | -------- | ---------- |
| m1 = move(m2) | 移动     | O(1)O(1)   |
| m1 = m2       | 拷贝     | O(N)O(N)   |
| swap(m1, m2)  | 交换     | O(1)O(1)   |
| m.clear()     | 清空     | O(N)O(N)   |

------

| 函数或写法                   | 解释说明                         | 时间复杂度       |
| ---------------------------- | -------------------------------- | ---------------- |
| m.insert({key, val})         | 插入键值对                       | O(logN)O(log⁡N)   |
| m.insert(pos, {key, val})    | 带提示的插入，如果位置提示准确   | O(1)O(1)+        |
| m.insert(pos, {key, val})    | 带提示的插入，如果位置提示不准确 | O(logN)O(log⁡N)   |
| m[key] = val                 | 插入或覆盖                       | O(logN)O(log⁡N)   |
| m.insert_or_assign(key, val) | 插入或覆盖                       | O(logN)O(log⁡N)   |
| m.insert({vals…})            | 设 M 为待插入元素（vals）的数量  | O(MlogN)O(Mlog⁡N) |
| map m =                      | 如果 vals 无序                   | O(NlogN)O(Nlog⁡N) |
| map m =                      | 如果 vals 已事先从小到大排列     | O(N)O(N)         |

------

| 函数或写法         | 解释说明                                                     | 时间复杂度         |
| ------------------ | ------------------------------------------------------------ | ------------------ |
| m.at(key)          | 根据指定的键，查找元素，返回值的引用                         | O(logN)O(log⁡N)     |
| m.find(key)        | 根据指定的键，查找元素，返回迭代器                           | O(logN)O(log⁡N)     |
| m.count(key)       | 判断是否存在指定键元素，返回相同键的元素数量（只能为 0 或 1） | O(logN)O(log⁡N)     |
| m.equal_range(key) | 根据指定的键，确定上下界，返回区间                           | O(logN)O(log⁡N)     |
| m.size()           | map 中所有元素的数量                                         | O(1)O(1)           |
| m.erase(key)       | 根据指定的键，删除元素                                       | O(logN)O(log⁡N)     |
| m.erase(it)        | 根据找到的迭代器，删除元素                                   | O(1)+O(1)+         |
| m.erase(beg, end)  | 批量删除区间内的元素，设该区间（beg 和 end 之间）有 M 个元素 | O(M+logN)O(M+log⁡N) |
| erase_if(m, cond)  | 批量删除所有符合条件的元素                                   | O(N)O(N)           |

------

| 函数或写法                      | 解释说明                               | 时间复杂度       |
| ------------------------------- | -------------------------------------- | ---------------- |
| m.insert(node)                  |                                        | O(logN)O(log⁡N)   |
| node = m.extract(it)            |                                        | O(1)+O(1)+       |
| node = m.extract(key)           |                                        | O(logN)O(log⁡N)   |
| m1.merge(m2)                    | 合并两个 map，清空 m2，结果写入 m1     | O(NlogN)O(Nlog⁡N) |
| m1.insert(m2.begin(), m2.end()) | 合并两个 map，m2 保持不变，结果写入 m1 | O(NlogN)O(Nlog⁡N) |

## 哈希表 unordered_map

C++11 新增：基于哈希 (hash) 的映射表 unordered_map

### unordered_map 与 map 之争：适用场景不同

之前提到，map 底层基于红黑树，大多数操作的复杂度都是 O(logN)O(log⁡N) 级别的，其中部分按迭代器的插入和删除的复杂度可以降低到 O(1)O(1)。

![hash_map](五、map和他的朋友们/hash_map.png)

而 unordered_map 则是基于哈希表的更高效查找，只需 O(1)O(1) 复杂度！他能实现如此高效查找得益于哈希函数可以把散列唯一定位到一个数组的下标中去，而数组的索引是 O(1)O(1) 的。缺点是哈希值可能产生冲突，而且哈希数组可能有空位没有填满，浪费一部分内存空间。总的来说哈希表在平均复杂度上（O(1)O(1)）比红黑树这类基于树的复杂度（O(logN)O(log⁡N)）更低，虽然固有延迟高，占用空间大，还容易被哈希冲突攻击。

- 哈希表结构简单无脑，在巨量的键值对的存储时会发挥出明显的性能优势，常用于需要高吞吐量但不太在乎延迟的图形学应用。
- 而各种基于树的数据结构，复杂度更加稳定，看似适合小规模数据，但是因为保持有序的特性，非常适合数据库这种需要范围查询的情况，且有序性反而有利于缓存局域性，无序的哈希表难以胜任。
- 最近新提出的一种数据结构——跳表，也是有序的，但基于链表，更加高效，在 Redis 等软件中都有应用。别担心，小彭老师之后的数据结构课程会专门介绍并带你手搓所有这些！

------

### 原理：unordered_map 中的“桶”

unordered_map 如何快速检索数据？高效的秘诀在于 unordered_map 内部是一个数组，一个由许多“桶”组成的数组。插入时把键值对存到键的 hash 对应编号的桶去，查询时就根据 hash 去线性地查找桶（这一操作是 O(1)O(1) 的）。

例如键为 “hello”，假设算出他的 hash 为 42。而当前桶的数量是 32 个，则会把 “hello” 存到 42 % 32 = 10 号桶去。查询时，同样计算出 hash(“hello”) % 32 = 10 号桶，然后就可以从 10 号桶取出 “hello” 对应的数据了。

```cpp
template <class K, class V>
class unordered_map {
    array<pair<K, V>, 32> buckets;

    void insert(pair<K, V> kv) {
        size_t h = hash(kv.first) % buckets.size();  // 计算出来的 hash 可能很大，取模可以防止 h >= buckets.size()
        buckets[h] = kv;
    }

    V &at(K k) {
        size_t h = hash(k) % buckets.size();
        auto &kv = buckets[h];
        if (k != kv.first) throw out_of_range{};
        return kv.second;
    }
};
```

### 哈希冲突 (hash-collision)

但是这里有一个问题，如果两个不同的字符串，刚好 hash 以后的模相同怎么办？这种现象称为 hash 冲突。

C++ 标准库的解决方案是采用链表法：一个桶不是单独的一个 K-V 对，而是数个 K-V 对组成的单链表（forward_list）。一个桶不是只存储一个数据，而是可以存储任意多个数据（0到∞个）。

插入时，找到对应的桶，并往链表的头部插入一个新的 K-V 对。查找时，先找到对应的桶，在这个桶里的链表里顺序遍历查找，由于第一步的桶查找是 O(1)O(1) 的，虽然最后还是链表暴力查找，但是已经被桶分摊了一个维度，因此查找的平均复杂度还是 O(1)+O(1)+ 的。

```cpp
    void insert(pair<K, V> kv) {
        size_t h = hash(kv.first) % buckets.size();  // 计算 hash 的模（所在桶的编号）
        buckets[h].push_front(kv);                // 单链表的头插，是最高效的
    }

    V &at(K k) {
        size_t h = hash(k) % buckets.size();         // 计算 hash 的模（所在桶的编号）
        for (auto &kv: buckets[h]) {
            if (k == kv.first)  // 可能有多个不同的键刚好有相同的 hash 模，需要进一步判断键确实相等才能返回
                return kv.second;
        }
        throw out_of_range{};
    }
```

------

这里还是有一个问题，hash 冲突时，对链表的暴力遍历查找复杂度是 O(N)O(N) 的，随着越来越多的元素被插入进来，32 个桶将会拥挤不堪。假设有 n 个元素，则平均每个桶都会有 n / 32 个元素，需要 n / 32 次遍历。所以元素数量充分大时 unordered_map 又会退化成暴力遍历的 O(N)O(N) 复杂度，满足不了我们用他加速查找的目的。

桶的数量相比元素的数量越是不足，越是拥挤，越是容易退化成链表。

因此 C++ 标准库规定，插入时，当检测到平均每个桶里都有 1 个元素时，也就是元素数量大于桶的数量时，就会发生自动扩容，一次性让桶的数量扩充 2 倍，并重新计算每个元素的 hash 模（桶编号）全部重新插入一遍。

> 元素数量除以桶的数量被称为“负载率（load factor），对于链表法的哈希表 unordered_map 来说，负载率可以高于 1；对于线性地址法的 flat_hash_map 则最高为 1。C++ 标准库通常的 unordered_map 实现中，负载率高于 1 时，就会发生自动扩容。可以通过 `.load_factor()` 函数查询一个 unordered_map 的负载率。

```cpp
template <class K, class V>
class unordered_map {
    vector<forward_list<pair<K, V>>> buckets;  // 因为需要动态扩容，桶数组变成了动态数组 vector
    size_t size = 0;  // 记录当前容器共有多少个元素

    void insert(pair<K, V> kv) {
        if (size + 1 > buckets.size()) reserve(n);  // 如果插入后的元素数量大于桶的容量，则扩容
        size_t h = hash(kv.first) % buckets.size();
        buckets[h].push_front(kv);
        size++;     // insert 时 size 自动加 1，erase 时也要记得减 1
    }

    void reserve(size_t n) {
        if (n <= buckets.size()) return;  // 如果要求的大小已经满足，不需要扩容
        buckets.resize(max(n, buckets.size() * 2));  // 把桶数组至少扩大 2 倍（避免重复扩容），至多扩到 n
        此处省略 rehash 的具体实现  // 桶的数量发生变化了，需要重新计算一遍所有元素 hash 的模，并重新插入
    }
};
```

> 每个 key 所在的桶编号计算公式：bucket_index(key) = hash(key) % bucket_count()

还是存在问题，刚刚的 insert 根本没有检测要插入的键是否已经存在了。如果已经存在还插入，那就变成 unordered_multimap 了！我们是普通的需要去重的 unordered_map，所以插入时先需要遍历下链表检测一下。

```cpp
template <class K, class V>
class unordered_map {
    vector<forward_list<pair<K, V>>> buckets;
    size_t size = 0;

    struct iterator {
        explicit iterator(pair<K, V> &kv) { /* ... */ }
        // ...
    };

    pair<iterator, bool> insert(pair<K, V> kv) {
        if (size + 1 > buckets.size()) reserve(size + 1);
        size_t h = hash(kv.first) % buckets.size();
        for (auto &kv2: buckets[h]) {
            if (kv.first == kv2.first)  // 检测是否发生了冲突
                return {iterator(kv2), false};  // 发生冲突则返回指向已存在的键的迭代器
        }
        buckets[h].push_front(kv);
        size++;
        return {iterator(buckets.front()), true};  // 没发生冲突则返回成功插入元素的迭代器
    }
};
```

## unordered_map 与 map 的异同

用法上，unordered_map 基本与 map 相同，以下着重介绍他们的不同点。

### 区别 1：有序性

- map 基于红黑树，元素从小到大顺序排列，遍历时也是从小到大的，键类型需要支持比大小（std::less 或 <）。
- unordered_map 基于哈希散列表，里面元素顺序随机，键类型需要支持哈希值计算（std::hash）和判断相等（std::equal_to 或 ==）。

map 中的元素始终保持有序，unordered_map 里面的元素是随机的。

这也意味着 std::set_union 这类要求输入区间有序的 algorithm 函数无法适用于 unordered_map/set。

#### hash 和 equal_to

map 只需要 K 类型支持一个 less 就能工作。

而 unordered_map 需要 K 支持的 trait 有两个：hash 和 equal_to。

`unordered_map<K, V>` 的完整形态是：

```cpp
unordered_map<K, V, hash<K>, equal_to<K>, allocator<pair<const K, V>>>
```

- 其中 allocator 我们照例先跳过不讲，之后分配器专题课中会介绍。
- hash 说的是，如何求键的哈希值？hash 仿函数接受一个 K 类型的键，返回一个 size_t（在 64 位系统上是个无符号 64 位整数，表示哈希值）。
- equal_to 说的是，如何判断两个键相等？如果两个键完全相等，他会返回 true。

这里对 hash 的实现只有一个要求，**如果两个键相等，则他们的哈希必定也相等，反之则不一定**。

这个假设构成了 unordered_map 得以高效的基石，他使得 unordered_map 可以更快排除不可能的答案，而不必像 vector 的查找那样需要去暴力遍历全部元素，只需要遍历哈希相等的那一部分元素就够了。

```cpp
template<typename _Key, typename _Tp,
   typename _Hash = hash<_Key>,      // 默认的哈希函数实现，支持了 int, void *, string 等类型
   typename _Pred = equal_to<_Key>,  // 默认的 == 运算符
   typename _Alloc = allocator<pair<const _Key, _Tp>>>
class unordered_map
```

换言之，只要 unordered_map 发现两个键不相等，就不用再做具体值的比较了，他们不可能相等了！

#### 哈希函数的思想

hash 返回的 size_t 这个整数可以理解为一个对任意类型的“摘要”。

把一个很复杂的类型（例如 string）压缩成一个 unordered_map 很轻易就能比较的 size_t 整数，整数比较起来就很容易，而且还能直接作为数组的下标（string 不能直接作为数组的下标）。

这种摘要的关键在于如何把一个极为复杂的类型“映射”到小小的 size_t 上去，并且分布得尽可能均匀，不要冲突。

这就需要我们把这个极为复杂类型的每个成员（对 string 而言就是每个字符）都加到最终结果的表达式中。

以字符串类型 string 为例，常见的一种生成“摘要”的方法是，用一个任意素数的乘方序列和各字符的 ASCII 码做点积：

```cpp
size_t hash_string(string const &s) {
    size_t h = 0;
    for (char c: s) {
        h = h * 37 + c;
    }
    return h;
}
```

例如对于字符串 “hello”，则 hash 可以生成这样一个摘要：

```cpp
size_t h = ((('h' * 37 + 'e') * 37 + 'l') * 37 + 'l') * 37 + 'o';
```

相当于 h⋅374+e⋅373+l⋅372+l⋅37+oh⋅374+e⋅373+l⋅372+l⋅37+o

> 也有其他更高效的生成摘要的方法，例如借助位运算。
>
> 甚至还有偷懒直接拿 strlen 当哈希函数的“世界上最好的哈希表”，我不说是谁。（其实是早期 PHP 啦）

#### 自动取模

```cpp
size_t h = ((('h' * 37 + 'e') * 37 + 'l') * 37 + 'l') * 37 + 'o';
```

随着字符串长度的增加，这个 h 肯定会超过 size_t 的表示范围，但是没关系，无符号整数的乘法、加法溢出不是未定义行为，他会自动 wrapping（取关于 264264 的模），也就是只保留乘法结果和 2^64 取模的部分。

取模也是对哈希值常见的一个操作，反正哈希值是随机的，取模以后也是随机的，但是缩小了范围。

> 基本假设：m 足够小时，一个均匀的分布取以 m 的模以后仍然应该是均匀的

unordered_map 中桶的数量是有限的，为了把范围从 00 到 264−1264−1 的哈希值映射为 0 到 bucket_count - 1 的桶序号，他内部会把键的哈希值取以桶数量的模，作为一个键要存储到的桶的序号：

```ini
bucket_index = hash(key) % bucket_count
```

### hash 是个 trait 类

std::hash 就是标准库用于计算哈希的仿函数类了，他和 std::less 一样，是一个 trait 类。

一些常见的类型有默认的实现，也可以针对自定义类型添加特化。

```cpp
template <class T>
struct hash {
    size_t operator()(T const &t) const noexcept;  // 有待实现
};

template <>
struct hash<int> {
    size_t operator()(int t) const noexcept {
        return t;  // 对 int 的特化
    }
};

template <class T>
struct hash<T *> {
    size_t operator()(T *t) const noexcept {
        return reinterpret_cast<uintptr_t>(t);  // 对 T * 的偏特化
    }
};
```

------

std::hash 针对每个不同的类型做了特化，例如当我们需要计算 string 类型的 hash 时：

```cpp
string str = "Hello, world";
size_t h = hash<string>()(str);
print(str, "的哈希是", h);
```

注意：这里有两个括号，第一个是空的。第一个括号创建仿函数对象，第二个用str作为实参调用仿函数的 `operator()`。当然还别忘了第一个尖括号，这个尖括号里的 string 表示的是 hash 仿函数接下来要接受参数的类型，之所以作为类的模板参数而不是模板函数，是为了方便特化和偏特化。同学们也可以自己写一个这样的函数，用起来就不用指定类型（如这里的 string）了，让模板函数自动推导参数类型（类似于 make_pair）：

```cpp
template <class T>
size_t do_hash(T const &t) {
    return hash<T>()(t);
}
int main() {
    string str = "Hello, world";
    size_t h = do_hash(str);
    print(str, "的哈希是", h);
}
"Hello, world" 的哈希是 14701251851404232991
```

------

对任意类型哈希的结果都是一个 size_t，其在 32 位系统上等同于 uint32_t，在我们 64 为系统上等同于 uint64_t。选择 size_t 是为了能哈希了以后直接用于 unordered_map 中桶的索引。

由于 hash 是用作哈希表的哈希函数，而不是用于加密领域（请你移步 md5），或是用于随机数生成（请移步 mt19937），因此对于任意类型，只需要根据他生成一个 size_t 的哈希值即可，只要保证哈希值分布均匀即可，不一定要有随机性。例如标准库对 int 的 hash 实现就是个恒等函数——直接返回其整数值，不用做任何计算：

```cpp
template <>
struct hash<int> {
    size_t operator()(int t) const noexcept {
        return t;  // 对 int 的特化真是什么也不做呢？
    }
};
```

而对于任意指针的实现则是直接把指针 bit-cast 成 size_t：

```cpp
template <class T>
struct hash<T *> {
    size_t operator()(T *t) const noexcept {
        return reinterpret_cast<uintptr_t>(t);  // 指针强制转换为整数
    }
};
```

------

```cpp
int i = 42;
int j = hash<int>()(i);  // 没想到罢！我系恒等函数哒
print(i, j);
42 42
```

记住，std::hash 不是为了加密或随机而生的，他的功能仅仅是尽可能快速地把任意类型 T 映射到 size_t 而已。

至于这对 unordered_map 的性能有何影响？通常没有什么影响，除非输入键故意设为和 bucket_count 同模，毕竟反正你也无法断定输入键的排布模式，不论选什么哈希函数只要保证均匀都是可以的。而恒等函数刚好是均匀的，又不用额外的花里胡哨位运算浪费时间，反而可能因为键有序而提升了缓存局域性，提升了性能，所以各大厂商的标准库都是这么做的。

### 区别 2：时间复杂度

- map 的查询和插入操作是 O(logN)O(log⁡N) 复杂度的。
- unordered_map 的查询和插入操作是 O(1)+O(1)+ 复杂度的。

看起来 unordered_map 更高效？那还要 map 干什么？完全上位替代啊？

但是我们要注意，上面所说的复杂度 O(1)O(1) 只是平均下来的，并不代表每一次 unordered_map 插入操作的复杂度都是 O(1)O(1)！所以，复杂度表示法里的这个 + 号就是这个意思，代表我这个复杂度只是多次运行取平均，如果只考虑单次最坏的情况，可能更高。

- map 的插入操作**最坏**也只是 O(logN)O(log⁡N) 复杂度的。
- unordered_map 的插入操作**最坏**可以是 O(N)O(N) 复杂度的。

处理很高的数据量时，这一点最坏的情况会被平摊掉，unordered_map 更高效。

#### 哈希表的复杂度不稳定

所以 unordered_map 不稳定，虽然平均是 O(1)O(1) 复杂度，但最坏可达到 O(N)O(N) 复杂度。背后的原因是什么呢？

原来 unordered_map 和 vector 一样，是一个需要不断动态扩容的容器。

如果不扩容，那么当很多元素挤在一个桶里，链表的压力就会变大，会很低效，因此 unordered_map 必须扩容。但是在扩容的时候是需要进行 rehash 操作的。一次扩容，就需要把所有的元素都移动一遍。

结果就是 unordered_map 的插入如果没触发 rehash，那就是 O(1)O(1) 的。触发了，那就是最坏的情况，O(N)O(N) 的。但是不触发的情况远多于触发了的，所以平均下来还是 O(1)O(1)，为了提醒人们他最坏的情况，所以写作 O(1)+O(1)+，读作“平摊 O1”（Amortized Constant）。

此外，不仅 unordered_map 的插入函数是 O(1)+O(1)+，他的查询函数也是 O(1)+O(1)+。为什么呢？设想你在编写一个富连网服务器，如果黑客已知你的 hash 函数，那他就可以通过构造一系列特殊设计好的 key，他们的哈希刚好相等（或同模），这样就使得所有 key 刚好全部落在一个桶里，导致 unordered_map 退化成线性的链表，所有的查询和插入都变成了这一个桶上的链表遍历操作，复杂度达到最坏的 O(N)O(N)，这一现象叫做 hash 退化。

因此 hash 函数的好坏决定着 unordered_map 性能，对于安全领域来说，还要保证 hash 函数无法被黑客破解。只要 hash 函数足够随机，就能保证键不冲突，就很快，一旦出现键冲突就会变慢。但需要频繁使用的 hash 函数计算难度又不能太大，那又会影响性能，因此 hash 也不能太过复杂。

> 标准库里存在这种“平摊复杂度”的例子还有很多，例如 vector 的 push_back 不 reserve 的话，就是 O(1)+O(1)+ 的，因为他需要动态扩容。

#### 哈希表的应用限制

一些实时性要求很高的领域就不能用 unordered_map。例如你造了个火箭，规定：火箭控制程序需要在 1000 μs 内对外界变化做出实时反应，如果不能及时做出反应，火箭就会做托马斯回旋给你看。

你在火箭控制程序中用了 unordered_map，这个程序会不断运行，以便控制火箭做出正确的机动，对抗侧向风干扰。第一次运行他在 180 μs 内反应了，第二次在 250 μs 内反应了，第三次 245 μs 内反应了，你觉得他很高效。

但是突然有一次，unordered_map 觉得他内部“桶太乱”了，急需重新扩容并 rehash 一下“忧化性能”。然后，他把所有的元素都移动了一遍，移动完了，把处理完的数据返回给火箭姿态调控系统，认为大功告成。但当他睁开眼睛一看，刚想要控制一下姿态呢？却发现自己已经在做托马斯回旋了！原来我这一“忧化”就忧了 4000 μs，超出了火箭实时响应的硬性指标，导致西装骰子人卷款跑路，小彭老师破产。

小彭老师重新创业，这次他选用了稳定的 map，第一次他在 810 μs 内反应了，第二次在 680 μs 内反应了，第三次 730 μs 内反应了，你觉得他很低效。但是他每一次都能成功卡点给你完成任务，从来不会突然超过 O(logN)O(log⁡N)，他的最坏情况是可控的，从而避免了托马斯破产回旋。小彭老师最终创业成功，1000 年后，我司成功建造完成 Type-II 文明所急需的戴森球，向星辰大海进军。

对实时性要求高的这类领域包括，音视频，造火箭，量化交易等。这类低延迟低吞吐量的领域对平摊复杂度很反感，他们只看重最坏的复杂度，而不是平均的。

但对于主打一个高吞吐量无所谓延迟的离线图形学，离线科学计算，实时性不重要的生态化反场景，我们可以认为 unordered_map 的平摊 O(1)+O(1)+ 就是比 map 高效的。

------

### 区别 3：迭代器失效条件

- map 和 unordered_map 都是只有当删除的刚好是迭代器指向的那个元素时才会失效，这点相同。
- 但 unordered_map 扩容时候的 rehash 操作会造成所有迭代器失效。

> insert 可能导致 unordered_map 扩容，其他只读操作不会。

迭代器指向的那个元素被删除时，不论 map 和 unordered_map 都会失效。

unordered_map 在 insert 时如果发生扩容，之前保存的迭代器可能失效，可以通过调用 reserve 避免 insert 时扩容。

小彭老师编写好了迭代器失效表，方便你记忆:

| 容器          | clear | swap | opeartor= | rehash |
| ------------- | ----- | ---- | --------- | ------ |
| vector        | 是    | 否   | 是        | -      |
| map           | 是    | 否   | 是        | -      |
| unordered_map | 是    | 否   | 是        | -      |

| 容器          | find | count | at   | []                                                           |
| ------------- | ---- | ----- | ---- | ------------------------------------------------------------ |
| vector        | 否   | 否    | 否   | 否                                                           |
| map           | 否   | 否    | 否   | 否                                                           |
| unordered_map | 否   | 否    | 否   | 是，如果创建了新元素且 size / bucket_count > max_load_factor |

小彭老师编写好了迭代器失效表，方便你记忆:

| 容器          | push_back                | insert                                               | erase                                                        | reserve |
| ------------- | ------------------------ | ---------------------------------------------------- | ------------------------------------------------------------ | ------- |
| vector        | 是，如果 size > capacity | 是，如果插入位置在当前迭代器之前，或 size > capacity | 是，如果删除的元素在当前迭代器之前，或刚好是当前迭代器指向的 | 是      |
| map           | -                        | 否                                                   | 是，如果删除的刚好是当前迭代器指向的元素                     | -       |
| unordered_map | -                        | 是，如果 size / bucket_count > max_load_factor       | 是，如果删除的刚好是当前迭代器指向的元素                     | 是      |

也可以查看官方版《迭代器失效表》：https://en.cppreference.com/w/cpp/container#Iterator_invalidation

### 负载率（load_factor）

计算公式：负载因子(load_factor) = 当前元素数量(size) ÷ 当前桶的数量(bucket_count)

插入新元素后，当检测到负载因子大于最大负载因子（默认 1.0）时，就会自动进行 rehash 操作。

为了避免重复小规模扩容浪费时间，这次 rehash 会一次性扩容两倍（跟 vector 的 push_back 扩容类似）。

> 最大负载因子可以通过 max_load_factor 函数调整。当前负载因子可以通过 load_factor 函数查询。

直观理解：当每个桶平均都有一个元素时，unordered_map 就会认为已经很满了，就会扩容并重新分配位置。

> 由于默认最大负载因子是 1.0，所以扩容条件等价于 size > bucket_count

------

### rehash 函数

在操作 unordered_map 容器过程（尤其是向容器中添加新键值对）中，一旦当前容器的负载因子超过最大负载因子（默认值为 1.0），该容器就会适当增加桶的数量（通常是翻一倍），并自动执行 rehash() 成员方法，重新调整各个键值对的存储位置（此过程又称“重哈希”），此过程很可能导致之前创建的迭代器失效。[1](https://142857.red/book/stl_map/#fn:1)

> 除了扩容时自动的 rehash，确认数据插入完毕不会再改动时，我们也可以手动调用 rehash() 函数来优化容器中元素的排布，提升性能。

```cpp
unordered_map<int, int> umap;
for (int i = 1; i <= 50; i++) {
    umap.emplace(i, i);
}
auto pair = umap.equal_range(49);  //获取键为 49 的键值对所在的区间，由于不是 multimap，区间大小只能为 0 或 1
for (auto iter = pair.first; iter != pair.second; ++iter) { //输出 pair 范围内的每个键值对的键的值
    cout << iter->first << '\n';
}
umap.rehash(10); //手动调用 rehash() 函数重哈希为 10 个桶
for (auto iter = pair.first; iter != pair.second; ++iter) { // 重哈希之后，之前保存的迭代器可能会发生变化
    cout << iter->first << '\n';
}
49 
Segmentation fault (core dumped)
```

### hash 需要特化

基于红黑树的映射表 map 只需支持比较运算的 less 即可，而 unordered_map 需要哈希和相等两个 trait，他们分别名叫 std::hash 和 std::equal_to。

虽然两者都是仿函数，但也有很多区别：

1. hash 只接受一个参数，而 equal_to 接受两个参数。
2. hash 返回 size_t，而 equal_to 返回 bool 类型。
3. equal_to 有默认的实现，那就是调用运算符 ==。而 hash 没有默认实现，也没相应的运算符，只能手动特化。

正因为如此，通常我们需要让一个类（例如 Student）支持 equal_to 或 less 这些有相应运算符的仿函数时，直接在类型内部定义 `operator==` 或 `operator<` 即可，而 hash 则是只能用特化的方法才能支持上。

```cpp
template <class T>
struct hash {
    size_t operator()(T const &t) const noexcept;  // 有待实现
};

template <class T>
struct equal_to {
    bool operator()(T const &x, T const &y) const noexcept {
        return x == y;
    }
};
```

有些类型能用作 map 的键，但不能用作 unordered_map 的键。这是因为偷懒的标准库没对他们的 hash 特化！

例如 tuple 支持 < 运算符，支持 less。

但是 tuple 没有 hash 的特化，不支持 hash。

```cpp
tuple<int, int> tup;
size_t h = hash<tuple<int, int>>()(tup);  // 编译期报错：查无此函数！
unordered_map<tuple<int, int>>
```

### 给 tuple 等复合类型自定义哈希函数

和 less 的情形一样，也是有三种解决方案：

1. 自定义一个 hash 的特化，equal_to 的特化

```cpp
template <>
struct std::hash<Student> {
    bool operator()(Student const &x) const {
        return hash<string>()(x.name) ^ hash<int>(x.id) ^ hash<int>(x.sex);
    }
};

template <>
struct std::equal_to<Student> {
    bool operator()(Student const &x, Student const &y) const {
        return x.name == y.name && x.id == y.id && x.sex == y.sex;
    }
};

unordered_map<Student, int> stutab;
```

1. 自定义一个 hash 的仿函数类，一个 equal_to 的仿函数类，然后传入 unordered_map 做模板参数

```cpp
template <>
struct HashStudent {
    bool operator()(Student const &x) const {
        return hash<string>()(x.name) ^ hash<int>(x.id) ^ hash<int>(x.sex);
    }
};

struct EqualStudent {
    bool operator()(Student const &x, Student const &y) const {
        return x.name == y.name && x.id == y.id && x.sex == y.sex;
    }
};

unordered_map<Student, int, HashStudent, EqualStudent> stutab;
```

> 注：如果 Student 已经定义了 `operator==`，则这里不用 EqualStudent，默认的 equal_to 会自动调用 == 运算符的。

1. 对于 tuple 而言，tuple 已经有了 == 运算符，不用特化 equal_to 了，只需要特化或指定 hash 即可

```cpp
template <class ...Ts>
inline size_t hash_combine(Ts const &...ts) {
    return (std::hash<Ts>()(ts) ^ ...);
}

template <class ...Ts>
struct std::hash<std::tuple<Ts...>> {
    size_t operator()(std::tuple<Ts...> const &x) const {
        return std::apply(hash_combine<Ts...>, x);
    }
};

unordered_map<tuple<string, int, int>, int> stutab;
```

#### 试试看效果吧！

```cpp
template <class ...Ts>
inline size_t hash_combine(Ts const &...ts) {
    return (std::hash<Ts>()(ts) ^ ...); // 把任意多个元素哈希通过“位异或(^)”拼凑成一个单独的哈希
}

template <class ...Ts>
struct std::hash<std::tuple<Ts...>> {
    size_t operator()(std::tuple<Ts...> const &x) const {
        // std::apply 会把 tuple 里的元素全部展开来调用 hash_combine，相当于 Python 里的 *args
        return std::apply(hash_combine<Ts...>, x);
    }
};

int main() {
    tuple<int, int> t(42, 64);
    size_t h = hash<tuple<int, int>>()(t);
    print(t, "的哈希值是:", h);
    return 0;
}
{42, 64} 的哈希值是: 106
```

这里的计算是：42 ^ 64 = 106，位异或的知识可以去 Bing 搜索一下，或者问一下 GPT，CS 学生应该都知道的。

### 更好的 hash_combine

但是简简单单用一个位异或 ^ 来把两个成员的哈希组合起来，有个严重的问题，如果 `tuple<int, int>` 里的两个成员值刚好一样，则其两个哈希值也会一样，那么他们通过位异或 ^ 合并的结果就会始终为 0。

例如不论 (42, 42) 还是 (64, 64) 这两个 tuple，他们的哈希值都会为 0。明明具体值不同哈希值却相同，这就是发生了哈希冲突，这会严重影响 unordered_map 的性能，是必须避免的。

用 + 来组合也有这个问题，如果第一个成员刚好是另一个的相反数，或只要是两个数加起来和相等，就会冲突。

例如如果我们用 unordered_map 构建一张地图的话，就发现当玩家在往斜上方移动时就会变得特别卡顿，原来是因为玩家的历史轨迹刚好是一条 y = x 的曲线，斜率为 1，由于我们采用 ^ 来组合哈希，就导致刚好这条线上所有的点都会塌缩到 0 号桶去，让 unordered_map 退化成了 O(N)O(N) 复杂度。

#### 最先进的是 boost::hash_combine 的方法

```cpp
template <class ...Ts>
inline size_t hash_combine(Ts const &...ts) {
    size_t h = 0;
    ((h ^= std::hash<Ts>()(ts) + 0x9e3779b9 + (h << 6) + (h >> 2)), ...);
    return h;
}

template <class ...Ts>
struct std::hash<std::tuple<Ts...>> {
    size_t operator()(std::tuple<Ts...> const &x) const {
        return std::apply(hash_combine<Ts...>, x);
    }
};

int main() {
    tuple<int, int> t(42, 64);
    size_t h = hash<tuple<int, int>>()(t);
    print(t, "的哈希值是:", h);
    return 0;
}
{42, 64} 的哈希值是: 175247763666
```

可以看到随机性大大提升了。

#### 应用

用 hash_combine 改进刚刚 Student 的哈希函数。

```cpp
template <>
struct std::hash<Student> {
    bool operator()(Student const &x) const {
        return hash_combine(hash<string>()(x.name), hash<int>(x.id), hash<int>(x.sex));
    }
};
```

同理可得 array 的特化

```cpp
template <class T, size_t N>
struct std::hash<std::array<T, N>> {
    size_t operator()(std::array<T, N> const &x) const {
        std::hash<T> hasher;
        size_t h = 0;
        for (T const &t: x) {
            h ^= hasher(t);
        }
        return h;
    }
};

unordered_map<array<string, 3>, int> stutab;
```

采用素数乘方法来提升哈希函数的均匀性和随机性：

```cpp
template <class T, size_t N>
struct std::hash<std::array<T, N>> {
    size_t operator()(std::array<T, N> const &x) const {
        std::hash<T> hasher;
        size_t h = 0;
        for (T const &t: x) {
            h = h * 18412483 + hasher(t);
        }
        return h;
    }
};

unordered_map<array<string, 3>, int> stutab;
```

采用最高级的，基于位运算的，最高效的，boost::hash_combine 的实现：

```cpp
template <class T, size_t N>
struct std::hash<std::array<T, N>> {
    size_t operator()(std::array<T, N> const &x) const {
        std::hash<T> hasher;
        size_t h = 0;
        for (T const &t: x) {
            h ^= hasher(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

unordered_map<array<string, 3>, int> stutab;
```