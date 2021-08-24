# 线段树

### 一些引理

性质 1：任意两个结点的区间或者为包含关系，或者不交。

区间拆分：区间 [l,r][l,r] 可以表示为线段树上若干个区间的不交并，称这些区间构成 [l,r][l,r] 的区间拆分。

性质 2：任何一个包含于 [1,n][1,n] 的区间的区间拆分存在，且大小最小的区间拆分唯一。

性质 3：最小区间拆分的大小不超过 2\log n+c，其中 c 是与 n 无关的常数。

引理 1：若 l\leq xl 或 r\ge xr，则 |Solve(l,r,xl,xr)|\leq \log (xr-xl+1)+c。

引理 2：递归过程中至多存在一个点 [xl,xr] 使得 $$ xl l \leq mid < r < xr $$。

### 基本功能

区间赋值，查询区间和、区间乘积、区间最值、最大字段和、最大不相邻子集和。

#### 最大子段和

```cpp
struct Node{
    int l, r, maxl, maxr, maxval;
    Node *ls, *rs;
    bool InRange(const int L, const int R){
        return L <= l && r <= R;
    }
    bool OutofRange(const int L, const int R){
        return l > R || L > r;
    }
    void pushup(){
        maxl = max(ls->maxl, ls->sum + rs->maxl);
        maxr = max(rs->maxr, rs->sum + ls->maxr);
        maxval = max(max(ls->maxval, rs->maxval)), ls->maxr + rs->maxl);
    }
    int query(int L, int R){
        if(InRange(L, R)){
            return maxval;
        }
        else if(OutofRange(L, R)){
            return -inf;
        }
        else{
            int M = (l + r) / 2;
            if(L <= M && M < R)
                return max(ls->query(L, M) , rs->query(M + 1, R), ls->rmax + rs.lmax);
            else if(R <= M) return ls->query(L, R);
            else if(L > M)  return rs->query(L, R);
        }
    }
}Mem[200010], *pool = Mem;
Node* New(){
    return pool++;
}
Node* Build(const int L, const int R){
    Node *u = New();
    if(L == R){
        u->maxl = a[L];
        u->maxr = a[L];
        u->maxval = a[L];
    }
    else{
        int M = (L + R) / 2;
        u->ls = build(L, M);
        u->rs = build(M + 1, R);
        u->pushup();
    }
    return u;
}

```

# 树状数组

[树状数组 1：单点修改，区间查询](https://loj.ac/p/130)
[树状数组 2：区间修改，单点查询](https://loj.ac/p/131)
[树状数组 3：区间修改，区间查询](https://loj.ac/p/132)
[二维树状数组 1：单点修改，区间查询](https://loj.ac/p/133)
[二维树状数组 3：区间修改，区间查询](https://loj.ac/p/135)

区间求和，单点修改。

```cpp
void add(int p, int val){
    while(p <= n){
        bit[p] += val;
        p += p & (-p);
    }
}

int sum(int p){查询前缀和
    int res = 0;
    while(p >= 1){
        res += bit[p];
        p -= p & (-p);
    }
    return res;
}
```

树状数组bit[x]维护的是区间(prex, x]的和，其中prex=x - lowbit(x)。

## 区间修改区间查询

原数组$a[]$，设$\sum{i=1}^x b[i] = a[x]$，$b[]$是那个差分数组，那么：
$\sum_{i=1}^x a[i]=\sum_{i=1}^x \sum_{j=1}^{i} b[i] = \sum_{j=1}^x (x-j+1) * b[j]$
那么我们维护$c[j]=b[j]$和$d[j]=j*b[j]$的树状数组。
对于区间查询a[l...r]，只需树状数组基操区间查询$\sum_{j=1}^l c[j]-d[j]$和区间查询$\sum_{j=1}^r c[j]-d[j]$。
对于区间修改a[l...r]，只需树状数组基操单点修改b[l]和b[r+1]，同时单点修改c[l]，c[r+1]和d[l]，d[r+1]。

区间加a[l...r]+x=>单点修改b[l]+x,b[r+1]-x=>单点修改c[l]+=x,d[l]+=l*x,c[r+1]-=x,d[r+1]-=x *(r+1)。
区间查询sum a[l...r]=>区间查询a[1...l-1]和区间查询a[1...r]=>区间查询c[1...l-1]*(l)-d[1...l-1]和区间查询c[1...r] *(r+1)-d[1...r]。

## 二维树状数组

```cpp
int bit[maxn][maxn];
void Add(int x1, int y1, int x2, int y2, int x){
    for(int i = x1; i <= x2; i += i & (-i)){
        for(int j = y1; j <= y2; j += j & (-j)){
            bit[i][j] += x;
        }
    }
}
int query(int x1, int y1, int x2, int y2){
    int res = 0;
    for(int i = x2; i >= x1; i -= i & (-i)){
        for(int j = y2; j >= y1; j -= j & (-j)){
            res += bit[i][j];
        }
    }
    return res;
}

```

# P4868 Preprefix sum

做法一：

分别维护 {S} 和 {SS} 的差分序列 {DS} 和 {DSS}。其实 {DS} = {a}, {DSS} = {S}。

ai + x 即 DSi + x 并且 DSS_{i,i+1,...,n} + x，单点修改和区间修改线段树维护即可。

询问SSi等价于询问区间和DSS_{1...i}。复杂度O(log)。

做法二：

SS_i = \sum_{j = 1}^{j <= i} a[j] * (i - j + 1) = \sum a[j] * (i + 1) - \sum a[j] * j。

分别维护$$\sum a[j]$$ 和 $$\sum a[j] * j$$， 可以用树状数组维护这两个序列，支持单点修改和前缀和查询。


# Matrix

二维树状数组

# CF1527E Partition Game

n^2 预处理cost数组，之后就可以O(1)查询了。
朴素的做法是设dp[i][j]为划分前i个元素并且第i个元素划分作为第j段的结尾所能得到的最小值。
$$dp[i][j] = min_{j \leq k < i} (dp[k][j - 1] + cost(k + 1, j) )$$
这个东西的复杂度是$O(kn^2)$ 的。

# 习题5

维护一个长度为 n 的数组 a[]，支持：
1.单点修改。
2.给定 l,r，查询$$max (a[y]-a[x]) | l <= x < y <= r$$

n <= 1e5 

维护每个区间节点的最大值、最小值和最大顺差(只是个叫法，即题目所求)。

这个东西是可以合并的。

```cpp
maxval = max(ls->maxval, rs->maxval);
minval = min(ls->minval, rs->minval);
ans = max(max(ls->ans, rs->ans), rs->maxval - ls->minval);
```

合并询问区间所拆出来的log个区间即为答案。

# 习题5扩展到树上 Poj 3728 The merchant

序列上的区间移动到树上。考虑倍增，维护每个点到他的 $$2^k$$ 级祖先的信息，视作上一个题的区间即可。同样维护最值，由于树上的先后顺序是有输入的节点顺序定义的，所以每一个区间都有两个方向，维护最大顺逆差即可。

询问时只需求出lca,合并到lca的两个区间即可。

# HDU6315 Naive Operations

#### 答案阈值

最大的答案是qlogq级别的

#### 

称一个位置被激活当且仅当$$\floor(a[i]/b[i])$$的值改变。称一个元素的激活值为a[i]/b[i]改变的值。维护一个区间内激活值的最小值。对于激活值为0的区间进行暴力dfs计算。

最多qlogq次激活，每次激活要影响logn个点，故复杂度上限是 O(qlogqlogn)。

# CF5634 Rikka with Phi

引理： 一个数在取 phi log 次后就会变成1。

证明：
$$当n为奇数时， \phi (n) = x, 2 | x$$

$$当n为偶数时，\phi (n) <= \frac{n}{2}$$

由于取phi操作减小幅度之大，同时还有区间加操作，所以区间内元素有趋于相等的趋势。区间加操作可以暴力。如果一个区间内的数全部相等，可以直接打标记用一个数来代表这个区间。否则递归儿子。区间和就常规操作。

这就是这个题的做法，但是这个题的价值在于通过势能分析来计算复杂度。

# CF1149C Tree Generator™

考虑一个点经过一段括号序列后到达的点距离它的距离，即将这个括号序列内匹配的括号去掉后剩下的括号个数。
定义一个子段的权值为将这个子段内匹配的括号去掉之后剩下的括号个数，题目即求最大子段权值。

后面就不懂了。