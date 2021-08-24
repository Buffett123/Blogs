Day 4 线段树大礼包

高级操作：

-区间立方和

标记优先级：
区间覆盖>区间乘>区间加
区间覆盖的时候，乘标记和加标记都清空。
区间乘的时候，乘标记和加标记都乘。
区间加的时候，只修改加标记。

# 2019 Shanghai ICPC Onsite F

支持树上操作：
链加，链乘，链覆盖，链上立方和

$(ai+x)^2 = ai^2+2*ai*x+x^2$,只需维护区间平方和、区间和。
$(ai+x)^3 = ai^3+3*ai^2*x+3*ai*x^2+x^3$，只需维护区间立方和、平方和、区间和。
$(ai*x)^3 = ai^3*x^3$。

```cpp
void maketagadd(int x){
    sumfang3+=3*sumfang*sumfang*x+3*sumfang*x*x+x*x*x;
    sumfang+=2*sum*x+(r-l+1)*x*x;
    sum+=(r-l+1)*x;
    tagadd=x;
}
void maketagx(int x){
    sumfang3*=x*x*x;
    sumfang*=x*x;
    sumfang*=x;
    tagx*=x;
    tagadd*=x;
}
```
# 2019 Multi-University HDU Day 6 
框一个矩形使得矩形内点的权值和最大。
二维平面$|x_i|<=1e9, |y_i|<=1e9,n<=2000$

n^2确定上下边界，问题转换为求最大子段和。
下边界下移的时候修改（add）每一列值同时维护最大子段和。上边界下移的时候修改（减）每一列值同时维护最大子段和。由于一共n^2个点，所以复杂度为$O(n^2logn)$。

最大子段和等价于Day2的$max (a[y]-a[x]) | l <= x < y <= r$。

# 2019 Multi-University Nowcoder Day 1 I
一部分点赋值为a_i，一部分点赋值为b_i,求最大的价值和。
不存在B_i在A_i的左上方，也即是说在每一个A点划分出来的四个半平面，左上方没有B点，也即是说用一条折线划分平面，A点都在左上角。
n<=1e5。

用线段树维护DP。钦定折线经过一个点(x_i,y_i)，那么它可由其左下方的所有点(x_j,y_j)转移而来。新增的价值即为在[x_j,x_i]这个区间内的$\sum b_k *[y_k < y_i] + \sum a_k*[y_k \ge y_i]$。这个用二维树状数组维护。


# BZOJ1503 郁闷的出纳员
用权值线段树实现平衡树。离散化可能出现的所有工资，开一个权值线段树。可以使用线段树里的tag标记，也可以开一个全局标记。维护区间内数的个数用于查询第k大。

# HDU6606 Distribution of books
前缀分成k组，这个题并不要求把n个数都分完，只要选择一个前缀做。二分最大值mid，check分的组数是否不多于k。dp[i]表示把前i个数按区间和不大于mid最少能分成几组。dp[i]=max(dp[j]+1)其中sum[i]-sum[j]<=mid。用线段树log求max(sum[i]-mid, maxnum)。

# HDU6609 Find the answer
离散化动态开点。
???
# 扫描线
标记可持久化！！然而目前我还是没过板子题。
# 2019 Multi-University Nowcoder Day 8 I
有一个树区间，一开始每个元素都只有根节点1，接下来每次操作对于一个连续的区间[l,r]给这些树上的节点u（保证有）产生一个v儿子。
考虑建一棵树来表示所有的数。树的结构按照输入的u->v构建，树上每一个节点都保存了一个区间信息[l,r]即表示输入的树序列的区间。这棵树上的节点可能有重复的，但是即便节点编号一样，保存的区间不一样，本质上还是不一样的。接下来求出这棵树的dfs序，这样以x为根的子树就对应dfs序上的一段。把dfs序放在二维平面上做扫描线问题即可。

# 2019 Multi-University,HDU Day 9 B
给出一些从边界出发的直线，求平面被分割成了几块。
(1≤n,m≤10^9,1≤K≤10^5),
矩阵大小 n m ，有 k 条线。
可以发现"十"字形的个数加1即为答案。

# 合并线段树

```cpp
int merge(int L, int R, int A, int B){
    if(!A) return B;
    if(!B) return A;
    if(L == R){
        t[A].v=a[L];
        t[A].sum+=t[B].sum;
        return A;
    }
    int mid=L+R>>1;
    t[A].ls=merge(L,mid,A.ls,B.ls);
    t[A].rs=merge(mid+1,R,A.rs,B.rs);
    pushup(A);
    return A;
}
```
复杂度是均摊O(logn)的
# Codeforces 600 E

每一个节点建一颗关于颜色的权值线段树。自底向上更新。

用a[i]表示颜色i出现的节点个数，b[i]表示颜色i的编号之和。树上启发式合并。

# POI2011 Tree Rotations
给定一颗有n个叶节点的二叉树。每个叶节点都有一个权值p; (注意，根不是叶节点)，所有叶节点的权值构成了一个1 ~ n的排列。对于这棵二叉树的任何一个结点，保证其要么是叶节点，要么左右两个孩子都存在。现在你可以任选一些节点，交换这些节点的左右子树。在最终的树上，按照先序遍历遍历整棵树并依次写下遇到的叶结点的权值构成一个长度为n的排列，你需要最小化这个排列的逆序对数。

掉线了/kk
