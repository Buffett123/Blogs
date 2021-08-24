# RMQ Range maxium/minum Query

## ST

时间复杂度： O(nlogn)
空间复杂度： O(nlogn)
```cpp
maxlen = lg2[maxn];
int lg2[maxn], st[maxn][maxlen];
void getlg2();
void getst(int n){
    for(int i = 1; i <= n; ++i)  st[i][0] = a[i];
    for(int j = 1; j <= len; ++j){
        for(int i = 1; i <= n; ++i){
            st[i][j] = st[i][j - 1];
            if(i + (1 << j - 1) <= n)   st[i][j] = max(st[i][j], st[i + (1 << j - 1)][j - 1]);
        }
    }
}
int query(const int L, const int R){
    int x = lg2[R - L + 1];
    return max(st[L][x], st[R - (1 << x) + 1][x]);
}
```

# LCA Lowest Common Ancestor

```cpp
void dfs(int u, int fa){
    dep[u] = dep[fa] + 1;
    an[u][0] = fa;
    for(int i = 1; i <= lg2[dep[u]]; ++i){
        an[u][i] = an[an[u][i - 1]][i - 1];
    }
    for(int e = fst[u]; e; e = ed[e].nxt){
        int v = ed[e].to;
        if(v == fa) continue;
        dfs(v, u);
    }
}
int LCA(int x, int y){
    if(dep[x] < dep[y]) swap(x, y);
    for(int i = lg2[dep[x]]; i >= 0; --i){
        if(dep[an[x][i]] >= dep[y]) x = an[x][i];
    }
    if(x == y)  return x;
    for(int i = lg2[dep[x]]; i >= 0; --i){
        if(){

        }
    }
    return 
}
```

## 给予ST表的LCA

在欧拉序对应的深度区间中查询区间最小值即为LCA的深度。

O(1)询问。

```cpp
int euler[maxn], tot, dep[maxn], lg2[maxn], star[maxn], end[maxn];
pi st[maxn][18];
void dfs(int u, int fa){
    euler[++tot] = u;
    star[u] = tot;
    dep[u] = dep[fa] + 1;
    for(int e = fst[u]; e; e = ed[e].nxt){
        int v = ed[e].to;
        if(v != fa) dfs(v, u);
    }
    erler[++tot] = u;
    end[u] = tot;
    return;
}
void StEuler(){
    for(int i = 1; i <= tot; ++i){
        st[i][0] = mp(dep[euler[i]], erler[i]);
    }
    for(int j = 1; j <= lg2[n]; ++j){
        for(int i = 1; i <= n; ++i){
            st[i][j] = st[i][j - 1];
            if(i + (1 << j) <= n){
                if(st[i + (1 << j)][j - 1].fi < st[i][j].fi)    s[i][j] = s[i + (1 << j)][j - 1];
            }
        }
    }
    return;
}
int query(const int L, const int R){
    int x = lg2[R - L + 1];
    if(st[L][x].fi < st[L + (1 << x)][x].fi)    return st[L][x].se;
    else return st[L + (1 << x)][x].se;
}
void LCA(int x, int y){
    return query(min(star[x], star[y]), max(end[x], end[y]));
}
```

# Euler Tour

长度为 2 * n - 1

# HDU3183 A Magic Lamp

证明一次删除m个数等价于m次删除1个数。即证明m=2的中间步骤一定是m=1。即证明m=1删掉的那个数一定是m=2时也要删掉的。

19284756

m=2时，124756
m=1时，1284756  m=1时，124756

注意到每次删除的都是第一个单间区间的峰值。

证明：由于单间序列的峰值一定要删掉，m=2的删除和两次m=1的删除过程完全一样，得证。

证明删数的操作不能用区间最值解决。

想法三：进行n - m次选数，从高位到低位选，每次在待选区间内询问最小值，需要满足待选区间的长度不小于未确定的位数。时间复杂度的瓶颈是预处理区间最值。

```cpp
const int maxn = 1000010;
int n, m, top, b[1010], ans[1010], pos, lg2[1010], st[21][1010];
string a;
int query(const int L, const int R){
	int x = lg2[R - L + 1];
	return min(st[x][L], st[x][R - (1 << (x)) + 1]);
}
int main(){
	lg2[0] = -1;
	for(int i = 1; i <= 1000; ++i)	lg2[i] = lg2[i / 2] + 1;
	while(cin >> a){
		pos = 1; top = 0;
		for(int i = a.size() + 1; i >= 1; --i)	a[i] = a[i - 1];
		m = rd();
		n = a.size();
		for(int i = 1; i <= n; ++i)	b[i] = a[i] - '0';
		for(int i = 1; i <= n; ++i)	st[0][i] = b[i];
		for(int j = 1; j <= lg2[n]; ++j){
			for(int i = 1; i + (1 << j) - 1 <= n; ++i)
				st[j][i] = min(st[j - 1][i], st[j - 1][i + (1 << (j - 1))]);	
		}
		m = n - m;
		int l = 1, r = n - m + 1;
		while(m--){
			ans[++top] = query(l, r);
			pos = l;
			while(b[pos] != ans[top]) ++pos;
			l = pos + 1;
			r = n - m + 1; 
		}
		int i = 1;
		while(ans[i] == 0 && i <= top)	i++;
		if(i > top)	printf("0");
		for(i; i <= top; ++i)	printf("%d", ans[i]);
		printf("\n");
	}
	return 0;
}
```

# POJ3419 Difference Is Beautiful

美丽区间的前驱点和后继点的坐标的差值最大，前驱点和后继点一定是美丽区间内出现的数字。

左端点从小到大改变的时候右端点一定是单调不减的，双指针从头扫，开一个桶记录当前区间内每个数字出现的次数判断可行性。储存每一个左端点对应的右端点。

对于一个询问[L, R]，答案具有二分性，二分区间长度x，判断[R - x + 1, R]区间内是否存在a[i]使得i - a[i] >= L，对于i - a[i]查询区间最大值即可。或判断[L, L + x - 1]区间内是否存在i使得a[i] >= x， 只需查询这个区间内的最大值判断即可。

```cpp

const int maxn = 200010;
int n, m, L, R, a[maxn], len[maxn], lf, rt, bkt[2000010];
int st[2000010][23], lg2[200010];
void Getlg2(){
	lg2[0] = -1;
	for(int i = 1; i <= n; ++i)	lg2[i] = lg2[i / 2] + 1;
	return;
}
void Getst(){
	for(int i = 1; i <= n; ++i){
			st[i][0] = len[i];
	}
	for(int j = 1; j <= lg2[n]; ++j){
		for(int i = 1; i + (1 << j) - 1 <= n; ++i){
			st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]); 
		}
	}
	return;
}
int query(int l, int r){
	int x = lg2[r - l + 1];
	return max(st[l][x], st[r - (1 << x) + 1][x]);
}
bool check(int mid){
	if(query(L, R - mid + 1) >= mid)	return 1;
	else return 0;
}
int main(){
	n = rd(); m = rd();
	for(int i = 1; i <= n; ++i)	{
		a[i] = rd();
		a[i] += 1000000;
	}
	rt = 0;
	for(lf = 1; lf <= n; bkt[a[lf++]]--){
		while(rt + 1 <= n && bkt[a[rt + 1]] <= 0)	bkt[a[++rt]]++;
		len[lf] = rt - lf + 1;
	}
	Getlg2();
	Getst();
	while(m--){
		L = rd(); R = rd();
		L++; R++;
		int l = 1, r = R - L + 1;
		while(l < r){
			int mid = (l + r + 1) >> 1;
			if(check(mid))	l = mid;
			else r = mid - 1;
		}
		printf("%d\n", l);
	}
	return 0;
}


```

# CF1301E Nanosoft

题意：给定一个矩形，每个元素是{'R','G','Y','B"}中的一个。给定q次询问，每次询问给定一个子矩形，问这个子矩形内部最大的合法正方形面积。合法正方形是指边长为偶数的，由完全相同的四个子正方形以组成的，四个子正方形中上左为红，上右为绿，左下为黄，右下为蓝的正方形。

视每个合法矩形的最后一个红点为重要的点，考虑这个点能够扩散形成的最大面积的合法矩形。只需计算以这个红点为基准向左上能够扩散得到的最大的红色纯色方形的面积，绿黄蓝同理，四个面积取最小即可。

如何判断一个矩形内是否都是同一种颜色？二维前缀和即可。判断二维前缀和是否等于矩形面积乘上单位颜色的值。


O(n ^ 2)搞出所有的纯色矩形，O(n ^ 2)搞出所有的合法四色矩形。

## 二维ST表

考虑预处理 f[x][y][a][b] 表示横坐标在 [x, min(n, x+2^a-1)]，纵坐标在 [y, min(m, y+2^b-1)] 的矩形内的最大值。转移即：

f[x][y][a][b] = min(f[x][y][a-1][b], f[x+2^(a-1)][y][a-1][b]),x+2^(a-1) <= n 且 a > 0

= f[x][y][a-1][b],x+2^(a-1) > n 且 a > 0

= min(f[x][y][a][b-1],f[x][y+2^(b-1)][a][b-1]),y+2^(b-1) <= m 且 b > 0

= f[x][y][a][b-1],y+2^(b-1) > m 且 b > 0

= val[x][y] , a = 0 且 b = 0


处理询问：

对于横坐标 [xl,xr]，纵坐标 [yl,yr] 的询问，设

$$kx = \log_2^{xr - xl + 1}, ky = \log_2^{yr - yl + 1}$$

则该矩形内的最大值为：

max(f[xl][yl][kx][ky], f[xr-2^kx+1][yl][kx][ky], f[xl][yr-2^ky+1][kx][ky], f[xr-2^kx+1][yr-2^ky+1][kx][ky])

```cpp
int st[25][25][maxn][maxm], squ[maxn][maxm];
void Getst(){
    for(int i=1;i<=n;++i){
        for(int j=1;j<=m;++j)
            for(int k=1;k<=4;++k)   st[0][0][i][j] = squ[i][j];
    }
    for(int p = 1; p <= lg2[n]; ++p){
        for(int q = 1; q <= lg2[m]; ++q){
            for(int i = 1; i + (1 << p) - 1 <= n; ++i){
                for(int j = 1; j + (1 << q) - 1 <= m; ++j){
                    
                    st[p][q][i][j] = min(min(st[p-1][q][i][j], st[p-1][q][i+(1<<(p-1))][j]),min(st[p][q-1][i][j], st[p][q-1][i][j+(1<<(q-1))]));
                }
            }
        }
    }
}
int query(int x1, int y1, int x2, int y2){
    int kx = lg2[x2 - x1 + 1], ky = lg2[y2 - y1 + 1];
    return min(min(st[kx][ky][x1][y1], st[kx][ky][x2-(1<<kx)][y1]),st[kx][ky][x1][y2-(1<<ky)]);
}
```

# CF1527D MEX Tree

题意:简单路径的条数

树上简单路径等价于一条链等价于端点-LCA-端点。钦定0为根节点，则对于mex>0的每条路径上的点的lca都为0。

涉及容斥。设f(x)为仅包含0~x节点的简单路径的条数，则满足mex=x的路径条数即为f(x-1)-x。前者满足0~x-1都出现过，后者限制了x不能出现。答案一定是单调不增的。

注意钦定0为根。

单独考虑mex=0的情况。即为 
$$\sum_{fa[i] = 0} (s[i] * (s[i] - 1) / 2)$$

考虑维护一条链满足端点是0~x且lca为0，链内部可以有大于x的点无妨。f(x)即为两端点的子树大小的乘积。注意如果两个端点是祖孙关系（这种情况一旦出现0必为端点之一），祖先的"子树"大小要减去孙子的子树大小。

维护链即可。考虑f(x-1)对应的链如何扩充为f(x)对应的链，即插入点x。分三种情况讨论。

- x在当前链上。链不变。
- x在端点的子树内，则新增对应端点到x的简单路径。
- x在其他位置（当前链上除了端点以外的其他点不包含根节点的子树内），则链无了。






# Uva1707

给定一个长度为 n 的环，有 k 个区域被覆盖，求最小的满足环被完全覆盖的区域数量。

对于链的情况，是很简单的贪心。发现链接区间和扫描线很像。

破环为链，长度乘二，暴力做n次总复杂度是O(n ^ 2)的。

考虑倍增。