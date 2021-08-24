# T1

棋盘 
【题目描述】
小可爱的生日得到了一块棋盘。棋盘有 N 行 M 列，每个位置有一个英文字母的
小写字母。在他的生日聚会上，每个人都感到无聊，所以他们决定玩一个简单
的棋盘游戏。
游戏开始时，在左上角坐标（1，1）的位置放置一个芯片。在每一回合中，我
们都必须将芯片向右或向下移动一个位置。游戏结束时，将芯片移动到标有坐
标（N，M）的棋盘右下角区域。
在游戏中，我们注意到我们通过移动芯片形成的字符串构造一个单词。游戏的
目标是找到所有情况下，移动构成的字典顺序排列的最小单词。
【输入格式】
输入的第一行包含两个正整数 N，N（1≤N，M≤2000），表示棋盘大小。
接下来的 N 行每行一个长度为 M 的字符串，表示棋盘内容。
【输出格式】
输出可能的字典序最小的单词。
【输入输出样例 1】
in:
```
4 5 
ponoc
ohoho
hlepo
mirko
```
out:
```
pohlepko
```
【输入输出样例 2】
in：
```
4 5 
bbbbb
bbbbb
bbabb
bbbbb
```
out:
```
bbbbabbb
```
【样例解释】
对于第一个样例：
【数据范围】
对于其中 40%的数据，保证每一个位置下边和右边的位置中字母都不同。

试图DP，发现没法存字符串。
如果遇到下面和右边是相同的字符的时候，无法确定选哪一个，所以两个都要保留。考虑bfs。用f[x][y]表示这个位置的前驱结点，ans[dep]表示一共走了dep步（向下和向右一共）所能得到的所有串中第dep位最下能是什么字母。每次只有第dep位是ans[dep]的状态才能向下转移。转移的规则是没有搜索过并且字符小于等于另一个方向。

```cpp
#include<map>
#include<set>
#include<cmath>
#include<ctime>
#include<queue>
#include<stack>
#include<vector>
#include<cstdio>
#include<string>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
#include<functional>
#define mod 998244353
#define inf 0x3fffffff
#define pi pair<int,int>
#define mp(a,b) make_pair(a, b)
#define fi first
#define se second
using namespace std;
typedef long long ll;
int rd(){
	int res = 0, fl = 1;
   char c = getchar();
   while(!isdigit(c)){
       if(c == '-') fl = -1;
       c = getchar();
   }
   while(isdigit(c)){
      res = (res << 3) + (res << 1) + c - '0';
      c = getchar();
   }
   return res * fl;
}
namespace force{
int main(){
	return 0;
}
}
const int maxn=2020;
struct data{
	int x, y, dep;
	
}f[maxn][maxn];
queue<data> q;
int n, m;
char bd[maxn][maxn];
int ans[maxn*2], vst[maxn][maxn];
int tit=0;
void print(data u){
	if(u.x==1&&u.y==1){
		printf("%c",bd[1][1]); return;
	}
	print(f[u.x][u.y]);
	printf("%c",bd[u.x][u.y]); return;
}
int main(){
	n=rd();m=rd();
	for(int i=1;i<=n;++i)
		for(int j=1;j<=m;++j)
			cin>>bd[i][j]; 
	for(int i=0;i<=n+m;++i)	ans[i]=200; 
	q.push((data){1,1,2});ans[2]=bd[1][1]-'a';
	while(q.size()){
		data u=q.front();q.pop();
		int s=u.dep;
		if(bd[u.x][u.y]>ans[s]+'a') continue;
		if(!vst[u.x+1][u.y]&&u.x+1<=n&&((u.y+1<=m&&bd[u.x+1][u.y]<=bd[u.x][u.y+1])||u.y==m)){
			vst[u.x+1][u.y]=1;
			ans[s+1]=min(ans[s+1], bd[u.x+1][u.y]-'a');
			q.push((data){u.x+1,u.y,s+1});
			f[u.x+1][u.y]=u;
		}
		if(!vst[u.x][u.y+1]&&u.y+1<=m&&((u.x+1<=n&&bd[u.x][u.y+1]<=bd[u.x+1][u.y])||u.x==n)){
			vst[u.x][u.y+1]=1;
			ans[s+1]=min(ans[s+1], bd[u.x][u.y+1]-'a');
			q.push((data){u.x,u.y+1,s+1});
			f[u.x][u.y+1]=u;
		}
	}
	if(!(n==1&&m==1))print(f[n][m]);
	printf("%c\n",bd[n][m]);
	return 0;
}
```



# T2

网络流，流量加和变为取max,所有的流量只有0/1。
分层建图，建n层绝对够用。//1~(n-1)*n //400
超级起点向所有第一层的点连边，流量限制为1，费用为0。S=522
相邻层之间按照给定的邻接矩阵连边，流量限制为0，费用为矩阵值。
最后一层全部链接到超级终点，流量限制为inf,费用为0。//E=667
超级终点全部连接到超级超级终点，流量限制为k,费用为0。//FE=777

但其实假了。

状压。设dp[s]表示有水的状态为s所需的最小花费。最后的答案就是$\min_{numof1(s) = k}{dp[s]}$。
转移的时候枚举倒水的过程(i->j)，其中s的i这一位是空的表示i的水已经被倒到了j，j这一位是1。转移方程为$dp[s]=\min dp[s0] + G[i][j]$。
复杂度即为$$O(2 ^ n * n ^ 2)$$

```cpp
#include<map>
#include<set>
#include<cmath>
#include<ctime>
#include<queue>
#include<stack>
#include<vector>
#include<cstdio>
#include<string>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
#include<functional>
#define mod 998244353
#define inf 0x3fffffff
#define pi pair<int,int>
#define mp(a,b) make_pair(a, b)
#define fi first
#define se second
using namespace std;
typedef long long ll;
int rd(){
	int res = 0, fl = 1;
   char c = getchar();
   while(!isdigit(c)){
       if(c == '-') fl = -1;
       c = getchar();
   }
   while(isdigit(c)){
      res = (res << 3) + (res << 1) + c - '0';
      c = getchar();
   }
   return res * fl;
}
namespace force{
int main(){
	return 0;
}
}
const int maxn = 22;
int n, G[maxn][maxn], k, num1[1<<maxn], dp[1<<maxn], ans;
int main(){
	n = rd(); k = rd();
	for(int i = 1; i <= n; ++i){
		for(int j = 1; j <= n; ++j){
			G[i][j] = rd();
		}
	}
	ans=inf;
	memset(dp,0x3f,sizeof(dp));
	for(int s = 0; s <= (1<<n)-1; ++s)	num1[s]=num1[s>>1]+(s&1);
	dp[(1<<n)-1]=0;
	for(int s = (1 << n) - 1; s > 0; s--){
		for(int i = 1; i <= n; ++i){
			if(s & (1 << (i - 1))) continue;
			for(int j = 1; j <= n; ++j){
				if(!(s & (1<< (j - 1))))	continue;
//				cout<<(s|(1<<(i-1)))<<"->"<<s<<endl;
//				cout<<dp[s|(1<<(i-1))]<<"->";
				dp[s] = min(dp[s], dp[s | (1 << (i - 1))] + G[i][j]);
//				cout<<dp[s]<<endl;
			}
		}
		if(num1[s] == k)	ans = min(ans, dp[s]);
	}
	printf("%d\n", ans);
	return 0;
}
```


# 愤怒的小鸟

# 币

# T3

钦定一个元素x一定在序列中。按顺序考虑x后面的元素，一个简单的想法是比x小的放开头比x大的放结尾，这样就只需要记录x~n的最长上升序列放到x后面，x~n的最长下降子序列放到x前面，这两个序列的交一定只有x，那么二者拼起来得到的单增序列的长度即为二者长度和减去1。至于方案数，这个序列之外的数都是有前后两种选择的。

方案数即为$$downlen[i]*uplen[i]*2^{n-downlen[i]-uplen[i]+1}$$。离散化后权值线段树或者树状数组LIS总复杂度$O(nlogn)$。

当前区间是[i,n]，枚举i递减。维护两棵权值线段树分别维护uplen[]和down[]，uplen[i]表示在区间[i,n]里最长的上升子序列，down[i]表示在区间[i,n]里最长的下降子序列。下面只考虑上升子序列。对于a[i],询问线段树里[a[i]+1,maxa]的最大值maxlen，即最长长度，修改线段树里a[i]的位置为uplen[i]=maxlen+1。用f[len]表示长度为len的上升子序列的个数，那么f[uplen[i]]++。

贴一波xie老板的阴间std:
```cpp
#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
using namespace std;

#define TRACE(x) cout << #x << " = " << x << endl;

typedef long long int llint;

typedef pair<int, int> par;

#define X first
#define Y second

const int MAXN = 500010, MOD = 1000000007;

inline int add(int a, int b)
{
  int ret = a + b;
  if(ret >= MOD) ret -= MOD;
  return ret;
}

inline int mul(int a, int b)
{
  llint ret = (llint)a * b;
  if(ret >= MOD) ret %= MOD;
  return ret;
}

int n;
int niz[MAXN], dva[MAXN];

par A[MAXN], B[MAXN];
par FWT_gore[MAXN], FWT_dolje[MAXN];

par rj;

par spoji(par a, par b)
{
 if(b.X > a.X)
 {
   a.X = b.X;
   a.Y = b.Y;
 } 
 else if(b.X == a.X)
   a.Y = add(a.Y, b.Y);
 return a;
}

void ubaci_gore(int x, par v)
{
  x += 5;
  for(; x < MAXN; x += x & -x)
    FWT_gore[x] = spoji(FWT_gore[x], v);
}

par upit_gore(int x)
{
  x += 5;
  par ret(0, 0);
  for(; x > 0; x -= x & -x)
    ret = spoji(ret, FWT_gore[x]);
  return ret;
}

void ubaci_dolje(int x, par v)
{
  x += 5;
  for(; x > 0; x -= x & -x)
    FWT_dolje[x] = spoji(FWT_dolje[x], v);
}

par upit_dolje(int x)
{
  x += 5;
  par ret(0, 0);
  for(; x < MAXN; x += x & -x)
    ret = spoji(ret, FWT_dolje[x]);
  return ret;
}

void sazmi()
{
  vector<int> v;
  for(int i = 0; i < n; i++)
    v.push_back(niz[i]);
  sort(v.begin(), v.end());
  v.resize(unique(v.begin(), v.end()) - v.begin());
  for(int i = 0; i < n; i++)
    niz[i] = lower_bound(v.begin(), v.end(), niz[i]) - v.begin();
}

void sredi_gore()
{
  for(int i = n - 1; i >= 0; i--)
  {
    par p = upit_gore(niz[i] - 1);
    if(p.X == 0)
    {
      A[i] = par(0, 1);
      ubaci_gore(niz[i], par(1, 1));
    }
    else 
    {
      A[i] = p;
      p.X++;
      ubaci_gore(niz[i], p);
    }
  }
}

void sredi_dolje()
{
  for(int i = n - 1; i >= 0; i--)
  {
    par p = upit_dolje(niz[i] + 1);
    if(p.X == 0)
    {
      B[i] = par(0, 1);
      ubaci_dolje(niz[i], par(1, 1));
    }
    else
    {
      B[i] = p;
      p.X++;
      ubaci_dolje(niz[i], p);
    }
  }
}

void postavi()
{
  dva[0] = 1;
  for(int i = 1; i < MAXN; i++)
    dva[i] = mul(dva[i - 1], 2);
}

void glavno()
{
  for(int i = 0; i < n; i++)
    rj = spoji(rj, par(A[i].X + 1 + B[i].X, mul(A[i].Y, B[i].Y)));
  rj.Y = mul(rj.Y, dva[n - rj.X]);
}

int main()
{
  postavi();
  scanf("%d", &n);
  for(int i = 0; i < n; i++)
    scanf("%d", &niz[i]);
  sazmi();
  sredi_gore();
  sredi_dolje();
  glavno();
  printf("%d %d\n", rj.X, rj.Y);
  return 0;
}

```

# T4

打蚊子
【题目描述】
夏天房间里的蚊子总是让人很烦。小可爱决定打蚊子。
小可爱的蚊子拍是一个 K 边形，现在有 N 个蚊子停在窗户上。窗口是一个矩形，其左下角的顶点位于坐标系的中心（0，0）。在小可爱拿拍子击中窗户 后，多边形（蚊子拍）的顶点必须位于整数坐标上，而且蚊子拍必须位于窗户内，不能有任意一部分超出窗户边界。
蚊子位于顶点、边缘或蚊子拍内部，都认为它受伤了。
现在小可爱想知道有多少不同的可能性，她会一个蚊子都无法击中。
【输入格式】
输入的第一行包含三个整数 Xp,Yp,N（1≤Xp,Yp≤500,0≤N≤Xp*Yp）,分表表示窗户的上边和右边的边长，以及蚊子的数量。
接下来的 N 行，每行两个整数 X，Y（0<X<Xp,0<Y<Yp），表示蚊子的位置。接下来一行一个整数 K（3≤K≤10000），表示蚊子拍的边数量。
接下来的 K 行，每行两个整数 Xi，Yi（-10^9≤Xi,Yi≤10^9）,表示蚊子拍的第 i 个顶点。 蚊子拍顶点是按照连接线的顺序提供的，因此相邻的顶点通过直线连接。
【输出格式】
输出击打窗户而不伤害一只蚊子的方式数量。
【样例输入输出 1】
6	7	2	1
2	5		
4	5		
8			
1	4		
3	3		
4	1		
5	3		
7	4		
5	5		
4	7		
3	5		
【样例输入输出 2】
4	5	2	4
1	3		
3	4		
4			
0	0		
2	0		
2	2		
0	2		

如何判断一个点与多边形的位置关系？
多边形内一点发出任意一条射线一定与多边形有奇数个交点，多边形外一点发出任意一条射线一定与多边形有偶数个交点。复杂度O(边数)。判断一个点是否在多边形边界上也是O(边数)。

为了好操作要平移多边形使得左下角在(0,0)

### 二维图形的是移动转化为一位01串的移动

状压表示，01串的第(2*n*i+j)位为1表示坐标为(i,j)的点在多边形内部（含），为0表示在多边形外部。

用01串表示拍子和所有蚊子的集合，或起来为0及说明没有蚊子在拍子上。用bitset实现。

然后？？？

看了眼标程要写FFT。



