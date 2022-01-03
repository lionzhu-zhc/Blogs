## python 记录
- `collections.defaultdict(list)` 字典每个key对应的value是list，而普通字典的一个key无法对应多个值

- `collections.defaultdict(lsit)` 可以使用 a[key].append(v)添加元素
   普通dict 使用 dict.update(key:v) 添加新的字典对

- `dict.get(key)` 获取值

- `del dict['name']`, `.pop(key)` 删除

## 栈
### 155. 最小栈
```python
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = [math.inf]  #用一个附加的栈保存最小值
    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))	
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()
    def top(self) -> int:
        return self.stack[-1]
    def getMin(self) -> int:
        return self.min_stack[-1]
```
### 20. 有效的括号
```python
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s)%2 == 1:
            return False
        pairs = {")":"(", "]":"[", "}":"{"}
        stack = list()
        for key in s:
            if key in pairs: #如果是左括号，入栈，如果是右括号，判断有没有对应的然后出栈
                if len(stack) == 0 or stack[-1] != pairs[key]:
                    return False
                stack.pop()
            else:
                stack.append(key)
        return not stack
```
### 94. 二叉树中序遍历
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        stack = []
        seq = []
        while (len(stack) >0 or root != None ):
            if (root != None):	# 不断寻找左儿子，入栈
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()	#左二子依序出栈，再找右儿子
                seq.append(root.val)
                root = root.right
        return seq
```
### 739. 每日温度
用一个栈来保存温度列表的下标，一个list 保存隔几天最高温度的结果
```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        lens = len(T)
        ans = [0]*lens
        stack = []
        for i in range(lens):
            temp = T[i]
            while stack and temp>T[stack[-1]]:   #新的元素温度比栈顶还高
                pre_indx = stack.pop();
                ans[pre_indx] = i-pre_indx;
            stack.append(i)
        return ans
```
### 394. 字符串解码
用一个栈来保存上一次出现的括号对 里面的字符串
```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, mul = [], "", 0
        for ch in s:
            if '0' <= ch <= '9':
                mul = mul*10 + int(ch)
            elif ch == '[':
                stack.append([mul, res])
                mul = 0
                res = ""
            elif ch == ']':		# 遇到右括号，提取上次结果，再加上这次
                multi, last_res = stack.pop()
                res =  last_res + res*multi
            else:
                res = res + ch
        return res
```
### 42. 接雨水
使用栈来保存height 列表的下标
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        stack = []
        for i in range(len(height)):
            while (len(stack) > 0 and height[i] > height[stack[-1]]): 
            #栈非空并且当前高度大于前一个高度
                top = stack[-1]
                stack.pop()
                if (len(stack) == 0):
                    break 
                    #连续升高，则把前面的高度pop，栈为空，退出循环后当前高度入栈
                distance = i - stack[-1] - 1
                bound_height = min(height[i], height[stack[-1]]) - height[top]
                ans = ans + distance * bound_height
            stack.append(i)

        return ans
```

## 堆
### 215. 数组中的第k大元素值
可以用堆排序直接排序后去-k。或者建立大根堆，每次删除堆尾
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def adjustHeap (nums, i, size):
            left_child  = 2*i+1
            right_child = 2*i+2
            largest = i
            if (left_child < size) and nums[left_child] > nums[largest]:
                largest = left_child
            if (right_child < size) and nums[right_child] > nums[largest]:
                largest = right_child
            if largest != i:
                nums[largest], nums[i] = nums[i], nums[largest]
                adjustHeap(nums, largest, size)

        def buildHeap(nums, size):
            for i in range(len(nums) // 2)[::-1]:
                adjustHeap(nums, i, size)
        
        size = len(nums)
        buildHeap(nums, size)
        for i in range(k-1):  # 寻找第k大
            nums[0], nums[-1] = nums[-1], nums[0]  #将最大的放到堆尾
            nums.pop()	#移出前一次最大的
            adjustHeap(nums, 0, len(nums))	# 重新调整为大根堆
        return nums[0]
```
## 贪婪算法
### 406. 根据身高重建队列
身高有大到小排列，该身高的人数从小到大排列。 对于第i个person，只对队列中已排的有影响，不影响后面未排的人，所以直接在person的第二个参数对于的坐标insert到list就行
```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        ans = []
        people.sort(key=lambda x: (-x[0], x[1]))

        for person in people:
            ans.insert(person[1], person)
        return ans
```
### 55. 跳跃游戏
贪心算法，对于之前可以到达的位置i，i+nums[i]对应的就是能到达的最远端，如果最远端覆盖了最后一位，则返回true
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_len = 0
        for i in range(len(nums)):
            if i <= max_len:  # 当前i位置可以到达
                max_len = max(i+nums[i], max_len)  #更新最远到达
                if max_len >= len(nums)-1:
                    return True
        return False
```
## 排序
### 148. 排序链表
用归并排序的方法对链表进行排序。用slow 和fast作为指针，slow每移动一次 fast移动两次，最后slow即为mid的位置，然后分成(head, mid) (mid, tail)两半排序
```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
    	def merge(left, right):
            oriHead = ListNode(0)
            temp, temp1, temp2 = oriHead, left, right
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return oriHead.next
        def sortF(head, tail):
            if not head:
                return head
            if head.next == tail:
                head.next = None  # 到末尾就直接返回
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortF(head,mid), sortF(mid, tail))
        return sortF(head, None)
```
### 31. 下一个排列
从后往前搜索，找到后一个比前一个大的位置j, 表明从j-1:end可以排成一个更大的。然后再从最后往前找出第一个比j-1大的位置k， 交换k和j-1上的数，然后为了交换后的序列尽量小，将j-1后面的序列从小到大排序。

> https://leetcode-cn.com/problems/next-permutation/solution/31xia-yi-ge-pai-lie-c-by-ctrlcccctrlvvvv-0hrz/

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        size = len(nums)
        if size == 1:
            return nums
        j = size-1
        while j >= 1:  #找到后一个比前一个大的数下标j
            if nums[j] > nums[j-1]: break
            j -= 1
        if j == 0:   # 当已经是最大排序了
            nums.sort()
            return nums
        k = size - 1 
        while k > j-1:   # 找到第一个比j-1还大的数，用于交换
            if nums[k] > nums[j-1]:
                break
            k -= 1
        nums[j-1], nums[k] = nums[k], nums[j-1]         
        nums[j:] =  sorted(nums[j:])   
```
### 56. 合并区间
将集合区间按开始位从小到大排序，然后比较后一个的开始与前一个的结尾的大小, 判断是否要合并
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x:x[0]）
        res = []
        for i in intervals:
            if len(res) == 0 or i[0] > res[-1][1]:  #不重叠，直接添加
                res.append(i)
            else:
                res[-1][1] = max(res[-1][1], i[1])
        return res
```
## 位运
### 461.汉明距离
按位异或，然后 按位与，然后右移一位
```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        xor = x ^ y
        dist = 0
        while xor:
            if xor & 1 == 1:
                dist += 1
            xor = xor >> 1
        return dist
```
### 136. 数组中唯一一个只出现一次的数字，其他出现两次
两个相同的数字在不同的异或位置运算还是0，最后剩下的就是唯一一个一次的数字
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        single = 0
        for n in nums:
            single = single ^n
        return single
```
### 338.比特位计数
使用动态规划，x<=y，x表示最接近y, 且是2的n次幂，则y比y-x在二进制上只多了最高一个1位，动态规划bit(y)=bit(y-x)+1. 当 x&(x-1) ==0 则表示x是2的n次幂
```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        bit = [0]
        high_bit = 0
        for i in range(1, n+1):
            if i & (i-1) == 0:
                high_bit = i
            bit.append(bit[i-high_bit]+1)
        return bit
```
## 树
### 617. 合并二叉树
深度优先（递归）， 用类方法不断创建新的节点
```python
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1:
            return root2
        if not root2:
            return root1
        merged = TreeNode(root1.val+ root2.val)
        merged.left = self.mergeTrees(root1.left, root2.left)
        merged.right = self.mergeTrees(root1.right, root2.right)
        return merged
```
广度优先法（队列）：
### 226. 翻转二叉树
用递归的方式，先翻转左子树，再翻转右子树
```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root ==None:
            return root
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```
### 543. 二叉树的直径
以某个点为根的路径上的最大节点数为左深度L+右深度R+1，一个节点的最大深度路上节点数为max(L, R) +1
```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 1
        def depth(node):
            if node == None:
                return 0
            L = depth(node.left)
            R = depth(node.right)
            self.ans = max(self.ans, L+R+1)
            return max(L,R)+1
        depth(root)
        return self.ans-1
```
### 114. 二叉树展开为链表
使用前序遍历将节点保存到List中，然后从list中读取连续的两个节点，设置前节点的右子节点
```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            return root
        stack = []
        cur = root
        preorderlist = []
        while cur or stack:  # 中序遍历
            if cur:
                preorderlist.append(cur)
                stack.append(cur)
                cur = cur.left
            else:
            	cur = stack.pop()
            	cur = cur.right        
        size = len(preorderlist)
        for i in range(1, size):
            pre_node, cur_node = preorderlist[i-1], preorderlist[i]
            pre_node.left = None
            pre_node.right = cur_node
```
## 深度优先
### 104.二叉树的最大深度
```python
class Solution:
	def maxDepth(self, root):
		if root is None:
			return 0
		left_depth = self.maxDepth(root.left)
		right_depth = self.maxDepth(root.right)
		dept = max(left_depth, right_depth) + 1
		return dept
```
### 22. 括号生成
深度优先搜索，当左括号<n时，str添加一个左括号，然后深度优先搜索；当右括号小于<n且左括号>右括号时，str加入一个右括号，然后再继续深度优先
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def dfs(n, lc, rc, s):
            if (lc == n) and rc == n:
                res.append(s)
            else:
                if lc < n:        
                    dfs(n, lc+1, rc, s+'(')
                if rc<n and rc < lc:
                    dfs(n, lc, rc+1, s+')')
        dfs(n, 0,0, '')
        return res
```

## 广度优先
###  101. 对称的二叉树
采用队列的方式判断二叉树是否对称，是广度优先。即，以一个队列保存同一层的左右儿子，比较左右儿子是否对称，然后再进行下一层操作
```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True
        queue = []
        queue.append(root.left)
        queue.append(root.right)
        while len(queue) > 0:
            left_node = queue.pop(0)
            right_node = queue.pop(0)
            if ((left_node is None) and (right_node is None)): # 左右都为空，跳出此次循环
                continue
            if ( (left_node is None) or (right_node is None )): #左右一个为空
                return False
            if (left_node.val != right_node.val): # 左右值不等
                return False
            queue.append(left_node.left)
            queue.append(right_node.right)
            queue.append(left_node.right)
            queue.append(right_node.left)
        return True
```
### 102. 二叉树的层序遍历
使用队列，直接中序遍历
```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:        
        if root== None:
            return []
        queue = [root]
        res = []
        while queue:
            result = []
            size = len(queue)
            for _ in range (size):  # 当前层size控制住
                cur = queue.pop(0)
                result.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(result)
        return res
```
## 并查集
### 并查集模板
并查集指的是有个根节点，有n条链指向根节点，一条链就是一串字符。同一跟节点他们都互相有关系
```python
class UnionFind:
    def __init__(self):
        self.father = {} #以一个字典记录key所对应的跟节点value
    def find_father(self, x):
        root = x
        while(self.father[root] != None): # 遍历查找根节点
            root = self.father[root]
        while x!=root:	# 路径压缩，即将当前节点直接指向根节点
            ori_father = self.father[x]
            self.father[x] = root
            x = ori_father
        return root
    def merge(self, x, y, val):
        root_x, root_y =  self.find_father(x), self.find_father(y)
        if root_x != root_y :
            self.father[root_x] = root_y
    def is_connected(self, x, y):
        return self.find_father(x) == self.find_father(y)
    def add_node(self, x):	#添加一个独立节点
        if x not in self.father:
            self.father[x] = None
```
### 399. 除法求值
用并查集指定两个之间的联系。 增加一个value字典保存两个点之间的weight
```python
class UnionFind:
    def __init__(self):
        self.father = {} # 记录每个点的根节点
        self.value = {}  # 记录每个点到根节点的权重
    def find(self,x):
        root = x
        base = 1	#节点在压缩路径时权重变化的基数
        while self.father[root] != None:
            root = self.father[root]
            base *= self.value[root] 	# 查找根节点同时改变它到根节点的变化基数
        while x != root:
            original_father = self.father[x]	
            self.value[x] *= base	#改变节点时需要同时权重
            base /= self.value[original_father]
            self.father[x] = root	#改变节点
            x = original_father   
        return root
    def merge(self,x,y,val):
        root_x,root_y = self.find(x),self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y
            self.value[root_x] = self.value[y] * val / self.valsue[x]	#根据四边形规则计算合并后的权重
    def is_connected(self,x,y):
        return x in self.value and y in self.value and self.find(x) == self.find(y)
    def add(self,x):
        if x not in self.father:
            self.father[x] = None
            self.value[x] = 1.0
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        uf = UnionFind()
        for (a,b),val in zip(equations,values):
            uf.add(a)
            uf.add(b)
            uf.merge(a,b,val)
        res = [-1.0] * len(queries)
        for i,(a,b) in enumerate(queries):
            if uf.is_connected(a,b):
                res[i] = uf.value[a] / uf.value[b] # 如果ab相连，计算他们的比值
        return res
```
## 图
有向图有入度和出度两个属性
### 207. 课程表（也是拓扑排序）
广度搜索，用一个字典映射a->[b,c]，a必须在b,c前。同时一个list记录每个点的入度。当入度为0时，可以入队列学习，学习完出队列，与之对应的下一个点入度减1.
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    edges = collections.defaultdict(list)
    in_degree = [0] * numCourses
   	for elem in prerequisites:
   		edges[elem[1]].append(elem[0])	#构建有向图字典
   		in_degree(elem[0]) += 1			# 入度自加
   	
   	que = collections.deque(i for i in range(numCourses) if in_degree[i] == 0)	# 将入度为0的入列待学习
   	nums = 0
   	while que:
   		done_num = que.popleft() #学习完一个，出列
   		nums +=1
   		for i in edges[done_num]:	# 对于学习结束的课程，找到它下一个对于的课程，使得下一课程入度自减
   			in_degree[i] -= 1
   			if in_degree[i] ==0:
   				que.append(i)
   	return nums = numCourses
```
## 设计
### 208.前缀树
前缀树是一个根节点开始，指向多条通道，每个通道链表示一串字符，并且包含一个end的标记。用一个字典key：value 对应字符串链。模板如下
```python
class TreeNode(dict): #新建一个节点
    def __init__(self):
        self.end = False
        super(TreeNode, self).__init__() #继承使用父类dict的方法
class Trie:
    def __init__(self):
        self.root = TreeNode()
    def insert(self, word: str) -> None: #插入一个新单词
        cur = self.root
        for ch in word:
            node = cur.get(ch)
            if node is None:	#如果是之前没有的单词串，新建一个节点
                node = TreeNode()
                cur[ch] = node
            cur = node
        cur.end = True #结束标记
    def search(self, word: str) -> bool:
        cur = self.root
        for ch in word:
            cur = cur.get(ch)
            if cur is None:	#依次遍历串中的节点
                return False
        return cur.end
    def startsWith(self, prefix: str) -> bool:	#查找是否有前缀
        cur = self.root
        for ch in prefix:
            cur = cur.get(ch)
            if cur is None:
                return False
        return True
```
### 48. 旋转图像
顺时针旋转90度，等于先沿着\对角线交换，然后从竖着中间|再两边交换
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        size = len(matrix)
        for i in range(1, size):
            for j in range(0, i):
                matrix[i][j], matrix[j][i] =  matrix[j][i], matrix[i][j]  # 对角线交换
        for i in range(size):
            for j in range(size // 2):
                matrix[i][j], matrix[i][size-j-1] = matrix[i][size-j-1], matrix[i][j] #竖直中心交换
        return matrix
```
## 二叉搜索树
二叉搜索树的中序遍历是按从小到大排列好的
### 538. 将二叉搜索树换成累加树
将二叉搜索树中序遍历逆序后累加，再对节点赋值，即可
```python
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        def dfs(root):
        	nonlocal total
        	if root is not None:
        		dfs(root.right)		#先深度搜索右儿子
        		total += root.val
        		root.val = total
        		dfs(root.left)		# 再深度搜索左边
        total = 0
        dfs(root)
        return root
```
### 98. 验证二叉搜索树
采用中序遍历的方法，比较中序遍历法当前点与前一个点值大小
```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        cur = root
        pre =  None
        while cur or stack:
            if cur != None:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if pre and cur.val <= pre.val:
                    return False
                pre = cur
                cur = cur.right
        return True
```
## 递归
### 21.合并两个有序列表
```python
class Solution:
	def mergeList(self, l1, l2):
		if l1 is None:
			return l2
		elif l2 is None:
			return l1
		elif l1.val < l2.val:
			l1.next = self.mergeList(1l.next, l2)  # L1的下一个节点为原L1下个节点与L2之间合并的结果
			return l1
		else:
			l2.next = self.mergeList(l1, l2.next)
			return l2
```
### 105. 从前序和中序遍历中构造二叉树
由前序可以得到根节点，然后结合中序可以得到左子树的节点长度，然后构造左子树，在构造右子树
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        size = len(preorder)
        index = {elem:i for i,elem in enumerate(inorder)}

        def buildtree(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:
                return None
            preorder_root = preorder_left  #由前序遍历得到根节点
            inorder_root = index[preorder[preorder_root]] #从根节点找出中序节点在遍历中的坐标
            root = TreeNode(preorder[preorder_root])
            size_left_subtree = inorder_root - inorder_left #左子树的节点个数
            root.left = buildtree(preorder_left+1, preorder_left + size_left_subtree, inorder_left, inorder_left+size_left_subtree)
            root.right = buildtree(preorder_left+size_left_subtree+1, preorder_right, inorder_left+size_left_subtree+1, inorder_right)
            return root
        res = buildtree(0, size-1, 0, size-1)
        return res
```
## 队列
### 621. 任务调度器
需要找出执行最多的指令，然后计算出执行这个最多的指令共需要多少时间。 在执行这个指令的间隙可以一次抽空做其他指令。如果间隙不够，就直接需要任务串长度的时间。
```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = collections.Counter(tasks)
        max_exec = max(freq.values())
        max_exec_count = sum(1 for i in freq.values() if i == max_exec) # 执行max_exec次的任务的数量
        time = max((n+1)*(max_exec-1) + max_exec_count, len(tasks)) # 计算执行最多次的任务共需要的时间，比较
        return time
```
## 数组
### 169. 多数元素
用一个hashmap（字典），key表示原数组中的元素值，value表示该值出现的次数
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
    	counts = collections.Counter(nums) # 返回一个字典
    	max_c = 0
    	for k in counts.keys():
    		if counts[k] > max_c:
    			max_c = counts[k]
    			ind = k
    	return k
```
## 哈希表
### 1. 两数之和
用一个hashtable （字典），key表示原数列中的值，value表示该值在原数组中的下标。当target - nums[i] 在hash表的key中，直接返回hash[key]就是另一个数的位置
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
    	hashtab = dict()
    	for i, num in enumerate (nums):
    		if (target-num) in hashtab.keys():
    			return [hashtab[target-num], i]
    		hashtab[num] = i 	# 添加新键值
    	return []
```
### 49. 字母异位词分组
将每个单词排序，排序结果作为hash字典的key，然后将同样的key的单词加入该key的value
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            mp[key].append(s)
        return list(mp.values())
```
### 146. LRU缓存机制
用一个字典的哈希表来保存key:node查找起来更快, 其中node包含key:value。构建虚拟的head tail节点可以再处理时更加方便
```python
class LinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = dict()
        self.head = LinkedNode()
        self.tail = LinkedNode()
        self.head.next =self.tail
        self.tail.next = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1        
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = LinkedNode(key, value)
            self.cache[key] = node
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
        
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)
    
    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
```
## 链表
### 206.反转链表
用递归法
```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
    	if head is None or head.next is None:
    		return head
    	new_head = self.reverseList(head.next)	#递归
    	head.next.next = head		#将倒数第二个点和最后一个点链接反向
    	head.next = None	#在最后添加一个空
    	return new_head
```
### 234.回文链表
将链表数据添加到list里面，然后对比list前后和后前顺序是否一样
```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if head  == None or head.next == None:
            return True
        list1 = []
        node = head
        while node != None:
            list1.append(node.val)
            node = node.next       
        return list1 == list1[::-1]
```
### 141. 环形链表 
### 142. 环形链表2
用快慢指针，快指针如果有环，则在环内循环，一定会套圈慢指针. 然后一个从投开始，一个从套圈交点开始，再相遇即是环的起点
```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if head == None or head.next == None:
            return False        
        slow = fast = head
        while fast!=None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        if slow != fast:
            return False
       	pre = head   # 下面这段是用来找链表开始点的，
        while(pre!=slow):
            pre= pre.next
            slow = slow.next
        return slow
```
### 160. 相交链表
把一个链表保存到set中，再遍历第二个链表，有就返回node
```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        s  = set()
        temp = headA
        if headA == None or headB ==None:
            return None
        while temp != None:
            if temp not in s:
                s.add(temp)
                temp = temp.next
        temp = headB
        while temp != None:
            if temp in s:
                return  temp
            temp = temp.next
        return None
```
### 2. 两数相加
用一个指针指向结果的开头，按L1和L2按位计算和，计算结果取模为本位，整除10为有没有进位
```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode()
        temp = head
        s = 0
        while l1 or l2 or s!= 0:
            s = s + (l1.val if l1!=None else 0) + (l2.val if l2!= None else 0)
            if l1 != None:
                l1 = l1.next
            if l2 != None:
                l2 = l2.next
            temp.next = ListNode(s % 10)
            s = s // 10
            temp = temp.next  #不断往链表下一位
        return head.next
```
### 19. 删除链表倒数第n个节点
可以先全放到stack中，然后pop出去n个
```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        prehead = ListNode()
        prehead.next = head
        stack = []
        temp = prehead
        while temp != None:
            stack.append(temp)
            temp = temp.next
        for i in range(n):
            stack.pop()
        top = stack[-1]
        top.next =  top.next.next
        return prehead.next
```
## 双指针
### 283. 移动零
用left right两个下标， left左边都是非0， right若非零，与left值交换
```python
class Solution:
	def moveZeroes(self, nums: List[int]) -> None:
		lens = len(nums)
		left = right = 0
		while right < lens:
			if nums[right] != 0:
				nums[left], nums[right] = nums[right], nums[left]
				left += 1
			right += 1
		return nums
```
### 11. 盛最多水的容器
left和right指针分别指向list左右，然后计算直接的值，再将小一点数的指针往里面移动一位。因为如果移动大的数的指针话，新的结果不会超过之前的结果。
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        size = len(height)
        right = size-1
        max_vol = 0
        if size == 0 or size == 1:
            return 0
        if size == 2:
            return min(height[0], height[1]) * 1
        while(left < right):
            area = min(height[left], height[right]) * (right - left)
            max_vol = max(area, max_vol)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_vol
```
### 75. 颜色分类
“荷兰国旗问题”。 循环，将0，1分别移动到前面合适的位置。双指针p0, p1分别指向最后一个0的后一位和最后一个1的后一位，用于下次交换时的位置。当遇到1时，交换且p1++，遇到0时交换p0且p0++ & p1++。
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        p0 = p1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1
            elif nums[i] ==0:
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1: #不相等时，表明p0原来指向1，p1原来指向2，上面交换后p0指向0了，nums(i)指向1了，需要跟p1再交换一下
                    nums[i], nums[p1] = nums[p1], nums[i]
                p0+=1
                p1 += 1
        return nums
```
### 15. 三数之和
三个坐标指针，先将数组由小到大排序。第一个指针k从0到size-2，第二个指针i从k+1开始，第三个指针从最后一个往前。计算三者的和，调整i，j的大小看看有没有和为0
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        size  = len(nums)
        if size == 0:
            return []
        nums.sort()
        res = []
        for k in range(size-2):
            if nums[k] > 0:  #nums[k]大于0，三数和肯定大于0，pass
                break
            if k > 0 and nums[k] == nums[k-1]:  #第k个跟第k-1个数相等的话，k-1的已经查找过，直接pass
                continue
            i = k+1
            j = size - 1
            while i < j:
                s = nums[k] + nums[i] + nums[j]
                if s == 0:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i<j and nums[i-1] == nums[i]: # 必须是同时满足
                        i += 1
                    while i<j and nums[j+1] == nums[j]:
                        j -= 1
                if s > 0:
                    j -= 1
                    while i<j and nums[j+1] == nums[j]:
                        j -= 1
                if s < 0:
                    i += 1
                    while i<j and nums[i-1] == nums[i]:
                        i += 1
        return res
```

## 数学
## 字符串
## 二分查找
### 33. 搜索旋转排序数组
排序数字旋转后，会有一半是有序的。如果[0, mid]是有序的，且target在0-mid之间，则在0-mid之间继续二分查找，同理，判断右半的二分查找
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l+r) //2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target <nums[mid]:
                    r = mid-1
                else:
                    l = mid+1
            else:
                if nums[mid] < target <= nums[len(nums)-1]:
                    l = mid + 1
                else:
                    r = mid -1
        return -1
```
### 34. 在排序数组中查找元素的第一个和最后一个位置
采用二分查找法，找出target元素出现的位置，然后找target+1，如果target+1不存在，则返回的超过target的第一个下标
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binary_search(nums, target):
            n = len(nums) -1 
            left = 0
            right = n 
            while(left <= right):
                mid = (left + right) //2
                if nums[mid] >= target:  #必须是>=
                    right = mid -1
                else:
                    left = mid +1
            return left
        a = binary_search(nums, target)
        b= binary_search(nums, target+1)
        if a == len(nums) or nums[a] != target:  # 没有找到target
            return [-1, -1]
        else:
            return [a, b-1] 
```
## 集合
### 287. 寻找重复数
```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
    	s = set()
    	for i in nums:
    		if i in s:
    			return i
    		else:
    			s.add(i)
```
### 448.找到消失的数字
用集合的结构
```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        s = set(nums)
        res = []
        for i in range(1,len(nums)+1):
            if i not in s:
                res.append(i)
        return res
```
### 128. 连续最长的序列
当x,...y 是一个序列，那么x+1,..y也是序列 就重复了。因此判断一下可以省去很多时间复杂度
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        s = set(nums)
        for num in s:
            if num-1 in s:  #num-1 已经是存在的序列，则不考虑
                continue
            else:
                length = 1
                while num+1 in s:
                    length += 1
                    num = num +1
                max_len = max(max_len, length)
        return max_len

```
## 分治法
### 53.最大子串
分三种情况，1. 最大子串在左边，2，最大子串在右边，3，最大子串跨中间。递归后返回最大的
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
    	if len(nums) == 1:
    		return nums[0] # 递归终止
    	mid = lens(nums) // 2
    	left_max = self.maxSubArray(nums[:mid])
    	right_max = self.maxSubArray(nums[mid:])
    	temp = 0		# 横跨中间时，只需求出mid向左最大子序和，向右求出最大子序和，然后相加
    	max_L = nums[mid-1]
    	for i in range(mid)[::-1]:
    		temp += nums[i]
    	max_L = max(temp, max_L)
    	temp = 0
    	max_R = nums[mid]
    	for i in range(mid, len(nums)):
    		temp += nums[i]
    	max_R = max(temp, max_R)
    	return max(left_max, right_max, (max_L + max_R))
```
## 动态规划
动态规划的做题顺序
1. 明确dp(i)数组应该表示什么(二维情况 dp(i)(j))
2. 根据dp(i)和dp(i-1)关系得出状态转移方程
3. 确定初始条件，如dp(0)
### 121.买卖股票的最佳时机
状态转移方程 dp(i) = max( dp(i-1),  当日价格-历史最低)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
    	size = len(prices)
    	if size == 1:
    		return 0
    	dp = [0] * size
    	min_price = prices[0]		# 历史最低价
    	for i in range(prices):
    		min_price = min(prices[i], min_price)
    		dp[i] = max(dp[i-1], prices[i]-min_price)
    	return dp[-1]
```
### 70.爬楼梯
采用动态规划，当前楼梯的方法数是下一个楼梯的方法数+下两个楼梯的方法数
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0:
            return 1
        if n==1:
            return 1
        dp = [1] * (n+1)
        for i in range(2,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
```
### 5. 最长回文子串
设计动态规划数组，dp(j, i)从下标j到i的子串是回文串，则表示j~i里面的子串j+1到i-1是回文串且s[j]==s[i]. 记录最大字串长度和开始的字串下标
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        size = len(s)
        max_len = 1
        dp = [[False]*size for _ in range(size)]
        start = 0
        for i in range(size):
            dp[i][i] = True
        for i in range(1, size):
            if s[i] == s[i-1]:  # 两个字符的字串，如果相等则是回文
                dp[i-1][i] = True
                start = i-1
                max_len = 2
        for i in range(2, size):
            for j in range(i-2, -1, -1):   #从i-2的坐标（尾部）一直向s的开头进行迭代计算
                if dp[j+1][i-1] and (s[i] == s[j]):
                    dp[j][i] = True
                    cur_len = i-j+1
                    if cur_len > max_len:
                        start = j
                        max_len = cur_len
        res = s[start:start+max_len]
        return res
```
### 62. 不同路径
对于一点(i, j) ，到它的路径条数dp(i,j) = dp(i-1, j)+ dp(i, j-1)。初始化最上行和最左列均为1种路径
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n] + [[1]+ [0]*(n-1) for _ in range(m-1)]  #初始化
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]         
        return dp[m-1][n-1] #最右下角的坐标
```
### 64. 最小路径和
到某一点的最小路径和dp(i,j) = min(dp(i-1, j), dp(i, j-1)) + grid(i,j) ，对最上行和最左列先进行初始化
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row = len(grid)
        col = len(grid[0])
        if row == 0  or col == 0:
            return 0 
        dp = [[0]*col for _ in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1, row):      #最左列初始化
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, col): 	 #最上行初始化
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[row-1][col-1] 
```
### 96. 不同的二叉搜索数
F(j,n)表示以j为根节点长度为n的二叉搜索数个数，G(n)表示n长的二叉搜索树个数。 F(j, n) = G(j-1)· G (n-j)，即i左半边子树类型个数* i右半子树个数。 G(n) = F(1,n) + F(2,n) + ... + F(n,n) = ΣG(j-1)G(n-j)
```python
class Solution:
    def numTrees(self, n: int) -> int:
        G = [0]*(n+1)
        G[0] = 1
        G[1] = 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]  #不断求G[i]
        return G[n]
```
###  139. 单词拆分
动态方程 dp[j]为true需要之前的dp[i]为true并且[i,j]之间的字符串在字典中
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1)  
        dp[0] = True
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                if dp[i] == True and (s[i:j] in wordDict):
                    dp[j] = True
        return dp[-1]        
```
### 416. 分割等和子集  （0-1背包）
dp(i)(j) 表示从数组[0,i]的坐标里面找到和为j的状态是否为True。如果`j>=nums[i]` 则`nums[i]`可选可不选，选与不选两种方式or。如果`j<nums[i]`则和超过了j，必不能选。
```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        size = len(nums)
        if size < 2:
            return False
        total = sum(nums)
        max_num = max(nums)
        if total %2 != 0:
            return False      
        target = total //2 
        if max_num > target:
            return False      #排除一些无法二分的情况
        
        dp = [[False]*(target+1) for _ in range(size)]
        for i in range(size):
            dp[i][0] = True
        dp[0][nums[0]] = True   # 初始化dp状态
        
        for i in range(1, size):
            for j in range(1, target+1):
                if j>=nums[i]:
                    dp[i][j] = dp[i-1][j] | dp[i-1][j-nums[i]] #不选或选nums[i]
                else:
                    dp[i][j] = dp[i-1][j]  # 不选
        
        return dp[size-1][target]

```
## 回溯算法

### 17.电话号码的字母组合
使用回溯法，这个index是记录遍历第几个数字了，就是用来遍历digits的（题目中给出数字字符串），同时index也表示树的深度。不是组合问题中的start_idx
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
         letterMap = {'2': 'abc','3': 'def','4': 'ghi','5': 'jkl','6': 'mno','7': 'pqrs','8': 'tuv','9': 'wxyz'
        }
        ans = []
        s = ''
        def backtracking( digits, index):
            nonlocal s  # list不需要定义为	nonlocal，但是变量需要
            if index == len(digits):
                ans.append(s)
                return
            else:
                letters = letterMap[digits[index]]  # 取出数字对应的字符集
                for letter in letters:
                    s = s + letter  # 处理
                    backtracking(digits, index + 1)
                    s = s[:-1]      # 回溯
        if digits == '':
            return []
        backtracking(digits, 0)
        return ans
```
> https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/solution/dai-ma-sui-xiang-lu-17-dian-hua-hao-ma-d-ya2x/
### 39. 组合总和
使用回溯+剪枝法，需要控制元素是可以重复的
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        def traceback(candidates, target, total, start_idx):
            if total > target:
                return  
            if total == target:
                res.append(path[:]) #如果不加[:]，则res和path公用地址，会出错
                return   #可以不加 return res
            for i in range(start_idx, len(candidates)):
                if total + candidates[i] > target: # 超过了目标，剪枝
                    return
                total += candidates[i]
                path.append(candidates[i])
                traceback(candidates, target, total, i) # 参数start_idx为当前i，则表示可重复用，i+1不重复
                total -= candidates[i]	#回溯
                path.pop()	#回溯
        candidates = sorted(candidates)
        traceback(candidates, target, 0, 0)
        return res
```
### 77.  组合
回溯+剪枝，元素不可以重复
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        path  = []
        res = []
        def backtracking(n, k, start_idx):
            if len(path) == k:
                res.append(path[:])
                return 
            for i in range(start_idx, n-(k-len(path))+1): # 第二个参数用来剪枝，将后续元素数目不够的全都丢弃
                path.append(i+1)
                backtracking(n, k , i+1)	#k+1作为startidx控制不重复元素
                path.pop()
        backtracking(n,k,0)
        return res
```
### 78. 子集
子集是搜集所有的节点因此不需要判断加入的条件，而组合是搜集叶子节点。由于跟排列问题不同，不考虑顺序，所以要使用start_idx来指定下一次搜索的开始数字
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res =[]
        path = []
        def backtracking(nums, start_idx):
            res.append(path[:])
            for i in range(start_idx, len(nums)):
                path.append(nums[i])
                backtracking(nums, i+1)
                path.pop()
        backtracking(nums,0)
        return res
```
### 46. 全排列
使用回溯法，排列的问题因为需要顺序不同所以不使用start_idx，但是需要判断path中是否已经有了元素，控制元素不重复。
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        def backtracking(nums):
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if nums[i] in path:  #判断path中是否已经使用过一个元素,去掉则会重复使用
                    continue
                path.append(nums[i])
                backtracking(nums)
                path.pop()
        backtracking(nums)
        return res
```
### 79. 单词搜索
用一个二维数组记录有没有被访问过。深度优先搜索路径，当有一个字符不对时，就回溯路径
```python
		m, n = len(board), len(board[0])
        used = [[False] * n for _ in range(m)]
        def dfs(row, col, i): # 判断当前点是否是目标路径上的点. row col 当前点的坐标，i当前考察的word字符索引
            if i == len(word): # 递归结束条件 : 找到了单词的最后一个
                return True
            if row < 0 or row >= m or col < 0 or col >= n: # 越界
                return False
            if used[row][col] or board[row][col] != word[i]:# 已经访问过,或者不是word里的字母
                return False
            # 排除掉上面的false情况，当前点是合格的，可以继续递归考察
            used[row][col] = True #  记录一下当前点被访问了
            if dfs(row+1, col, i+1) or dfs(row-1, col, i+1) or dfs(row, col+1, i+1) or dfs(row, col-1, i+1): # 基于当前点[row,col]，可以为剩下的字符找到路径
                return True
            used[row][col] = False # 不能为剩下字符找到路径，返回false，撤销当前点的访问状态，继续考察别的分支


        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0] and dfs(i, j, 0):
                    return True
        return False
```
##  滑动窗口
### 3. 无重复字符的最长子串
一个滑动窗口，类型为set，不重复的字符就加入窗口，如果有重复，则不断从滑动窗口中把左边的字符移除直至没有重复，然后继续往后添加滑动窗。
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
    	max_len = cur_len =0
    	left  = 0
    	lookup = set()
    	for i in range(len(s)):
    		while s[i] in lookup:
    			lookup.remove(s[left])
    			left += 1
    		lookup.add(s[i])
    		cur_len = len(lookup)
    		if cur_len>max_len: max_len = cur_len
    	return max_len
```

















