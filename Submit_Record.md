# 1(itemcf填充)
- itmecf:10
- itemcf recall:34.117%
- 线上：0.1065

# 2
- itemcf:25
- hot:15
- itemcf recall:46.5345%
- hot recall:10.7035%
- total recall:48.874%
- 线上:0.0720

# 3
- itemcf:25
- hot:15
- itemcf recall:46.4875%
- hot recall:10.7035%
- total recall:48.862%
- 线上:0.0660

# 4
- itemcf:25
- hot:15
- itemcf recall:46.4875%
- hot recall:10.7035%
- total recall:48.862%
- 线上:0.0285

# 5(itemcf填充)
- itmecf:10
- itemcf recall:34.117%
- fit添加eval_set
- 线下：0.1000
- 线上：0.1000

# 6
- itmecf:25
- hot:10
- itemcf recall:46.4875%
- hot recall:8.512%
- total recall:48.311%
- 线下：0.30809289688187347
- 线上：0.0613

# 7
- hot recall去掉了过去出现过的文章
- itmecf:25
- hot:10
- itemcf recall:46.4875%
- hot recall:8.4725%
- total recall:48.2715%
- 线下：0.31048661249069187
- 线上：0.1592

# 8(base 7)
- hot recall将热门文章缩短为前后1天
- itmecf:25
- hot:10
- itemcf recall:46.4875%
- hot recall:26.518000000000004%
- total recall:54.2365%
- 线下：0.3376045615464144
- 线上：0.1956[BEST]

# 9(base 8)
- hot recall优化了生成复杂度，并且train和test的热门商品分开计算
- itmecf:25
- hot:10
- itemcf recall:46.4875%
- hot recall:27.3935%
- total recall:54.4215%
- 线下：0.3382830553994129
- 线上：0.1921 ↓
