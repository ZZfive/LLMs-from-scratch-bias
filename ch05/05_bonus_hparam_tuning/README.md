# 优化预训练的超参数

基于[附录D：为训练循环添加更多花哨功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)中的扩展训练函数，hparam_search.py 脚本旨在通过网格搜索找到最佳超参数。

>[!NOTE]
这个脚本运行时间会很长。可适当减少在顶部`HPARAM_GRID`字典中探索的超参数配置数量。