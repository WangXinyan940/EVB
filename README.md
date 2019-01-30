# EVB

EVB的简单实现。MD engine使用OpenMM。其中OpenMM对AMOEBA的stretch bend force支持有个小bug，希望他们下一版改了吧……

用法：

1. 制作用于描述体系拓扑的Tinker xyz文件和key文件。流程看这里：https://sites.google.com/site/amoebaworkshop/exercise-2
2. 使用key2xml.py将上述文件转化为pdb文件和xml描述文件，作为EVB Halmitonian的对角项。
3. 编辑conf.json，描述非对角项CV和所需常数。（示例文件施工中）
4. 调用evb.py中的EVBHalmitonian类，将conf.json作为字典项传入，进行初始化。