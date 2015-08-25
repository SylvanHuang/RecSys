数据集：MovieLens 1M
训练集与测试集比例：7:1


参数：

- UserCF：n=10，k=80
- ItemCF：n=10，k=10


算法|召回率|准确率|覆盖率|流行度
:-:|:-:|:-:|:-:|:-:
UserCF|0.12489092576779834|0.22983029801324506|0.20559308957311873|7.297909285598981
UserCF-IIF|0.12545051126912252|0.23086092715231787|0.21669640555232994|7.269115979978174
ItemCF|0.10948852463255011|0.2014859271523179|0.19293123297459927|7.262564861248097
ItemCF-Norm|0.11111858862401412|0.20448468543046358|0.23237113970737175|7.2125878155328
ItemCF-IUF|0.1119347026899766|0.2059871688741722|0.1754963656148635|7.362082131629706