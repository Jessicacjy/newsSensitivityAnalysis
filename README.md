# newsSensitivityAnalysis
新闻情感分析

1. **east_yanbao.ipynb**
  - 从东方财富网爬取研报
  - 运用虚拟web来获取java借口的研报
  
2. **textSens.ipynb**
  
  * 分析新闻正负情感值：

     - 读取新闻并且进行文本处理和分词
     - 建立模型 build_and_evaluate(X,y,classifier,outpath='model')
          - 采用了gridsearch， 允许同时运训多种参数同时进行
          - 也可以gridsearch = False 如果不需要调参数
          - 根据precision 和 recall 值， 对比Multibinomial, logistic regression, SGD等。最终采用随机森林 模型
          - outpath 允许模型保存到本地
     - 用随机森林模型，得出每个词的重要性，也就是feature importance
     - 寻找对于模型预测最有决定性的文字：show_most_informative_features(model, text=None, n=20):
          - 这个function只适用于pipline结构，也就是在训练模型的时候，需要吧gridsearch关掉训练出来的模型才能用
          - 随机森里不可用
          - 会产出哪些词是正面导向和负面导向以及向量

3. **Feature_Importance.ipynb**
  - 训练随机森里模型，并且计算出所有词语在模型中决定正负面的重要性
  
4. **SentimentlAnalysis.ipynb**
  - 读进一篇研报，用已训练好的模型，预测这是一篇正面还是负面的新闻，并且正负面的值
