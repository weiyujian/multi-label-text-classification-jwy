# multi-label-text-classification-jwy
multi label classification for my own data

#多标签文本分类，支持每个问句拥有不同个数的标签

#两种方式计算模型准确率：1.取得分最高的topk个结果（固定标签结果的个数）2.设置sigmoid得分阈值，得分
#超过阈值（自己设定，如0.5）的都作为结果返回（不需要固定标签个数）


#多标签分类的几个关键点：
1.数据：每个单独的标签都是one-hot编码，整个句子的标签就是所有one-hot编码标签的累加，eg,[0,1,1,0,0,1]
2.模型：模型的损失函数用sigmoid_cross_entropy_with_logits代替softmax_cross_entropy_with_logits
3.准确率计算：可以取topk个最高的结果，也可以取超过某个阈值的所有标签
