#### 1. Tokenizer和LLM是两个独立的阶段

tokenizer也有自己的训练集（不要求与LLM的一致），在预处理阶段通过BPE算法进行训练。

![image-20260602204206500](C:\Users\Law B J\AppData\Roaming\Typora\typora-user-images\image-20260602204206500.png)

本质上是一个raw text到token sequence的转换层，向上执行encode操作，向下执行decode操作。

**BPE 之后，将认为Tokenizer已经被dataset（text）训练好了（产出了一个表达了合并规则的`merges`字典），encoder 和 decoder 则按照text的信息背景来执行相应功能**。

> 为什么LLM目前做算术运算这么差，因为输入的数字会以比较**随机的方式进行划分**，且在BPE的过程中可能合并成更大的一个token → 失去了本来的数学意义