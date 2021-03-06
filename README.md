## MXOP: MXNet-OpSummary    
It only works for **gluon** yet.     
    
Reference: [THOP: PyTorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)    

### Installation    
* PyPi    
    ```bash
    pip install --index-url https://pypi.org/simple/ mxop
    ```
* Github (latest)    
    ```bash
    pip install --upgrade git+https://github.com/hey-yahei/OpSummary.MXNet.git
    ```

### Usage
#### Gluon
* Count OPs for model **(default)**  
    ```python
    from mxnet import nd
    from mxop.gluon import count_ops
    inputs = (nd.ones(shape=(1, 3, 224, 224)), )
    op_counter = count_ops(net, inputs, [, exclude])   # `net` is the gluon model you want to count OPs 
    ```
* Count OPs for every block       
    ```python
    from mxop.gluon import count_ops
    op_counters = count_ops(net, inputs, per_block=True)
    # Print muls
    mul_counter = {k: v['muls'] for k, v in op_counters.items()}
    total = sum(mul_counter.values())
    for k, v in mul_counter.items():
        print(f"{k.name}\t{v/total*100:.2f}%")  
    ```
* Count parameters    
    ```python
    from mxop.gluon import count_params
    params_counter = count_params(net [, exclude])   # `net` is the gluon model you want to count parameters
                                                                    # `exclude` is the list of blocks to be excluded 
    ```
* Print summary     
    ```python
    from mxop.gluon import op_summary
    op_summary(net [, inputs] [, exclude])   # `net` is the gluon model you want to count
                                             # `inputs` is the inputs to feed net 
                                             # `exclude` is the list of blocks to be excluded
    ```

#### Custom OPs      
Defining count function for your own blocks is supported.         

1. Define a function `hook(m, x, y) --> None` in which you should set values of OPs for dict `m.ops`, for example,      
    ```python
    def count_bn(m, x, y):
        x = x[0]
        n_elem = x.size
        m.ops["adds"] = 2*n_elem
        m.ops["muls"] = n_elem
        m.ops["divs"] = n_elem
        m.ops["exps"] = 0
    ```     
2. Use it as a parameter when you call function `count_op` or `op_summary`, for example,       
    ```python
    from mxnet.gluon import nn
    from mxop.gluon import count_ops, op_summary
    custom_ops = {nn.BatchNorm: count_bn}
    counter = count_ops(net, inputs, custom_ops=custom_ops)
    # op_summaary(net, inputs, custom_ops=custom_ops)
    ```       
    
You can also use your own count function for blocks that I have setted count function for, because functions list in `custom_ops` is given higher priority.      

### Test

Run `tests/test_gluon_utils.py` to count OPs and parameters for all models in model zoo of MXNet.   

#### Result:
| Model   | Params(M) | Muls(G) | \*Params(M) | *Muls(G) | Top1 Acc | Top5 Acc |
|---:|---:|---:|---:|---:|---:|---:|
|AlexNet|61.10|0.71|2.47|0.66|0.5492|0.7803|
|VGG11|132.86|7.61|9.22|7.49|0.6662|0.8734|
|VGG13|133.04|11.30|9.40|11.18|0.6774|0.8811|
|VGG16|138.63|15.47|14.71|15.35|0.7323|0.9132|
|VGG19|143.67|19.63|20.02|19.51|0.7411|0.9135|
|VGG11_bn|132.87|7.62|9.23|7.49|0.6859|0.8872|
|VGG13_bn|133.06|11.32|9.42|11.20|0.6884|0.8882|
|VGG16_bn|138.37|15.48|14.73|15.36|0.7310|0.9176|
|VGG19_bn|143.69|19.65|20.05|19.52|0.7433|0.9185|
|Inception_v3|23.87|5.72|21.82|5.72|0.7755|0.9364|
|ResNet18_v1|11.70|1.82|11.19|1.82|0.7093|0.8992|
|ResNet34_v1|21.81|3.67|21.3|3.67|0.7437|0.9187|
|ResNet50_v1|25.63|3.87|23.58|3.87|0.7647|0.9313|
|ResNet101_v1|44.70|7.59|42.65|7.58|0.7834|0.9401|
|ResNet152_v1|60.40|11.30|58.36|11.30|0.7900|0.9438|
|ResNet18_v2|11.70|1.82|11.18|1.82|0.7100|0.8992|
|ResNet34_v2|21.81|3.67|21.30|3.67|0.7440|0.9208|
|ResNet50_v2|25.60|4.10|23.55|4.10|0.7711|0.9343|
|ResNet101_v2|44.64|7.82|42.59|7.81|0.7853|0.9417|
|ResNet152_v2|60.33|11.54|58.28|11.53|0.7921|0.9431|
|DenseNet121|8.06|2.85|7.04|2.85|0.7497|0.9225|
|DenseNet161|28.90|7.76|26.69|7.76|0.7770|0.9380|
|DenseNet169|14.31|3.38|12.64|3.38|0.7617|0.9317|
|DenseNet201|20.24|4.32|18.32|4.31|0.7732|0.9362|
|MobileNet_v1_1.00|4.25|0.57|3.23|0.57|0.7105|0.9006|
|MobileNet_v1_0.75|2.60|0.33|1.83|0.33|0.6738|0.8782|
|MobileNet_v1_0.50|1.34|0.15|0.83|0.15|0.6307|0.8475|
|MobileNet_v1_0.25|0.48|0.04|0.22|0.04|0.5185|0.7608|
|MobileNet_v2_1.00|3.54|0.32|2.26|0.32|0.7192|0.9056|
|MobileNet_v2_0.75|2.65|0.19|1.37|0.19|0.6961|0.8895|
|MobileNet_v2_0.50|1.98|0.10|0.70|0.09|0.6449|0.8547|
|MobileNet_v2_0.25|1.53|0.03|0.25|0.03|0.5074|0.7456|
|SqueezeNet1_0|1.25|0.82|0.74|0.73|0.5611|0.7909|
|SqueezeNet1_1|1.24|0.35|0.72|0.26|0.5496|0.7817|

**To compare classification models used as backbone--**   
**\*Params col shows the number of parameters for models without last several layers.**    
**\*Muls col shows the number of multiplications for models without last several layers.**     
    
![Parameters](http://hey-yahei.cn/imgs/MXNet-OpSummary/Parameters.jpg)
    
![Multiplication](http://hey-yahei.cn/imgs/MXNet-OpSummary/Multiplication.jpg)    
     
***The data above may help with design when you choose such CNNs as backbone.***     

### TODO
    
- [x] Count OPs and parameters for each layer.
- [ ] Support Symbol model for MXNet.      
- [ ] Support quantized models.

--------------------------     
***More details refer to 《[模型参数与计算量 | Hey~YaHei!](https://www.yuque.com/yahei/hey-yahei/opsummary.mxnet)》***