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
* Count OPs    
    ```python
    from mxop.gluon import count_ops
    op_counter = count_ops(net)   # net is the gluon model you want to count OPs 
    ```
* Count parameters    
    ```python
    from mxop.gluon import count_params
    op_counter = count_params(net, input_size)   # net is the gluon model you want to count OPs
                                                 # input_size is the shape of your input 
    ```
* Print summary     
    ```python
    from mxop.gluon import op_summary
    op_counter = op_summary(net, input_size)   # net is the gluon model you want to count OPs 
                                               # input_size is the shape of your input 
    ```

### Test

Run `tests/test_gluon_utils.py` to count OPs and parameters for all models in model zoo of MXNet.   

#### Result:
| Model   | Parameters(M) | MAC(G) |
|:---:|:---:|:---:|
|AlexNet|61.10|0.71|
|VGG11|132.86|7.61|
|VGG13|133.04|11.30|
|VGG16|138.63|15.47|
|VGG19|143.67|19.63|
|VGG11_bn|132.87|7.62|
|VGG13_bn|133.06|11.32|
|VGG16_bn|138.37|15.48|
|VGG19_bn|143.69|19.65|
|Inception_v3|23.87|5.72|
|ResNet18_v1|11.70|1.82|
|ResNet34_v1|21.81|3.67|
|ResNet50_v1|25.63|3.87|
|ResNet101_v1|44.70|7.59|
|ResNet152_v1|60.40|11.30|
|ResNet18_v2|11.70|1.82|
|ResNet34_v2|21.81|3.67|
|ResNet50_v2|25.60|4.10|
|ResNet101_v2|44.64|7.82|
|ResNet152_v2|60.33|11.54|
|DenseNet121|8.06|2.85|
|DenseNet161|28.90|7.76|
|DenseNet169|14.31|3.38|
|DenseNet201|20.24|4.32|
|MobileNet_v1_1.00|4.25|0.57|
|MobileNet_v1_0.75|2.60|0.33|
|MobileNet_v1_0.50|1.34|0.15|
|MobileNet_v1_0.25|0.48|0.04|
|MobileNet_v2_1.00|3.54|0.32|
|MobileNet_v2_0.75|2.65|0.19|
|MobileNet_v2_0.50|1.98|0.10|
|MobileNet_v2_0.25|1.53|0.03|
|SqueezeNet1_0|1.25|0.82|
|SqueezeNet1_1|1.24|0.35|

### TODO
    
- [ ] Count OPs and parameters for each layer.
- [ ] Support Symbol model for MXNet.      
- [ ] Support different data type.
 