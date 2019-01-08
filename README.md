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
    params_counter = count_params(net, input_size)   # net is the gluon model you want to count parameters
                                                     # input_size is the shape of your input 
    ```
* Print summary     
    ```python
    from mxop.gluon import op_summary
    op_summary(net, input_size)   # net is the gluon model you want to count
                                  # input_size is the shape of your input 
    ```

### Test

Run `tests/test_gluon_utils.py` to count OPs and parameters for all models in model zoo of MXNet.   

#### Result:
| Model   | Parameters(M) | MAC(G) | Top1 Acc | Top5 Acc |
|---:|---:|---:|---:|---:|
|AlexNet|61.10|0.71|0.5492|0.7803|
|VGG11|132.86|7.61|0.6662|0.8734|
|VGG13|133.04|11.30|0.6774|0.8811|
|VGG16|138.63|15.47|0.7323|0.9132|
|VGG19|143.67|19.63|0.7411|0.9135|
|VGG11_bn|132.87|7.62|0.6859|0.8872|
|VGG13_bn|133.06|11.32|0.6884|0.8882|
|VGG16_bn|138.37|15.48|0.7310|0.9176|
|VGG19_bn|143.69|19.65|0.7433|0.9185|
|Inception_v3|23.87|5.72|0.7755|0.9364|
|ResNet18_v1|11.70|1.82|0.7093|0.8992|
|ResNet34_v1|21.81|3.67|0.7437|0.9187|
|ResNet50_v1|25.63|3.87|0.7647|0.9313|
|ResNet101_v1|44.70|7.59|0.7834|0.9401|
|ResNet152_v1|60.40|11.30|0.7900|0.9438|
|ResNet18_v2|11.70|1.82|0.7100|0.8992|
|ResNet34_v2|21.81|3.67|0.7440|0.9208|
|ResNet50_v2|25.60|4.10|0.7711|0.9343|
|ResNet101_v2|44.64|7.82|0.7853|0.9417|
|ResNet152_v2|60.33|11.54|0.7921|0.9431|
|DenseNet121|8.06|2.85|0.7497|0.9225|
|DenseNet161|28.90|7.76|0.7770|0.9380|
|DenseNet169|14.31|3.38|0.7617|0.9317|
|DenseNet201|20.24|4.32|0.7732|0.9362|
|MobileNet_v1_1.00|4.25|0.57|0.7105|0.9006|
|MobileNet_v1_0.75|2.60|0.33|0.6738|0.8782|
|MobileNet_v1_0.50|1.34|0.15|0.6307|0.8475|
|MobileNet_v1_0.25|0.48|0.04|0.5185|0.7608|
|MobileNet_v2_1.00|3.54|0.32|0.7192|0.9056|
|MobileNet_v2_0.75|2.65|0.19|0.6961|0.8895|
|MobileNet_v2_0.50|1.98|0.10|0.6449|0.8547|
|MobileNet_v2_0.25|1.53|0.03|0.5074|0.7456|
|SqueezeNet1_0|1.25|0.82|0.5611|0.7909|
|SqueezeNet1_1|1.24|0.35|0.5496|0.7817|

### TODO
    
- [ ] Count OPs and parameters for each layer.
- [ ] Support Symbol model for MXNet.      
- [ ] Support quantized models.
 