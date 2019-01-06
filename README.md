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
| Model   | Parameters | Multiplication | Addition |
|:---:|:---:|:---:|:---:|
|VGG13|133,047,848|11,308,456,984|11,308,447,792|
|VGG16|138,357,544|15,470,255,128|15,470,245,936|
|VGG19|143,667,240|19,632,053,272|19,632,044,080|
|VGG11_bn|132,874,344|7,616,506,904|7,623,923,760|
|VGG13_bn|133,059,624|11,320,699,928|11,332,933,680|
|VGG16_bn|138,374,440|15,483,802,648|15,497,340,976|
|VGG19_bn|143,689,256|19,646,905,368|19,661,748,272|
|Inception_v3|23,869,000|5,722,182,584|5,738,329,808|
|ResNet18_v1|11,699,112|1,816,556,056|1,816,555,056|
|ResNet34_v1|21,814,696|3,667,498,520|3,667,497,520|
|ResNet50_v1|25,629,032|3,868,559,384|3,875,457,584|
|ResNet101_v1|44,695,144|7,585,898,520|7,597,061,680|
|ResNet152_v1|60,404,072|11,304,441,880|11,320,873,520|
|ResNet18_v2|11,695,796|1,816,731,672|1,816,906,288|
|ResNet34_v2|21,811,380|3,667,674,136|3,667,848,752|
|ResNet50_v2|25,595,060|4,099,143,192|4,097,988,144|
|ResNet101_v2|44,639,412|7,816,482,328|7,815,327,280|
|ResNet152_v2|60,329,140|11,535,025,688|11,533,870,640|
|DenseNet121|8,062,504|2,849,828,120|2,859,171,376|
|DenseNet161|28,900,936|7,757,320,184|7,776,608,080|
|DenseNet169|14,307,880|3,378,865,304|3,391,212,208|
|DenseNet201|20,242,984|4,316,070,296|4,333,597,616|
|MobileNet_v1_1.00|4,253,864|573,782,040|573,781,040|
|MobileNet_v1_0.75|2,601,976|329,181,464|329,180,464|
|MobileNet_v1_0.50|1,342,536|152,017,432|152,016,432|
|MobileNet_v1_0.25|475,544|42,289,944|42,288,944|
|MobileNet_v2_1.00|3,539,136|320,698,848|320,697,848|
|MobileNet_v2_0.75|2,653,864|191,975,848|191,974,848|
|MobileNet_v2_0.50|1,983,104|95,842,160|95,841,160|
|MobileNet_v2_0.25|1,526,856|32,297,784|32,296,784|
|SqueezeNet1_0|1,248,424|818,924,576|819,092,576|
|SqueezeNet1_1|1,235,496|349,151,936|349,319,936|

### TODO
    
- [ ] Count OPs and parameters for each layer.
- [ ] Support Symbol model for MXNet.      
 