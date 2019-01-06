## MXOP: MXNet-OpSummary    
It only works for **gluon** yet.     
    
Reference: [THOP: PyTorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)    

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

### TODO
    
- [ ] Count OPs and parameters for each layer.
- [ ] Support Symbol model for MXNet.      
 