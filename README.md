# graduation_thesis
Optimization and Implementation of  Left and Right Hand Recognition Based on YOLOv2   




南  阳  理  工  学  院

本科生毕业设计（论文）
                        （二号、黑体、居中）

























（外封皮）


南阳理工学院本科生毕业设计（论文）
（四号、宋体、居中）



基于YOLOv2的左右手识别优化研究与实现
（二号、黑体、居中）

Optimization and Implementation of
 Left and Right Hand Recognition Based on YOLOv2    
英文题目
(Times  New  Roman  16)











总计：  毕业设计（论文） 页
表    格：         个
插    图 ：        幅
（五号、宋体）

   
（内封皮）


目   录

一、 摘要	2
2．目标：要完成和实现的系统。	2
3．方法：方法和步骤。	2
4．结果：系统可以完成什么工作。	2
5．结论：对算法的论断。	2
Abstract	3
二、关键词	3
Keyword	3
三、绪论	4
1．课题的背景、目的和意义。	4
2．要完成的目标。	4
3．主要内容。	4
4．论文的组织结构。	4
四、理论基础	4
1．算法用到的理论和知识。	4
2．与算法相关或相似的技术介绍。	4
五、算法描述	5
1．相关的概念。	5
2．算法描述：输入、输出和流程。	5
六、设计与实现	6
1．复杂和核心流程的程序流程图。	6
2．核心功能的主要代码及说明。	6
1．算法实现采用的工具和技术。	6
2．界面的设计。	6
3．数据结构或数据库的描述。	6
七、结果和结论	6
八、结束语	6
九、参考文献	6
（1）科技书籍和专著注录格式：	6
（2）科技论文的注录格式：	7

一、 摘要

（1．目的：算法的价值和意义。
2．目标：要完成和实现的系统。
3．方法：方法和步骤。
4．结果：系统可以完成什么工作。
5．结论：对算法的论断。
中文摘要约300字，英文摘要约250个词。）

基于YOLOv2的左右手识别优化研究与实现，是我实习公司的一个人工智能类科研项目，是我们的触控产品从单点触摸通向多点触控的桥梁，是我们公司跻身于人工智能科技行业的敲门砖。YOLOv2是开源深度学习框架DarkNet的网络模型，能够实现端到端的实时的识别出左右手的位置，开创了目标检测的新思路。首先我们需要创建一个网络模型，然后通过大量标定数据进行模型训练，得到权重文件，最后运用这个训练好的权重文件对需要图片中的走右手识别出来。DarkNet相当快。基础模型可以达到45fps，一个更小的模型（Fast YOLO），可以达到155fps，同时mAP是其他可以实现实时检测方法的两倍。和目前的最先进的方法相比，DarkNet的位置检测误差较大，但是对于背景有更低的误检。同时，DarkNet能够学习到更加泛化的特征，例如艺术品中物体的检测等任务，表现很好。
Abstract
Based on YOLOv2 left and right hand recognition optimization and implementation, is an artificial intelligence research project of my internship company. It is our touch product from single touch to multi-touch bridge, our company is among the artificial intelligence technology industry. The doorstep. YOLOv2 is a network model of the open source deep learning framework DarkNet. It can realize end-to-end real-time recognition of the position of right and left hands, and creates a new idea for target detection. First, we need to create a network model, and then train the model through a large number of calibration data to obtain a weight file. Finally, we use this trained weight file to identify the right hand in the required image. DarkNet is pretty fast. The basic model can reach 45fps, a smaller model (Fast YOLO) can reach 155fps, and mAP is twice that of other real-time detection methods. DarkNet's position detection error is larger than the current state-of-the-art method, but it has a lower false detection for the background. At the same time, DarkNet can learn more general features, such as the detection of objects in artworks, and it performs well.
二、关键词
YOLO 神经网络 深度学习 优化 实现

Keyword
YOLO  Neural_Networks  Deep_Learning Optimization Realization

三、绪论
1．课题的背景、目的和意义。
2．要完成的目标。
3．主要内容。
4．论文的组织结构。
四、理论基础
1．算法用到的理论和知识。
2．与算法相关或相似的技术介绍。
人们看到图像后能立即识别出图像中的对象，它们在哪里以及它们有什么样的联系。人类的视觉系统是快速和准确的，使我们能够执行复杂的任务。同样的，快速准确的目标检测算法可以让计算机在没有专门传感器的情况下能够执行复杂的任务。
如今市面上的检测系统将分类器重新用于检测平台。为了检测对象，这些系统为该对象提供一个分类器，在图片的不同位置评估它，并在测试图像中进行缩放。像DPM（可变形零件模型）这样的系统使用滑动窗口方法，让分类器在整个图像上间隔均匀的移动位置运行检测。
最近的方法，如R-CNN使用区域提案方法首先在图像中生成潜在的边界框，然后在这些提出的框上运行分类器。分类后，后处理用于细化边界框，消除重复检测，并根据场景中的其他对象重新定位框[13]。这些复杂的管道很慢并且难以优化，因为每个单独的组件都必须单独进行培训。




	YOLO将对象检测重新设计为单一回归问题，从图像像素得出边界框坐标和类概率。YOLO（You Only Look Once）只需要在图像上跑一次，就能预测出现物体的类别和位置。


图1：YOLO检测系统。用YOLO处理图像简单而直接。YOLO（1）将输入图像调整为448×448，（2）在图像上运行单个卷积网络，以及（3）用模型的置信度与阈值对比。

YOLO很简单：参见图1.单个卷积网络可同时预测这些盒子的多个边界框和类概率。
	YOLO训练全图像并直接优化检测性能。这种联合的模型与传统的物体检测方法相比有三个优点和一个缺点：
一、YOLO速度非常快。由于它将检测视为回归问题，因此不需要复杂管路。测试的时候它只是在一幅新图像上运行神经网络来预测检测结果。在Titan X GPU上没有批处理的情况下，它的基础网络运行速度为每秒45帧，快速版本运行速度超过每秒150帧。这意味着它可以在不到25毫秒每帧的效率实时处理媒体视频流。此外，YOLO实现了其他实时系统平均精度的两倍以上。
二、YOLO进行预测时，与基于滑动窗口和RPN(区域提议网络)的技术不同，YOLO在训练和测试时间期间能覆盖到整个图像，因此它能够隐式地编码分类的上下文信息以及它们的表层特征。Fast R-CNN是一种顶级的检测方法[14]，因为它看不到更大的上下文，所以在图像中错误的出现对象的背景补丁。与Fast R-CNN相比，YOLO的背景错误数量少了一半。
三、YOLO学习物体的一般化表示。在对自然图像进行训练并在艺术品上进行测试时，YOLO大幅优于DPM和R-CNN等顶级检测方法。由于YOLO具有高度概括性，因此在应用于新域或意外输入时发生故障的概率很小。
YOLO的准确度仍然落后于最先进的检测系统。虽然它可以快速识别图像中的物体，但它正努力精确定位某些物体，尤其是小物体。

五、算法描述
1．相关的概念。
2．算法描述：输入、输出和流程。
第一，作者使用了一系列的方法对原来的YOLO多目标检测框架进行了改进，在保持原有速度的优势之下，精度上得以提升。VOC 2007数据集测试，67FPS下mAP达到76.8%，40FPS下mAP达到78.6%，基本上可以与Faster R-CNN和SSD一战。这一部分是本文主要关心的地方。
第二，作者提出了一种目标分类与检测的联合训练方法，通过这种方法，YOLO9000可以同时在COCO和ImageNet数据集中进行训练，训练后的模型可以实现多达9000种物体的实时检测。这一方面本文暂时不涉及，待后面有时间再补充。

六、设计与实现
1．复杂和核心流程的程序流程图。
联合检测
YOLO将对象检测的两个单独组件集成到一个神经网络中。我们的网络使用整个图像的特征来预测每个边界框。它还同时预测图像中所有类的所有边界框。这意味着我们的网络在全球范围内关于整个图像和图像中的所有对象的原因。

YOLO设计可实现端到端培训和实时速度，同时保持较高的平均精度。





2．核心功能的主要代码及说明。
卷积层
前向传播：
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

反向传播：

void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, state);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = state.workspace;
        float *c = l.weight_updates;

        float *im = state.input+i*l.c*l.h*l.w;

        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(state.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = state.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
        }
    }
}



池化层
前向传播：
void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}


反向传播：
void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}



根据实际情况,该部分还可以包含以下内容:
1．算法实现采用的工具和技术。
2．界面的设计。
3．数据结构或数据库的描述。
七、结果和结论
1．测试与运行效果。
2．性能或者效果与同类算法的比较。
八、结束语
对课题的总结
九、参考文献
最少12篇，不能全是教材。
参考文献一律放在文后，参考文献的书写格式要按国家标准GB-T 7714-2005规定。注明引用文献的方式通常有三种，文中注，正文中在引用的地方用括号说明文献的出处；脚注，正文中只在引用地方写一个脚注标号，在当页最下方以脚注方式按标号顺序说明文献出处；文末注，正文中在引用的地方标号（一般以出现的先后次序编号，编号以方括号括起，放在右上角，如[1]，[3～5]），然后在全文末单设“参考文献”一节，按标号顺序一一说明文献出处。不同学科可以有不同要求，但均要按国标注录。科技文献一般用文末注的方式，其注录格式为：
（1）科技书籍和专著注录格式：
作者．书名．版本（版本为第一版时可省略），出版地：出版社，出版日期．引用内容所在页码。
例如：
 [1] 高景德，王祥珩，李发海．交流电机及其系统的分析．北京：清华大学出版社，1993年8月．120～125
 [2] Tugomir Surina, Clyde Herrick. Semiconductor Electronics. Copyright 1964 by Holt, Rinehart and Winston, Inc., 120～250
（2）科技论文的注录格式：
作者．论文篇名．刊物名，出版年，卷（期）号：论文在刊物中的页码。
　　例如：
 [1] 李永东．PWM供电的异步电机电压定向矢量控制．电气传动，1991，4（1）：52～56
 [2] Colby R．S．A State Analysis of LCI fed Synchronous Motor Drive System.IEEE Trans, 1984, 21（4）

[1] Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi.You Only Look Once:Unified, Real-Time Object Detection.arXiv:1506.02640v5 ,2016,1～10
[2]Joseph Redmon, Ali Farhadi.YOLO9000:Better, Faster, Stronger.arXiv:1612.08242v1.2016,1～9
