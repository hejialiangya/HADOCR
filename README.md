# HADOCR 
path = ../openrec/modeling/decoders/rctc_decoder.py

增加class FRMVBlock和class FRMHBlock

path = ../opendet/modeling/backbones/repvit.py

增加class SEModule(nn.Module) 

path = ../tools/data/ratio_sampler.py

DRS 根据文本图片的长宽比，通过动态比例采样调节，动态选择适合采样批次的输入尺寸，避免了在固定尺寸重采样时对文本图像的拉伸/压缩造成形变引起的失真

path = ../openrec/modeling/encoders/svtrv2.py

主要增强代码块

path = ../tools/visualize.py（新增）

根据result坐标和文本，将识别结果在图片上显示

# 数据集：

Synthetic datasets (MJ+ST)

Union14M-L-Filter

Common Benchmarks（evaluation）

LTB and OST

Chinese Benckmark

数据集来源
https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2

注意：复现时numpy2.x.x可能会报错，如出错请降版本至1.24.x

# dataset_structure
<details>
<summary>dataset structure</summary>
benchmark_bctr<br>
├── benchmark_bctr_test<br>
│  　　├── document_test<br>
│   　　├── handwriting_test<br>
│   　　├── scene_test<br>
│   　　└── web_test<br>
└── benchmark_bctr_train<br>
　　    ├── document_train<br>
 　　   ├── handwriting_train<br>
  　　  ├── scene_train<br>
  　 　└── web_train<br>
evaluation<br>
├── CUTE80<br>
├── IC13_857<br>
├── IC15_1811<br>
├── IIIT5k<br>
├── SVT<br>
└── SVTP<br>
filter_jsonl_mmocr0.x<br>
ltb<br>
OST<br>
├── heavy<br>
└── weak<br>
synth<br>
├── MJ<br>
│　　├── test<br>
│　　├── train<br>
│　　└── val<br>
└── ST<br>
test<br>
├── ArT<br>
├── COCOv1.4<br>
├── CUTE80<br>
├── IC13_1015<br>
├── IC13_1095<br>
├── IC13_857<br>
├── IC15_1811<br>
├── IC15_2077<br>
├── IIIT5k<br>
├── SVT<br>
├── SVTP<br>
└── Uber<br>
train_data_set  # individual dataset<br>
u14m<br>
├── artistic<br>
├── contextless<br>
├── curve<br>
├── general<br>
├── multi_oriented<br>
├── multi_words<br>
└── salient<br>
Union14M-L-LMDB-Filtered<br>
├── filter_train_challenging<br>
├── filter_train_easy<br>
├── filter_train_hard<br>
├── filter_train_medium<br>
└── filter_train_normal<br>
</details>
![img.png](dataset_str.png)
