# COMP6258-Reproducibility-Challenge

This repository is a reproduction of [Model LEGO: Creating Models Like Disassembling and Assembling Building Blocks](https://github.com/jiaconghu/Model-LEGO) for research/learning purposes.
All credits to the original authors.

## Environment

+ Python Version: 3.9
+ GPU: NVIDIA RTX 3090

## Modifications 

- Modify .py files in core/
- Add preprocess_cifar_10.py and preprocess_cifar_100.py for dataset preprocessing.
- Outputs are stored in outputs_resnet50_cifar100/ and outputs_vgg16_cifar10/

## Discussion

During the process of running and debugging the code of the paper, we found that there are still some problems worth improving related to the reproducibility. Firstly, there is incompatibility of package path in different operating systems. In some files, due to the lack of flexible management of module paths, directly running the program will give an error. To fix this, we manually added the \texttt{sys.path.append(str(Path(\_\_file\_\_).resolve().parents[1]))} statement to make sure the program correctly identifies module paths inside the project. Secondly, the project repository is unclear about the usage instructions of the dataset. Specifically, it is not clear how the sample labels are stored, such as whether the label information is stored in a file or if the samples with the same label are divided into corresponding subdirectories. This causes the program to fail to run. Finally, some of the core code files were not logical enough. For example, the \texttt{relevant\_feature\_identified.py} file does not check whether the path to save the results exists before running the long task of identifying features. If the target path does not exist, the program will not automatically create the directory. This will give a running error and the running results can not be saved, resulting in the loss of time and efficiency. The above questions show that the project still has room for improvement in terms of code robustness and user friendliness.
