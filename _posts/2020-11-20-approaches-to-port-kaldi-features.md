---
layout: post
title: Approaches to port Kaldi features
categories: []
published: true
comment_id: 1

---

[Kaldi](https://github.com/kaldi-asr/kaldi) is a software for Automatic Speech Recognition (ASR), which had enormous impact on both research of and production system.

Getting started with Kaldi is not easy. The code base is composed of a lot of submodules for a variety of tasks from feature extractoin to model training, and it depends on many third party libraries such as BLAS libraries, build tools and domain-specific libraries like OpenFST.

Nowadays, it has become a standard to use Deep Learning frameworks, such as PyTorch and Tensoflow, which come with an automatic differentiation, and utility functions for training/inference/deployment. Even [K2 FST](https://github.com/k2-fsa/k2) and [Lhotse Speech](https://github.com/lhotse-speech/lhotse), the successors of Kaldi, have adopted PyTorch as the standard framework for NN model authoring and data preparation.

These frameworks are accessible as Python package, so they are very easy to install, but we can not readily migrate away from Kaldi yet. Kaldi still provides values that are not available in these frameworks. Some of the features are available outside of the DL frameworks, i.e. as NumPy library. Some examples are [PyKaldi](https://github.com/pykaldi/pykaldi) and [Librosa](https://github.com/librosa/librosa). Of cource, typical DL frameworks are compatible with NumPy so one can always convert the array object to NumPy's NDArray and use these libraries, but couldn't it be simpler?

In [one of the Kaldi's town meeting](https://www.kaldi.dev/industry.html), I got an impression that people working in the industry are worried about being left behind in the transition from Kaldi to K2 FST. It is not like I have a comprehensive solution to them, but I think making it asy to move the recipe, model, preprocessing pipeline etc from Kaldi to DL framework with minimum fliction could be a way.

Talking about the transition, there is another exciting project. [GTN](https://github.com/facebookresearch/gtn) is a library which emerged independently at the same time as K2 FST with very similar goal, and it allows to train FST-based Language Model as part of PyTorch model.

So, the technological scene of Speech Recogniton shaped by K2 FST, Lhoste Speech, GTN and PyTorch, how can we close the loop so that everything is differential? And how can we make it accessible to from people in research to people in industry?

Okay, enough about introductory remarks. Let's move onto the fun part. How can we port the missing ingredients from Kaldi to PyTorch realm?

Looking at the existing libraries, I think there are two approaches;

1. Wrap Kaldi libraries (PyKaldi)
2. Re-implement functionalities in Python (torchaudio, Librosa)

Let's look into these approaches and think about the pros and cons.

<ol start="1">
  <li>Wrap Kaldi libraries</li>
</ol>

This approach implements a bridging code between array implementations of Python (NumPy NDArray in the case of PyKaldi, Tensor for PyTorch) and Kaldi's Vector/Matrix class, then make the interface available in Python.  

- **Pros**
   - Can reuse the Kaldi's source code as-is.
      - Easy to incorporate the changes in upstream.

- **Cons**
   - If statically built, the binary contains another BLAS library.
   - If dynamically built, redistribution of the binary is not straightforward.  
     (Loading C++ extension module in Python which depends on another one could be tricky)
   - If using an interface language (like SWIG or CLIF), the build process is more complicated.
   - Could incur cost to copy data to Kaldi's Vector/Matrix format.
   - Cannot leverage the PyTorch features.
       - CUDA
       - autograd

<ol start="2">
  <li>Re-implement functionalities in Python</li>
</ol>

This approach implements the same logic in Python.

- **Pros**
   - No new dependency.
   - No build process.
   - Can leverage the PyTorch features.
      - CUDA (though the performance improvement depends on the algorithm)
      - autograd (if re-written correctly)
- **Cons**
   - Does not scale well for porting multiple features.
      - No systematic way to translate C++ code into Pyhton code.
      - One has to understand all the logics of each Kaldi feature written in C++.
   - No control on low level implementatoin.

After spending some time trying the approach 1, I realized that there is another way to port Kaldi features.

<ol start="3">
  <li>Re-implement Kaldi's Vector/Matrix class with PyTorch's C++ API.</li>
</ol>

Kaldi's Vector/Matrix class uses a BLAS library (such as OpenBLAS, ATLAS, Intel MKL etc). PyTorch's Python distribution comes with one, so if we write an interface that mimics Kaldi's Vector/Matrix class on top of PyTorch's Tensor class, the rest of the Kaldi code base should compile.  
Of cource this approach is not necessarily efficient, or compatible with CUDA, but if we can write a numerically consistant implementation of Kaldi's Vector/Matrix class, we can mostly copy-paste the features built on top of it and we get the feature compiled and it should be numerically consistant, then we can modify the implementation to improve its performance.

- **Pros**
   - Can use the Kaldi's source cod of high level feature implementation with minimum modification.
   - Scalable  
      - once the Vector/Matrix classes are implemented, it is easy to add features.
   - No new dependency.
   - Can leverage the PyTorch features.
      - CUDA
      - autograd
   - Can apply low level optimization
      - Potential to prarllerize / vectorize the implementation for further speed up
   - No cost to copy the memory hold by Tensor to Vector/Marix classes
- **Cons**
   - Have to implement fundations (Vector/Matrix/utilities)
   - Build process

The approach 3 is somewhat a hybrid of the previous approaches. When it comes to porting a library, it is very important to quickly get to the point where the ported implementation produces the same result as the original library. I feel that the approach 3 is the simplest to achieve this.
