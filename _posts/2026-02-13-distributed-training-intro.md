---
layout: post
title: "Understanding Distributed Model Training at Scale - From Academic Experiments to Production Systems"
date: 2026-02-15 12:00:00
description: Understanding what actually changes when training moves from a handful of GPUs to large-scale distributed systems.
tags: datasets machine-learning distributed-systems
categories: blog
---

A central technical aspect driving the rapid changes and impact of AI across domains is the increasing availability of compute. Today’s frontier models — whether large language models or data-driven climate simulations — are trained on clusters spanning hundreds or thousands of GPUs.

As a PhD student working in applied earth observation and climate research, and exploring roles in research institutes and industry, I have become increasingly aware that many of these roles require a skill set that is not typically developed in academia: a deep understanding of distributed model training at scale.

In many academic environments, computational resources are constrained. Training might happen on 1–8 GPUs, perhaps 16 if one is fortunate. Code is often written to validate a hypothesis, demonstrate a result, or support a publication. It is rarely written with long-term maintainability, fault tolerance, or infrastructure constraints in mind. In contrast, production systems operate under very different assumptions.

When training scales beyond a handful of GPUs, the nature of the problem changes. At some point, training is no longer primarily a modeling problem, it becomes a distributed systems problem shaped by communication overhead, memory and hardware constraints, and failure modes.

## A Motivating Example

Consider a large collection of optical and multispectral satellite imagery that could be used to monitoring deforestation, wildfires, flooding, or urbanization. The data volume may span terabytes or petabytes. It is not feasible to simply download the full dataset to a local cluster. Even if that were possible, training a sufficiently expressive model on a small number of GPUs would be prohibitively slow, or might not fit into memory at all.

You might encounter:

- Out-of-memory (OOM) errors even with modest batch sizes  
- Unacceptable training times  
- Data loading pipelines that cannot saturate the GPUs  
- Synchronization overhead dominating compute time  

At that scale, questions arise that rarely appear in small-scale academic experiments:

- How do we distribute the model across devices?
- How do we minimize communication overhead?
- How do we design data pipelines that keep hundreds of GPUs busy?
- What happens when a node fails mid-training?


## What Actually Changes When You Scale?

In introductory material, distributed training is often framed in terms of two approaches:

- **Data parallelism**: Replicating the model across devices and synchronizing gradients during backpropagation.
- **Model parallelism**: Partitioning the model across multiple devices when it cannot fit into a single GPU.

In practice, modern large-scale systems combine multiple forms of parallelism simultaneously through various strategies


## Outline

Rather than attempting to cover everything at once, I want to focus on three core dimensions that fundamentally change once we move beyond a handful of GPUs: scaling behavior, parallelism strategies, and dataloading I/O.


### **Part 1: What Changes When You Scale?**

We begin by examining the conceptual shift from 1 GPU to 8 GPUs to 128+ GPUs. At small scale, distributed training often feels like a simple extension of single-GPU training. However, beyond a certain point, naive scaling stops working. The goal of this part is to develop intuition for where and why scaling breaks down.


### **Part 2: Parallelism Strategies**

Next, we take a closer look at the different ways computation itself can be distributed.

This includes:
- Data parallelism and gradient synchronization  
- Model and tensor parallelism  
- Pipeline parallelism and micro-batching  
- Hybrid strategies used in large-scale training  

Here, the focus will also be on understanding their trade-offs and get an intuitive understanding of how they work and fit together

### **Part 3: The Data Problem: I/O as a Bottleneck**

Returning to large-scale satellite and climate datasets:

When data volumes grow to terabytes or petabytes, input pipelines often become the limiting factor. GPUs may sit idle not because the model is slow, but because the data cannot be delivered fast enough.

Here, I want to cover:
- Map-style vs iterable-style datasets  
- Streaming and sequential access patterns  
- Shuffle buffers for approximate global randomization  
- Asynchronous workers and prefetch queues  
- Data locality and cloud-native storage  

The objective of this part is to understand how to design data pipelines that reduce latency, hide I/O bottlenecks, and maximize GPU utilization.

## Conclusion

This series is an attempt to document my learning process toward understanding large-scale distributed systems.

The goal is not to provide definitive answers, but to develop a clearer understanding of what training at scale actually entails, especially in domains such as earth observation and climate science, where data volume and computational demand are intrinsic to the problem.

I aim to combine conceptual explanations with illustrations and practical code snippets that make the underlying mechanics more approachable.
