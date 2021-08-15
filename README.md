# LIDN
Lightweight Color Image Demosaicking with Multi-Core Feature Extraction(VCIP2020)

# Abstract
Convolutional neural network (CNN)-based color image demosaicking methods have achieved great success recently. However, in many applications where the computation resource is highly limited, it is not practical to deploy large-scale networks. This paper proposes a lightweight CNN for color image demosaicking. Firstly, to effectively extract shallow features, a multi-core feature extraction module, which takes the Bayer sampling positions into consideration, is proposed. Secondly, by taking advantage of inter-channel correlation, an attention-aware fusion module is presented to efficiently reconstruct the full color image. Moreover, a feature enhancement module, which contains several cascading attention-aware enhancement blocks, is designed to further refine t he i nitial r econstructed i mage. To demonstrate the effectiveness of the proposed network, several state-of-the-art demosaicking methods are compared. Experimental results show that with the smallest number of parameters, the proposed network outperforms the other compared methods in terms of both objective and subjective qualities.


Detailed source code will be available soon.
